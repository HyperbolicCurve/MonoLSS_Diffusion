import os
import sys
from datetime import datetime
import yaml
import logging
import argparse
import torch
import torch.distributed as dist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester

parser = argparse.ArgumentParser(description='implementation of MonoLSS')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--config', type=str, default='lib/configs/dla34_bs16_lr0.001.yaml')
parser.add_argument("--exp", type=str, default=None)
args = parser.parse_args()

local_rank = int(os.environ.get("LOCAL_RANK", -1))
distributed = local_rank != -1

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def main():
    print(f"Local rank: {local_rank}")
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda")

    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    if distributed:
        world_size = dist.get_world_size()
        batch_size = cfg['dataset']['batch_size']
        per_process_batch_size = batch_size // world_size
        cfg['dataset']['batch_size'] = per_process_batch_size

        base_learning_rate = cfg['optimizer']['lr']
        adjusted_learning_rate = base_learning_rate * (per_process_batch_size / batch_size)
        cfg['optimizer']['lr'] = adjusted_learning_rate

    os.makedirs(cfg['trainer']['log_dir'], exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg['trainer']['log_dir'] = os.path.join(cfg['trainer']['log_dir'], args.exp, timestamp)
    log_file = os.path.join(cfg['trainer']['log_dir'], 'train.log')

    if not os.path.exists(cfg['trainer']['log_dir']):
        os.makedirs(cfg['trainer']['log_dir'])

    logger = create_logger(log_file)
    import shutil
    if not args.evaluate:
        if not args.test and (not distributed or local_rank == 0):
            if os.path.exists(os.path.join(cfg['trainer']['log_dir'], 'lib/')):
                shutil.rmtree(os.path.join(cfg['trainer']['log_dir'], 'lib/'))
            shutil.copytree('./lib', os.path.join(cfg['trainer']['log_dir'], 'lib/'))

    train_loader, val_loader, test_loader = build_dataloader(cfg['dataset'], distributed)

    model = build_model(cfg['model'], train_loader.dataset.cls_mean_size)
    model = model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)

    if args.evaluate:
        tester = Tester(cfg['tester'], cfg['dataset'], model, val_loader, logger)
        tester.test()
        return

    if args.test:
        tester = Tester(cfg['tester'], cfg['dataset'], model, test_loader, logger)
        tester.test()
        return

    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg,
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      local_rank=local_rank if distributed else None)
    try:
        trainer.train()
    except Exception as e:
        if not distributed or local_rank == 0:
            logger.error(f"Error on rank {local_rank}: {e}")
        raise e


if __name__ == '__main__':
    main()
