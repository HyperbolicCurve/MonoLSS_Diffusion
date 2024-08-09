import os
import torch
import numpy as np
from lib.helpers.save_helper import get_checkpoint_state, save_checkpoint, load_checkpoint
from lib.losses.loss_function import LSS_Loss, Hierarchical_Task_Learning
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
from tqdm import tqdm
from tools import eval
from torch.cuda.amp import GradScaler, autocast  # 用于混合精度训练

class Trainer(object):
    def __init__(self, cfg, model, optimizer, train_loader, test_loader, lr_scheduler, warmup_lr_scheduler, logger, local_rank):
        self.cfg_train = cfg['trainer']
        self.cfg_test = cfg['tester']
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.local_rank = local_rank
        self.distributed = local_rank is not None
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and self.distributed else "cuda" if torch.cuda.is_available() else "cpu")
        self.class_name = test_loader.dataset.class_name
        self.label_dir = cfg['dataset']['label_dir']
        self.eval_cls = cfg['dataset']['eval_cls']
        self.scaler = GradScaler()  # 混合精度训练的GradScaler

        if self.cfg_train.get('resume_model', None):
            assert os.path.exists(self.cfg_train['resume_model'])
            self.epoch = load_checkpoint(self.model, self.optimizer, self.cfg_train['resume_model'], self.logger, map_location=self.device)
            self.lr_scheduler.last_epoch = self.epoch - 1

        self.model = self.model.to(self.device)
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    def train(self):
        start_epoch = self.epoch
        ei_loss = self.compute_e0_loss()
        loss_weightor = Hierarchical_Task_Learning(ei_loss)

        if not self.distributed or self.local_rank == 0:
            print("Epochs: ", self.cfg_train['max_epoch'])
            print("Trained Parameters(M): ", sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6)
            print("Total Parameters(M): ", sum(p.numel() for p in self.model.parameters()) / 1e6)

        for epoch in tqdm(range(start_epoch, self.cfg_train['max_epoch']), desc="Training Epochs"):
            if not self.distributed or self.local_rank == 0:
                self.logger.info('------ TRAIN EPOCH %03d ------' % (epoch + 1))
                if self.warmup_lr_scheduler is not None and epoch < 5:
                    self.logger.info('Learning Rate: %f' % self.warmup_lr_scheduler.get_lr()[0])
                else:
                    self.logger.info('Learning Rate: %f' % self.lr_scheduler.get_lr()[0])

            np.random.seed(np.random.get_state()[1][0] + epoch)
            loss_weights = loss_weightor.compute_weight(ei_loss, self.epoch)

            log_str = 'Weights: '
            for key in sorted(loss_weights.keys()):
                log_str += ' %s:%.4f,' % (key[:-4], loss_weights[key])
            if not self.distributed or self.local_rank == 0:
                self.logger.info(log_str)

            try:
                ei_loss = self.train_one_epoch(loss_weights)
            except Exception as e:
                if not self.distributed or self.local_rank == 0:
                    self.logger.error(f"Error during training epoch {epoch + 1} on rank {self.local_rank}: {e}")
                raise e

            self.epoch += 1

            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            if ((self.epoch % self.cfg_train['eval_frequency']) == 0 and self.epoch >= self.cfg_train['eval_start']):
                if not self.distributed or self.local_rank == 0:
                    self.logger.info('------ EVAL EPOCH %03d ------' % self.epoch)
                try:
                    Car_res, results, disp_dict = self.eval_one_epoch()
                    if not self.distributed or self.local_rank == 0:
                        self.logger.info(str(Car_res))
                except Exception as e:
                    if not self.distributed or self.local_rank == 0:
                        self.logger.error(f"Error during evaluation epoch {self.epoch} on rank {self.local_rank}: {e}")
                    raise e

            if ((self.epoch % self.cfg_train['save_frequency']) == 0 and self.epoch >= self.cfg_train['eval_start'] and (not self.distributed or self.local_rank == 0)):
                os.makedirs(self.cfg_train['log_dir'] + '/checkpoints', exist_ok=True)
                ckpt_name = os.path.join(self.cfg_train['log_dir'] + '/checkpoints', 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name, self.logger)

        return None

    def train_one_epoch(self, loss_weights=None):
        self.model.train()

        disp_dict = {}
        stat_dict = {}
        for batch_idx, (inputs, calibs, coord_ranges, targets, info) in enumerate(self.train_loader):
            if self.distributed:
                self.train_loader.sampler.set_epoch(batch_idx)
            inputs, calibs, coord_ranges, targets = self._move_to_device(inputs, calibs, coord_ranges, targets)

            self.optimizer.zero_grad()

            with autocast():  # 混合精度上下文管理器
                criterion = LSS_Loss(self.epoch)
                outputs = self.model(inputs, coord_ranges, calibs, targets)

                # 确保 outputs 中的每个张量都在同一个设备上
                outputs = {key: outputs[key].to(self.device) for key in outputs.keys()}

                total_loss, loss_terms = criterion(outputs, targets)

                if loss_weights is not None:
                    total_loss = torch.zeros(1, device=self.device)
                    for key in loss_weights.keys():
                        total_loss += loss_weights[key].detach() * loss_terms[key]

            self.scaler.scale(total_loss).backward()  # 混合精度训练：缩放损失
            self.scaler.step(self.optimizer)
            self.scaler.update()

            trained_batch = batch_idx + 1

            for key in loss_terms.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0
                stat_dict[key] += loss_terms[key].detach() if isinstance(loss_terms[key], torch.Tensor) else loss_terms[key]

            for key in loss_terms.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                disp_dict[key] += loss_terms[key].detach() if isinstance(loss_terms[key], torch.Tensor) else loss_terms[key]

            if trained_batch % self.cfg_train['disp_frequency'] == 0 and (not self.distributed or self.local_rank == 0):
                log_str = 'BATCH[%04d/%04d]' % (trained_batch, len(self.train_loader))
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / self.cfg_train['disp_frequency']
                    log_str += ' %s:%.4f,' % (key, disp_dict[key])
                    disp_dict[key] = 0
                self.logger.info(log_str)

        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch

        # 在每个 epoch 结束后清理缓存
        torch.cuda.empty_cache()

        return stat_dict

    def eval_one_epoch(self):
        self.model.eval()

        results = {}
        disp_dict = {}
        progress_bar = tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
        with torch.no_grad():
            for batch_idx, (inputs, calibs, coord_ranges, _, info) in enumerate(self.test_loader):
                inputs, calibs, coord_ranges, _ = self._move_to_device(inputs, calibs, coord_ranges)

                with autocast():  # 混合精度上下文管理器
                    outputs = self.model(inputs, coord_ranges, calibs, K=50, mode='val')

                dets = extract_dets_from_outputs(outputs, K=50)
                dets = dets.detach().cpu().numpy()

                calibs = [self.test_loader.dataset.get_calib(index) for index in info['img_id']]
                info = {key: val.detach().cpu().numpy() for key, val in info.items()}
                cls_mean_size = self.test_loader.dataset.cls_mean_size
                dets = decode_detections(dets=dets, info=info, calibs=calibs, cls_mean_size=cls_mean_size, threshold=self.cfg_test['threshold'])
                results.update(dets)
                progress_bar.update()
            progress_bar.close()
        out_dir = os.path.join(self.cfg_train['out_dir'], 'EPOCH_' + str(self.epoch))
        self.save_results(results, out_dir)
        Car_res = eval.eval_from_scrach(self.label_dir, os.path.join(out_dir, 'data'), self.eval_cls, ap_mode=40)
        return Car_res, results, disp_dict
z
    def _move_to_device(self, inputs, calibs, coord_ranges, targets=None):
        if type(inputs) != dict:
            inputs = inputs.to(self.device)
        else:
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)
        calibs = calibs.to(self.device)
        coord_ranges = coord_ranges.to(self.device)
        if targets:
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)
        return inputs, calibs, coord_ranges, targets
