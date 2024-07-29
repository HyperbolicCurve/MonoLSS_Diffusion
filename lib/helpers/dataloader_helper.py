from torch.utils.data import DataLoader
from lib.datasets.kitti import KITTI
import torch

def build_dataloader(cfg, distributed):
    # --------------  build kitti dataset ----------------
    if cfg['type'] == 'kitti':
        train_set = KITTI(root_dir=cfg['root_dir'], split='train', cfg=cfg)
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        else:
            train_sampler = None
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=cfg['num_workers'],
                                  pin_memory=True,
                                  drop_last=True,
                                  sampler=train_sampler,
                                  shuffle=(train_sampler is None))

        val_set = KITTI(root_dir=cfg['root_dir'], split='val', cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                batch_size=cfg['batch_size'],
                                num_workers=cfg['num_workers'],
                                shuffle=False,
                                pin_memory=True,
                                drop_last=cfg['drop_last_val'])

        test_set = KITTI(root_dir=cfg['root_dir'], split='test', cfg=cfg)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        return train_loader, val_loader, test_loader


    elif cfg['type'] == 'waymo':
        train_set = Waymo(root_dir=cfg['root_dir'], split='train', cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=cfg['num_workers'],
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        test_set = Waymo(root_dir=cfg['root_dir'], split='test', cfg=cfg)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=cfg['num_workers'],
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False)
        return train_loader, train_loader, test_loader

    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])
