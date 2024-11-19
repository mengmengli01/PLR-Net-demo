import torch
import math

def make_optimizer(cfg, model):

    params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr=cfg.SOLVER.BASE_LR

        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


    if cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(params,
                                    cfg.SOLVER.BASE_LR,
                                    momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER == 'ADAM' or 'ADAMcos':
        optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                     amsgrad=cfg.SOLVER.AMSGRAD)
    elif cfg.SOLVER.OPTIMIZER == 'ADAMW':
        optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                     amsgrad=cfg.SOLVER.AMSGRAD)
    else:
        raise NotImplementedError()
    return optimizer

def make_lr_scheduler(cfg, optimizer):

   return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.SOLVER.MAX_EPOCH, eta_min=1e-5)