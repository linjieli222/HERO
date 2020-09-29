"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Copied from UNITER
(https://github.com/ChenRocks/UNITER)

Misc lr helper
"""
from torch.optim import Adam, Adamax
from .adamw import AdamW


def build_optimizer(model, opts):
    # Prepare optimizer
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 'v_encoder' in n and p.requires_grad]
    # top layer has larger learning rate
    param_top = [(n, p) for n, p in model.named_parameters()
                 if 'v_encoder' not in n and p.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
            'lr': opts.lr_mul*opts.learning_rate,
            'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
            'lr': opts.lr_mul*opts.learning_rate,
            'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
            'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer
