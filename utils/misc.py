"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Copied from UNITER
(https://github.com/ChenRocks/UNITER)

Misc utilities
"""
import random

import torch
import numpy as np

from utils.logger import LOGGER


class Struct(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


class NoOp(object):
    """ useful for distributed training No-Ops """
    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def set_dropout(model, drop_p):
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != drop_p:
                module.p = drop_p
                LOGGER.info(f'{name} set to {drop_p}')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
