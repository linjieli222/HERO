"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

some functions are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
import torch
from torch import nn
import logging
logger = logging.getLogger(__name__)


def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters)
        to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(
        new_size[1], new_size[0], bias=layer.bias is not None).to(
            layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def mask_logits(target, mask, eps=-1e4):
    return target * mask + (1 - mask) * eps


def load_partial_checkpoint(checkpoint, n_layers, skip_layers=True):
    if skip_layers:
        new_checkpoint = {}
        gap = int(12/n_layers)
        prefix = "roberta.encoder.layer."
        layer_range = {str(l): str(i) for i, l in enumerate(
            list(range(gap-1, 12, gap)))}
        for k, v in checkpoint.items():
            if prefix in k:
                layer_name = k.split(".")
                layer_num = layer_name[3]
                if layer_num in layer_range:
                    layer_name[3] = layer_range[layer_num]
                    new_layer_name = ".".join(layer_name)
                    new_checkpoint[new_layer_name] = v
            else:
                new_checkpoint[k] = v
    else:
        new_checkpoint = checkpoint
    return new_checkpoint


def load_pretrained_weight(model, state_dict):
    # Load from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = ({} if metadata is None
                          else metadata.get(prefix[:-1], {}))
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys,
            unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    start_prefix = ''
    if not hasattr(model, 'roberta') and\
            any(s.startswith('roberta.') for s in state_dict.keys()):
        start_prefix = 'roberta.'

    load(model, prefix=start_prefix)
    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from "
                    "pretrained model: {}".format(
                        model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in "
                    "{}: {}".format(
                        model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for '
                            '{}:\n\t{}'.format(
                                model.__class__.__name__,
                                "\n\t".join(error_msgs)))
    return model


def pad_tensor_to_mul(tensor, dim=0, mul=8):
    """ pad tensor to multiples (8 for tensor cores) """
    t_size = list(tensor.size())
    n_pad = mul - t_size[dim] % mul
    if n_pad == mul:
        n_pad = 0
        padded_tensor = tensor
    else:
        t_size[dim] = n_pad
        pad = torch.zeros(*t_size, dtype=tensor.dtype, device=tensor.device)
        padded_tensor = torch.cat([tensor, pad], dim=dim)
    return padded_tensor, n_pad
