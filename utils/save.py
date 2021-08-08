"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Modified from UNITER
(https://github.com/ChenRocks/UNITER)

saving utilities
"""
import json
import os
from os.path import abspath, dirname, exists, join, realpath
import subprocess
from apex import amp
import torch

from utils.logger import LOGGER
from utils.basic_utils import save_json, make_zipfile, load_json


def save_training_meta(args):
    # Comment out, since rank is not saved to args. Safeguard save_training_meta already in training scripts.
    # if args.rank > 0:
    #    return

    # args is an EasyDict object, treat it the same as a normal dict
    os.makedirs(join(args.output_dir, 'log'), exist_ok=True)
    os.makedirs(join(args.output_dir, 'ckpt'), exist_ok=True)

    # training args
    save_args_path = join(args.output_dir, 'log', 'hps.json')
    save_json(vars(args), save_args_path, save_pretty=True)

    # model args
    model_config = load_json(args.model_config)
    save_model_config_path = join(args.output_dir, 'log', 'model_config.json')
    save_json(model_config, save_model_config_path, save_pretty=True)
    # git info
    try:
        LOGGER.info("Waiting on git info....")
        c = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                           timeout=10, stdout=subprocess.PIPE)
        git_branch_name = c.stdout.decode().strip()
        LOGGER.info("Git branch: %s", git_branch_name)
        c = subprocess.run(["git", "rev-parse", "HEAD"],
                           timeout=10, stdout=subprocess.PIPE)
        git_sha = c.stdout.decode().strip()
        LOGGER.info("Git SHA: %s", git_sha)
        git_dir = abspath(dirname(__file__))
        git_status = subprocess.check_output(
            ['git', 'status', '--short'],
            cwd=git_dir, universal_newlines=True).strip()
        with open(join(args.output_dir, 'log', 'git_info.json'),
                  'w') as writer:
            json.dump({'branch': git_branch_name,
                       'is_dirty': bool(git_status),
                       'status': git_status,
                       'sha': git_sha},
                      writer, indent=4)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        LOGGER.exception(e)
        LOGGER.warn("Git info not found. Saving code into zip instead...")
        # save a copy of the codebase.
        # !!!Do not store heavy file in your codebase when using it.
        code_dir = dirname(dirname(realpath(__file__)))
        code_zip_filename = os.path.join(args.output_dir, "code.zip")
        LOGGER.info(f"Saving code from {code_dir} to {code_zip_filename}...")
        make_zipfile(code_dir, code_zip_filename,
                     enclosing_dir="code",
                     exclude_dirs_substring="results",
                     exclude_dirs=["results", "debug_results", "__pycache__"],
                     exclude_extensions=[".pyc", ".ipynb", ".swap"])
        LOGGER.info("Saving code done.")


def _to_cuda(state):
    """ usually load from cpu checkpoint but need to load to cuda """
    if isinstance(state, torch.Tensor):
        ret = state.cuda()  # assume propoerly set py torch.cuda.set_device
        if 'Half' in state.type():
            ret = ret.float()  # apex O2 requires it
        return ret
    elif isinstance(state, list):
        new_state = [_to_cuda(t) for t in state]
    elif isinstance(state, tuple):
        new_state = tuple(_to_cuda(t) for t in state)
    elif isinstance(state, dict):
        new_state = {n: _to_cuda(t) for n, t in state.items()}
    else:
        return state
    return new_state


def _to_cpu(state):
    """ store in cpu to avoid GPU0 device, fp16 to save space """
    if isinstance(state, torch.Tensor):
        ret = state.cpu()
        if 'Float' in state.type():
            ret = ret.half()
        return ret
    elif isinstance(state, list):
        new_state = [_to_cpu(t) for t in state]
    elif isinstance(state, tuple):
        new_state = tuple(_to_cpu(t) for t in state)
    elif isinstance(state, dict):
        new_state = {n: _to_cpu(t) for n, t in state.items()}
    else:
        return state
    return new_state


class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt'):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix

    def save(self, model, step, optimizer=None):
        output_model_file = join(self.output_dir,
                                 f"{self.prefix}_{step}.{self.suffix}")
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.state_dict().items()}
        for k, v in state_dict.items():
            if 'word_embeddings.weight' in k or 'decoder.weight' in k:
                assert v.size(0) % 8 == 0
                state_dict['vocab_padded'] = True
                break
        else:
            state_dict['vocab_padded'] = False
        torch.save(state_dict, output_model_file)
        if optimizer is not None:
            dump = {'step': step, 'optimizer': optimizer.state_dict()}
            torch.save(dump, f'{self.output_dir}/train_state_{step}.pt')


class TrainingRestorer(object):
    def __init__(self, opts, model, optimizer):
        if exists(f'{opts.output_dir}/log/hps.json'):
            restore_opts = json.load(open(
                f'{opts.output_dir}/log/hps.json', 'r'))
            assert vars(opts) == restore_opts
        # keep 2 checkpoints in case of corrupted
        self.save_path = f'{opts.output_dir}/restore.pt'
        self.backup_path = f'{opts.output_dir}/restore_backup.pt'
        self.model = model
        self.optimizer = optimizer
        self.save_steps = opts.save_steps
        self.amp = opts.fp16
        if exists(self.save_path) or exists(self.backup_path):
            LOGGER.info('found previous checkpoint. try to resume...')
            self.restore(opts)
        else:
            self.global_step = 0

    def step(self):
        self.global_step += 1
        if self.global_step % self.save_steps == 0:
            self.save()

    def save(self):
        checkpoint = {'global_step': self.global_step,
                      'model_state_dict': _to_cpu(self.model.state_dict()),
                      'optim_state_dict': _to_cpu(self.optimizer.state_dict())}
        if self.amp:
            checkpoint['amp_state_dict'] = amp.state_dict()
        if exists(self.save_path):
            os.rename(self.save_path, self.backup_path)
        torch.save(checkpoint, self.save_path)

    def restore(self, opts):
        try:
            checkpoint = torch.load(self.save_path)
        except Exception:
            checkpoint = torch.load(self.backup_path)
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(_to_cuda(checkpoint['model_state_dict']))
        self.optimizer.load_state_dict(
            _to_cuda(checkpoint['optim_state_dict']))
        if self.amp:
            amp.load_state_dict(checkpoint['amp_state_dict'])
        LOGGER.info(f'resume training from step {self.global_step}')
