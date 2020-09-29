"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training Video QA
"""
from collections import defaultdict
import os
from os.path import exists, join
from time import time

import torch
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
from apex import amp
from horovod import torch as hvd


from data import QaQueryTokLmdb, PrefetchLoader, MetaLoader
from load_data import (
    get_video_ids, load_video_sub_dataset, build_downstream_dataloaders)
from model.videoQA import HeroForVideoQA
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta, TrainingRestorer
from utils.misc import NoOp, set_dropout, set_random_seed
from utils.const import VFEAT_DIM, MAX_FRM_SEQ_LEN
from utils.basic_utils import save_json
from config.config import shared_configs
from eval_videoQA import validate_videoQA


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    opts.n_gpu = n_gpu
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))
    if hvd.rank() != 0:
        LOGGER.disabled = True
    set_random_seed(opts.seed)

    # train_examples = None
    LOGGER.info(f"Loading the whole video dataset {opts.sub_txt_db}, "
                f"{opts.vfeat_db}")
    video_db = load_video_sub_dataset(
        opts.vfeat_db, opts.sub_txt_db, opts.vfeat_interval, opts)

    # data loaders
    # train
    LOGGER.info(f"Loading the train QA dataset {opts.train_query_txt_db}")
    video_ids = get_video_ids(opts.train_query_txt_db)
    train_q_txt_db = QaQueryTokLmdb(opts.train_query_txt_db,
                                    opts.max_txt_len)
    train_dataloaders = build_downstream_dataloaders(
        [opts.task], video_db, video_ids,
        True, opts, q_txt_db=train_q_txt_db,
        shuffle=True)
    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

    # val
    LOGGER.info(f"Loading the val QA dataset {opts.val_query_txt_db}")
    video_ids = get_video_ids(opts.val_query_txt_db)
    val_q_txt_db = QaQueryTokLmdb(opts.val_query_txt_db, -1)
    val_dataloaders = build_downstream_dataloaders(
        [opts.task], video_db, video_ids,
        False, opts, q_txt_db=val_q_txt_db)
    if opts.test_query_txt_db:
        LOGGER.info(f"Loading the test QA dataset {opts.test_query_txt_db}")
        video_ids = get_video_ids(opts.test_query_txt_db)
        test_q_txt_db = QaQueryTokLmdb(opts.test_query_txt_db, -1)
        test_dataloaders = build_downstream_dataloaders(
            [opts.task], video_db, video_ids,
            False, opts, q_txt_db=test_q_txt_db)

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}
    img_pos_embed_weight_key = "v_encoder.f_encoder.img_embeddings" +\
        ".position_embeddings.weight"
    if img_pos_embed_weight_key in checkpoint:
        max_frm_seq_len = len(checkpoint[img_pos_embed_weight_key])
    else:
        max_frm_seq_len = MAX_FRM_SEQ_LEN

    model = HeroForVideoQA.from_pretrained(
            opts.model_config,
            state_dict=checkpoint,
            vfeat_dim=VFEAT_DIM,
            max_frm_seq_len=max_frm_seq_len)

    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}
    model, optimizer = amp.initialize(model, optimizer,
                                      num_losses=len(task2scaler),
                                      enabled=opts.fp16, opt_level='O2')
    restorer = TrainingRestorer(opts, model, optimizer)
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step
    if hvd.rank() == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        if not exists(join(opts.output_dir, 'results')):
            # store tvqa predictions
            os.makedirs(join(opts.output_dir, 'results'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()
        restorer = NoOp()

    if global_step > 0:
        pbar.update(global_step)
    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    task2loss = {task: RunningMeter(f'loss/{task}')
                 for task in train_dataloaders.keys()}

    for obj in (f'{opts.task}_qa', f'{opts.task}_st_ed'):
        task2loss[obj] = RunningMeter(f'loss/{obj}')

    model.train()
    n_examples = defaultdict(int)
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    if global_step == 0:
        optimizer.step()
    for step, (task, batch) in enumerate(meta_loader):
        n_examples[task] += opts.train_batch_size

        loss = model(batch, task=task, compute_loss=True)

        loss_qa, loss_st_ed = loss
        loss = loss_qa + opts.lw_st_ed * loss_st_ed
        for n, ls in (('st_ed', loss_st_ed),
                      ('qa', loss_qa)):
            ls = ls.item()
            task2loss[f'{task}_{n}'](ls)

        loss = loss.mean()
        task2loss[task](loss.item())

        delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale,
                            loss_id=task2scaler[task]) as scaled_loss:
            scaled_loss.backward()
            if not delay_unscale:
                # gather gradients from every processes
                # do this before unscaling to make sure every process uses
                # the same gradient scale
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

        if (step + 1) % opts.gradient_accumulation_steps == 0:
            global_step += 1

            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, opts)
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 0 or i == 1:
                    param_group['lr'] = lr_this_step * opts.lr_mul
                elif i == 2 or i == 3:
                    param_group['lr'] = lr_this_step
                else:
                    raise ValueError()
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            TB_LOGGER.log_scaler_dict({temp_loss.name: temp_loss.val
                                       for temp_loss in task2loss.values()
                                       if temp_loss.val is not None})
            TB_LOGGER.step()

            # update model params
            if opts.grad_norm != -1:
                grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                            opts.grad_norm)
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            optimizer.step()
            optimizer.zero_grad()
            restorer.step()
            pbar.update(1)

            if global_step % 100 == 0:
                # monitor training throughput
                LOGGER.info('-------------------------------------------')
                LOGGER.info(f'Step {global_step}:')
                for t in train_dataloaders.keys():
                    tot_ex = sum(all_gather_list(n_examples[t]))
                    ex_per_sec = int(tot_ex / (time()-start))
                    LOGGER.info(f'{t}: {tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar(f'perf/{t}_ex_per_s', ex_per_sec,
                                         global_step)

            if global_step % opts.valid_steps == 0:
                LOGGER.info('===========================================')
                LOGGER.info(f"Step {global_step}: start running validation")
                validate(model, val_dataloaders, "val",
                         opts, global_step=global_step)
                if opts.test_query_txt_db:
                    validate(model, test_dataloaders, "test",
                             opts, global_step=global_step)
                LOGGER.info('===========================================')
                model_saver.save(model, global_step)
        if global_step >= opts.num_train_steps:
            break

    LOGGER.info('===========================================')
    if global_step % opts.valid_steps != 0:
        LOGGER.info('===========================================')
        LOGGER.info(f"Step {global_step}: start running validation")
        validate(model, val_dataloaders, "val",
                 opts, global_step=global_step)
        if opts.test_query_txt_db:
            validate(model, test_dataloaders, "test",
                     opts, global_step=global_step)
        LOGGER.info('===========================================')
    model_saver.save(model, f'{global_step}_final')


def validate(model, val_dataloaders, split, opts, global_step=0):
    model.eval()
    task = opts.task
    loader = val_dataloaders[task]
    LOGGER.info(f"validate on {task} task, split {split}")
    val_log, results, _ = validate_videoQA(
        model, loader, task=task, split=split,
        save_logits=False)
    save_json(
        results,
        f'{opts.output_dir}/results/'
        f'{split}_results_{global_step}'
        f'_rank{hvd.rank()}.json')
    val_log = {f'{task}_{split}_{k}': v for k, v in val_log.items()}
    TB_LOGGER.log_scaler_dict(
        {f'{split}_{task}/{k}': v for k, v in val_log.items()})
    model.train()


if __name__ == "__main__":
    args = shared_configs.get_videoQA_args()

    main(args)
