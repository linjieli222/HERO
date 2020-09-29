"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training VR model
"""
from collections import defaultdict
import os
from os.path import exists, join

from time import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from data import (
    MsrvttQueryTokLmdb,
    VrFullEvalDataset, vr_full_eval_collate,
    VrVideoOnlyFullEvalDataset,
    PrefetchLoader, MetaLoader)
from load_data import (
    get_video_ids, load_video_sub_dataset,
    build_downstream_dataloaders,
    load_video_only_dataset)

from model.vr import HeroForVr
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta, TrainingRestorer
from utils.misc import NoOp, set_dropout, set_random_seed
from utils.const import VFEAT_DIM, MAX_FRM_SEQ_LEN
from utils.basic_utils import save_json, load_json
from config.config import shared_configs
from eval_vr import validate_full_vr


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
    if opts.task != "msrvtt_video_only":
        video_db = load_video_sub_dataset(
            opts.vfeat_db, opts.sub_txt_db,
            opts.vfeat_interval, opts)
    else:
        txt_meta = load_json(
            join(opts.train_query_txt_db, "meta.json"))
        video_db = load_video_only_dataset(
            opts.vfeat_db, txt_meta,
            opts.vfeat_interval, opts)

    # data loaders
    # train
    video_ids = get_video_ids(opts.train_query_txt_db)
    train_q_txt_db = MsrvttQueryTokLmdb(
        opts.train_query_txt_db, opts.max_txt_len)
    train_dataloaders = build_downstream_dataloaders(
        [opts.task], video_db, video_ids,
        True, opts, shuffle=True,
        q_txt_db=train_q_txt_db)
    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

    # val
    video_ids = get_video_ids(opts.val_query_txt_db)
    val_q_txt_db = MsrvttQueryTokLmdb(opts.val_query_txt_db, -1)
    val_dataloaders = build_downstream_dataloaders(
        [opts.task], video_db, video_ids,
        False, opts, q_txt_db=val_q_txt_db)

    if opts.task != "msrvtt_video_only":
        inf_dataset = VrFullEvalDataset
    else:
        inf_dataset = VrVideoOnlyFullEvalDataset
    LOGGER.info(f"Loading Inference Dataset {opts.val_query_txt_db} (val)")
    val_dset = inf_dataset(
        video_ids, video_db, val_q_txt_db,
        distributed=opts.distributed_eval)
    inf_loader_val = DataLoader(val_dset,
                                batch_size=opts.vr_eval_q_batch_size,
                                num_workers=opts.n_workers,
                                pin_memory=opts.pin_mem,
                                collate_fn=vr_full_eval_collate)
    inf_loader_val = PrefetchLoader(inf_loader_val)
    if opts.test_query_txt_db:
        LOGGER.info(
            f"Loading Inference Dataset {opts.test_query_txt_db} (test)")
        video_ids = get_video_ids(opts.test_query_txt_db)
        test_q_txt_db = MsrvttQueryTokLmdb(opts.test_query_txt_db, -1)
        test_dset = inf_dataset(
            video_ids, video_db, test_q_txt_db,
            distributed=opts.distributed_eval)
        inf_loader_test = DataLoader(
            test_dset, batch_size=opts.vr_eval_q_batch_size,
            num_workers=opts.n_workers,
            pin_memory=opts.pin_mem,
            collate_fn=vr_full_eval_collate)
        inf_loader_test = PrefetchLoader(inf_loader_test)

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

    model = HeroForVr.from_pretrained(
            opts.model_config,
            state_dict=checkpoint,
            vfeat_dim=VFEAT_DIM,
            max_frm_seq_len=max_frm_seq_len,
            lw_neg_ctx=opts.lw_neg_ctx,
            lw_neg_q=opts.lw_neg_q,
            ranking_loss_type=opts.ranking_loss_type,
            use_hard_negative=False,
            hard_pool_size=opts.hard_pool_size,
            margin=opts.margin,
            use_all_neg=opts.use_all_neg)

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
            # store tvr predictions
            os.makedirs(join(opts.output_dir, 'results'))
        if opts.nms_thd != -1:
            # store tvr-nms predictions
            if not exists(join(opts.output_dir, 'results_nms')):
                os.makedirs(join(opts.output_dir, 'results_nms'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
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

    for obj in (f'{opts.task}_neg_ctx',
                f'{opts.task}_neg_q'):
        task2loss[obj] = RunningMeter(f'loss/{obj}')
    model.train()
    n_examples = defaultdict(int)
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    if global_step == 0:
        optimizer.step()
    for step, (task, batch) in enumerate(meta_loader):
        if len(opts.hard_negtiave_start_step) > 0:
            for i, hn_step in enumerate(opts.hard_negtiave_start_step):
                if global_step >= hn_step and hn_step != -1:
                    model.set_hard_negative(
                        True, opts.hard_pool_size[i], opts.hard_neg_weights[i])

        n_examples[task] += opts.train_batch_size

        loss = model(batch, task=task, compute_loss=True)

        loss_neg_ctx, loss_neg_q = loss
        loss = loss_neg_ctx + loss_neg_q
        for n, ls, w in (('neg_ctx', loss_neg_ctx, opts.lw_neg_ctx),
                         ('neg_q', loss_neg_q, opts.lw_neg_q)):
            ls = ls.item()
            if w:
                ls /= w
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
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
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
                validate(model, val_dataloaders, opts)
                if hvd.rank() == 0 or opts.distributed_eval:
                    log, results = validate_full_vr(
                            model, inf_loader_val,
                            'val', opts, model_opts=opts)
                    save_json(
                        results, f'{opts.output_dir}/results/'
                        f'val_results_{global_step}_rank{hvd.rank()}.json')
                    TB_LOGGER.log_scaler_dict(log)
                    if opts.test_query_txt_db:
                        log, results = validate_full_vr(
                            model, inf_loader_test,
                            'test', opts, model_opts=opts)
                        save_json(
                            results, f'{opts.output_dir}/results/'
                            f'test_results_{global_step}_rank{hvd.rank()}.json')
                        TB_LOGGER.log_scaler_dict(log)
                LOGGER.info('===========================================')
                model_saver.save(model, global_step)

            # step restorer in the end to prevent missing validation checkpoint
            restorer.step()
        if global_step >= opts.num_train_steps:
            break

    LOGGER.info('===========================================')
    if global_step % opts.valid_steps != 0:
        if hvd.rank() == 0 or opts.distributed_eval:
            log, results = validate_full_vr(
                    model, inf_loader_val,
                    'val', opts, model_opts=opts)
            save_json(results,
                      f'{opts.output_dir}/results/'
                      f'val_results_{global_step}'
                      f'_rank{hvd.rank()}_final.json')
            TB_LOGGER.log_scaler_dict(log)
            if opts.test_query_txt_db:
                log, results = validate_full_vr(
                    model, inf_loader_test,
                    'test', opts, model_opts=opts)
                save_json(
                    results, f'{opts.output_dir}/results/'
                    f'test_results_{global_step}_rank{hvd.rank()}.json')
                TB_LOGGER.log_scaler_dict(log)
    model_saver.save(model, f'{global_step}_final')


def validate(model, val_dataloaders, opts):
    model.eval()
    task = opts.task
    loader = val_dataloaders[task]
    LOGGER.info(f"validate on {task} task")
    val_log = validate_vr(model, loader, opts)
    val_log = {f'{task}_{k}': v for k, v in val_log.items()}
    TB_LOGGER.log_scaler_dict(
        {f'valid_{task}/{k}': v for k, v in val_log.items()})
    model.train()


@torch.no_grad()
def validate_vr(model, val_loader, opts):
    LOGGER.info(
        "start running validation (easy version with loss computed)...")
    val_loss = 0
    val_loss_neg_ctx = 0
    val_loss_neg_q = 0
    n_ex = 0
    n_ex_pos = 0
    st = time()

    for i, batch in enumerate(val_loader):
        if 'qids' in batch:
            # qids = batch['qids']
            del batch['qids']
        n_ex += len(batch['q_vidx'])

        loss_neg_ctx, loss_neg_q =\
            model(batch, opts.task, compute_loss=True)

        if opts.lw_neg_ctx != 0 or opts.lw_neg_q != 0:
            n_pos = len(loss_neg_ctx)
            val_loss_neg_ctx += loss_neg_ctx.sum().item()
            val_loss_neg_q += loss_neg_q.sum().item()
            n_ex_pos += n_pos

    val_loss_neg_ctx = sum(all_gather_list(val_loss_neg_ctx))
    val_loss_neg_q = sum(all_gather_list(val_loss_neg_q))
    n_ex = sum(all_gather_list(n_ex))
    n_ex_pos = sum(all_gather_list(n_ex_pos))
    tot_time = time()-st
    if n_ex_pos > 0 and opts.lw_neg_q > 0 and\
            opts.lw_neg_ctx > 0:
        val_loss_neg_ctx /= n_ex_pos
        val_loss_neg_q /= n_ex_pos
        val_loss_neg_ctx /= opts.lw_neg_ctx
        val_loss_neg_q /= opts.lw_neg_q

    val_loss = opts.lw_neg_ctx * val_loss_neg_ctx +\
        opts.lw_neg_q * val_loss_neg_q
    val_log = {
        'valid/loss_overall': val_loss,
        'valid/loss_neg_ctx': val_loss_neg_ctx,
        'valid/loss_neg_q': val_loss_neg_q,
        'valid/ex_per_s': n_ex/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.2f}")
    return val_log


if __name__ == "__main__":
    args = shared_configs.get_vcmr_args()
    main(args)
