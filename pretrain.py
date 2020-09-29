"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pre-Training HERO using TV and HowTo100M data
"""
from collections import defaultdict
import json
from os.path import join
from time import time

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from data import (SubTokLmdb,
                  VideoMlmDataset, mlm_collate,
                  MfmDataset, mfm_collate,
                  VsmDataset, vsm_collate,
                  FomDataset, fom_collate,
                  FomEvalDataset, fom_eval_collate,
                  PrefetchLoader, MetaLoader)
from model.model import VideoModelConfig
from model.pretrain import HeroForPretraining
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta, TrainingRestorer
from utils.misc import NoOp, set_dropout, set_random_seed
from utils.const import VFEAT_DIM, MAX_FRM_SEQ_LEN
from config.config import shared_configs
from .load_data import load_video_sub_dataset


def build_target_loaders(target, tgt_ratio, opts):
    if 'vfeat_shards' in target:
        sub_txt_db = SubTokLmdb(f"{opts.txt_db}/{target['sub_txt_db']}",
                                opts.max_clip_len)
        video_db = [
            load_video_sub_dataset(
                f"{target['vfeat_db']}/{shard}", sub_txt_db,
                target['vfeat_interval'], opts)
            for shard in target['vfeat_shards']
        ]
    else:
        video_db = load_video_sub_dataset(
            f"{opts.img_db}/{target['vfeat_db']}",
            f"{opts.txt_db}/{target['sub_txt_db']}",
            target['vfeat_interval'], opts)
    train_loaders = {}
    val_loaders = {}
    for split in target['splits']:
        if 'ratio' not in split:
            split['ratio'] = [1] * len(split['tasks'])
        assert len(split['tasks']) == len(split['ratio'])
        for task, r in zip(split['tasks'], split['ratio']):
            name = f"{task}_{target['name']}_{split['name']}"
            LOGGER.info(f'loading {name} ...')
            ratio = tgt_ratio * r
            if isinstance(video_db, list):
                all_train_ids = [
                    json.load(open(f"{opts.txt_db}/{ids}"))
                    for ids in split['train_idx']
                ]
            else:
                train_ids = json.load(
                    open(f"{opts.txt_db}/{split['train_idx']}"))
            val_ids = json.load(open(f"{opts.txt_db}/{split['val_idx']}"))
            if task == 'mlm':
                if isinstance(video_db, list):
                    train_dset = ConcatDataset([
                        VideoMlmDataset(ids, vid_db, opts.mask_prob,
                                        sub_ctx_len=opts.sub_ctx_len)
                        for ids, vid_db in zip(all_train_ids, video_db)
                    ])
                    val_dset = VideoMlmDataset(
                        val_ids, video_db[0], opts.mask_prob,
                        sub_ctx_len=opts.sub_ctx_len)
                else:
                    train_dset = VideoMlmDataset(
                        train_ids, video_db, opts.mask_prob,
                        sub_ctx_len=opts.sub_ctx_len)
                    val_dset = VideoMlmDataset(
                        val_ids, video_db, opts.mask_prob,
                        sub_ctx_len=opts.sub_ctx_len)
                train_collate = mlm_collate
                val_collate = mlm_collate
            elif task == 'mfm-nce' or task == 'mffr':
                if isinstance(video_db, list):
                    train_dset = ConcatDataset([
                        MfmDataset(ids, vid_db, opts.mask_prob)
                        for ids, vid_db in zip(all_train_ids, video_db)
                    ])
                    val_dset = MfmDataset(val_ids, video_db[0], opts.mask_prob)
                else:
                    train_dset = MfmDataset(train_ids, video_db,
                                            opts.mask_prob)
                    val_dset = MfmDataset(val_ids, video_db, opts.mask_prob)
                train_collate = mfm_collate
                val_collate = mfm_collate
            elif task == 'fom':
                if isinstance(video_db, list):
                    train_dset = ConcatDataset([
                        FomDataset(ids, vid_db, opts.mask_prob)
                        for ids, vid_db in zip(all_train_ids, video_db)
                    ])
                    val_dset = FomEvalDataset(val_ids, video_db[0],
                                              opts.mask_prob)
                else:
                    train_dset = FomDataset(train_ids, video_db,
                                            opts.mask_prob)
                    val_dset = FomEvalDataset(val_ids, video_db,
                                              opts.mask_prob)
                train_collate = fom_collate
                val_collate = fom_eval_collate
            elif task == 'vsm':
                if isinstance(video_db, list):
                    train_dset = ConcatDataset([
                        VsmDataset(ids, vid_db, sub_ctx_len=opts.sub_ctx_len)
                        for ids, vid_db in zip(all_train_ids, video_db)
                    ])
                    val_dset = VsmDataset(val_ids, video_db[0],
                                          sub_ctx_len=opts.sub_ctx_len)
                else:
                    train_dset = VsmDataset(train_ids, video_db,
                                            sub_ctx_len=opts.sub_ctx_len)
                    val_dset = VsmDataset(val_ids, video_db,
                                          sub_ctx_len=opts.sub_ctx_len)
                train_collate = vsm_collate
                val_collate = vsm_collate
            else:
                raise ValueError(f'undefined task {task}')
            train_loader = DataLoader(train_dset,
                                      batch_size=opts.train_batch_size,
                                      num_workers=opts.n_workers,
                                      pin_memory=opts.pin_mem,
                                      collate_fn=train_collate, shuffle=True)
            val_loader = DataLoader(val_dset, batch_size=opts.val_batch_size,
                                    num_workers=opts.n_workers,
                                    pin_memory=opts.pin_mem,
                                    collate_fn=val_collate, shuffle=False)
            train_loaders[name] = (train_loader, ratio)
            val_loaders[name] = PrefetchLoader(val_loader)
    return train_loaders, val_loaders


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

    # data loaders
    train_dataloaders = {}
    val_dataloaders = {}
    for target, t_r in zip(opts.targets, opts.targets_ratio):
        train_loaders, val_loaders = build_target_loaders(target, t_r, opts)
        train_dataloaders.update(train_loaders)
        val_dataloaders.update(val_loaders)
    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

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

    if opts.load_partial_pretrained:
        # from roberta
        model = HeroForPretraining(
            VideoModelConfig(opts.model_config),
            vfeat_dim=VFEAT_DIM,
            max_frm_seq_len=max_frm_seq_len,
            lw_neg_ctx=opts.lw_neg_ctx,
            lw_neg_q=opts.lw_neg_q, lw_st_ed=0,
            ranking_loss_type=opts.ranking_loss_type,
            use_hard_negative=False,
            hard_pool_size=opts.hard_pool_size,
            margin=opts.margin,
            use_all_neg=opts.use_all_neg,
            drop_svmr_prob=opts.drop_svmr_prob)
        model.load_partial_pretrained(
            checkpoint, VFEAT_DIM, max_frm_seq_len,
            skip_layers=opts.skip_layer_loading)
    else:
        # continue training
        model = HeroForPretraining.from_pretrained(
            opts.model_config,
            state_dict=checkpoint,
            vfeat_dim=VFEAT_DIM,
            max_frm_seq_len=max_frm_seq_len,
            lw_neg_ctx=opts.lw_neg_ctx,
            lw_neg_q=opts.lw_neg_q, lw_st_ed=0,
            ranking_loss_type=opts.ranking_loss_type,
            use_hard_negative=False,
            hard_pool_size=opts.hard_pool_size,
            margin=opts.margin,
            use_all_neg=opts.use_all_neg,
            drop_svmr_prob=opts.drop_svmr_prob)

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
    all_gather_list(None)   # sync to prevent slower rank to read training meta
    global_step = restorer.global_step
    TB_LOGGER.global_step = global_step
    if hvd.rank() == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
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
    for task in train_dataloaders.keys():
        if task.startswith('vsm'):
            for obj in ('st_ed', 'neg_ctx', 'neg_q'):
                task2loss[f"{task}_{obj}"] = RunningMeter(f'loss/{task}_{obj}')
    model.train()
    n_examples = defaultdict(int)
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    if global_step == 0:
        optimizer.step()
    assert all(global_step == s for s in all_gather_list(global_step))
    for step, (task, batch) in enumerate(meta_loader):
        LOGGER.debug(f"Task: {task}")

        # hard negative in VSM
        if len(opts.hard_negtiave_start_step) > 0:
            for i, hn_step in enumerate(opts.hard_negtiave_start_step):
                if global_step >= hn_step and hn_step != -1:
                    model.set_hard_negative(
                        True, opts.hard_pool_size[i], opts.hard_neg_weights[i])

        # start-end loss
        if opts.train_span_start_step != -1 and\
                global_step >= opts.train_span_start_step:
            model.set_train_st_ed(opts.lw_st_ed)

        train_task = task.split('_')[0]
        n_examples[task] += opts.train_batch_size

        loss = model(batch, task=train_task, compute_loss=True)
        if train_task == 'vsm':
            loss_st_ed, loss_neg_ctx, loss_neg_q = loss
            loss = loss_st_ed + loss_neg_ctx + loss_neg_q
            for n, ls, w in (('st_ed', loss_st_ed, opts.lw_st_ed),
                             ('neg_ctx', loss_neg_ctx, opts.lw_neg_ctx),
                             ('neg_q', loss_neg_q, opts.lw_neg_q)):
                ls = ls.item()
                if w:
                    ls /= w
                task2loss[f'{task}_{n}'](ls)
        elif train_task == "mffr":
            loss = torch.sqrt(loss.sum(dim=1))

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
                LOGGER.debug("before reduce grad")
                all_reduce_and_rescale_tensors(grads, float(1))
                LOGGER.debug("after reduce grad")

        if (step + 1) % opts.gradient_accumulation_steps == 0:
            global_step += 1

            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, opts)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
            # NOTE: only consider rank 0 for speed
            TB_LOGGER.log_scaler_dict({ll.name: ll.val
                                       for ll in task2loss.values()
                                       if ll.val is not None})
            TB_LOGGER.step()

            LOGGER.debug("before norm grad")
            # update model params
            if opts.grad_norm != -1:
                grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                            opts.grad_norm)
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            LOGGER.debug("after norm grad")
            LOGGER.debug("before optim step")
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)
            LOGGER.debug("after optim step")

            if global_step % 100 == 0:
                LOGGER.debug("after gather stats")
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
                LOGGER.debug("after gather stats")

            if global_step % opts.valid_steps == 0:
                LOGGER.info('===========================================')
                LOGGER.info(f"Step {global_step}: start running validation")
                validate(model, val_dataloaders, opts)
                LOGGER.info('===========================================')
                model_saver.save(model, global_step)

            # step restorer in the end to prevent missing validation checkpoint
            restorer.step()
        if global_step >= opts.num_train_steps:
            break

    LOGGER.info('===========================================')
    if global_step % opts.valid_steps != 0:
        LOGGER.info('===========================================')
        LOGGER.info(f"Step {global_step}: start running validation")
        validate(model, val_dataloaders, opts)
        LOGGER.info('===========================================')
        model_saver.save(model, global_step)


def validate(model, val_dataloaders, opts):
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate on {task} task")
        if task.startswith('mlm'):
            val_log = validate_mlm(model, loader)
        elif task.startswith('mffr'):
            val_log = validate_mffr(model, loader)
        elif task.startswith('mfm-nce'):
            val_log = validate_mfm_nce(model, loader)
        elif task.startswith('fom'):
            val_log = validate_fom(model, loader)
        elif task.startswith('vsm'):
            val_log = validate_vsm(model, loader, opts)
        else:
            raise ValueError(f'Undefined task {task}')
        val_log = {f'{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scaler_dict(
            {f'valid_{task}/{k}': v for k, v in val_log.items()})
    model.train()


@torch.no_grad()
def validate_vsm(model, val_loader, opts):
    LOGGER.info("start running VSM validation...")
    val_loss = 0
    val_loss_st_ed = 0
    val_loss_neg_ctx = 0
    val_loss_neg_q = 0
    n_ex = 0
    n_ex_pos = 0
    st = time()

    for i, batch in enumerate(val_loader):
        n_ex += len(batch['q_vidx'])

        loss_st_ed, loss_neg_ctx, loss_neg_q =\
            model(batch, 'vsm', compute_loss=True)

        val_loss_st_ed += loss_st_ed.item()
        if opts.lw_neg_ctx != 0 or opts.lw_neg_q != 0:
            n_pos = len(loss_neg_ctx)
            val_loss_neg_ctx += loss_neg_ctx.sum().item()
            val_loss_neg_q += loss_neg_q.sum().item()
            n_ex_pos += n_pos

    val_loss_st_ed = sum(all_gather_list(val_loss_st_ed))
    val_loss_neg_ctx = sum(all_gather_list(val_loss_neg_ctx))
    val_loss_neg_q = sum(all_gather_list(val_loss_neg_q))
    n_ex = sum(all_gather_list(n_ex))
    n_ex_pos = sum(all_gather_list(n_ex_pos))
    tot_time = time()-st
    if opts.lw_st_ed:
        val_loss_st_ed /= n_ex
        val_loss_st_ed /= opts.lw_st_ed
    if n_ex_pos > 0 and opts.lw_neg_q > 0 and\
            opts.lw_neg_ctx > 0:
        val_loss_neg_ctx /= n_ex_pos
        val_loss_neg_q /= n_ex_pos
        val_loss_neg_ctx /= opts.lw_neg_ctx
        val_loss_neg_q /= opts.lw_neg_q

    val_loss = opts.lw_st_ed * val_loss_st_ed +\
        opts.lw_neg_ctx * val_loss_neg_ctx +\
        opts.lw_neg_q * val_loss_neg_q
    val_log = {'valid/loss_overall': val_loss,
               'valid/loss_st_ed': val_loss_st_ed,
               'valid/loss_neg_ctx': val_loss_neg_ctx,
               'valid/loss_neg_q': val_loss_neg_q,
               'valid/ex_per_s': n_ex/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.2f}")
    return val_log


@torch.no_grad()
def validate_mlm(model, val_loader):
    LOGGER.info("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='mlm', compute_loss=False)
        labels = batch['txt_labels']
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
    val_loss = sum(all_gather_list(val_loss))
    n_correct = sum(all_gather_list(n_correct))
    n_word = sum(all_gather_list(n_word))
    tot_time = time()-st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log


@torch.no_grad()
def validate_mffr(model, val_loader):
    LOGGER.info("start running MFFR validation...")
    val_loss = 0
    cosine = 0
    n_feat = 0
    st = time()
    for i, batch in enumerate(val_loader):
        targets = batch['feat_targets']
        pred_feat = model(batch, task='mffr', compute_loss=False)
        loss = F.mse_loss(pred_feat, targets, reduction='none')
        loss = torch.sqrt(loss.sum(dim=1))
        val_loss += loss.sum().item()
        cosine += F.cosine_similarity(pred_feat, targets, dim=-1).sum().item()
        n_feat += batch['c_v_masks'].sum().item()
    val_loss = sum(all_gather_list(val_loss))
    cosine = sum(all_gather_list(cosine))
    n_feat = sum(all_gather_list(n_feat))
    tot_time = time()-st
    val_loss /= n_feat
    val_log = {'loss': val_loss,
               'cosine': cosine / n_feat,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.2f}")
    return val_log


@torch.no_grad()
def validate_mfm_nce(model, val_loader):
    LOGGER.info("start running MFM-NCE validation...")
    val_loss = 0
    val_l2 = 0
    n_correct = 0
    cosine = 0
    n_feat = 0
    n_neg = 0
    st = time()
    for i, batch in enumerate(val_loader):
        feats, neg_feats = model(batch, task='mfm-nce', compute_loss=False)
        pos_feats = batch['feat_targets']
        logits = model.v_encoder.mfm_nce(feats, pos_feats, neg_feats,
                                         compute_loss=False)
        targets = torch.arange(0, logits.size(0),
                               dtype=torch.long, device=logits.device)
        val_loss += F.cross_entropy(logits, targets, reduction='sum').item()
        val_l2 += F.mse_loss(feats, pos_feats, reduction='none'
                             ).sum(dim=1).sqrt().sum().item()
        n_correct += (logits.max(dim=-1)[1] == targets).sum().item()
        cosine += F.cosine_similarity(feats, pos_feats, dim=-1).sum().item()
        nf = pos_feats.size(0)
        n_feat += nf
        n_neg += neg_feats.size(0) * nf

    val_loss = sum(all_gather_list(val_loss))
    val_l2 = sum(all_gather_list(val_l2))
    n_correct = sum(all_gather_list(n_correct))
    cosine = sum(all_gather_list(cosine))
    n_feat = sum(all_gather_list(n_feat))
    n_neg = sum(all_gather_list(n_neg))
    tot_time = time()-st
    val_loss /= n_feat
    val_acc = n_correct / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'l2': val_l2 / n_feat,
               'cosine': cosine / n_feat,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.2f}, acc: {val_acc*100:.2f} "
                f"(average {n_neg/n_feat:.0f} negatives)")
    return val_log


@torch.no_grad()
def validate_fom(model, val_loader):
    LOGGER.info("start running FOM validation...")
    val_loss = 0
    n_ex = 0
    n_valid_ex = 0
    tot_score = 0
    st = time()

    for i, batch in enumerate(val_loader):
        targets = batch['targets']
        batch_size, seq_len = targets.size()
        vids = batch['vids']
        del batch['targets']
        del batch['vids']

        scores = model(batch, task='fom', compute_loss=False)

        targets_valid = targets.view(scores.shape[0], )
        loc = (targets_valid != -1).nonzero().squeeze()

        scores_valid = scores[loc, :]
        targets_valid = targets_valid[loc]
        loss = F.cross_entropy(scores_valid, targets_valid, reduction='sum')
        val_loss += loss.item()
        tot_score += (
            scores_valid.max(dim=-1, keepdim=False)[1] == targets_valid
            ).sum().item()
        n_valid_ex += len(targets_valid)
        n_ex += len(vids)

    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    n_valid_ex = sum(all_gather_list(n_valid_ex))
    tot_time = time()-st
    val_loss /= n_valid_ex
    val_acc = tot_score / n_valid_ex
    val_log = {
        'valid/loss': val_loss,
        'valid/acc': val_acc,
        'valid/ex_per_s': n_ex/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log


if __name__ == "__main__":
    args = shared_configs.get_pretrain_args()
    assert hasattr(args, "targets"), "No pretraining targets are given"
    if args.targets_ratio is None:
        args.targets_ratio = [1] * len(args.targets)
    assert len(args.targets) == len(args.targets_ratio)

    main(args)
