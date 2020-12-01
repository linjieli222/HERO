"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training TVCaption model
"""
import argparse
import os
from os.path import exists, join
from time import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd
from transformers import RobertaTokenizer

from tqdm import tqdm

from data import (CaptionTokLmdb, PrefetchLoader,
                  TvcTrainDataset, TvcValDataset)
from model.tvc import HeroForTvc, TvcGenerator
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_dropout, set_random_seed
from utils.const import VFEAT_DIM, MAX_FRM_SEQ_LEN
from utils.basic_utils import save_jsonl

from eval.tvc import TVCEval
from config.config import parse_with_config

from load_data import load_video_sub_dataset


def build_dataloader(dataset, batch_size, collate_fn, is_train, opts):
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=opts.n_workers,
                        pin_memory=opts.pin_mem,
                        collate_fn=collate_fn,
                        shuffle=is_train)
    return PrefetchLoader(loader)


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    if hvd.rank() != 0:
        LOGGER.disabled = True

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)
    opts.task = 'tvc'

    # train_examples = None
    LOGGER.info(f"Loading the whole video dataset {opts.sub_txt_db}, "
                f"{opts.vfeat_db}")
    video_db = load_video_sub_dataset(opts.vfeat_db, opts.sub_txt_db,
                                      opts.vfeat_interval, opts)

    # data loaders
    # train
    LOGGER.info(f"Loading train dataset {opts.train_db}")
    train_cap = CaptionTokLmdb(opts.train_db, opts.max_txt_len)
    train_dset = TvcTrainDataset(video_db, train_cap, opts.max_cap_per_vid)
    LOGGER.info(f"{sum(all_gather_list(len(train_dset)))} samples loaded")
    train_loader = build_dataloader(train_dset, opts.train_batch_size,
                                    TvcTrainDataset.collate, True, opts)

    # val
    LOGGER.info(f"Loading val dataset {opts.val_db}")
    val_cap = CaptionTokLmdb(opts.val_db, -1)
    val_dset = TvcValDataset(video_db, val_cap, -1)
    val_loader = build_dataloader(val_dset, opts.val_batch_size,
                                  TvcValDataset.collate, False, opts)
    if hvd.rank() == 0:
        evaluator = TVCEval(opts.val_ref)
    else:
        evaluator = NoOp()

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

    model = HeroForTvc.from_pretrained(opts.model_config,
                                       state_dict=checkpoint,
                                       vfeat_dim=VFEAT_DIM,
                                       max_frm_seq_len=max_frm_seq_len,
                                       lsr=opts.lsr)

    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2')

    # assumes roberta tokenizer only
    if hvd.local_rank() == 0:
        # quick hack to prevent multi-process download collision
        toker = RobertaTokenizer.from_pretrained('roberta-base')
        all_gather_list(None)
    else:
        all_gather_list(None)
        toker = RobertaTokenizer.from_pretrained('roberta-base')
    bos = toker.convert_tokens_to_ids(['<s>'])[0]
    eos = toker.convert_tokens_to_ids(['</s>'])[0]
    generator = TvcGenerator(model, opts.max_gen_step, bos, eos, opts.fp16)

    global_step = 0
    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        os.makedirs(join(opts.output_dir, 'results'))  # store val predictions
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    train_loss = RunningMeter('loss')
    n_vid = 0
    n_cap = 0
    n_epoch = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    model.train()
    while True:
        for step, batch in enumerate(train_loader):
            n_vid += opts.train_batch_size
            n_cap += batch['cap_input_ids'].size(0)

            loss = model(batch, compute_loss=True)
            loss = loss.mean()
            train_loss(loss.item())

            delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                                ) as scaled_loss:
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

                # log loss
                TB_LOGGER.add_scalar(train_loss.name, train_loss.val,
                                     global_step)
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
                    tot_vid = sum(all_gather_list(n_vid))
                    vid_per_sec = int(tot_vid / (time()-start))
                    LOGGER.info(f'{tot_vid} videos trained at '
                                f'{vid_per_sec} vid/s')
                    tot_cap = sum(all_gather_list(n_cap))
                    cap_per_sec = int(tot_cap / (time()-start))
                    TB_LOGGER.add_scalar(f'perf/vid_per_s', vid_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/cap_per_s', cap_per_sec,
                                         global_step)

                if global_step % opts.valid_steps == 0:
                    LOGGER.info('===========================================')
                    LOGGER.info(f"Step {global_step}: start validation")
                    val_log, results = validate(val_loader, generator,
                                                toker, evaluator)
                    if hvd.rank() == 0:
                        save_jsonl(results, f"{opts.output_dir}/results/"
                                            f"/results_{global_step}.jsonl")
                    TB_LOGGER.log_scaler_dict(val_log)
                    LOGGER.info('===========================================')
                    model_saver.save(model, global_step)
            if global_step >= opts.num_train_steps:
                break
        n_epoch += 1
        LOGGER.info(f"finished {n_epoch} epochs")
        if global_step >= opts.num_train_steps:
            break

    LOGGER.info('===========================================')
    if global_step % opts.valid_steps != 0:
        val_log, results = validate(val_loader, generator, toker, evaluator)
        if hvd.rank() == 0:
            save_jsonl(results, f"{opts.output_dir}/results/"
                                f"/results_{global_step}.jsonl")
        TB_LOGGER.log_scaler_dict(val_log)
        model_saver.save(model, global_step)


@torch.no_grad()
def validate(loader, generator, tokenizer, evaluator):
    st = time()
    generator.model.eval()
    results = []
    for batch in loader:
        vids = batch['vid_names']
        cids = batch['clip_ids']
        all_ts = batch['all_ts']
        outputs = generator.greedy_decode(batch)
        for vid, cid, ts, out_ids in zip(vids, cids, all_ts, outputs):
            output = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(out_ids))
            results.append({'vid_name': vid, 'clip_id': cid, 'ts': ts,
                            'descs': [{'desc': output}]})
    results = [r for rs in all_gather_list(results) for r in rs]
    LOGGER.info(f'decoding finished in {int(time() - st)} seconds')
    if hvd.rank() == 0:
        val_log = evaluator(results)
        LOGGER.info(f'Validation finished in {int(time() - st)} seconds')
        LOGGER.info(f'CIDEr: {val_log["CIDEr"]}')
    else:
        val_log = {}
    generator.model.train()
    return val_log, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model arch parameters
    parser.add_argument("--model_config", default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="pretrained model")

    # Required parameters
    parser.add_argument("--sub_txt_db",
                        default=None, type=str,
                        help="The input video subtitle corpus. (LMDB)")
    parser.add_argument("--vfeat_db",
                        default=None, type=str,
                        help="The input video frame features.")
    parser.add_argument("--train_db",
                        default=None, type=str,
                        help="The input train query corpus. (LMDB)")
    parser.add_argument("--val_db",
                        default=None, type=str,
                        help="The input validation query corpus. (LMDB)")
    parser.add_argument("--val_ref", default=None, type=str,
                        help="original raw json file")
    parser.add_argument("--test_db",
                        default=None, type=str,
                        help="The input test query corpus. (LMDB)")
    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters
    parser.add_argument('--max_clip_len', type=int, default=100,
                        help='max number of frames in video')
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--max_gen_step', type=int, default=30,
                        help='max number of generation step')
    parser.add_argument("--vfeat_version",
                        default="resnet_slowfast", type=str,
                        help="video frame feature's version")
    parser.add_argument("--vfeat_interval",
                        default=1.5, type=float,
                        help="every ** second to extract one vfeat")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')

    # training parameters
    parser.add_argument("--train_batch_size",
                        default=16, type=int,
                        help="Total batch size for training. "
                             "(batch by number of videos)")
    parser.add_argument("--val_batch_size",
                        default=20, type=int,
                        help="Total batch size for validation. "
                             "(batch by number of videos)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lr_mul", default=1.0, type=float,
                        help="lr multiplier for non-pretrained layers")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adamw',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+', type=float,
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")
    # TVC training parameters
    parser.add_argument('--max_cap_per_vid', type=int, default=-1,
                        help='max number of captions per video')
    parser.add_argument('--lsr', default=0.1, type=float,
                        help='label smoothing regularization')

    parser.add_argument("--sub_ctx_len", default=1, type=int,
                        help="consider 'sub_ctx_len' subtitles before and "
                             "after the current one")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    main(args)
