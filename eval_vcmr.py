"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run evaluation of VCMR or infenrece of TVR for submission
"""
import argparse
import os
from os.path import exists
from time import time

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import pprint
from apex import amp
from horovod import torch as hvd

from data import (VcmrFullEvalDataset, vcmr_full_eval_collate,
                  VcmrVideoOnlyFullEvalDataset,
                  PrefetchLoader, QueryTokLmdb,
                  video_collate)
from load_data import (
    get_video_ids, load_video_sub_dataset,
    load_video_only_dataset)
from data.loader import move_to_cuda
from model.vcmr import HeroForVcmr

from utils.logger import LOGGER
from utils.const import VFEAT_DIM, VCMR_IOU_THDS
from utils.tvr_standalone_eval import eval_retrieval
from utils.distributed import all_gather_list
from utils.misc import Struct
from cytoolz import concat
from utils.basic_utils import (
    load_json, save_json)
from utils.tvr_eval_utils import (
    find_max_triples_from_upper_triangle_product,
    generate_min_max_length_mask,
    get_submission_top_n, post_processing_vcmr_nms,
    post_processing_svmr_nms)


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))
    if hvd.rank() != 0:
        LOGGER.disabled = True
    hps_file = f'{opts.output_dir}/log/hps.json'
    model_opts = Struct(load_json(hps_file))
    model_config = f'{opts.output_dir}/log/model_config.json'

    # load DBs and image dirs
    video_ids = get_video_ids(opts.query_txt_db)
    if opts.task != "didemo_video_only":
        video_db = load_video_sub_dataset(
            opts.vfeat_db, opts.sub_txt_db, model_opts.vfeat_interval,
            model_opts)
    else:
        txt_meta = load_json(
            os.path.join(opts.query_txt_db, "meta.json"))
        video_db = load_video_only_dataset(
            opts.vfeat_db, txt_meta,
            model_opts.vfeat_interval,
            model_opts)
    assert opts.split in opts.query_txt_db
    q_txt_db = QueryTokLmdb(opts.query_txt_db, -1)
    if opts.task != "didemo_video_only":
        inf_dataset = VcmrFullEvalDataset
    else:
        inf_dataset = VcmrVideoOnlyFullEvalDataset
    eval_dataset = inf_dataset(
        video_ids, video_db, q_txt_db,
        distributed=model_opts.distributed_eval)

    # Prepare model
    if exists(opts.checkpoint):
        ckpt_file = opts.checkpoint
    else:
        ckpt_file = f'{opts.output_dir}/ckpt/model_step_{opts.checkpoint}.pt'
    checkpoint = torch.load(ckpt_file)
    img_pos_embed_weight_key = (
        "v_encoder.f_encoder.img_embeddings" +
        ".position_embeddings.weight")
    assert img_pos_embed_weight_key in checkpoint
    max_frm_seq_len = len(checkpoint[img_pos_embed_weight_key])

    model = HeroForVcmr.from_pretrained(
        model_config,
        state_dict=checkpoint,
        vfeat_dim=VFEAT_DIM,
        max_frm_seq_len=max_frm_seq_len,
        lw_neg_ctx=model_opts.lw_neg_ctx,
        lw_neg_q=model_opts.lw_neg_q, lw_st_ed=0,
        ranking_loss_type=model_opts.ranking_loss_type,
        use_hard_negative=False,
        hard_pool_size=model_opts.hard_pool_size,
        margin=model_opts.margin,
        use_all_neg=model_opts.use_all_neg,
        drop_svmr_prob=model_opts.drop_svmr_prob)
    model.to(device)
    if opts.fp16:
        model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')

    eval_dataloader = DataLoader(eval_dataset, batch_size=opts.batch_size,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 collate_fn=vcmr_full_eval_collate)
    eval_dataloader = PrefetchLoader(eval_dataloader)

    _, results = validate_full_vcmr(
        model, eval_dataloader, opts.split, opts, model_opts)
    result_dir = f'{opts.output_dir}/results_{opts.split}'

    if not exists(result_dir) and rank == 0:
        os.makedirs(result_dir)

    all_results = list(concat(all_gather_list(results)))
    if hvd.rank() == 0:
        save_json(
            all_results,
            f'{result_dir}/results_{opts.checkpoint}_all.json')
        LOGGER.info('All results written......')


@torch.no_grad()
def validate_full_vcmr(model, val_loader, split, opts, model_opts):
    LOGGER.info("start running  full VCMR evaluation"
                f"on {opts.task} {split} split...")
    model.eval()
    n_ex = 0
    st = time()
    val_log = {}
    has_gt_target = True
    val_vid2idx = val_loader.dataset.vid2idx
    if split in val_vid2idx:
        video2idx_global = val_vid2idx[split]
    else:
        video2idx_global = val_vid2idx
    video_ids = sorted(list(video2idx_global.keys()))
    video2idx_local = {e: i for i, e in enumerate(video_ids)}
    query_data = val_loader.dataset.query_data

    partial_query_data = []
    total_frame_embeddings = None
    video_batch, video_idx = [], []
    max_clip_len = 0
    for video_i, (vid, vidx) in tqdm(enumerate(video2idx_local.items()),
                                     desc="Computing Video Embeddings",
                                     total=len(video2idx_local)):
        video_item = val_loader.dataset.video_db[vid]
        video_batch.append(video_item)
        video_idx.append(vidx)
        if len(video_batch) == opts.vcmr_eval_video_batch_size or\
                video_i == len(video2idx_local) - 1:
            video_batch = move_to_cuda(video_collate(video_batch))
            # Safeguard fp16
            for k, item in video_batch.items():
                if isinstance(item, torch.Tensor) and\
                        item.dtype == torch.float32:
                    video_batch[k] = video_batch[k].to(
                        dtype=next(model.parameters()).dtype)
            curr_frame_embeddings = model.v_encoder(video_batch, 'repr')
            curr_c_attn_masks = video_batch['c_attn_masks']
            curr_clip_len = curr_frame_embeddings.size(-2)
            assert curr_clip_len <= model_opts.max_clip_len

            if total_frame_embeddings is None:
                feat_dim = curr_frame_embeddings.size(-1)
                total_frame_embeddings = torch.zeros(
                    (len(video2idx_local), model_opts.max_clip_len, feat_dim),
                    dtype=curr_frame_embeddings.dtype,
                    device=curr_frame_embeddings.device)
                total_c_attn_masks = torch.zeros(
                    (len(video2idx_local), model_opts.max_clip_len),
                    dtype=curr_c_attn_masks.dtype,
                    device=curr_frame_embeddings.device)
            indices = torch.LongTensor(video_idx)
            total_frame_embeddings[indices, :curr_clip_len] =\
                curr_frame_embeddings
            total_c_attn_masks[indices, :curr_clip_len] =\
                curr_c_attn_masks
            max_clip_len = max(max_clip_len, curr_clip_len)
            video_batch, video_idx = [], []
    total_frame_embeddings = total_frame_embeddings[:, :max_clip_len, :]
    total_c_attn_masks = total_c_attn_masks[:, :max_clip_len]

    svmr_st_probs_total, svmr_ed_probs_total = None, None
    sorted_q2c_indices, sorted_q2c_scores = None, None
    flat_st_ed_sorted_scores, flat_st_ed_scores_sorted_indices = None, None
    total_qids, total_vids = [], []
    for batch in tqdm(val_loader, desc="Computing q2vScores"):
        qids = batch['qids']
        vids = batch['vids']
        targets = batch['targets']
        if has_gt_target and targets.min() < 0:
            has_gt_target = False
            LOGGER.info("No GT annotations provided, only generate predictions")
        del batch['targets']
        del batch['qids']
        del batch['vids']

        total_qids.extend(qids)
        total_vids.extend(vids)
        for qid in qids:
            partial_query_data.append(query_data[qid])
        # Safeguard fp16
        for k, item in batch.items():
            if isinstance(item, torch.Tensor) and item.dtype == torch.float32:
                batch[k] = batch[k].to(
                    dtype=next(model.parameters()).dtype)

        # FIXME
        _q2video_scores, _st_probs, _ed_probs =\
            model.get_pred_from_raw_query(
                total_frame_embeddings, total_c_attn_masks, **batch,
                cross=True, val_gather_gpus=False)

        _st_probs = F.softmax(_st_probs, dim=-1)
        _ed_probs = F.softmax(_ed_probs, dim=-1)
        n_ex += len(qids)

        if "SVMR" in opts.full_eval_tasks and has_gt_target:
            row_indices = torch.arange(0, len(_st_probs))
            svmr_gt_vidx = torch.LongTensor(
                [video2idx_local[e] for e in vids])
            svmr_st_probs = _st_probs[
                row_indices, svmr_gt_vidx].float().cpu().numpy()
            svmr_ed_probs = _ed_probs[
                row_indices, svmr_gt_vidx].float().cpu().numpy()
            if svmr_st_probs_total is None:
                svmr_st_probs_total = svmr_st_probs
                svmr_ed_probs_total = svmr_ed_probs
            else:
                svmr_st_probs_total = np.concatenate(
                    (svmr_st_probs_total, svmr_st_probs),
                    axis=0)
                svmr_ed_probs_total = np.concatenate(
                    (svmr_ed_probs_total, svmr_ed_probs),
                    axis=0)

        if "VR" not in opts.full_eval_tasks or _q2video_scores is None:
            continue

        _q2video_scores = _q2video_scores.float()
        # To give more importance to top scores,
        # the higher opt.alpha is the more importance will be given
        q2video_scores = torch.exp(model_opts.q2c_alpha * _q2video_scores)
        _sorted_q2c_scores, _sorted_q2c_indices = \
            torch.topk(q2video_scores, model_opts.max_vcmr_video,
                       dim=1, largest=True)
        if sorted_q2c_indices is None:
            sorted_q2c_indices = _sorted_q2c_indices.cpu().numpy()
            sorted_q2c_scores = _sorted_q2c_scores.cpu().numpy()
        else:
            sorted_q2c_indices = np.concatenate(
                (sorted_q2c_indices, _sorted_q2c_indices.cpu().numpy()),
                axis=0)
            sorted_q2c_scores = np.concatenate(
                (sorted_q2c_scores, _sorted_q2c_scores.cpu().numpy()),
                axis=0)

        if "VCMR" not in opts.full_eval_tasks:
            continue

        row_indices = torch.arange(
            0, len(_st_probs), device=_st_probs.device).unsqueeze(1)
        _st_probs = _st_probs[
            row_indices, _sorted_q2c_indices]  # (_N_q, max_vcmr_video, L)
        _ed_probs = _ed_probs[row_indices, _sorted_q2c_indices]
        # (_N_q, max_vcmr_video, L, L)
        _st_ed_scores = torch.einsum("qvm,qv,qvn->qvmn", _st_probs,
                                     _sorted_q2c_scores, _ed_probs)
        valid_prob_mask = generate_min_max_length_mask(
            _st_ed_scores.shape, min_l=model_opts.min_pred_l,
            max_l=model_opts.max_pred_l)
        _st_ed_scores *= torch.from_numpy(
            valid_prob_mask).to(
                _st_ed_scores.device)  # invalid location will become zero!
        # sort across the top-max_n_videos videos (by flatten from the 2nd dim)
        # the indices here are local indices, not global indices
        _n_q = _st_ed_scores.shape[0]
        _flat_st_ed_scores = _st_ed_scores.reshape(
            _n_q, -1)  # (N_q, max_vcmr_video*L*L)
        _flat_st_ed_sorted_scores, _flat_st_ed_scores_sorted_indices = \
            torch.sort(_flat_st_ed_scores, dim=1, descending=True)

        if flat_st_ed_sorted_scores is None:
            flat_st_ed_scores_sorted_indices =\
                _flat_st_ed_scores_sorted_indices[
                    :, :model_opts.max_before_nms].cpu().numpy()
            flat_st_ed_sorted_scores =\
                _flat_st_ed_sorted_scores[
                    :, :model_opts.max_before_nms].cpu().numpy()
        else:
            flat_st_ed_scores_sorted_indices = np.concatenate(
                (flat_st_ed_scores_sorted_indices,
                 _flat_st_ed_scores_sorted_indices[
                     :, :model_opts.max_before_nms].cpu().numpy()),
                axis=0)
            flat_st_ed_sorted_scores = np.concatenate(
                (flat_st_ed_sorted_scores,
                 _flat_st_ed_sorted_scores[
                     :, :model_opts.max_before_nms].cpu().numpy()),
                axis=0)

    svmr_res, vr_res, vcmr_res = [], [], []
    if "SVMR" in opts.full_eval_tasks and has_gt_target:
        st_ed_prob_product = np.einsum(
            "bm,bn->bmn", svmr_st_probs_total,
            svmr_ed_probs_total)  # (N, L, L)
        valid_prob_mask = generate_min_max_length_mask(
            st_ed_prob_product.shape, min_l=model_opts.min_pred_l,
            max_l=model_opts.max_pred_l)
        # invalid location will become zero!
        st_ed_prob_product *= valid_prob_mask
        batched_sorted_triples =\
            find_max_triples_from_upper_triangle_product(
                st_ed_prob_product, top_n=model_opts.max_before_nms,
                prob_thd=None)
        for svmr_i, (qid, vid) in tqdm(
                enumerate(zip(total_qids, total_vids)),
                desc="[SVMR] Loop over queries to generate predictions",
                total=len(total_qids)):
            vidx = video2idx_global[vid]
            _sorted_triples = batched_sorted_triples[svmr_i]
            # as we redefined ed_idx, which is inside the moment.
            _sorted_triples[:, 1] += 1
            _sorted_triples[:, :2] = (_sorted_triples[:, :2]
                                      * model_opts.vfeat_interval)
            cur_ranked_predictions = [
                [vidx, ] + row for row in _sorted_triples.tolist()]
            cur_query_pred = dict(desc_id=int(qid),
                                  desc="",
                                  predictions=cur_ranked_predictions)
            svmr_res.append(cur_query_pred)

    if "VR" in opts.full_eval_tasks:
        for vr_i, (_sorted_q2c_scores_row, _sorted_q2c_indices_row) in tqdm(
                    enumerate(
                        zip(sorted_q2c_scores[:, :100],
                            sorted_q2c_indices[:, :100])),
                    desc="[VR] Loop over queries to generate predictions",
                    total=len(total_qids)):
            cur_vr_redictions = []
            for v_score, v_meta_idx in zip(_sorted_q2c_scores_row,
                                           _sorted_q2c_indices_row):
                video_idx = video2idx_global[video_ids[v_meta_idx]]
                cur_vr_redictions.append([video_idx, 0, 0, float(v_score)])
            cur_query_pred = dict(desc_id=int(total_qids[vr_i]),
                                  desc="",
                                  predictions=cur_vr_redictions)
            vr_res.append(cur_query_pred)
    if "VCMR" in opts.full_eval_tasks:
        for vcmr_i, (
                _flat_st_ed_scores_sorted_indices,
                _flat_st_ed_sorted_scores) in tqdm(
                enumerate(zip(
                    flat_st_ed_scores_sorted_indices,
                    flat_st_ed_sorted_scores)),
                desc="[VCMR] Loop over queries to generate predictions",
                total=len(total_qids)):  # i is query_idx
            # list([video_idx(int), st(float),
            #       ed(float), score(float)])
            video_meta_indices_local, pred_st_indices, pred_ed_indices = \
                np.unravel_index(
                    _flat_st_ed_scores_sorted_indices,
                    shape=(model_opts.max_vcmr_video, model_opts.max_clip_len,
                           model_opts.max_clip_len))
            # video_meta_indices_local refers to
            # the indices among the top-max_vcmr_video
            # video_meta_indices refers to
            # the indices in all the videos,
            # which is the True indices
            video_meta_indices = sorted_q2c_indices[
                vcmr_i, video_meta_indices_local]

            pred_st_in_seconds = pred_st_indices.astype(
                np.float32) * model_opts.vfeat_interval
            pred_ed_in_seconds = pred_ed_indices.astype(
                np.float32
                ) * model_opts.vfeat_interval + model_opts.vfeat_interval
            cur_vcmr_redictions = []
            for j, (v_meta_idx, v_score) in enumerate(
                    zip(video_meta_indices,
                        _flat_st_ed_sorted_scores)):  # videos
                video_idx = video2idx_global[video_ids[v_meta_idx.item()]]
                cur_vcmr_redictions.append(
                    [video_idx, float(pred_st_in_seconds[j]),
                     float(pred_ed_in_seconds[j]), float(v_score)])

            cur_query_pred = dict(
                desc_id=int(total_qids[vcmr_i]),
                desc="",
                predictions=cur_vcmr_redictions)
            vcmr_res.append(cur_query_pred)

    eval_res = dict(SVMR=svmr_res, VCMR=vcmr_res, VR=vr_res)
    eval_res = {k: v for k, v in eval_res.items() if len(v) != 0}
    eval_res["video2idx"] = video2idx_global

    eval_submission = get_submission_top_n(
        eval_res, top_n=model_opts.max_after_nms)

    if has_gt_target:
        metrics = eval_retrieval(eval_submission, partial_query_data,
                                 iou_thds=VCMR_IOU_THDS,
                                 match_number=True,
                                 verbose=False,
                                 use_desc_type=model_opts.eval_with_query_type)

        if model_opts.distributed_eval:
            n_ex_per_rank = all_gather_list(n_ex)
            metrics_per_rank = all_gather_list(metrics)
        else:
            n_ex_per_rank = [n_ex]
            metrics_per_rank = [metrics]
        n_ex = sum(n_ex_per_rank)
        val_log = {}
        gathered_metrics = {}
        for task_type, task_metric in metrics.items():
            gathered_metrics[task_type] = {}
            for k in task_metric.keys():
                if k == "desc_type_ratio":
                    continue
                gathered_v = 0
                for idx, n in enumerate(n_ex_per_rank):
                    gathered_v += n*metrics_per_rank[idx][task_type][k]
                gathered_v = gathered_v / n_ex
                gathered_metrics[task_type][k] = gathered_v
                val_log[
                    f'valid_{split}_{task_type}/{task_type}_{k}'] = gathered_v
        if "VCMR" in gathered_metrics:
            LOGGER.info("metrics_no_nms_VCMR \n{}".format(pprint.pformat(
                    gathered_metrics["VCMR"], indent=4)))
        elif "SVMR" in gathered_metrics:
            LOGGER.info("metrics_no_nms_SVMR \n{}".format(pprint.pformat(
                gathered_metrics["SVMR"], indent=4)))

        if model_opts.nms_thd != -1:
            LOGGER.info(
                "Performing nms with nms_thd {}".format(
                    model_opts.nms_thd))
            eval_res_after_nms = dict(
                video2idx=eval_res["video2idx"])
            if "SVMR" in eval_res:
                eval_res_after_nms["SVMR"] =\
                    post_processing_svmr_nms(
                    eval_res["SVMR"], nms_thd=model_opts.nms_thd,
                    max_before_nms=model_opts.max_before_nms,
                    max_after_nms=model_opts.max_after_nms)
            if "VCMR" in eval_res:
                eval_res_after_nms["VCMR"] =\
                    post_processing_vcmr_nms(
                    eval_res["VCMR"], nms_thd=model_opts.nms_thd,
                    max_before_nms=model_opts.max_before_nms,
                    max_after_nms=model_opts.max_after_nms)
            metrics_nms = eval_retrieval(
                eval_res_after_nms, partial_query_data,
                iou_thds=VCMR_IOU_THDS,
                match_number=True,
                verbose=False,
                use_desc_type=model_opts.eval_with_query_type)

            if model_opts.distributed_eval:
                metrics_nms_per_rank = all_gather_list(metrics_nms)
            else:
                metrics_nms_per_rank = [metrics_nms]
            gathered_metrics_nms = {}

            for task_type, task_metric in metrics_nms.items():
                gathered_metrics_nms[task_type] = {}
                for k in task_metric.keys():
                    if k == "desc_type_ratio":
                        continue
                    gathered_v_nms = 0
                    for idx, n in enumerate(n_ex_per_rank):
                        gathered_v_nms += (
                            n*metrics_nms_per_rank[idx][task_type][k])
                    gathered_v_nms = gathered_v_nms / n_ex
                    gathered_metrics_nms[task_type][k] = gathered_v_nms
                    val_log[f'valid_{split}_{task_type}'
                            f'_nms_{model_opts.nms_thd}/'
                            f'{task_type}_{k}'] = gathered_v_nms
            if "VCMR" in gathered_metrics_nms:
                LOGGER.info("metrics_nms_VCMR \n{}".format(pprint.pformat(
                    gathered_metrics_nms["VCMR"], indent=4)))
            elif "SVMR" in gathered_metrics_nms:
                LOGGER.info("metrics_nms_SVMR \n{}".format(pprint.pformat(
                    gathered_metrics_nms["SVMR"], indent=4)))

        tot_time = time()-st
        val_log.update(
            {f'valid/vcmr_{split}_ex_per_s': n_ex/tot_time})
        LOGGER.info(f"validation finished in {int(tot_time)} seconds")
    model.train()
    return val_log, eval_submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--sub_txt_db",
                        default="/txt/tv_subtitles.db",
                        type=str,
                        help="The input video subtitle corpus. (LMDB)")
    parser.add_argument("--vfeat_db",
                        default="/video/tv", type=str,
                        help="The input video frame features.")
    parser.add_argument("--query_txt_db",
                        default="/txt/tvr_val.db",
                        type=str,
                        help="The input test query corpus. (LMDB)")
    parser.add_argument("--split", choices=["val", "test_public", "test"],
                        default="val", type=str,
                        help="The input query split")
    parser.add_argument("--task", choices=["tvr", "how2r", "didemo_video_sub",
                                           "didemo_video_only"],
                        default="tvr", type=str,
                        help="The evaluation vcmr task")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model checkpoint steps")
    parser.add_argument("--batch_size",
                        default=80, type=int,
                        help="number of queries in a batch")
    parser.add_argument("--vcmr_eval_video_batch_size",
                        default=50, type=int,
                        help="number of videos in a batch")
    parser.add_argument(
            "--full_eval_tasks", type=str, nargs="+",
            choices=["VCMR", "SVMR", "VR"], default=["VCMR", "SVMR", "VR"],
            help="Which tasks to run."
            "VCMR: Video Corpus Moment Retrieval;"
            "SVMR: Single Video Moment Retrieval;"
            "VR: regular Video Retrieval. "
            "    (will be performed automatically with VCMR)")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # device parameters
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    args = parser.parse_args()

    # options safe guard
    # TODO

    main(args)
