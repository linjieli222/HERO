from torch.utils.data import DataLoader
from utils.basic_utils import load_json
from data import (
    VideoFeatLmdb, SubTokLmdb, VrSubTokLmdb,
    VideoFeatSubTokDataset, VideoFeatDataset,
    VcmrDataset, vcmr_collate, VcmrEvalDataset, vcmr_eval_collate,
    VrVideoOnlyDataset, VrVideoOnlyEvalDataset,
    vr_collate, vr_eval_collate,
    VrDataset, VrEvalDataset,
    VcmrVideoOnlyDataset, VcmrVideoOnlyEvalDataset,
    VideoQaDataset, video_qa_collate,
    VideoQaEvalDataset, video_qa_eval_collate,
    ViolinDataset, violin_collate,
    ViolinEvalDataset, violin_eval_collate,
    PrefetchLoader)
from utils.logger import LOGGER
from utils.distributed import all_gather_list
import os


def get_video_ids(query_txt_db):
    if os.path.exists(f'{query_txt_db}/query2video.json'):
        q2v = load_json(f'{query_txt_db}/query2video.json')
        qids = load_json(f'{query_txt_db}/id2len.json').keys()
        video_ids = list(set([q2v[qid] for qid in qids]))
    else:
        video_ids = load_json(f'{query_txt_db}/video_ids.json')
    return video_ids


def load_video_sub_dataset(v_feat_path, sub_txt_db, vfeat_interval, opts):
    vfeat_db = VideoFeatLmdb(
        v_feat_path, opts.vfeat_version,
        vfeat_interval,  opts.compressed_db,
        opts.max_clip_len)
    if not isinstance(sub_txt_db, SubTokLmdb):
        if "msrvtt" in opts.task:
            sub_txt_db = VrSubTokLmdb(sub_txt_db, opts.max_clip_len)
        else:
            sub_txt_db = SubTokLmdb(sub_txt_db, opts.max_clip_len)
    video_db = VideoFeatSubTokDataset(
        sub_txt_db, vfeat_db,
        sub_ctx_len=opts.sub_ctx_len)
    return video_db


def load_video_only_dataset(v_feat_path, txt_meta, vfeat_interval, opts):
    vfeat_db = VideoFeatLmdb(
        v_feat_path, opts.vfeat_version,
        vfeat_interval,  opts.compressed_db,
        opts.max_clip_len)
    video_db = VideoFeatDataset(
        txt_meta, vfeat_db)
    return video_db


def build_downstream_dataloaders(
        tasks, video_db, video_ids, is_train, opts,
        q_txt_db=None, shuffle=False):
    dataloaders = {}
    assert q_txt_db is not None
    for i, task in enumerate(tasks):
        if is_train:
            LOGGER.info(f"Loading {task} train dataset "
                        f"{video_db.img_db.img_dir}")
            batch_size = opts.train_batch_size
        else:
            batch_size = opts.val_batch_size
            LOGGER.info(f"Loading {task} validation dataset"
                        f"{video_db.img_db.img_dir}")
        if task in ["tvqa", "how2qa"]:
            if is_train:
                dataset = VideoQaDataset(
                    video_ids, video_db, q_txt_db)
                collate_fn = video_qa_collate
            else:
                dataset = VideoQaEvalDataset(
                    video_ids, video_db, q_txt_db)
                collate_fn = video_qa_eval_collate
        elif task in ["tvr", "how2r", "didemo_video_sub"]:
            if is_train:
                dataset = VcmrDataset(
                    video_ids, video_db, q_txt_db)
                collate_fn = vcmr_collate
            else:
                dataset = VcmrEvalDataset(
                    video_ids, video_db, q_txt_db)
                collate_fn = vcmr_eval_collate
        elif task == "didemo_video_only":
            if is_train:
                dataset = VcmrVideoOnlyDataset(
                    video_ids, video_db, q_txt_db)
                collate_fn = vcmr_collate
            else:
                dataset = VcmrVideoOnlyEvalDataset(
                    video_ids, video_db, q_txt_db)
                collate_fn = vcmr_eval_collate
        elif task == "msrvtt_video_only":
            if is_train:
                dataset = VrVideoOnlyDataset(
                    video_ids, video_db, q_txt_db)
                collate_fn = vr_collate
            else:
                dataset = VrVideoOnlyEvalDataset(
                    video_ids, video_db, q_txt_db)
                collate_fn = vr_eval_collate
        elif task == "msrvtt_video_sub":
            if is_train:
                dataset = VrDataset(
                    video_ids, video_db, q_txt_db)
                collate_fn = vr_collate
            else:
                dataset = VrEvalDataset(
                    video_ids, video_db, q_txt_db)
                collate_fn = vr_eval_collate
        elif task == "violin":
            if is_train:
                dataset = ViolinDataset(
                    video_ids, video_db, q_txt_db)
                collate_fn = violin_collate
            else:
                dataset = ViolinEvalDataset(
                    video_ids, video_db, q_txt_db)
                collate_fn = violin_eval_collate
        else:
            raise ValueError(f'Undefined task {task}')
        LOGGER.info(f"{sum(all_gather_list(len(dataset)))} samples loaded")
        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem,
                            collate_fn=collate_fn,
                            shuffle=shuffle)
        if is_train:
            ratio = 1
            dataloaders[task] = (loader, ratio)
        else:
            dataloaders[task] = PrefetchLoader(loader)
    return dataloaders
