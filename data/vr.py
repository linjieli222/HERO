"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VR dataset
"""
import torch
import horovod.torch as hvd
from utils.basic_utils import load_jsonl
import os
import json
from .data import (VideoFeatSubTokDataset, TxtTokLmdb, SubTokLmdb,
                   get_ids_and_lens, _check_ngpu)
from .vcmr import VcmrDataset, vcmr_collate, vcmr_full_eval_collate


class VrSubTokLmdb(SubTokLmdb):
    def __init__(self, db_dir, max_clip_len=-1):
        super().__init__(db_dir, max_clip_len=-1)
        self.max_clip_len = max_clip_len
        self.vid2max_len = json.load(
            open(f'{db_dir}/vid2max_frame_sub_len.json'))
        self.id2len = json.load(
            open(f'{db_dir}/vid2len.json'))
        self.vid2dur, self.vid2idx = {}, {}


class VrQueryTokLmdb(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len=-1):
        super().__init__(db_dir, max_txt_len)
        if os.path.exists(f'{self.db_dir}/query2video.json'):
            self.query2video = json.load(
                open(f'{self.db_dir}/query2video.json'))
            self.video2query = {}
            for k, v in self.query2video.items():
                if v not in self.video2query:
                    self.video2query[v] = [k]
                else:
                    self.video2query[v].append(k)
        else:
            self.query2video = {}
            self.video2query = {}
        self.query_data_f = load_jsonl(f'{self.db_dir}/query_data.jsonl')

    def __getitem__(self, id_):
        txt_dump = self.db[id_]
        return txt_dump


class MsrvttQueryTokLmdb(VrQueryTokLmdb):
    @property
    def query_data(self):
        try:
            data = {
                str(item["sen_id"]): item
                for item in self.query_data_f}
        except Exception:
            data = {
                str(item["retrieval_key"]): item
                for item in self.query_data_f}
        return data


class VrDataset(VcmrDataset):
    def __init__(self, video_ids, video_db, query_db, max_num_query=5,
                 sampled_by_q=True):
        assert isinstance(query_db, VrQueryTokLmdb)
        assert isinstance(video_db, VideoFeatSubTokDataset)
        self.video_db = video_db
        self.query_db = query_db
        self.vid2dur = self.video_db.img_db.name2nframe
        self.query_data = query_db.query_data
        self.max_clip_len = video_db.txt_db.max_clip_len
        self.frame_interval = video_db.img_db.frame_interval
        self.max_num_query = max_num_query
        self.sampled_by_q = sampled_by_q
        self.vids = video_ids
        self.global_vid2idx = {
            vid_name: idx for idx, vid_name in
            enumerate(sorted(list(self.vid2dur.keys())))}
        self.vid2idx = {
            vid_name: self.global_vid2idx[vid_name]
            for vid_name in video_ids}
        if sampled_by_q:
            self.lens, self.qids = get_ids_and_lens(query_db)
            # FIXME
            if _check_ngpu() > 1:
                # partition data by rank
                self.qids = self.qids[hvd.rank()::hvd.size()]
                self.lens = self.lens[hvd.rank()::hvd.size()]
        else:
            # FIXME
            if _check_ngpu() > 1:
                # partition data by rank
                self.vids = self.vids[hvd.rank()::hvd.size()]
            self.lens = [video_db.vid2dur[vid] for vid in self.vids]

    def __getitem__(self, i):
        vid, qids = self.getids(i)

        video_inputs = self.video_db.__getitem__(vid)
        (frame_level_input_ids, frame_level_v_feats,
         frame_level_attn_masks,
         clip_level_v_feats, clip_level_attn_masks, num_subs,
         sub_idx2frame_idx) = video_inputs

        query_and_targets = []
        for qid in qids:
            example = self.query_db[qid]
            target = torch.LongTensor([-1, -1])
            query_input_ids = example["input_ids"]
            query_input_ids = torch.tensor(
                [self.query_db.cls_] + query_input_ids)

            query_attn_mask = torch.tensor([1]*len(query_input_ids))

            query_and_targets.append(
                (query_input_ids, query_attn_mask, vid, target))

        return (video_inputs, vid, tuple(query_and_targets))


def vr_collate(inputs):
    return vcmr_collate(inputs)


class VrEvalDataset(VrDataset):
    def __getitem__(self, i):
        vid, qids = self.getids(i)
        outs = super().__getitem__(i)
        return qids, outs


def vr_eval_collate(inputs):
    qids, batch = [], []
    for id_, tensors in inputs:
        qids.extend(id_)
        batch.append(tensors)
    batch = vr_collate(batch)
    batch['qids'] = qids
    return batch


class VrFullEvalDataset(VrDataset):
    def __init__(self, video_ids, video_db, query_db, max_num_query=5,
                 distributed=False):
        super().__init__(video_ids, video_db, query_db, sampled_by_q=True)
        qlens, qids = get_ids_and_lens(query_db)
        # this dataset does not support multi GPU
        del self.vids
        self.vid2idx = {
            vid_name: self.global_vid2idx[vid_name]
            for vid_name in video_ids}

        # FIXME
        if _check_ngpu() > 1 and distributed:
            # partition data by rank
            self.qids = qids[hvd.rank()::hvd.size()]
            self.lens = qlens[hvd.rank()::hvd.size()]
        else:
            self.qids = qids
            self.lens = qlens

    def __len__(self):
        return len(self.qids)

    def getids(self, i):
        qid = self.qids[i]
        if len(self.query_db.query2video):
            vid = self.query_db.query2video[qid]
        else:
            vid = -1
        return vid, [qid]

    def __getitem__(self, i):
        vid, qids = self.getids(i)
        if vid != -1:
            video_inputs = self.video_db.__getitem__(vid)
            (frame_level_input_ids, frame_level_v_feats,
             frame_level_attn_masks,
             clip_level_v_feats, clip_level_attn_masks, num_subs,
             sub_idx2frame_idx) = video_inputs
        query_and_targets = []
        for qid in qids:
            example = self.query_db[qid]
            target = torch.LongTensor([-1, -1])
            query_input_ids = example["input_ids"]

            query_input_ids = torch.tensor(
                [self.query_db.cls_] + query_input_ids)

            query_attn_mask = torch.tensor([1]*len(query_input_ids))

            query_and_targets.append(
                (query_input_ids, query_attn_mask, vid, target))
        return (qid, query_and_targets)


def vr_full_eval_collate(inputs):
    return vcmr_full_eval_collate(inputs)
