"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VCMR video-only dataset
"""

import torch
import horovod.torch as hvd
from .data import (QueryTokLmdb, get_ids_and_lens,  _check_ngpu)
from .vr_video_only import VideoFeatDataset
from .vcmr import VcmrDataset


class VcmrVideoOnlyDataset(VcmrDataset):
    def __init__(self, video_ids, video_db, query_db, max_num_query=5,
                 sampled_by_q=True):
        assert isinstance(query_db, QueryTokLmdb)
        assert isinstance(video_db, VideoFeatDataset)
        self.video_db = video_db
        self.query_db = query_db
        self.vid2dur = self.video_db.vid2dur
        self.vids = video_ids
        self.global_vid2idx = video_db.vid2idx
        self.vid2idx = {
            vid_name: self.global_vid2idx[vid_name]
            for vid_name in video_ids}
        self.query_data = query_db.query_data
        self.frame_interval = video_db.img_db.frame_interval
        self.max_num_query = max_num_query
        self.sampled_by_q = sampled_by_q

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
            self.lens = [video_db.txt_db.id2len[vid] for vid in self.vids]


class VcmrVideoOnlyEvalDataset(VcmrVideoOnlyDataset):
    def __getitem__(self, i):
        vid, qids = self.getids(i)
        outs = super().__getitem__(i)
        return qids, outs


class VcmrVideoOnlyFullEvalDataset(VcmrVideoOnlyDataset):
    def __init__(self, video_ids, video_db, query_db, max_num_query=5,
                 distributed=False):
        super().__init__([], video_db, query_db, sampled_by_q=True)
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
            nframes = len(clip_level_v_feats)
        query_and_targets = []
        for qid in qids:
            example = self.query_db[qid]
            if example['target'] is not None:
                st_idx, ed_idx = self.get_st_ed_label(
                    example['target'], max_idx=nframes-1)
                target = torch.LongTensor(
                    [st_idx, ed_idx])
            else:
                target = torch.LongTensor([-1, -1])
            query_input_ids = example["input_ids"]
            query_input_ids = torch.tensor(
                [self.query_db.cls_] + query_input_ids)

            query_attn_mask = torch.tensor([1]*len(query_input_ids))

            query_and_targets.append(
                (query_input_ids, query_attn_mask, vid, target))
        return (qid, query_and_targets)
