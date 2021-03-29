"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VCMR dataset
"""
import math
import random

from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import horovod.torch as hvd

from .data import (VideoFeatSubTokDataset, QueryTokLmdb,
                   get_ids_and_lens, video_collate, _check_ngpu)


class VcmrDataset(Dataset):
    def __init__(self, video_ids, video_db, query_db, max_num_query=5,
                 sampled_by_q=True):
        assert isinstance(query_db, QueryTokLmdb)
        assert isinstance(video_db, VideoFeatSubTokDataset)
        self.video_db = video_db
        self.query_db = query_db
        if len(video_db.vid2dur):
            self.vid2dur = self.video_db.vid2dur
            self.vid2idx = self.video_db.vid2idx
            self.global_vid2idx = self.vid2idx
        else:
            self.vid2dur = self.video_db.img_db.name2nframe
            self.global_vid2idx = {
                vid_name: idx for idx, vid_name in
                enumerate(sorted(list(self.vid2dur.keys())))}
            self.vid2idx = {
                vid_name: self.global_vid2idx[vid_name]
                for vid_name in video_ids}
        self.query_data = query_db.query_data
        self.max_clip_len = video_db.txt_db.max_clip_len
        self.frame_interval = video_db.img_db.frame_interval
        self.max_num_query = max_num_query
        self.sampled_by_q = sampled_by_q
        self.vids = video_ids
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

    def getids(self, i):
        if not self.sampled_by_q:
            vid = self.vids[i]
            assert len(self.query_db.video2query) > 0
            # TVR video loss assumes fix number of queries
            qids = self.query_db.video2query[vid][:self.max_num_query]
            if len(qids) < self.max_num_query:
                qids += random.sample(qids, self.max_num_query - len(qids))
        else:
            qids = [self.qids[i]]
            assert len(self.query_db.query2video) > 0
            vid = self.query_db.query2video[qids[0]]
        return vid, qids

    def __getitem__(self, i):
        vid, qids = self.getids(i)

        video_inputs = self.video_db.__getitem__(vid)
        (frame_level_input_ids, frame_level_v_feats,
         frame_level_attn_masks,
         clip_level_v_feats, clip_level_attn_masks, num_subs,
         sub_idx2frame_idx) = video_inputs
        nframes = len(clip_level_v_feats)

        query_and_targets = []
        for qid in qids:
            example = self.query_db[qid]
            st_idx, ed_idx = self.get_st_ed_label(
                example['target'], max_idx=nframes-1)
            target = torch.LongTensor(
                [st_idx, ed_idx])
            query_input_ids = example["input_ids"]
            query_input_ids = torch.tensor(
                [self.query_db.cls_] + query_input_ids)

            query_attn_mask = torch.tensor([1]*len(query_input_ids))

            query_and_targets.append(
                (query_input_ids, query_attn_mask, vid, target))

        return (video_inputs, vid, tuple(query_and_targets))

    def __len__(self):
        if self.sampled_by_q:
            return len(self.qids)
        return len(self.vids)

    def get_st_ed_label(self, ts, max_idx):
        """
        Args:
            ts: [st (float), ed (float)] in seconds, ed > st
            max_idx: length of the video

        Returns:
            [st_idx, ed_idx]: int,

        Given ts = [3.2, 7.6], st_idx = 2, ed_idx = 6,
        clips should be indexed as [2: 6),
        the translated back ts should be [3:9].
        # TODO which one is better, [2: 5] or [2: 6)
        """
        st_idx = min(math.floor(ts[0]/self.frame_interval), max_idx)
        ed_idx = min(max(math.ceil(ts[1]/self.frame_interval)-1,
                         st_idx+1), max_idx)
        return st_idx, ed_idx


def query_collate(query_input_ids, query_attn_mask, targets):
    # hard_coded padding value, TODO: check correctness
    query_pad_values = 1 if len(query_input_ids[0].size()) == 1 else 0
    query_input_ids = pad_sequence(
        query_input_ids, batch_first=True, padding_value=query_pad_values)
    query_pos_ids = torch.arange(0, query_input_ids.size(1), dtype=torch.long
                                 ).unsqueeze(0)
    query_attn_masks = pad_sequence(
        query_attn_mask, batch_first=True, padding_value=0)
    targets = pad_sequence(
        targets, batch_first=True, padding_value=-1)

    batch = {'query_input_ids': query_input_ids,
             'query_pos_ids': query_pos_ids,
             'query_attn_masks': query_attn_masks,
             'targets': targets}
    return batch


def vcmr_collate(inputs):
    (video_inputs, vids,
     query_and_targets) = map(list, unzip(inputs))
    video_batch = video_collate(video_inputs)

    (query_input_ids,
     query_attn_mask, q_vids, targets) = map(
        list, unzip(concat(outs for outs in query_and_targets)))
    batch = query_collate(query_input_ids, query_attn_mask, targets)
    batch.update(video_batch)
    batch["vids"] = vids
    vid2idx = {vid: i for i, vid in enumerate(vids)}
    batch["q_vidx"] = torch.tensor([vid2idx[q_vid] for q_vid in q_vids],
                                   dtype=torch.long)
    return batch


class VcmrEvalDataset(VcmrDataset):

    def __getitem__(self, i):
        vid, qids = self.getids(i)
        outs = super().__getitem__(i)
        return qids, outs


def vcmr_eval_collate(inputs):
    qids, batch = [], []
    for id_, tensors in inputs:
        qids.extend(id_)
        batch.append(tensors)
    batch = vcmr_collate(batch)
    batch['qids'] = qids
    return batch


class VcmrFullEvalDataset(VcmrDataset):
    def __init__(self, video_ids, video_db, query_db, max_num_query=5,
                 distributed=False):
        super().__init__([], video_db, query_db, sampled_by_q=True)
        qlens, qids = get_ids_and_lens(query_db)
        # this dataset does not support multi GPU
        del self.vids
        try:
            self.vid2idx = {
                vid_name: self.global_vid2idx[vid_name]
                for vid_name in video_ids}
        except Exception:
            self.vid2idx = self.vid2idx

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


def vcmr_full_eval_collate(inputs):
    (qids, query_and_targets) = map(list, unzip(inputs))
    (query_input_ids,
     query_attn_mask, q_vids, targets) = map(
        list, unzip(concat(outs for outs in query_and_targets)))
    batch = query_collate(query_input_ids, query_attn_mask, targets)
    batch["vids"] = q_vids
    batch["qids"] = qids
    return batch
