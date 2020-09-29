"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Video QA dataset
"""
import random

from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
import horovod.torch as hvd

from .data import (VideoFeatSubTokDataset, QaQueryTokLmdb,
                   get_ids_and_lens, video_collate, _check_ngpu,
                   txt_input_collate)
import math


class VideoQaDataset(Dataset):
    def __init__(self, video_ids, video_db, query_db, max_num_query=5,
                 sampled_by_q=True):
        assert isinstance(query_db, QaQueryTokLmdb)
        assert isinstance(video_db, VideoFeatSubTokDataset)
        self.video_db = video_db
        self.query_db = query_db
        self.vid2dur = self.video_db.vid2dur
        self.vid2idx = self.video_db.vid2idx
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
            # TVR video loss assumes fix number of queries
            qids = self.query_db.video2query[vid][:self.max_num_query]
            if len(qids) < self.max_num_query:
                qids += random.sample(qids, self.max_num_query - len(qids))
        else:
            qids = [self.qids[i]]
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

        all_vids = []
        all_targets = []
        all_ts_targets = []
        all_qa_input_ids = []
        all_qa_attn_masks = []
        all_video_qa_inputs = []
        for qid in qids:
            example = self.query_db[qid]
            if example['target'] is not None:
                target = torch.LongTensor([example['target']])
            else:
                target = torch.LongTensor([-1])
            if example['ts'] is not None:
                st_idx, ed_idx = self.get_st_ed_label(
                    example['ts'], max_idx=nframes-1)
                ts_target = torch.LongTensor(
                    [st_idx, ed_idx])
            else:
                ts_target = torch.LongTensor([-1, -1])

            input_ids = example["input_ids"]
            q_input_ids = input_ids[0]
            for a_input_ids in input_ids[1:]:
                f_sub_qa_input_ids = []
                f_sub_qa_attn_masks = []
                curr_qa_input_id = torch.tensor(
                    [self.query_db.sep] + q_input_ids + [
                        self.query_db.sep] + a_input_ids)
                curr_qa_attn_masks = torch.tensor([1]*len(curr_qa_input_id))
                all_qa_input_ids.append(curr_qa_input_id)
                all_qa_attn_masks.append(curr_qa_attn_masks)
                for f_sub_input_ids, f_attn_masks in zip(
                        frame_level_input_ids, frame_level_attn_masks):
                    curr_f_sub_qa_input_ids = torch.cat((
                        f_sub_input_ids, curr_qa_input_id))
                    curr_f_sub_qa_attn_masks = torch.cat((
                        f_attn_masks, curr_qa_attn_masks))
                    f_sub_qa_input_ids.append(curr_f_sub_qa_input_ids)
                    f_sub_qa_attn_masks.append(curr_f_sub_qa_attn_masks)
                curr_video_qa_inputs = (
                    f_sub_qa_input_ids, frame_level_v_feats,
                    f_sub_qa_attn_masks,
                    clip_level_v_feats, clip_level_attn_masks, num_subs,
                    sub_idx2frame_idx)
                all_video_qa_inputs.append(curr_video_qa_inputs)
            all_vids.append(vid)
            all_targets.append(target)
            all_ts_targets.append(ts_target)
        out = (all_video_qa_inputs, all_qa_input_ids, all_qa_attn_masks,
               all_vids, all_targets, all_ts_targets)
        return out

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
        try:
            ts = ts.split("-")
            st = float(ts[0])
            ed = float(ts[1])
            st_idx = min(math.floor(st/self.frame_interval), max_idx)
            ed_idx = min(max(math.ceil(ed/self.frame_interval)-1,
                             st_idx+1), max_idx)
        except Exception:
            st_idx, ed_idx = -1, -1

        return st_idx, ed_idx


def video_qa_collate(inputs):
    (video_qa_inputs, qa_input_ids, qa_attn_masks,
     vids, target, ts_target) = map(
        list, unzip(inputs))
    all_video_qa_inputs = []
    all_target, all_ts_target = [], []
    all_qa_input_ids, all_qa_attn_masks = [], []
    for i in range(len(video_qa_inputs)):
        all_video_qa_inputs.extend(video_qa_inputs[i])
        all_qa_input_ids.extend(qa_input_ids[i])
        all_qa_attn_masks.extend(qa_attn_masks[i])
    for j in range(len(vids)):
        all_target.extend(target[j])
        all_ts_target.extend(ts_target[j])
    batch = video_collate(all_video_qa_inputs)

    targets = pad_sequence(
        all_target, batch_first=True, padding_value=-1)
    ts_targets = pad_sequence(
        all_ts_target, batch_first=True, padding_value=-1)
    input_ids, pos_ids, attn_masks =\
        txt_input_collate(all_qa_input_ids, all_qa_attn_masks)
    batch["targets"] = targets
    batch["ts_targets"] = ts_targets
    batch['qa_input_ids'] = input_ids
    batch['qa_pos_ids'] = pos_ids
    batch['qa_attn_masks'] = attn_masks
    return batch


class VideoQaEvalDataset(VideoQaDataset):
    def __getitem__(self, i):
        vid, qids = self.getids(i)
        outs = super().__getitem__(i)
        return qids, outs


def video_qa_eval_collate(inputs):
    qids, batch = [], []
    for id_, tensors in inputs:
        qids.extend(id_)
        batch.append(tensors)
    batch = video_qa_collate(batch)
    batch['qids'] = qids
    return batch
