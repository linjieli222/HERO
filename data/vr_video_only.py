"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VR video-only dataset
"""
from torch.utils.data import Dataset
import torch
import horovod.torch as hvd
from .data import (VideoFeatLmdb,
                   get_ids_and_lens, _check_ngpu)
from .vr import VrQueryTokLmdb, VrDataset


class VideoFeatDataset(Dataset):
    def __init__(self, meta, img_db):
        assert isinstance(img_db, VideoFeatLmdb)
        self.img_db = img_db
        self.max_clip_len = self.img_db.max_clip_len
        self.vid2dur = self.img_db.name2nframe
        self.vids = sorted(list(self.vid2dur.keys()))
        self.vid2idx = {vid: idx for idx, vid in enumerate(self.vids)}
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, vid_):
        v_feat, nframes = self._get_v_feat(vid_)
        num_subs = 1  # fake an empty sub
        sub2frames = [(0, list(range(len(v_feat))))]
        frame_level_input_ids, frame_level_v_feats = (
            [torch.tensor([self.cls_])],
            [v_feat])
        frame_level_attn_masks = [
            torch.tensor([1] * (1+len(v_feat)))]  # [(fffwww)]

        clip_level_v_feats = v_feat
        clip_level_attn_masks = [1] * len(clip_level_v_feats)
        clip_level_attn_masks = torch.tensor(clip_level_attn_masks)

        out = (frame_level_input_ids,   # num_subs list[tensor(sep,w0,w1,...)]
               frame_level_v_feats,     # num_subs list[tensor(#sub_frames, d)]
               frame_level_attn_masks,  # num_subs list[L_sub + #sub_frames]
               clip_level_v_feats,      # tensor(num_frames, d)
               clip_level_attn_masks,   # #frames list[1]
               num_subs, sub2frames)    # num_subs, [(sub_ix, [frame_ix]) ]
        return out

    def _get_v_feat(self, fname):
        v_feat = self.img_db[fname]
        nframes = v_feat.size(0)
        return v_feat, nframes


class VrVideoOnlyDataset(VrDataset):
    def __init__(self, video_ids, video_db, query_db, max_num_query=5,
                 sampled_by_q=True):
        assert isinstance(query_db, VrQueryTokLmdb)
        assert isinstance(video_db, VideoFeatDataset)
        self.video_db = video_db
        self.query_db = query_db
        self.vid2dur = self.video_db.vid2dur
        self.query_data = query_db.query_data
        self.max_clip_len = video_db.max_clip_len
        self.frame_interval = video_db.img_db.frame_interval
        self.max_num_query = max_num_query
        self.sampled_by_q = sampled_by_q
        self.vids = video_ids
        self.global_vid2idx = video_db.vid2idx
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


class VrVideoOnlyEvalDataset(VrVideoOnlyDataset):
    def __getitem__(self, i):
        vid, qids = self.getids(i)
        outs = super().__getitem__(i)
        return qids, outs


class VrVideoOnlyFullEvalDataset(VrVideoOnlyDataset):
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
