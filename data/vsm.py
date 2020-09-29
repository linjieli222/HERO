"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pretrain VSM  dataset
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from toolz.sandbox import unzip
from cytoolz import concat
import horovod.torch as hvd
import copy

from .data import VideoFeatSubTokDataset, _check_ngpu, video_collate


class VsmDataset(Dataset):
    def __init__(self, video_ids, vid_sub_db, query_per_video=5,
                 sub_ctx_len=0):
        assert isinstance(vid_sub_db, VideoFeatSubTokDataset)
        self.query_per_video = query_per_video
        self.vid_sub_db = vid_sub_db
        if _check_ngpu() > 1:
            self.ids = video_ids[hvd.rank()::hvd.size()]
        else:
            self.ids = video_ids
        self.sub_ctx_len = sub_ctx_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        vid = self.ids[i]
        example = self.vid_sub_db.txt_db[vid]
        v_feat, nframes = self.vid_sub_db._get_v_feat(vid)
        sub2frames = self.vid_sub_db.vid_sub2frame[vid]

        frame_level_input_ids, frame_level_v_feats = [], []
        frame_level_attn_masks = []
        num_subs = len(sub2frames)

        sub_queries_and_targets = []
        matched_sub_idx = [sub_idx for sub_idx, matched_frames in sub2frames
                           if matched_frames]
        n_samples = min(len(matched_sub_idx), self.query_per_video)
        query_sub_ids = set(random.sample(matched_sub_idx, n_samples))
        for sub_idx, matched_frames in sub2frames:
            # text input
            if self.sub_ctx_len >= 0:
                curr_sub_ctx_input_ids = []
                for tmp_sub_idx in range(sub_idx-self.sub_ctx_len,
                                         sub_idx+1):
                    if tmp_sub_idx >= 0 and tmp_sub_idx < num_subs\
                            and tmp_sub_idx not in query_sub_ids:
                        in_ids = example['input_ids'][tmp_sub_idx]
                        if self.vid_sub_db.max_txt_len != -1:
                            in_ids = in_ids[:self.vid_sub_db.max_txt_len]
                        curr_sub_ctx_input_ids.extend(copy.deepcopy(in_ids))
            curr_sub_ctx_input_ids = [
                self.vid_sub_db.txt_db.sep] + curr_sub_ctx_input_ids

            n_frame = len(matched_frames)
            attn_masks_fill_0_pos = None
            if n_frame:
                matched_v_feats = torch.index_select(
                    v_feat, 0, torch.tensor(matched_frames))

                if sub_idx in query_sub_ids:
                    in_ids = example['input_ids'][sub_idx]
                    if self.vid_sub_db.max_txt_len != -1:
                        in_ids = in_ids[:self.vid_sub_db.max_txt_len]
                    sub_quries_input_ids = torch.tensor(
                        [self.vid_sub_db.txt_db.cls_] + copy.deepcopy(in_ids))
                    sub_query_attn_masks = torch.ones(
                        len(sub_quries_input_ids), dtype=torch.long)
                    st, ed = matched_frames[0], min(max(
                        matched_frames[0]+1, matched_frames[-1]), nframes-1)
                    assert st <= ed, "st frame must <= ed frame"
                    assert st >= 0, "st frame must >= 0"
                    assert ed < nframes, f"ed frame must < frame_len {nframes}"
                    targets = torch.tensor([st, ed], dtype=torch.long)
                    sub_queries_and_targets.append(
                        (sub_quries_input_ids, sub_query_attn_masks,
                         vid, targets))
                if len(curr_sub_ctx_input_ids) == 0:
                    curr_sub_ctx_input_ids = [self.vid_sub_db.txt_db.mask]
                    attn_masks_fill_0_pos = -1
                attn_masks = torch.ones(
                    len(curr_sub_ctx_input_ids) + n_frame,
                    dtype=torch.long)
            else:
                matched_v_feats = torch.zeros(1, v_feat.shape[1])
                attn_masks = torch.ones(
                    len(curr_sub_ctx_input_ids) + 1, dtype=torch.long)
                attn_masks_fill_0_pos = 0
            if attn_masks_fill_0_pos is not None:
                attn_masks.data[attn_masks_fill_0_pos].fill_(0)

            frame_level_input_ids.append(torch.tensor(curr_sub_ctx_input_ids))
            frame_level_attn_masks.append(attn_masks)
            frame_level_v_feats.append(matched_v_feats)
        while len(sub_queries_and_targets) < self.query_per_video:
            sub_queries_and_targets.append(
                copy.deepcopy(sub_queries_and_targets[-1]))
        clip_level_v_feats = v_feat
        clip_level_attn_masks = [1] * len(clip_level_v_feats)
        clip_level_attn_masks = torch.tensor(clip_level_attn_masks)
        video_inputs = (frame_level_input_ids, frame_level_v_feats,
                        frame_level_attn_masks,
                        clip_level_v_feats, clip_level_attn_masks,
                        num_subs, sub2frames)
        out = (video_inputs, vid, tuple(sub_queries_and_targets))

        return out


def vsm_collate(inputs):
    (video_inputs, vids, sub_queries_and_targets) = map(list, unzip(inputs))
    (input_ids, attn_masks, sub_vids, targets) = map(
        list, unzip(concat(outs for outs in sub_queries_and_targets)))

    batch = video_collate(video_inputs)
    vid2idx = {vid: i for i, vid in enumerate(vids)}
    batch["q_vidx"] = torch.tensor([vid2idx[s_vid] for s_vid in sub_vids],
                                   dtype=torch.long)

    # text batches
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    vsm_targets = pad_sequence(
        targets, batch_first=True, padding_value=-1)
    batch.update({
        'query_input_ids': input_ids,
        'query_pos_ids': position_ids,
        'query_attn_masks': attn_masks,
        'targets': vsm_targets,
        'vids': vids})

    return batch
