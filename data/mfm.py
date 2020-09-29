"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pretrain MFM  dataset
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from toolz.sandbox import unzip
from cytoolz import concat
import horovod.torch as hvd

from .data import VideoFeatSubTokDataset, video_collate, _check_ngpu


def _get_img_mask(mask_prob, num_frame):
    img_mask = [random.random() < mask_prob for _ in range(num_frame)]
    if not any(img_mask):
        # at least mask 1
        img_mask[random.choice(range(num_frame))] = True
    img_mask = torch.tensor(img_mask)
    return img_mask


def _get_feat_target(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)  # (n, m, d)
    feat_dim = img_feat.size(-1)
    feat_targets = img_feat[img_masks_ext].contiguous().view(
        -1, feat_dim)  # (s, d)
    return feat_targets


def _mask_img_feat(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked


class MfmDataset(Dataset):
    def __init__(self, video_ids, vid_sub_db, mask_prob=0.15):
        assert isinstance(vid_sub_db, VideoFeatSubTokDataset)
        self.mask_prob = mask_prob
        self.vid_sub_db = vid_sub_db
        if _check_ngpu() > 1:
            self.ids = video_ids[hvd.rank()::hvd.size()]
        else:
            self.ids = video_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        vid = self.ids[i]
        (all_input_ids, f_v_feats, f_attn_masks,
         c_v_feats, c_attn_masks,
         num_subs, sub2frames) = self.vid_sub_db[vid]

        c_frame_mask = _get_img_mask(self.mask_prob, c_v_feats.size(0))
        frame_masks = []
        for i, frames in sub2frames:
            if len(frames):
                frame_masks.append(
                    c_frame_mask.index_select(0, torch.tensor(frames)))
            else:
                frame_masks.append(torch.zeros(1, dtype=torch.bool))
        c_pos_ids = torch.tensor(range(len(c_v_feats)), dtype=torch.long)
        c_frame_mask = c_frame_mask.index_select(0, c_pos_ids)
        return ((all_input_ids, f_v_feats, f_attn_masks,
                 c_v_feats, c_attn_masks,
                 num_subs, sub2frames),
                frame_masks, c_frame_mask)


def mfm_collate(inputs):
    video_inputs, all_frame_masks, c_frame_masks = map(list, unzip(inputs))
    batch = video_collate(video_inputs)

    # mask features
    frame_masks = pad_sequence(list(concat(all_frame_masks)),
                               batch_first=True, padding_value=0)
    c_frame_masks = pad_sequence(c_frame_masks,
                                 batch_first=True, padding_value=0)
    f_v_feats = batch['f_v_feats']
    f_v_feats = _mask_img_feat(f_v_feats, frame_masks)
    c_v_feats = batch['c_v_feats']
    feat_targets = _get_feat_target(c_v_feats, c_frame_masks)
    c_v_feats = _mask_img_feat(c_v_feats, c_frame_masks)

    batch['f_v_feats'] = f_v_feats
    batch['f_v_masks'] = frame_masks
    batch['c_v_feats'] = c_v_feats
    batch['c_v_masks'] = c_frame_masks
    batch['feat_targets'] = feat_targets
    return batch
