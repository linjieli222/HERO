"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pretrain MLM  dataset
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from toolz.sandbox import unzip
from cytoolz import concat
import horovod.torch as hvd
import copy

from .data import (VideoFeatSubTokDataset,
                   pad_tensors, get_gather_index, _check_ngpu)


def random_word(tokens, vocab_range, mask, mask_prob=0.15):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_prob:
            prob /= mask_prob

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


def _get_txt_tgt_mask(txt_mask, n_frame):
    z = torch.zeros(n_frame, dtype=torch.bool)
    txt_mask_tgt = torch.cat([z, txt_mask], dim=0)
    return txt_mask_tgt


def create_mlm_io(input_ids, db, mask_prob, cls_tok=True):
    input_ids, txt_labels = random_word(
        input_ids, db.v_range, db.mask, mask_prob)
    if cls_tok:
        input_ids = [db.cls_] + input_ids
    else:
        input_ids = [db.sep] + input_ids
    txt_labels = torch.tensor([-1] + txt_labels)
    return input_ids, txt_labels


class VideoMlmDataset(Dataset):
    def __init__(self, video_ids, vid_sub_db, mask_prob=0.15,
                 sub_ctx_len=0):
        assert isinstance(vid_sub_db, VideoFeatSubTokDataset)
        self.mask_prob = mask_prob
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
        num_subs = len(sub2frames)
        outputs = []
        for sub_idx, matched_frames in sub2frames:
            # text input
            orig_input_ids = []
            for tmp_sub_idx in range(sub_idx-self.sub_ctx_len,
                                     sub_idx+1):
                if tmp_sub_idx >= 0 and tmp_sub_idx < num_subs:
                    in_ids = example['input_ids'][tmp_sub_idx]
                    if self.vid_sub_db.max_txt_len != -1:
                        in_ids = in_ids[:self.vid_sub_db.max_txt_len]
                    orig_input_ids.extend(copy.deepcopy(in_ids))
            input_ids, txt_labels = create_mlm_io(
                orig_input_ids, self.vid_sub_db.txt_db,
                self.mask_prob)

            # video input
            n_frame = len(matched_frames)
            if n_frame:
                matched_v_feats = torch.index_select(
                    v_feat, 0, torch.tensor(matched_frames))
                attn_masks = torch.ones(len(input_ids) + n_frame,
                                        dtype=torch.long)
                txt_mask_tgt = _get_txt_tgt_mask(txt_labels != -1, n_frame)
            else:
                matched_v_feats = torch.zeros(1, v_feat.shape[1])
                attn_masks = torch.ones(len(input_ids) + 1, dtype=torch.long)
                attn_masks.data[0].fill_(0)
                txt_mask_tgt = _get_txt_tgt_mask(txt_labels != -1, 1)
            input_ids = torch.tensor(input_ids)
            outputs.append((input_ids, matched_v_feats, attn_masks,
                            txt_mask_tgt, txt_labels))

        return outputs


def mlm_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (input_ids, v_feats, attn_masks, txt_masks, txt_labels
     ) = map(list, unzip(concat(inputs)))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    txt_mask_tgt = pad_sequence(txt_masks, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_fs = [f.size(0) for f in v_feats]
    v_feat = pad_tensors(v_feats, num_fs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_vl, _ = v_feat.size()
    out_size = attn_masks.size(1)
    if max_vl > 0:
        gather_index = get_gather_index(txt_lens, num_fs, bs, max_vl, out_size)
    else:
        gather_index = None
        v_feat = None

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'v_feat': v_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_mask_tgt': txt_mask_tgt,
             'txt_labels': txt_labels[txt_labels != -1]}
    return batch
