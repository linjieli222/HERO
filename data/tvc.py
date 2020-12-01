"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

TVC dataset

NOTE: known issue: can't handle video segments after 150 seconds
"""
from collections import defaultdict
import json
import math
import random

from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import horovod.torch as hvd

from .data import (VideoFeatSubTokDataset, TxtLmdb,
                   video_collate, _check_ngpu)


class CaptionTokLmdb(object):
    def __init__(self, db_dir, max_txt_len=-1):
        self.db_dir = db_dir
        self.cap_db = TxtLmdb(f"{db_dir}/cap.db", readonly=True)
        self.clip_db = TxtLmdb(f"{db_dir}/clip.db", readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.pad = meta['PAD']
        self.bos = meta['BOS']
        self.eos = meta['EOS']
        self.max_txt_len = max_txt_len

    def __getitem__(self, id_):
        return self.get_caption(id_)

    def get_caption(self, id_):
        txt_dump = self.cap_db[id_]
        cap_input_ids = txt_dump['input_ids']
        input_ids = [self.bos] + cap_input_ids
        tgt_ids = cap_input_ids + [self.eos]
        if self.max_txt_len != -1:
            input_ids = input_ids[:self.max_txt_len]
            tgt_ids = tgt_ids[:self.max_txt_len]
        txt_dump['input_ids'] = torch.tensor(input_ids)
        txt_dump['tgt_ids'] = torch.tensor(tgt_ids)
        return txt_dump

    def get_clip(self, id_):
        txt_dump = self.clip_db[id_]
        return txt_dump

    @property
    def cap2vid(self):
        return json.load(open(f'{self.db_dir}/cap.db/cap2vid.json'))

    @property
    def clip2vid(self):
        return json.load(open(f'{self.db_dir}/clip.db/clip2vid.json'))

    @property
    def vid2caps(self):
        return json.load(open(f'{self.db_dir}/cap.db/vid2caps.json'))

    @property
    def vid2clips(self):
        return json.load(open(f'{self.db_dir}/clip.db/vid2clips.json'))


class TvcTrainDataset(Dataset):
    def __init__(self, video_db, caption_db, max_cap_per_vid=-1):
        assert isinstance(video_db, VideoFeatSubTokDataset)
        assert isinstance(caption_db, CaptionTokLmdb)
        self.video_db = video_db
        self.caption_db = caption_db

        self.vid2caps = caption_db.vid2caps
        self.vids = list(self.vid2caps.keys())
        if _check_ngpu() > 1:
            # partition data by rank
            self.vids = self.vids[hvd.rank()::hvd.size()]

        self.max_cap_per_vid = max_cap_per_vid
        self.vid2dur = video_db.vid2dur
        self.vid2idx = video_db.vid2idx
        self.max_clip_len = video_db.txt_db.max_clip_len
        self.frame_interval = video_db.img_db.frame_interval

    def getids(self, i):
        vid = self.vids[i]
        # TVR video loss assumes fix number of queries
        cap_ids = self.vid2caps[vid]
        if self.max_cap_per_vid != -1 and len(cap_ids) > self.max_cap_per_vid:
            # random sample some captions if too many
            cap_ids = random.sample(cap_ids, self.max_cap_per_vid)
        return vid, cap_ids

    def __getitem__(self, i):
        vid, cids = self.getids(i)

        video_inputs = self.video_db.__getitem__(vid)
        nframes = len(video_inputs[4])  # clip_level_v_feats

        clip_ranges = []
        cap_inputs = []
        for cid in cids:
            ex = self.caption_db[cid]
            st, ed = self.get_st_ed_label(ex['ts'], nframes)
            clip_ranges.append((st, ed))
            attn_mask = torch.tensor([1]*(ed-st))
            cap_inputs.append((ex['input_ids'], ex['tgt_ids'], attn_mask))
        return video_inputs, clip_ranges, cap_inputs

    def __len__(self):
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
        # ed_idx = min(max(math.ceil(ts[1]/self.frame_interval)-1,
        #                  st_idx+1), max_idx)
        ed_idx = min(max(round(ts[1]/self.frame_interval),
                         st_idx+1), max_idx)
        return st_idx, ed_idx

    @staticmethod
    def collate(inputs):
        video_inputs, all_clip_ranges, cap_inputs = map(list, unzip(inputs))

        (all_input_ids, all_tgt_ids, all_attn_masks
         ) = map(list, unzip(concat(outs for outs in cap_inputs)))
        input_ids = pad_sequence(all_input_ids,
                                 batch_first=True, padding_value=1)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)
        tgt_ids = pad_sequence(all_tgt_ids, batch_first=True, padding_value=-1)
        attn_mask = pad_sequence(all_attn_masks,
                                 batch_first=True, padding_value=0)
        batch = {'cap_input_ids': input_ids,
                 'cap_pos_ids': position_ids,
                 'cap_tgt_ids': tgt_ids,
                 'cap_attn_mask': attn_mask,
                 'clip_ranges': tuple(map(tuple, all_clip_ranges))}

        vid_batch = video_collate(video_inputs)
        batch.update(vid_batch)
        return batch


class TvcValDataset(TvcTrainDataset):
    """ for validation """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vid2clips = self.caption_db.vid2clips

    def __getitem__(self, i):
        vid = self.vids[i]
        clip_ids = self.vid2clips[vid]

        video_inputs = self.video_db.__getitem__(vid)
        nframes = len(video_inputs[4])  # clip_level_v_feats

        clip_ranges = []
        all_ts = []
        gts = []
        attn_masks = []
        for clip_id in clip_ids:
            ex = self.caption_db.get_clip(clip_id)
            ts = ex['ts']
            st, ed = self.get_st_ed_label(ts, nframes)
            clip_ranges.append((st, ed))
            attn_masks.append(torch.tensor([1]*(ed-st)))
            all_ts.append(ts)
            gts.append(ex['captions'][0]['text'])
        return (video_inputs, clip_ranges, attn_masks,
                (vid, clip_ids, all_ts, gts))

    @staticmethod
    def collate(inputs):
        (video_inputs, all_clip_ranges, attn_masks_list, metas
         ) = map(list, unzip(inputs))

        all_attn_masks = list(concat(attn_masks_list))
        attn_mask = pad_sequence(all_attn_masks,
                                 batch_first=True, padding_value=0)
        batch = {'cap_attn_mask': attn_mask,
                 'clip_ranges': tuple(map(tuple, all_clip_ranges))}

        vid_batch = video_collate(video_inputs)
        batch.update(vid_batch)

        # meta
        vids, clip_ids, all_ts, all_gts = [], [], [], []
        for vid, cids, tss, gts in metas:
            for cid, ts, gt in zip(cids, tss, gts):
                vids.append(vid)
                clip_ids.append(int(cid))
                all_ts.append(ts)
                all_gts.append(gt)
        batch['vid_names'] = vids
        batch['clip_ids'] = clip_ids
        batch['all_ts'] = all_ts
        batch['gts'] = all_gts
        return batch


class TvcEvalDataset(TvcTrainDataset):
    """ for generating submission from JSON input
    """
    def __init__(self, video_db, data_jsonl):
        self.video_db = video_db

        self.vid2clips = defaultdict(list)
        self.clip2ex = {}
        for line in open(data_jsonl):
            example = json.loads(line)
            vid = example['vid_name']
            clip_id = example['clip_id']
            self.vid2clips[vid].append(clip_id)
            self.clip2ex[clip_id] = example
        self.vids = list(self.vid2clips.keys())

        if _check_ngpu() > 1:
            # partition data by rank
            self.vids = self.vids[hvd.rank()::hvd.size()]

        self.vid2dur = video_db.vid2dur
        self.vid2idx = video_db.vid2idx
        self.max_clip_len = video_db.txt_db.max_clip_len
        self.frame_interval = video_db.img_db.frame_interval

    def __getitem__(self, i):
        vid = self.vids[i]
        clip_ids = self.vid2clips[vid]

        video_inputs = self.video_db.__getitem__(vid)
        nframes = len(video_inputs[4])  # clip_level_v_feats

        clip_ranges = []
        all_ts = []
        attn_masks = []
        for clip_id in clip_ids:
            ex = self.clip2ex[clip_id]

            ts = ex['ts']
            st, ed = self.get_st_ed_label(ts, nframes)
            clip_ranges.append((st, ed))
            attn_masks.append(torch.tensor([1]*(ed-st)))
            all_ts.append(ts)
        return (video_inputs, clip_ranges, attn_masks,
                (vid, clip_ids, all_ts))

    @staticmethod
    def collate(inputs):
        (video_inputs, all_clip_ranges, attn_masks_list, metas
         ) = map(list, unzip(inputs))

        all_attn_masks = list(concat(attn_masks_list))
        attn_mask = pad_sequence(all_attn_masks,
                                 batch_first=True, padding_value=0)
        batch = {'cap_attn_mask': attn_mask,
                 'clip_ranges': tuple(map(tuple, all_clip_ranges))}

        vid_batch = video_collate(video_inputs)
        batch.update(vid_batch)

        # meta
        vids, clip_ids, all_ts = [], [], []
        for vid, cids, tss in metas:
            for cid, ts in zip(cids, tss):
                vids.append(vid)
                clip_ids.append(int(cid))
                all_ts.append(ts)
        batch['vid_names'] = vids
        batch['clip_ids'] = clip_ids
        batch['all_ts'] = all_ts
        return batch
