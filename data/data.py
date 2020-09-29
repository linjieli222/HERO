"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Dataset interfaces
"""
from torch.nn.utils.rnn import pad_sequence
from contextlib import contextmanager
import io
import json
from utils.basic_utils import load_jsonl
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
import horovod.torch as hvd
from tqdm import tqdm
import lmdb
from lz4.frame import compress, decompress
from toolz.sandbox import unzip
import os
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


def _fp16_to_fp32(feat_dict):
    out = {k: arr.astype(np.float32)
           if arr.dtype == np.float16 else arr
           for k, arr in feat_dict.items()}
    return out


def _check_distributed():
    try:
        dist = hvd.size() != hvd.local_size()
    except ValueError:
        # not using horovod
        dist = False
    return dist


def _check_ngpu():
    try:
        n_gpu = hvd.size()
    except ValueError:
        # not using horovod
        n_gpu = 1
    return n_gpu


class VideoFeatLmdb(object):
    def __init__(self, img_dir, feat_version="resnet_slowfast",
                 frame_interval=1.5, compress=True, max_clip_len=-1):
        self.img_dir = img_dir
        db_name = f'{feat_version}_{frame_interval}'
        self.name2nframe = json.load(
            open(f'{img_dir}/'
                 f'id2nframe.json', "r"))
        self.frame_interval = frame_interval
        self.pad = 0
        self.cls_ = 1
        self.mask = 2
        self.compress = compress
        self.max_clip_len = max_clip_len
        if compress:
            db_name += '_compressed'

        # only read ahead on single node training
        self.env = lmdb.open(f'{img_dir}/{db_name}',
                             readonly=True, create=False,
                             max_readers=4096 * 8,
                             readahead=False)
        self.txn = self.env.begin(buffers=True)
        if self.name2nframe is None:
            self.name2nframe = self._compute_nframe()

    def _compute_nframe(self):
        name2nframe = {}
        fnames = json.loads(self.txn.get(key=b'__keys__').decode('utf-8'))
        for fname in tqdm(fnames, desc='reading images'):
            dump = self.txn.get(fname.encode('utf-8'))
            if self.compress:
                with io.BytesIO(dump) as reader:
                    img_dump = np.load(reader, allow_pickle=True)
                    features = img_dump['features']
            else:
                img_dump = msgpack.loads(dump, raw=False)
                features = img_dump['features']
            nframe = len(features)
            name2nframe[fname] = self.max_clip_len\
                if nframe > self.max_clip_len else nframe

        return name2nframe

    def __del__(self):
        self.env.close()

    def get_dump(self, file_name):
        dump = self.txn.get(file_name.encode('utf-8'))
        if self.compress:
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                img_dump = _fp16_to_fp32(img_dump)
        else:
            img_dump = msgpack.loads(dump, raw=False)
            img_dump = _fp16_to_fp32(img_dump)
        return img_dump

    def __getitem__(self, file_name):
        dump = self.txn.get(file_name.encode('utf-8'))
        nframes = self.name2nframe[file_name]
        nframes = nframes if nframes < self.max_clip_len else self.max_clip_len
        if self.compress:
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                img_dump = {'features': img_dump['features']}
        else:
            img_dump = msgpack.loads(dump, raw=False)
        img_feat = torch.tensor(img_dump['features'][:nframes]).float()
        # img_feat_normalized = F.normalize(img_feat, dim=1)
        return img_feat


@contextmanager
def open_lmdb(db_dir, readonly=False):
    db = TxtLmdb(db_dir, readonly)
    try:
        yield db
    finally:
        del db


class TxtLmdb(object):
    def __init__(self, db_dir, readonly=True):
        self.readonly = readonly
        if readonly:
            # training
            self.env = lmdb.open(db_dir,
                                 readonly=True, create=False,
                                 max_readers=4096,
                                 readahead=False)
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            # prepro
            self.env = lmdb.open(db_dir, readonly=False, create=True,
                                 map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        return msgpack.loads(decompress(self.txn.get(key.encode('utf-8'))),
                             raw=False)

    def __setitem__(self, key, value):
        # NOTE: not thread safe
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'),
                           compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret


class TxtTokLmdb(object):
    def __init__(self, db_dir, max_txt_len=60):
        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.eos = meta['EOS']
        self.pad = meta['PAD']
        self.bos = meta['BOS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']
        id2len_path = f'{db_dir}/id2len.json'
        if os.path.exists(id2len_path):
            if max_txt_len == -1:
                self.id2len = json.load(open(f'{db_dir}/id2len.json'))
            else:
                self.id2len = {
                    id_: len_
                    for id_, len_ in json.load(
                        open(f'{db_dir}/id2len.json')).items()
                    if isinstance(len_, int) and len_ <= max_txt_len
                    or isinstance(len_, list) and
                    len_[0] + max(len_[1:]) <= max_txt_len
                }
        else:
            self.id2len = None

    def __getitem__(self, id_):
        txt_dump = self.db[id_]
        return txt_dump

    def combine_inputs(self, *inputs):
        input_ids = [self.cls_]
        for ids in inputs:
            input_ids.extend(ids + [self.sep])
        return torch.tensor(input_ids)


class SubTokLmdb(TxtTokLmdb):
    def __init__(self, db_dir, max_clip_len=-1):
        super().__init__(db_dir, max_txt_len=-1)
        self.max_clip_len = max_clip_len
        self.vid2max_len = json.load(
            open(f'{db_dir}/vid2max_frame_sub_len.json'))
        self.id2len = json.load(
            open(f'{db_dir}/vid2len.json'))
        self.vid2dur, self.vid2idx = {}, {}
        video_data_file = f'{db_dir}/vid2dur_idx.json'
        if os.path.exists(video_data_file):
            video_data = json.load(open(video_data_file, "r"))
            for key, info in video_data.items():
                self.vid2dur[key] = [
                    {"vid_name": k, "duration": v[0]} for k, v in info.items()]
                self.vid2idx[key] = {k: v[1] for k, v in info.items()}
        # else:
        #     raise ValueError(f"vid2dur_idx.json does not exists in {db_dir}")
        self.vid_sub2frame, self.vid2vonly_frames =\
            self.compute_sub2frames()

    def compute_sub2frames(self):
        vid_sub2frame = {}
        vid2vonly_frames = {}
        for vid in tqdm(list(self.id2len.keys()), desc='reading subtitles'):
            ex = self.db[vid]
            if 'unmatched_frames' not in ex:
                unmatched_frames = []
            else:
                unmatched_frames = ex["unmatched_frames"]
            curr_sub2frame = ex["unique_sub2frames"]
            cutoff = False

            sen2frame = []
            for _, (sub_idx, matched_frames) in enumerate(curr_sub2frame):
                if self.max_clip_len > -1:
                    in_range = []
                    for i in sorted(matched_frames):
                        if i < self.max_clip_len:
                            in_range.append(i)
                        else:
                            cutoff = True
                    if len(in_range) or len(matched_frames) == 0:
                        sen2frame.append((sub_idx, in_range))
                    if cutoff:
                        break
                else:
                    sen2frame.append((sub_idx, matched_frames))
            unmatched_frames = [i for i in unmatched_frames
                                if i < self.max_clip_len]
            self.id2len[vid] = min(self.id2len[vid], self.max_clip_len)

            vid_sub2frame[vid] = sen2frame
            vid2vonly_frames[vid] = unmatched_frames
        return vid_sub2frame, vid2vonly_frames


class QueryTokLmdb(TxtTokLmdb):
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
        if os.path.exists(f'{self.db_dir}/query_data.jsonl'):
            query_data = load_jsonl(
                f'{self.db_dir}/query_data.jsonl')
            self.query_data = {
                str(item["desc_id"]): item for item in query_data}
        else:
            self.query_data = {}

    def __getitem__(self, id_):
        txt_dump = self.db[id_]
        return txt_dump


class QaQueryTokLmdb(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len=-1):
        super().__init__(db_dir, max_txt_len=max_txt_len)
        self.query2video = json.load(open(f'{self.db_dir}/query2video.json'))
        self.video2query = {}
        for k, v in self.query2video.items():
            if v not in self.video2query:
                self.video2query[v] = [k]
            else:
                self.video2query[v].append(k)

    def __getitem__(self, id_):
        txt_dump = self.db[id_]
        return txt_dump


def get_ids_and_lens(db):
    assert isinstance(db, TxtTokLmdb)
    assert db.id2len is not None, "id2len does not exists"
    lens = []
    ids = []
    for id_, len_ in db.id2len.items():
        lens.append(len_)
        ids.append(id_)
    return lens, ids


class VideoFeatSubTokDataset(Dataset):
    def __init__(self, txt_db, img_db, max_txt_len=60, sub_ctx_len=0):
        assert isinstance(txt_db, SubTokLmdb)
        assert isinstance(img_db, VideoFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.max_txt_len = max_txt_len
        assert self.txt_db.max_clip_len == self.img_db.max_clip_len
        self.max_clip_len = self.img_db.max_clip_len
        self.clip_lens, self.vids = get_ids_and_lens(txt_db)  # num.frames, vid
        self.vid_sub2frame, self.vid2vonly_frames =\
            self.txt_db.vid_sub2frame, self.txt_db.vid2vonly_frames
        self.vid2dur = self.txt_db.vid2dur
        self.vid2idx = self.txt_db.vid2idx
        self.sub_ctx_len = sub_ctx_len
        assert self.sub_ctx_len >= 0

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, vid_):
        '''
        matched_sub_frames:
        [[txt, txt_position_ids, v_feat(frames), v_position_ids, attn_masks],
         [txt, txt_position_ids, v_feat(frames), v_position_ids, attn_masks]]
        all_frames:[v_feat(frames)]
        clip_level_position_ids
        clip_level_attn_masks
        '''
        example = self.txt_db[vid_]
        v_feat, nframes = self._get_v_feat(vid_)

        frame_level_input_ids, frame_level_v_feats = [], []
        frame_level_attn_masks = []  # [(fffwww)]
        sub2frames = self.vid_sub2frame[vid_]  # sub_ix -> [frame_ix]
        num_subs = len(sub2frames)

        for sub_idx, matched_frames in sub2frames:
            # text input
            input_ids = []
            input_ids = [self.txt_db.sep] + input_ids
            for tmp_sub_idx in range(sub_idx-self.sub_ctx_len,
                                     sub_idx+1):
                if tmp_sub_idx >= 0 and tmp_sub_idx < num_subs:
                    input_ids.extend(
                        copy.deepcopy(
                            example['input_ids'][tmp_sub_idx]))

            matched_frames = [f_idx for f_idx in matched_frames
                              if f_idx in range(v_feat.shape[0])]
            if len(matched_frames):
                matched_v_feats = torch.index_select(
                    v_feat, 0, torch.tensor(matched_frames))
                attn_masks = [1] * (len(input_ids) + len(matched_frames))
            else:
                matched_v_feats = torch.zeros(1, v_feat.shape[1])
                attn_masks = [0] + [1] * len(input_ids)

            frame_level_input_ids.append(torch.tensor(input_ids))
            frame_level_attn_masks.append(torch.tensor(attn_masks))
            frame_level_v_feats.append(matched_v_feats)

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


def video_collate(inputs):
    (frame_level_input_ids,
     frame_level_v_feats,
     frame_level_attn_masks,
     clip_level_v_feats,
     clip_level_attn_masks, num_subs,
     sub_idx2frame_idx) = map(list, unzip(inputs))

    # all_f_sub_input_ids: list[tensor(sep, w0, w1)]
    # whose list size = total number of subs
    all_f_sub_input_ids, all_f_v_feats, all_f_attn_masks = [], [], []
    for i in range(len(num_subs)):
        all_f_sub_input_ids += frame_level_input_ids[i]
        all_f_v_feats += frame_level_v_feats[i]
        all_f_attn_masks += frame_level_attn_masks[i]

    txt_lens = [i.size(0) for i in all_f_sub_input_ids]  # len. of each sub
    # hard_coded padding value, TODO: check correctness
    all_f_sub_input_ids = pad_sequence(
        all_f_sub_input_ids, batch_first=True, padding_value=1)

    all_f_sub_pos_ids = torch.arange(0, all_f_sub_input_ids.size(1),
                                     dtype=torch.long).unsqueeze(0)
    all_f_sub_pos_ids.data[all_f_sub_pos_ids > 511] = 511  # FIXME quick hack
    all_f_attn_masks = pad_sequence(
        all_f_attn_masks, batch_first=True, padding_value=0)

    v_lens = [i.size(0) for i in all_f_v_feats]
    all_f_v_feats = pad_tensors(all_f_v_feats, v_lens, 0)
    all_f_v_pos_ids = torch.arange(0, all_f_v_feats.size(1), dtype=torch.long
                                   ).unsqueeze(0)

    # all_f_sub_input_attn_masks (total_subs, max_sl) for subtitles only
    all_f_sub_input_attn_masks = [torch.tensor([1] * tl) for tl in txt_lens]
    all_f_sub_input_attn_masks = pad_sequence(
        all_f_sub_input_attn_masks, batch_first=True, padding_value=0)

    # TODO: How to calculate gather index at frame_level
    bs, max_vl, _ = all_f_v_feats.size()
    out_size = all_f_attn_masks.size(1)
    frame_level_gather_index = get_gather_index(
        txt_lens, v_lens, bs, max_vl, out_size)

    num_frames = [i.size(0) for i in clip_level_v_feats]
    clip_level_v_feats = pad_tensors(
        clip_level_v_feats, num_frames, pad=0)
    clip_level_pos_ids = torch.arange(
        0, clip_level_v_feats.size(1), dtype=torch.long
    ).unsqueeze(0).expand(clip_level_v_feats.size(0), -1).clone()

    clip_level_attn_masks = pad_sequence(
            clip_level_attn_masks, batch_first=True, padding_value=0)

    batch = {'f_sub_input_ids': all_f_sub_input_ids,  # (total_sub, max_sl)
             'f_sub_pos_ids': all_f_sub_pos_ids,      # (total_sub, max_sl)
             'f_v_feats': all_f_v_feats,              # (total_sub, max_vl, k)
             'f_v_pos_ids': all_f_v_pos_ids,          # (total_sub, max_vl)
             'f_attn_masks': all_f_attn_masks,        # (total_sub, max_vl+max_sl)
             'f_gather_index': frame_level_gather_index,  # (total_sub, max_vl+max_sl)
             'f_sub_input_attn_masks': all_f_sub_input_attn_masks, # (total_sub, max_sl)
             'c_v_feats': clip_level_v_feats,         # (bz, max_len, k)
             'c_pos_ids': clip_level_pos_ids,         # (bz, max_len) [matched, unmatched]
             'c_attn_masks': clip_level_attn_masks,   # (bz, max_len)
             'num_subs': num_subs,                    # [num_sub]
             'sub_idx2frame_idx': sub_idx2frame_idx}  # [ [(sub_ix, [frame_ix]) ] ]
    return batch


def txt_input_collate(input_ids, attn_masks):
    # hard_coded padding value, TODO: check correctness
    pad_values = 1 if len(input_ids[0].size()) == 1 else 0
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=pad_values)
    pos_ids = torch.arange(
        0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    pos_ids.data[pos_ids > 511] = 511  # FIXME quick hack
    attn_masks = pad_sequence(
        attn_masks, batch_first=True, padding_value=0)
    return input_ids, pos_ids, attn_masks


def pad_tensors(tensors, lens=None, pad=0, max_len=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    if max_len == 0:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, hid, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


def get_gather_index(txt_lens, num_frames, batch_size, max_len, out_size):
    assert len(txt_lens) == len(num_frames) == batch_size
    gather_index = torch.arange(0, out_size, dtype=torch.long
                                ).unsqueeze(0).expand(batch_size, -1).clone()

    for i, (tl, nframe) in enumerate(zip(txt_lens, num_frames)):
        gather_index.data[i, nframe:tl+nframe] = torch.arange(
            max_len, max_len+tl, dtype=torch.long).data
    return gather_index
