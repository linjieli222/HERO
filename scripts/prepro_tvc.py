"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
preprocess TVC annotations into LMDB
"""
import argparse
from collections import defaultdict
import json
import os
from os.path import exists

from cytoolz import curry
from tqdm import tqdm
from transformers import RobertaTokenizer

# quick hack for import
import sys
sys.path.insert(0, '/src')
from data.data import open_lmdb


@curry
def roberta_tokenize(tokenizer, text):
    words = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(words)
    return ids


def _compute_overlapped_subs(ts, subtitles):
    st, ed = ts
    inds = []
    for i, sub in enumerate(subtitles):
        if (st < sub['start'] < ed
                or st < sub['end'] < ed
                or sub['start'] < st < ed < sub['end']):
            inds.append(i)
    return inds


def process_tvc(cap_jsonl, sub_jsonl, cap_db, clip_db, tokenizer):
    # load subtitles
    vid2subs = {}
    for line in tqdm(sub_jsonl):
        sub_info = json.loads(line)
        vid2subs[sub_info['vid_name']] = sub_info['sub']

    id2len = {}
    cap2vid = {}
    clip2vid = {}
    vid2caps = defaultdict(list)
    vid2clips = defaultdict(list)
    for line in tqdm(cap_jsonl, desc='processing TVC data'):
        example = json.loads(line)
        vid = example['vid_name']
        ts = example['ts']
        clip_id = str(example['clip_id'])
        clip2vid[clip_id] = vid
        sub_inds = _compute_overlapped_subs(ts, vid2subs[vid])
        clip = {'vid_name': vid, 'ts': ts, 'sub_indices': sub_inds,
                'duration': example['duration'], 'captions': []}
        vid2clips[vid].append(clip_id)
        for cap in example['descs']:
            cap_id = str(cap['desc_id'])
            input_ids = tokenizer(cap['desc'])
            cap['input_ids'] = input_ids
            cap['vid_name'] = vid
            cap['clip_id'] = clip_id
            cap['ts'] = ts
            cap['sub_indices'] = sub_inds
            cap['duration'] = example['duration']
            cap_db[cap_id] = cap
            cap2vid[cap_id] = vid
            vid2caps[vid].append(cap_id)

            clip['captions'].append({'id': cap['desc_id'],
                                     'input_ids': input_ids,
                                     'text': cap['desc']})

        clip_db[clip_id] = clip
    return id2len, cap2vid, clip2vid, vid2caps, vid2clips


def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        print(opts.output)
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    toker = RobertaTokenizer.from_pretrained(opts.toker)
    tokenizer = roberta_tokenize(toker)
    meta['BOS'] = toker.convert_tokens_to_ids(['<s>'])[0]
    meta['EOS'] = toker.convert_tokens_to_ids(['</s>'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['</s>'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['<s>'])[0]
    meta['PAD'] = toker.convert_tokens_to_ids(['<pad>'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['<mask>'])[0]
    meta['UNK'] = toker.convert_tokens_to_ids(['<unk>'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids(['.'])[0],
                       toker.convert_tokens_to_ids(['<|endoftext|>'])[0]+1)
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_cap_db = curry(open_lmdb, f"{opts.output}/cap.db", readonly=False)
    open_clip_db = curry(open_lmdb, f"{opts.output}/clip.db", readonly=False)
    with open_cap_db() as cap_db, open_clip_db() as clip_db:
        with open(opts.annotation) as ann, open(opts.subtitles) as sub:
            (id2lens, cap2vid, clip2vid, vid2caps, vid2clips
             ) = process_tvc(ann, sub, cap_db, clip_db, tokenizer)

    with open(f'{opts.output}/cap.db/id2len.json', 'w') as f:
        json.dump(id2lens, f)
    with open(f'{opts.output}/cap.db/cap2vid.json', 'w') as f:
        json.dump(cap2vid, f)
    with open(f'{opts.output}/clip.db/clip2vid.json', 'w') as f:
        json.dump(clip2vid, f)
    with open(f'{opts.output}/cap.db/vid2caps.json', 'w') as f:
        json.dump(vid2caps, f)
    with open(f'{opts.output}/clip.db/vid2clips.json', 'w') as f:
        json.dump(vid2clips, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    parser.add_argument('--subtitles', required=True,
                        help='subtitle JSON')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--toker', default='roberta-base',
                        choices=["roberta-base", "roberta-large"],
                        help='which RoBerTa tokenizer to used')
    args = parser.parse_args()
    main(args)
