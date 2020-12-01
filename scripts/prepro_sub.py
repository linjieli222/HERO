"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Modified from from TVRetrieval implementation
(https://github.com/jayleicn/TVRetrieval)

preprocess TV subtitles into LMDB
"""
import argparse
import json
import os
from os.path import exists
import numpy as np
from cytoolz import curry
from tqdm import tqdm
from transformers import RobertaTokenizer
from utils.basic_utils import (
    flat_list_of_lists, load_jsonl, save_json, load_json)
from data.data import open_lmdb
import copy
import html


def load_process_sub_meta(sub_meta_path, vid2nframe, frame_length):
    """ which subtitles should be assigned to which frames
    Args:
        sub_meta_path: contains a jsonl file, each line is a dict
            {"vid_name": str, "sub": list(dicts)},
            each dict under "sub" is, e.g.,
            {'text': " Chase : That's all this is?",
             'start': 0.862, 'end': 1.862}.
            The dicts under "sub" are ordered
            the same as the original .srt files.
        frame_length: float, assign each subtitle to a frame segment
    Returns:
    """
    video2sub = {e["vid_name"]: e for e in load_jsonl(sub_meta_path)}
    total_overlapped_sub, total_sub = 0, 0
    max_sub_length, extra_long_subs = 0, 0
    max_gap_time, max_sub_duration = 0, 0
    max_matched_frame_len, max_unmatched_group_len = 0, 0
    max_overlap_time = 0
    for vid_name, sub_info in tqdm(
            video2sub.items(), desc="processing subtitles"):
        if isinstance(vid2nframe[vid_name], int):
            num_of_frames = vid2nframe[vid_name]
            if num_of_frames == 0:
                num_of_frames = int(
                    int(sub_info["sub"][-1]["end"])/frame_length)
        else:
            raise ValueError(
                f"{vid_name} in vid2nframe, but with unexpected format:\n" +
                f"{vid2nframe[vid_name]}")
        info, overlapped_sub = process_single_vid_sub(
            sub_info["sub"], frame_length, num_of_frames)
        # sub_info.update(info)
        video2sub[vid_name] = info
        total_overlapped_sub += overlapped_sub
        total_sub += len(sub_info["sub"])
        max_sub_length = max(max_sub_length, info["max_sub_length"])
        max_matched_frame_len = max(
            max_matched_frame_len, info["max_matched_frame_len"])
        max_sub_duration = max(
            max_sub_duration, info["max_sub_duration"])
        max_gap_time = max(
            max_gap_time, info["max_gap_time"])
        max_unmatched_group_len = max(
            max_unmatched_group_len, info["max_unmatched_group_len"])
        max_overlap_time = max(max_overlap_time, info["max_overlap_time"])
        extra_long_subs += info["extra_long_subs"]
    print(f"overlap/total: {total_overlapped_sub}/{total_sub}")
    print(f"max subtitle length: {max_sub_length}")
    print(f"max subtitle duration: {max_sub_duration}")
    print(f"max overlap between two subtitles:{max_overlap_time}")
    print(f"max gap time between two subtitles: {max_gap_time}")
    print(f"max number of matched frames: {max_matched_frame_len}")
    print(f"max len of unmatched frame group: {max_unmatched_group_len}")
    print(f"extra long subs: {extra_long_subs}")
    return video2sub


def temporal_iou(time1, time2):
    start_time1, end_time1 = time1
    start_time2, end_time2 = time2
    min_start_time = min(start_time1, start_time2)
    max_start_time = max(start_time1, start_time2)
    min_end_time = min(end_time1, end_time2)
    max_end_time = max(end_time1, end_time2)
    if min_end_time <= max_start_time:
        return 0
    iou = (min_end_time-max_start_time)/(
        max_end_time-min_start_time)
    return iou


def process_single_vid_sub(sub_listdicts, frame_length, num_of_frames):
    """
    Args:
        sub_listdicts: list(dicts), each dict is, e.g.,
            {'text': " Chase : That's all this is?",
             'start': 0.862, 'end': 1.862}
        frame_length: float
    Returns:
        frame_idx2subtitle_indices: dict, {frame_idx: [sub_idx1, sub_idx2, ...]},
            which subtitles are associated with which frames.
            The indices are in ascending order, i.e., sub_idx1 < sub_idx2 < ...
    """
    if len(sub_listdicts) == 0 or num_of_frames == 0:
        info = {"num_of_frames": num_of_frames,
                "max_sub_length": 0,
                "max_sub_duration": 0,
                "max_gap_time": 0,
                "max_overlap_time": 0,
                "max_matched_frame_len": 0,
                "max_unmatched_group_len": frame_length,
                "extra_long_subs": 0}
        return info, 0
    max_sub_length = max([len(e["text"].split(" ")) for e in sub_listdicts])
    orig_timestamps = np.array(
        [[e["start"], e["end"]] for e in sub_listdicts],
        dtype=np.float32)  # (n_subs, 2)
    # sorted_idx = np.argsort(orig_timestamps, axis=0)

    # Check if subs are sorted by start time
    # assert np.equal(orig_timestamps, np.sort(orig_timestamps, axis=0)).all()
    assert np.equal(
        orig_timestamps[:, 0], np.sort(orig_timestamps[:, 0])).all()
    timestamps = orig_timestamps / frame_length

    # r-th row of frame_indices is [st_idx, ed_idx),
    # where [st_idx, st_idx+1, ..., ed_idx-1]
    # should be with r-th frame, which is [r*frame_length, (r+1)*frame_length]
    subtitle2frame_st_ed = np.empty_like(timestamps, dtype=np.int)
    subtitle2frame_st_ed[:, 0] = np.floor(timestamps[:, 0])
    subtitle2frame_st_ed[:, 1] = np.ceil(timestamps[:, 1])

    overlapped_subtitles = 0
    subtitle_idx2frame_indices = {}
    previous_sub_idx = -1
    max_gap_time = 0
    max_overlap_time = 0
    max_duration = 0
    extra_long_subs = 0
    for sub_idx, (
            frame_st_idx, frame_ed_idx) in enumerate(subtitle2frame_st_ed):
        current_frame_set = [i for i in range(frame_st_idx, frame_ed_idx)]
        if previous_sub_idx > 0:
            overlapped_subtitles += (
                orig_timestamps[
                    previous_sub_idx][1] > orig_timestamps[sub_idx][0])
            gaptime = orig_timestamps[sub_idx][0] - orig_timestamps[
                previous_sub_idx][1]
            # if gaptime > 20:
            #     print(
            #         orig_timestamps[previous_sub_idx],
            #         sub_listdicts[previous_sub_idx])
            #     print(
            #         orig_timestamps[sub_idx],
            #         sub_listdicts[sub_idx])
            max_gap_time = max(max_gap_time, float(gaptime))
            max_overlap_time = max(max_overlap_time, float(-gaptime))
        start_time = orig_timestamps[sub_idx][0]
        end_time = min(
            orig_timestamps[sub_idx][1], frame_length*num_of_frames)
        if start_time >= frame_length*num_of_frames:
            continue

        duration = end_time - start_time
        if sub_idx == len(sub_listdicts) - 1 and duration > 16:
            extra_long_subs += 1
            current_frame_set = current_frame_set[:11]
        else:
            max_duration = max(max_duration, float(duration))
        subtitle_idx2frame_indices[sub_idx] = current_frame_set
        previous_sub_idx = sub_idx

    # all_frame_indices = set(flat_list_of_lists(
    #     list(subtitle_idx2frame_indices.values())))

    frame_idx2subtitle_indices = {}
    frame_idx2unique_sub_idx = {}
    unmatched_frames = []
    curr_unmatched_groups = []
    for frame_idx in range(num_of_frames):
        matched_sub_idx_list = []
        for k, v in subtitle_idx2frame_indices.items():
            if frame_idx in set(v):
                matched_sub_idx_list.append(k)
        if len(matched_sub_idx_list):
            matched_sub_idx = 0
            frame_idx2subtitle_indices[str(frame_idx)] = matched_sub_idx_list
            max_iou = 0
            for sub_idx in matched_sub_idx_list:
                # each frame is matched to the subtitle with max iou
                current_iou = temporal_iou(
                    [frame_idx, frame_idx+1], timestamps[sub_idx])
                if current_iou > max_iou:
                    max_iou = current_iou
                    matched_sub_idx = sub_idx
            frame_idx2unique_sub_idx[frame_idx] = matched_sub_idx
        else:
            if len(curr_unmatched_groups) > 0:
                if frame_idx > curr_unmatched_groups[-1]+1:
                    unmatched_frames.append(copy.copy(curr_unmatched_groups))
                    curr_unmatched_groups = []
            curr_unmatched_groups.append(frame_idx)
    if len(curr_unmatched_groups) > 0:
        unmatched_frames.append(copy.copy(curr_unmatched_groups))

    unique_sub_idx2frame_indices = {}
    for sub_idx in range(len(subtitle2frame_st_ed)):
        curr_frame_idx = []
        for k, v in frame_idx2unique_sub_idx.items():
            if sub_idx == v:
                curr_frame_idx.append(k)
        if len(curr_frame_idx):
            curr_frame_idx = sorted(curr_frame_idx)
        unique_sub_idx2frame_indices[sub_idx] = curr_frame_idx

    info = {
        "num_of_frames": num_of_frames,
        "unique_sub2frames": unique_sub_idx2frame_indices,
        "sub2frames": subtitle_idx2frame_indices,
        "frame2subs": frame_idx2subtitle_indices,
        "frame2unique_sub": frame_idx2unique_sub_idx,
        "unmatched_frames": flat_list_of_lists(unmatched_frames),
        "max_sub_length": max_sub_length,
        "max_sub_duration": max_duration,
        "max_gap_time": max_gap_time,
        "max_overlap_time": max_overlap_time,
        "max_matched_frame_len": max(
            [len(v) for _, v in unique_sub_idx2frame_indices.items()]),
        "max_unmatched_group_len": max(
            [len(v) for v in unmatched_frames])
        if len(unmatched_frames) else 0,
        "extra_long_subs": extra_long_subs
        }
    return info, overlapped_subtitles


@curry
def roberta_tokenize(tokenizer, text):
    text = html.unescape(text)
    if text.isupper():
        text = text.lower()
    words = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(words)
    return ids


def process_tv_subtitles(
        raw_data, video2sub_info, db, tokenizer, sep_id,
        allow_empty_sub=True):
    vid2len = {}
    vid2max_frame_sub_len = {}
    max_sub_seq_len = 0
    unmatched_sub_num = 0
    for line in tqdm(raw_data, desc='tokenizing subtitles'):
        example = json.loads(line)
        v_id = example["vid_name"]
        curr_info = video2sub_info[v_id]
        input_ids = []
        max_mtached_sub_len = 0

        uniq_sub2frames = []
        sub2frames = []
        n_frames = curr_info["num_of_frames"]
        if len(example["sub"]) and n_frames:
            curr_uniq_sub2frames = curr_info["unique_sub2frames"]
            curr_sub2frames = curr_info["sub2frames"]
            example['unmatched_frames'] = curr_info["unmatched_frames"]
            for sub_idx, sub in enumerate(example["sub"]):
                input_id = tokenizer(sub["text"])
                if sub_idx in curr_sub2frames:
                    matched_frames = curr_sub2frames[sub_idx]
                else:
                    matched_frames = []
                if len(matched_frames) == 0:
                    unmatched_sub_num += 1
                curr_len = len(input_id)+len(matched_frames)
                max_mtached_sub_len = max(max_mtached_sub_len, curr_len)
                input_ids.append(input_id)
                uniq_sub2frames.append(
                    (sub_idx, curr_uniq_sub2frames[sub_idx]))
                sub2frames.append((sub_idx, matched_frames))
        elif allow_empty_sub:
            start_f_inds = list(range(0, n_frames, 5))
            for sub_idx, st_f_idx in enumerate(start_f_inds):
                ed_f_idx = min(st_f_idx+5, n_frames)
                input_id = []
                matched_frames = list(range(st_f_idx, ed_f_idx))
                curr_len = len(input_id)+len(matched_frames)
                max_mtached_sub_len = max(max_mtached_sub_len, curr_len)
                input_ids.append(input_id)
                uniq_sub2frames.append((sub_idx, matched_frames))
                sub2frames.append((sub_idx, matched_frames))
            example['unmatched_frames'] = []

        max_sub_seq_len = max(max_sub_seq_len, max_mtached_sub_len)

        example['input_ids'] = input_ids
        example['unique_sub2frames'] = uniq_sub2frames
        example['sub2frames'] = sub2frames
        example['v_id'] = v_id
        example['duration'] = curr_info["num_of_frames"]
        example['max_mtached_sub_len'] = max_mtached_sub_len
        db[v_id] = example
        vid2len[v_id] = curr_info["num_of_frames"]
        vid2max_frame_sub_len[v_id] = max_mtached_sub_len
    print("max length of (tokenized subtitle + matched frames):"
          f" {max_sub_seq_len}")
    print("number of unmatched subtitles:"
          f" {unmatched_sub_num}")
    return vid2len, vid2max_frame_sub_len


def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    toker = RobertaTokenizer.from_pretrained(
        opts.toker)
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
    save_json(vars(opts), f'{opts.output}/meta.json', save_pretty=True)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:
        sub_info_cache_path = f'{opts.output}/sub_info.json'
        try:
            vid2nframe = load_json(opts.vid2nframe)
        except Exception:
            vid2nframe = None
        if not os.path.exists(sub_info_cache_path):
            video2sub_info = load_process_sub_meta(
                opts.annotation, vid2nframe, frame_length=args.frame_length)
            save_json(video2sub_info, sub_info_cache_path)
        else:
            video2sub_info = load_json(sub_info_cache_path)
        with open(opts.annotation) as ann:
            vid2len, vid2max_frame_sub_len = process_tv_subtitles(
                ann, video2sub_info, db,
                tokenizer, meta['SEP'])

        save_json(vid2len, f'{opts.output}/vid2len.json')
        save_json(vid2max_frame_sub_len,
                  f'{opts.output}/vid2max_frame_sub_len.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    parser.add_argument("--vid2nframe", type=str,
                        help="vid info file")
    parser.add_argument('--frame_length', default=1.5, type=float,
                        help='video frame length')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--toker', default='roberta-base',
                        help='which RoBerTa tokenizer to used')
    parser.add_argument('--task', default='tv',
                        help='which RoBerTa tokenizer to used')
    args = parser.parse_args()
    main(args)
