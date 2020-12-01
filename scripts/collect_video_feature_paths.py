"""
gather slowfast/resnet feature paths
"""
import os
import numpy as np
import pickle as pkl
import argparse
from tqdm import tqdm
from cytoolz import curry
import multiprocessing as mp


@curry
def load_npz(slowfast_dir, resnet_dir, slowfast_f):
    vid = slowfast_f.split("/")[-1].split(".npz")[0]
    folder_name = slowfast_f.split("/")[-2]
    resnet_f = slowfast_f.replace(slowfast_dir, resnet_dir)
    try:
        slowfast_data = np.load(slowfast_f, allow_pickle=True)
        slowfast_frame_len = max(0, len(slowfast_data["features"]))
    except Exception:
        slowfast_frame_len = 0
    resnet_frame_len = 0
    if slowfast_frame_len == 0:
        slowfast_f = ""
        print(f"Corrupted slowfast files for {vid}")
    # print(resnet_f)
    if not os.path.exists(resnet_f):
        resnet_f = ""
        print(f"resnet files for {vid} does not exists")
    else:
        try:
            resnet_data = np.load(resnet_f, allow_pickle=True)
            resnet_frame_len = len(resnet_data["features"])
        except Exception:
            resnet_frame_len = 0
            resnet_f = ""
            print(f"Corrupted resnet files for {vid}")
    frame_len = min(slowfast_frame_len, resnet_frame_len)
    return vid, frame_len, slowfast_f, resnet_f, folder_name


def main(opts):
    slowfast_dir = os.path.join(opts.feature_dir, "slowfast_features/")
    resnet_dir = os.path.join(opts.feature_dir, "resnet_features/")
    failed_resnet_files = []
    failed_slowfast_files = []
    loaded_file = []
    for root, dirs, curr_files in os.walk(f'{slowfast_dir}/'):
        for f in curr_files:
            if f.endswith('.npz'):
                slowfast_f = os.path.join(root, f)
                loaded_file.append(slowfast_f)
    print(f"Found {len(loaded_file)} slowfast files....")
    print(f"sample loaded_file: {loaded_file[:3]}")
    failed_resnet_files, failed_slowfast_files = [], []
    files = {}
    load = load_npz(slowfast_dir, resnet_dir)
    with mp.Pool(opts.nproc) as pool, tqdm(total=len(loaded_file)) as pbar:
        for i, (vid, frame_len, slowfast_f,
                resnet_f, folder_name) in enumerate(
                pool.imap_unordered(load, loaded_file, chunksize=128)):
            files[vid] = (frame_len, slowfast_f, resnet_f, folder_name)
            if resnet_f == "":
                video_file = os.path.join(folder_name, vid)
                failed_resnet_files.append(video_file)
            if slowfast_f == "":
                video_file = os.path.join(folder_name, vid)
                failed_slowfast_files.append(video_file)
            pbar.update(1)
    output_dir = os.path.join(opts.output, opts.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    pkl.dump(files, open(os.path.join(
        output_dir, "video_feat_info.pkl"), "wb"))
    if len(failed_slowfast_files):
        pkl.dump(failed_slowfast_files, open(os.path.join(
            output_dir, "failed_slowfast_files.pkl"), "wb"))
    if len(failed_resnet_files):
        pkl.dump(failed_resnet_files, open(os.path.join(
            output_dir, "failed_resnet_files.pkl"), "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir",
                        default="",
                        type=str, help="The input video feature dir.")
    parser.add_argument("--output", default=None, type=str,
                        help="output dir")
    parser.add_argument('--dataset', type=str,
                        default="")
    parser.add_argument('--nproc', type=int, default=10,
                        help='number of cores used')
    args = parser.parse_args()
    main(args)
