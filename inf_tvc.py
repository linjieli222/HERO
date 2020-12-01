"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

TVC inference
generate prediction from JSON file
"""
import argparse
import json
from time import time

import torch
from horovod import torch as hvd
from transformers import RobertaTokenizer
from apex import amp
from tqdm import tqdm

from data.tvc import TvcEvalDataset
from model.tvc import HeroForTvc, TvcGenerator
from eval.tvc import TVCEval
from utils.misc import Struct
from utils.distributed import all_gather_list
from utils.const import VFEAT_DIM, MAX_FRM_SEQ_LEN
from utils.basic_utils import save_jsonl

from load_data import load_video_sub_dataset
from train_tvc import build_dataloader


def main(opts):
    hvd.init()
    if hvd.rank() == 0:
        toker = RobertaTokenizer.from_pretrained('roberta-base')
        all_gather_list(None)
    else:
        all_gather_list(None)
        toker = RobertaTokenizer.from_pretrained('roberta-base')

    model_opts = Struct(json.load(open(f"{opts.model_dir}/log/hps.json")))
    model_config = f"{opts.model_dir}/log/model_config.json"

    video_db = load_video_sub_dataset(model_opts.vfeat_db,
                                      model_opts.sub_txt_db,
                                      model_opts.vfeat_interval,
                                      model_opts)
    dset = TvcEvalDataset(video_db, opts.target_clip)
    loader = build_dataloader(dset, opts.batch_size,
                              TvcEvalDataset.collate, False, opts)

    checkpoint = torch.load(f"{opts.model_dir}/ckpt/"
                            f"model_step_{opts.ckpt_step}.pt")

    img_pos_embed_weight_key = "v_encoder.f_encoder.img_embeddings" +\
        ".position_embeddings.weight"
    if img_pos_embed_weight_key in checkpoint:
        max_frm_seq_len = len(checkpoint[img_pos_embed_weight_key])
    else:
        max_frm_seq_len = MAX_FRM_SEQ_LEN

    model = HeroForTvc.from_pretrained(model_config,
                                       state_dict=checkpoint,
                                       vfeat_dim=VFEAT_DIM,
                                       max_frm_seq_len=max_frm_seq_len,
                                       lsr=model_opts.lsr)
    model.cuda()
    model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')

    bos = toker.convert_tokens_to_ids(['<s>'])[0]
    eos = toker.convert_tokens_to_ids(['</s>'])[0]
    model.eval()
    generator = TvcGenerator(model, opts.max_gen_step, bos, eos, opts.fp16)
    results = decode(loader, generator, toker)
    save_jsonl(results, opts.output)

    # evaluate score if possible
    if (hvd.rank() == 0
            and 'descs' in json.loads(next(iter(open(opts.target_clip))))):
        evaluator = TVCEval(opts.target_clip)
        score = evaluator(results)
        print(score)


def decode(loader, generator, tokenizer):
    st = time()
    results = []
    for batch in tqdm(loader, desc='decoding...'):
        vids = batch['vid_names']
        cids = batch['clip_ids']
        all_ts = batch['all_ts']
        outputs = generator.greedy_decode(batch)
        for vid, cid, ts, out_ids in zip(vids, cids, all_ts, outputs):
            output = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(out_ids))
            results.append({'vid_name': vid, 'clip_id': cid, 'ts': ts,
                            'descs': [{'desc': output}]})
    results = [r for rs in all_gather_list(results) for r in rs]
    print(f'decoding finished in {int(time() - st)} seconds')
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str,
                        help="dir root to trained model")
    parser.add_argument("--ckpt_step", required=True, type=int,
                        help="checkpoint step")
    parser.add_argument("--output", type=str, required=True,
                        help="output file name")

    parser.add_argument("--batch_size", default=8, type=int,
                        help="validation batch size (per GPU)")
    parser.add_argument("--max_gen_step", default=30, type=int,
                        help="max generation steps")

    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help="disable pin memory")
    parser.add_argument("--no_fp16", action='store_false', dest='fp16',
                        help="disable fp16")

    parser.add_argument("--target_clip", required=True, type=str,
                        help="jsonl annotation")

    args = parser.parse_args()

    main(args)
