"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
import sys
import json
import argparse


def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


class SharedConfigs(object):
    """Shared options for pre-training and downstream tasks.
    For each downstream task, implement a get_*_args function,
    see `get_pretraining_args()`

    Usage:
    >>> shared_configs = SharedConfigs()
    >>> pretraining_config = shared_configs.get_pretraining_args()
    """

    def __init__(self,
                 desc="shared config class for both pre-training and downstream tasks"):
        parser = argparse.ArgumentParser(description=desc)
        # model arch parameters
        parser.add_argument("--model_config",
                            default=None, type=str,
                            help="json file for model architecture")
        parser.add_argument("--checkpoint",
                            default=None, type=str,
                            help="pretrained model")

        # training parameters
        parser.add_argument("--train_batch_size",
                            default=16, type=int,
                            help="Total batch size for training. "
                                 "(batch by number of videos)")
        parser.add_argument("--val_batch_size",
                            default=20, type=int,
                            help="Total batch size for validation. "
                                 "(batch by number of videos)")
        parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=16,
                            help="Number of updates steps to accumualte before"
                                 "performing a backward/update pass.")
        parser.add_argument("--learning_rate",
                            default=3e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--valid_steps",
                            default=1000,
                            type=int,
                            help="Run validation every X steps")
        parser.add_argument("--save_steps", default=500, type=int,
                            help="save every X steps for "
                                 "continue after preemption")
        parser.add_argument("--optim", default='adam',
                            choices=['adam', 'adamax', 'adamw'],
                            help="optimizer")
        parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                            type=float, help="beta for adam optimizer")
        parser.add_argument("--dropout",
                            default=0.1,
                            type=float,
                            help="tune dropout regularization")
        parser.add_argument("--weight_decay",
                            default=0.0,
                            type=float,
                            help="weight decay (L2) regularization")
        parser.add_argument("--grad_norm",
                            default=0.25,
                            type=float,
                            help="gradient clipping (-1 for no clipping)")
        parser.add_argument("--warmup_steps",
                            default=4000,
                            type=int,
                            help="Number of training steps to perform linear "
                                 "learning rate warmup for.")
        parser.add_argument("--lr_mul",
                            default=1.0,
                            type=float,
                            help="Learning rate multiplier")
        parser.add_argument(
            "--num_train_steps",
            default=100000, type=int,
            help="Total number of training updates to perform.")
        parser.add_argument(
            "--output_dir", default=None, type=str,
            help="The output directory where the model checkpoints will be "
                 "written.")

        # data parameters
        parser.add_argument(
            "--sub_ctx_len", default=0, type=int,
            help="consider 'sub_ctx_len' subtitles before and "
                 "after the current one")

        # Prepro parameters
        parser.add_argument('--max_clip_len', type=int, default=100,
                            help='max number of frames in video')
        parser.add_argument('--max_txt_len', type=int, default=60,
                            help='max number of tokens in text (BERT BPE)')
        parser.add_argument("--vfeat_version",
                            default="resnet_slowfast", type=str,
                            help="video frame feature's version")
        parser.add_argument("--vfeat_interval",
                            default=1.5, type=float,
                            help="every ** second to extract one vfeat")
        parser.add_argument('--compressed_db', action='store_true',
                            help='use compressed LMDB')

        # device parameters
        parser.add_argument('--seed',
                            type=int,
                            default=42,
                            help="random seed for initialization")
        parser.add_argument('--n_workers', type=int, default=4,
                            help="number of data workers")
        parser.add_argument('--pin_mem', action='store_true',
                            help="pin memory")
        parser.add_argument(
            '--fp16', action='store_true',
            help="Whether to use 16-bit float precision instead "
                 "of 32-bit")

        # can use config files
        parser.add_argument('--config', help='JSON config files')
        self.parser = parser

    def parse_args(self):
        args = parse_with_config(self.parser)

        # basic checks

        assert args.gradient_accumulation_steps >= 1, \
            f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps} "

        # set options to easier debug mode
        return args

    def get_vsm_args(self):
        # VSM configs (shared with VCMR)
        self.parser.add_argument(
            "--drop_svmr_prob", default=0, type=float,
            help="Randomly drop svmr training by a certain prob")
        self.parser.add_argument(
            "--lw_neg_q", type=float, default=0,
            help="weight for ranking loss with "
            "negative query and positive context")
        self.parser.add_argument(
            "--lw_neg_ctx", type=float, default=0,
            help="weight for ranking loss with "
            "positive query and negative context")
        self.parser.add_argument(
            "--lw_st_ed", type=float, default=1,
            help="weight for st ed prediction loss")
        self.parser.add_argument(
                "--ranking_loss_type", type=str, default="hinge",
                choices=["hinge", "lse"],
                help="att loss type, can be hinge loss or "
                "its smooth approximation LogSumExp")
        self.parser.add_argument(
                "--margin", type=float, default=0.1,
                help="margin for   hinge loss")
        self.parser.add_argument(
                "--hard_pool_size", type=int, default=[20],
                nargs='+',
                help="hard negatives are still sampled"
                     "but from a harder pool.")
        self.parser.add_argument(
                "--hard_neg_weights", type=float, default=[10],
                nargs='+',
                help="weighting factor for sampled hard negatives")
        self.parser.add_argument(
                "--hard_negtiave_start_step", type=int, default=[10000],
                nargs='+',
                help="which epoch to start hard negative sampling "
                "for video-level ranking loss,"
                "use -1 to disable")
        self.parser.add_argument(
                "--train_span_start_step", type=int, default=-1,
                help="which epoch to start svmr training "
                "use -1 to disable")
        self.parser.add_argument(
            "--use_all_neg",
            action="store_true",
            help="Use all negatives for VR")

    def get_pretrain_args(self):
        self.parser.add_argument("--txt_db", default=None, type=str,
                                 help="path to TXT_DB")
        self.parser.add_argument("--img_db", default=None, type=str,
                                 help="path to IMG_DB")
        self.parser.add_argument("--targets_ratio", type=int, nargs='+',
                                 help="multi-dataset mix ratio")
        self.parser.add_argument("--load_partial_pretrained",
                                 action='store_true',
                                 help="load only for frame embedding model")
        self.parser.add_argument("--skip_layer_loading",
                                 action='store_true',
                                 help="allow skip layer loading"
                                 "in load_partial_pretrained")
        self.parser.add_argument(
            "--mask_prob", default=0.15, type=float,
            help="masking probability")
        self.get_vsm_args()

        args = self.parse_args()

        assert len(args.hard_negtiave_start_step) == len(
            args.hard_neg_weights)
        return args

    def get_vcmr_args(self):
        self.parser.add_argument(
            "--task", default='tvr',
            choices=['tvr', 'didemo_video_sub', 'how2r', 'didemo_video_only'],
            type=str, help="vcmr tasks")
        self.parser.add_argument("--vcmr_eval_video_batch_size",
                                 default=40, type=int,
                                 help="Total video batch size for validation. "
                                      "(batch by number of videos)")
        self.parser.add_argument("--vcmr_eval_q_batch_size",
                                 default=80, type=int,
                                 help="Total query batch size for validation. "
                                      "(batch by number of queries)")

        # Training parameters
        self.get_vsm_args()

        # Eval parameters
        self.parser.add_argument(
            "--eval_with_query_type", action='store_true',
            help="eval the retrieval results by query type")
        self.parser.add_argument(
            "--max_before_nms", default=200, type=int,
            help="maximum to keep before nms")
        self.parser.add_argument(
            "--max_after_nms", default=100, type=int,
            help="maximum to keep after nms")
        self.parser.add_argument(
            "--distributed_eval", action='store_true',
            help="Allow distributed evaluation on multi-gpus")
        self.parser.add_argument(
            "--nms_thd", default=0.5, type=float,
            help="eval nms threshold")
        self.parser.add_argument(
            "--q2c_alpha", type=float, default=20,
            help="give more importance to top scored videos' spans,  "
            "the new score will be: s_new = exp(alpha * s), "
            "higher alpha indicates more importance. Note s in [-1, 1]")
        self.parser.add_argument(
            "--max_vcmr_video", type=int, default=100,
            help="re-ranking in top-max_vcmr_video")
        self.parser.add_argument(
            "--full_eval_tasks", type=str, nargs="+",
            choices=["VCMR", "SVMR", "VR"], default=["VCMR", "SVMR", "VR"],
            help="Which tasks to run."
            "VCMR: Video Corpus Moment Retrieval;"
            "SVMR: Single Video Moment Retrieval;"
            "VR: regular Video Retrieval. "
            "    (will be performed automatically with VCMR)")
        self.parser.add_argument(
            "--min_pred_l", type=int, default=2,
            help="constrain the [st, ed] with ed - st >= 2"
            "(2 clips with length 1.5 each, 3 secs in total"
            "this is the min length for proposal-based method)")
        self.parser.add_argument(
            "--max_pred_l", type=int, default=16,
            help="constrain the [st, ed] pairs with ed - st <= 16, "
            "24 secs in total"
            "(16 clips with length 1.5 each, "
            "this is the max length for proposal-based method)")

        args = self.parse_args()
        assert len(args.hard_negtiave_start_step) == len(
            args.hard_neg_weights)
        return args

    def get_vr_args(self):
        self.parser.add_argument(
            "--task", default='msrvtt_video_sub',
            choices=['msrvtt_video_sub', 'msrvtt_video_only'],
            type=str, help="vr tasks")
        self.parser.add_argument("--vr_eval_video_batch_size",
                                 default=40, type=int,
                                 help="Total video batch size for validation. "
                                      "(batch by number of videos)")
        self.parser.add_argument("--vr_eval_q_batch_size",
                                 default=80, type=int,
                                 help="Total query batch size for validation. "
                                      "(batch by number of queries)")

        # Training parameters
        self.get_vsm_args()

        # Eval parameters
        self.parser.add_argument(
            "--distributed_eval", action='store_true',
            help="Allow distributed evaluation on multi-gpus")
        self.parser.add_argument(
            "--max_vr_video", type=int, default=100,
            help="re-ranking in top-max_vr_video")

        args = self.parse_args()
        assert len(args.hard_negtiave_start_step) == len(
            args.hard_neg_weights)
        del args.lw_st_ed
        del args.train_span_start_step
        del args.drop_svmr_prob

        return args

    def get_videoQA_args(self):
        self.parser.add_argument(
            "--task", default='tvqa',
            choices=['tvqa', 'how2qa'],
            type=str, help="video qa tasks")
        self.parser.add_argument(
            "--lw_st_ed", type=float, default=1,
            help="weight for st ed prediction loss")

        args = self.parse_args()
        return args

    def get_violin_args(self):
        self.parser.add_argument(
            "--task", default='violin',
            choices=['violin'],
            type=str, help="violin tasks")
        args = self.parse_args()
        return args


shared_configs = SharedConfigs()
