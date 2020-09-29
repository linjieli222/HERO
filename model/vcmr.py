"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

HERO for Video Corpus Moment Retrieval Tasks, shared by:
1. TVR
2. How2R
3. DiDeMo with video and sub
4. DiDeMo with video only
"""
from .pretrain import HeroForPretraining


class HeroForVcmr(HeroForPretraining):
    def __init__(self, config, vfeat_dim, max_frm_seq_len,
                 conv_stride=1, conv_kernel_size=5,
                 ranking_loss_type="hinge", margin=0.1,
                 lw_neg_ctx=0, lw_neg_q=0, lw_st_ed=0.01, drop_svmr_prob=0,
                 use_hard_negative=False, hard_pool_size=20,
                 hard_neg_weight=10, use_all_neg=True):
        super().__init__(
            config, vfeat_dim, max_frm_seq_len,
            conv_stride, conv_kernel_size,
            ranking_loss_type, margin,
            lw_neg_ctx, lw_neg_q, lw_st_ed, drop_svmr_prob,
            use_hard_negative, hard_pool_size,
            hard_neg_weight, use_all_neg)

    def forward(self, batch, task='tvr', compute_loss=True):
        if task in ['tvr', 'how2r', 'didemo_video_sub',
                    'didemo_video_only']:
            return super().forward(
                batch, task='vsm', compute_loss=compute_loss)
        else:
            raise ValueError(f'Unrecognized task {task}')

    def get_pred_from_raw_query(self, frame_embeddings, c_attn_masks,
                                query_input_ids, query_pos_ids,
                                query_attn_masks, cross=False,
                                val_gather_gpus=False):
        modularized_query = self.encode_txt_inputs(
                    query_input_ids, query_pos_ids,
                    query_attn_masks, attn_layer=self.q_feat_attn,
                    normalized=False)

        st_prob, ed_prob = self.get_pred_from_mod_query(
            frame_embeddings, c_attn_masks,
            modularized_query, cross=cross)

        if self.lw_neg_ctx != 0 or self.lw_neg_q != 0:
            q2video_scores = self.get_video_level_scores(
                modularized_query, frame_embeddings, c_attn_masks,
                val_gather_gpus)
        else:
            q2video_scores = None
        return q2video_scores, st_prob, ed_prob
