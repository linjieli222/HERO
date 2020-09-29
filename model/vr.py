"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

HERO for Video Retrieval Tasks, shared by:
1. MSR-VTT with video and sub
2. MSR-VTT with video only
"""
from .vcmr import HeroForVcmr


class HeroForVr(HeroForVcmr):
    def __init__(self, config, vfeat_dim, max_frm_seq_len,
                 ranking_loss_type="hinge", margin=0.1,
                 lw_neg_ctx=1, lw_neg_q=1,
                 use_hard_negative=False, hard_pool_size=20,
                 hard_neg_weight=10, use_all_neg=True):
        assert lw_neg_ctx != 0 or lw_neg_q != 0,\
            "Need to set lw_neg_ctx or lw_neg_q for VR training"
        super().__init__(
            config, vfeat_dim, max_frm_seq_len,
            ranking_loss_type=ranking_loss_type, margin=margin,
            lw_neg_ctx=lw_neg_ctx, lw_neg_q=lw_neg_q,
            lw_st_ed=0, drop_svmr_prob=1.0,
            use_hard_negative=use_hard_negative,
            hard_pool_size=hard_pool_size,
            hard_neg_weight=hard_neg_weight,
            use_all_neg=use_all_neg)
        assert self.lw_st_ed == 0, "For VR, lw_st_ed should be 0"

    def forward(self, batch, task='msrvtt_video_sub', compute_loss=True):
        if task in ['msrvtt_video_sub', 'msrvtt_video_only']:
            if compute_loss:
                _, loss_neg_ctx, loss_neg_q = super().forward(
                    batch, task='tvr', compute_loss=True)
                return loss_neg_ctx, loss_neg_q
            else:
                q2video_scores, _, _ = super().forward(
                    batch, task='tvr', compute_loss=False)
                return q2video_scores
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

        q2video_scores = self.get_video_level_scores(
            modularized_query, frame_embeddings, c_attn_masks,
            val_gather_gpus)
        return q2video_scores
