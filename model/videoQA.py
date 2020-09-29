"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

HERO for Video Question Answering Tasks, shared by:
1. TVQA
2. How2QA
"""
from collections import defaultdict
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .model import HeroModel
from .layers import MLPLayer
from .modeling_utils import mask_logits


class HeroForVideoQA(HeroModel):
    def __init__(self, config, vfeat_dim, max_frm_seq_len):
        super().__init__(
            config, vfeat_dim, max_frm_seq_len)

        hsz = config.c_config.hidden_size

        self.qa_pool = nn.Linear(
            in_features=hsz, out_features=1, bias=False)
        self.qa_pred_head = MLPLayer(hsz, 1)

        # in tvqa/how2qa, we also have annotations for st and ed frame idx
        self.st_ed_pool = copy.deepcopy(self.qa_pool)
        self.st_ed_pred_head = MLPLayer(hsz, 2)

    def get_modularized_video(self, frame_embeddings, frame_mask):
        """
        Args:
            frame_embeddings: (Nv, Nq, L, D)
            frame_mask: (Nv, Nq, L)
        """
        st_ed_attn_scores = self.st_ed_pool(
            frame_embeddings)  # (Nv, Nq, L, 1)
        qa_attn_scores = self.qa_pool(frame_embeddings)

        st_ed_attn_scores = F.softmax(
            mask_logits(st_ed_attn_scores,
                        frame_mask.unsqueeze(-1)), dim=1)
        qa_attn_scores = F.softmax(
            mask_logits(qa_attn_scores,
                        frame_mask.unsqueeze(-1)), dim=2)
        # TODO check whether it is the same
        st_ed_pooled_video = torch.einsum(
            "vqlm,vqld->vlmd", st_ed_attn_scores,
            frame_embeddings)  # (Nv, L, 1, D)
        qa_pooled_video = torch.einsum(
            "vqlm,vqld->vqmd", qa_attn_scores,
            frame_embeddings)  # (Nv, Nq, 1, D)
        return st_ed_pooled_video.squeeze(2), qa_pooled_video.squeeze(2)

    def forward(self, batch, task='tvqa', compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task == 'tvqa' or task == 'how2qa':
            targets = batch['targets'].squeeze(-1)
            c_attn_masks = batch["c_attn_masks"]
            ts_targets = batch["ts_targets"]
            # (num_video * 5, num_frames, hid_size)
            frame_embeddings = self.v_encoder.forward_repr(
                batch, encode_clip=False)
            frame_embeddings = self.v_encoder.c_encoder.embeddings(
                frame_embeddings,
                position_ids=None)
            qa_embeddings = self.v_encoder.f_encoder._compute_txt_embeddings(
                batch["qa_input_ids"], batch["qa_pos_ids"], txt_type_ids=None)
            frame_qa_embeddings = torch.cat(
                (frame_embeddings, qa_embeddings), dim=1)
            frame_qa_attn_mask = torch.cat(
                (c_attn_masks, batch["qa_attn_masks"]), dim=1)
            fused_video_qa = self.v_encoder.c_encoder.forward_encoder(
                frame_qa_embeddings, frame_qa_attn_mask)
            num_frames = c_attn_masks.shape[1]
            video_embeddings = fused_video_qa[:, :num_frames, :]

            num_videos = len(targets)
            num_frames, hid_size = video_embeddings.shape[1:3]
            video_embeddings = video_embeddings.view(
                num_videos, -1, num_frames, hid_size)
            video_masks = c_attn_masks.view(num_videos, -1, num_frames)
            video_masks = video_masks.to(dtype=video_embeddings.dtype)
            st_ed_pooled_video, qa_pooled_video = self.get_modularized_video(
                video_embeddings, video_masks)
            pred_st_ed = self.st_ed_pred_head(st_ed_pooled_video)
            st_prob = mask_logits(pred_st_ed[:, :, 0], video_masks[:, 0])
            ed_prob = mask_logits(pred_st_ed[:, :, 1], video_masks[:, 0])
            logits = self.qa_pred_head(qa_pooled_video).squeeze(-1)

            if compute_loss:
                st_target, ed_target = ts_targets[:, 0], ts_targets[:, 1]
                st_loss = F.cross_entropy(
                    st_prob, st_target, reduction="mean",
                    ignore_index=-1)
                ed_loss = F.cross_entropy(
                    ed_prob, ed_target, reduction="mean",
                    ignore_index=-1)
                temporal_loss = (st_loss + ed_loss)/2.
                qa_loss = F.cross_entropy(logits, targets, reduction='mean',
                                            ignore_index=-1)
                return qa_loss, temporal_loss
            else:
                return logits
        else:
            raise ValueError(f'Unrecognized task: {task}')
