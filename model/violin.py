"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

HERO for VIOLIN
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from .model import HeroModel
from .layers import MLPLayer
from .modeling_utils import mask_logits


class HeroForViolin(HeroModel):
    def __init__(self, config, vfeat_dim, max_frm_seq_len):
        super().__init__(
            config, vfeat_dim, max_frm_seq_len)
        hsz = config.c_config.hidden_size

        self.violin_pool = nn.Linear(
                in_features=hsz,
                out_features=1,
                bias=False)
        self.violin_pred_head = MLPLayer(hsz, 1)

    def get_modularized_video(self, frame_embeddings, frame_mask):
        """
        Args:
            frame_embeddings: (Nv, L, D)
            frame_mask: (Nv, L)
        """
        violin_attn_scores = self.violin_pool(
                frame_embeddings)  # (Nv, L, 1)

        violin_attn_scores = F.softmax(
            mask_logits(violin_attn_scores,
                        frame_mask.unsqueeze(-1)), dim=1)

        # TODO check whether it is the same
        violin_pooled_video = torch.einsum(
            "vlm,vld->vmd", violin_attn_scores,
            frame_embeddings)  # (Nv, 1, D)
        return violin_pooled_video.squeeze(1)

    def forward(self, batch, task='violin', compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task == 'violin':
            c_attn_masks = batch["c_attn_masks"]
            # (num_video * 5, num_frames, hid_size)
            frame_embeddings = self.v_encoder.forward_repr(
                batch, encode_clip=False)
            frame_embeddings = self.v_encoder.c_encoder.embeddings(
                frame_embeddings,
                position_ids=None)
            q_embeddings = self.v_encoder.f_encoder._compute_txt_embeddings(
                batch["q_input_ids"], batch["q_pos_ids"], txt_type_ids=None)
            frame_q_embeddings = torch.cat(
                (frame_embeddings, q_embeddings), dim=1)
            frame_q_attn_mask = torch.cat(
                (c_attn_masks, batch["q_attn_masks"]), dim=1)
            fused_video_q = self.v_encoder.c_encoder.forward_encoder(
                frame_q_embeddings, frame_q_attn_mask)
            num_frames = c_attn_masks.shape[1]
            video_embeddings = fused_video_q[:, :num_frames, :]

            video_masks = c_attn_masks.to(dtype=video_embeddings.dtype)
            violin_pooled_video = self.get_modularized_video(
                video_embeddings, video_masks)
            logits = self.violin_pred_head(violin_pooled_video)

            if compute_loss:
                targets = batch['targets']
                scores = torch.sigmoid(logits).squeeze(-1)
                targets = targets.squeeze(-1).to(dtype=scores.dtype)
                violin_loss = F.binary_cross_entropy(
                    scores, targets, reduction='mean')
                return violin_loss
            else:
                return logits
        raise ValueError(f'Unrecognized task: {task}')
