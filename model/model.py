"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

HERO Model

some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
import json
import logging
from collections import defaultdict
from io import open

import torch
import torch.nn as nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm

from .encoder import (
    RobertaModelConfig, RobertaPreTrainedModel)
from .encoder import CrossModalTrm
from .encoder import TemporalTrm
from .layers import (GELU, LinearLayer, MLPLayer)
from .modeling_utils import load_pretrained_weight, load_partial_checkpoint


logger = logging.getLogger(__name__)


class VideoModelConfig(object):
    def __init__(self, config_json_file):
        assert isinstance(config_json_file, str)
        with open(config_json_file,
                  "r", encoding='utf-8') as reader:
            config = json.loads(reader.read())
        self.f_config = RobertaModelConfig.from_dict(config["f_config"])
        self.c_config = RobertaModelConfig.from_dict(config["c_config"])
        if "q_config" in config:
            self.q_config = RobertaModelConfig.from_dict(config["q_config"])
        else:
            self.q_config = None
        self.initializer_range = self.f_config.initializer_range

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `VideoModelConfig` from a json file of parameters."""
        videoConfig = VideoModelConfig(json_file)
        return videoConfig

    def __log__(self):
        logger.info("Model config:")
        logger.info(f"     Cross-Modal Transformer config: {self.f_config}")
        logger.info(f"     Temporal Transformer config: {self.c_config}")
        logger.info(f"     QueryEncoder config: {self.q_config}")


class VideoPreTrainedModel(RobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config.f_config)
        if not isinstance(config, VideoModelConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `VideoModelConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    @classmethod
    def load_config(cls, config_file):
        # Load config
        config = VideoModelConfig.from_json_file(config_file)
        config.__log__()
        return config

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        """
        Instantiate a UniterPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Uniter class
        """
        config = cls.load_config(config_file)
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict == {}:
            logger.info("No pretrained weights loaded")
            return model
        model = load_pretrained_weight(model, state_dict)
        return model


class FrameFeatureRegression(nn.Module):
    def __init__(self, hidden_size, feat_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 FusedLayerNorm(hidden_size, eps=1e-5),
                                 nn.Linear(hidden_size, feat_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output


class HierarchicalVlModel(VideoPreTrainedModel):
    def __init__(self, config, vfeat_dim, max_frm_seq_len,
                 max_clip_len=100, nce_temp=1.0):
        super().__init__(config)
        self.f_encoder = CrossModalTrm(
            config.f_config, vfeat_dim, max_frm_seq_len)
        self.frame_transform = LinearLayer(
            vfeat_dim, config.f_config.hidden_size,
            layer_norm=True, dropout=config.f_config.hidden_dropout_prob,
            relu=True)
        self.c_encoder = TemporalTrm(
            config.c_config)

        self.feat_regress = FrameFeatureRegression(
            config.f_config.hidden_size, vfeat_dim)
        self.nce_temp = nce_temp  # NCE shares same head with regression
        self.mask_embedding = nn.Embedding(2, vfeat_dim, padding_idx=0)
        self.fom_output = MLPLayer(
            config.c_config.hidden_size, max_clip_len)

        self.register_buffer('pad',
                             torch.zeros(8, config.c_config.hidden_size))

    def forward(self, batch, task='repr', compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task == 'repr':
            # get frame representations
            return self.forward_repr(batch)
        elif task.startswith('mlm'):
            return self.f_encoder(batch, task, compute_loss)
        elif task == 'mffr':
            return self.forward_mfm(batch, compute_loss, loss='regression')
        elif task == 'mfm-nce':
            return self.forward_mfm(batch, compute_loss, loss='nce')
        elif task == 'fom':
            return self.forward_fom(batch, compute_loss)
        else:
            raise ValueError(f'Unrecognized task {task}')

    def collect_frame_outputs(self, out_shape, frame_sequence_output,
                              num_subs, sub_idx2frame_idx):
        """
        Inputs:
        :out_shape              (bz, #frames, hidden_size)
        :frame_sequence_output  tensor (total_subs, max_vl+max_sl, hidden_size)
        :num_subs               [num_sub]
        :sub_idx2frame_idx      [ [(sub_ix, [frame_ix])] ]
        Return:
        :matched_v_feats        tensor (bz, #frames, hidden_size)
        """
        matched_v_feats = torch.zeros(*out_shape,
                                      dtype=frame_sequence_output.dtype,
                                      device=frame_sequence_output.device)
        start_idx, end_idx = 0, 0
        for vid, num_sub in enumerate(num_subs):
            current_sub_n_f = sub_idx2frame_idx[vid]
            end_idx += num_sub
            frame_sub = frame_sequence_output[start_idx: end_idx]
            for sid, matched_frame_idx in current_sub_n_f:
                n_frame = len(matched_frame_idx)
                if n_frame == 0:
                    continue
                matched_frame_idx = torch.tensor(
                    matched_frame_idx, dtype=torch.long,
                    device=frame_sub.device)
                matched_v_feats[vid, matched_frame_idx] =\
                    matched_v_feats[vid, matched_frame_idx] +\
                    frame_sub[sid, :n_frame]

            start_idx = end_idx
        return matched_v_feats

    def reorder_frame(self, frame_feats, c_pos_ids):
        c_pos_ids_expanded = c_pos_ids.unsqueeze(-1).expand_as(frame_feats)
        output = torch.zeros_like(frame_feats).scatter_(
            1, c_pos_ids_expanded, frame_feats)
        return output

    def forward_repr(self, batch, encode_clip=True):
        frame_outputs = self.f_encoder(batch, 'repr')
        # (total_sub, sub_len, 768)
        frame_sequence_output = frame_outputs[0]

        c_v_feats = batch['c_v_feats']
        c_attn_masks = batch['c_attn_masks']
        num_subs = batch['num_subs']
        sub_idx2frame_idx = batch['sub_idx2frame_idx']

        # (bz, #frames, 768)
        shape = list(c_v_feats.size()[:2]) + [frame_sequence_output.size(-1)]
        matched_v_feats = self.collect_frame_outputs(
            shape, frame_sequence_output, num_subs, sub_idx2frame_idx)

        # residual connection (transformed_v_feat = raw_v_feat + fused_v_feat)
        transformed_c_v_feats = self.frame_transform(c_v_feats)
        transformed_c_v_feats = transformed_c_v_feats + matched_v_feats

        if encode_clip:
            # reordered_feats = self.reorder_frame(
            #   transformed_c_v_feats, c_pos_ids)
            reordered_feats = transformed_c_v_feats
            # compute pos_ids in embedding layer
            clip_outputs = self.c_encoder(
                clip_level_pos_ids=None,
                clip_level_frame_feat=reordered_feats,
                attention_mask=c_attn_masks)
            return clip_outputs
        return transformed_c_v_feats

    def forward_vsm(self, batch):
        # [num_videos, clip_len, 768]
        clip_outputs = self.forward_repr(batch)
        # tuple([num_sub_queries, sub_len, 768], [num_sub_queries, 768])
        sub_query_batch = {}
        sub_query_batch["input_ids"] = batch['vsm_query_input_ids']
        sub_query_batch["pos_ids"] = batch['vsm_query_pos_ids']
        sub_query_batch["attn_masks"] = batch['vsm_query_attn_masks']
        sub_query_outputs = self.f_encoder(sub_query_batch, "txt")
        # [num_sub_queries, sub_len, 768]
        query = sub_query_outputs[0]
        return clip_outputs, query

    def forward_mfm(self, batch, compute_loss=True, loss='regression'):
        assert loss in ['regression', 'nce']
        c_v_feats = batch['c_v_feats']
        c_v_mask = batch['c_v_masks']
        # apply mask to c_v_feats
        c_v_feats.masked_fill_(c_v_mask.unsqueeze(-1), 0)
        mask = self.mask_embedding(c_v_mask.long())
        c_v_feats_masked = c_v_feats + mask
        batch['c_v_feats'] = c_v_feats_masked
        clip_outputs = self.forward_repr(batch)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(clip_outputs, c_v_mask)
        prediction_feat = self._pad_layer_unpad(masked_output,
                                                self.feat_regress)
        if loss == 'nce':
            neg_output = self._compute_masked_hidden(clip_outputs, ~c_v_mask)
            neg_pred_feat = self._pad_layer_unpad(neg_output,
                                                  self.feat_regress)

        if compute_loss:
            feat_targets = batch['feat_targets']
            if loss == 'regression':
                mfm_loss = F.mse_loss(prediction_feat, feat_targets,
                                      reduction='none')
            else:
                mfm_loss = self.mfm_nce(prediction_feat,
                                        feat_targets, neg_pred_feat)
            return mfm_loss
        else:
            if loss == 'regression':
                return prediction_feat
            else:
                return prediction_feat, neg_pred_feat

    def mfm_nce(self, masked_output, pos_output, neg_output,
                compute_loss=True):
        # dot product of ground truth feature
        masked_score = masked_output.matmul(pos_output.t())
        # dot product of neative samples
        neg_score = masked_output.matmul(neg_output.t())

        logits = torch.cat([masked_score, neg_score], dim=1).float()
        if compute_loss:
            targets = torch.arange(0, masked_output.size(0),
                                   dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits/self.nce_temp, targets,
                                   reduction='none')
            return loss
        else:
            return logits

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def _pad_layer_unpad(self, input_, layer):
        n_pad = 8 - input_.size(-1) % 8
        pad = self.pad[:n_pad]
        input_ = torch.cat([input_, pad], dim=0)
        output = layer(input_)
        if n_pad:
            output = output[:-n_pad, :]
        return output

    def forward_fom(self, batch, compute_loss=True):
        shuffled_orders = batch['shuffled_orders']
        transformed_c_v_feats = self.forward_repr(batch, encode_clip=False)

        # Reshuffle c_v_feats according to targets
        shuffled_orders_expanded = shuffled_orders.unsqueeze(-1).expand_as(
            transformed_c_v_feats)
        c_v_feats_shuffled = torch.zeros_like(
            transformed_c_v_feats, dtype=transformed_c_v_feats.dtype,
            device=transformed_c_v_feats.device)
        c_v_feats_shuffled = c_v_feats_shuffled.scatter_(
            1, shuffled_orders_expanded, transformed_c_v_feats)

        # compute pos_ids in embedding layer
        encoded_clip = self.c_encoder(
            clip_level_pos_ids=None,
            clip_level_frame_feat=c_v_feats_shuffled,
            attention_mask=batch["c_attn_masks"])

        bs, seq_len, hid_size = encoded_clip.size()
        encoded_clip = encoded_clip.view(bs * seq_len, hid_size)

        frame_reorder_outputs = self.fom_output(encoded_clip)

        if compute_loss:
            targets = batch['targets'].view(frame_reorder_outputs.shape[0])
            loss = F.cross_entropy(
                frame_reorder_outputs, targets, ignore_index=-1,
                reduction='mean')
            return loss
        return frame_reorder_outputs

    def initialize(self):
        self.apply(self.init_weights)
        self.f_encoder.apply(self.f_encoder.init_weights)
        self.c_encoder.apply(self.c_encoder.init_weights)

    def init_type_embedding(self):
        self.f_encoder.init_type_embedding()
        self.mask_embedding.weight.data[0].fill_(0)


class HeroModel(VideoPreTrainedModel):
    def __init__(self, config, vfeat_dim, max_frm_seq_len):
        super().__init__(config)
        self.config = config
        self.v_encoder = HierarchicalVlModel(
            config, vfeat_dim, max_frm_seq_len)
        self.v_encoder.initialize()

    def load_partial_pretrained(self, checkpoint, vfeat_dim, max_frm_seq_len,
                                skip_layers=True):
        partial_checkpoint = load_partial_checkpoint(
            checkpoint, self.config.f_config.num_hidden_layers, skip_layers)
        self.v_encoder.f_encoder = CrossModalTrm.from_pretrained(
            self.config.f_config, state_dict=partial_checkpoint,
            vfeat_dim=vfeat_dim, max_img_seq_len=max_frm_seq_len)
        self.v_encoder.f_encoder.pad_vocab()
        self.v_encoder.init_type_embedding()
