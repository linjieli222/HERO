"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.


Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)

Encoders
1. CrossModalTrm
2. TemporalTrm
3. QueryFeatEncoder
"""
import torch
from torch import nn
from torch.nn import functional as F

import logging
import json
import copy
from io import open
from collections import defaultdict
from apex.normalization.fused_layer_norm import FusedLayerNorm

from .layers import (BertPooler, LinearLayer,
                     BertLMPredictionHead, BertAttention,
                     BertEncoder)
from .embed import (
    QueryFeatEmbeddings, SubEmbeddings,
    ImageEmbeddings, FrameEmbeddings)
from .modeling_utils import (
    mask_logits, pad_tensor_to_mul, load_pretrained_weight)


logger = logging.getLogger(__name__)


class RobertaModelConfig(object):
    """Configuration class to store the configuration of a `RobertaModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        """Constructs RobertaModelConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `Model`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer
                encoder.
            num_attention_heads: Number of attention heads for each attention
                layer in the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e.
                feed-forward) layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string)
                in the encoder and pooler. If string, "gelu", "relu" and
                "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully
                connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this
                model might ever be used with. Typically set this to something
                large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed
                into `Model`.
            initializer_range: The sttdev of the truncated_normal_initializer
                for initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file,
                      "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.output_attentions = kwargs.pop(
                'output_attentions', False)
            self.output_hidden_states = kwargs.pop(
                'output_hidden_states', False)
        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `RobertaModelConfig` from a
           Python dictionary of parameters."""
        config = RobertaModelConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `RobertaModelConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class RobertaPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
        Usage:
            if opts.checkpoint:
                    checkpoint = torch.load(opts.checkpoint)
                else:
                    checkpoint = {}

                model = ModelCls.from_pretrained(
                    opts.model_config, state_dict=checkpoint,
                    vfeat_dim=vfeat_dim)
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, RobertaModelConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `RobertaModelConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def load_config(cls, config):
        # Load config
        if isinstance(config, str):
            config = RobertaModelConfig.from_json_file(config)
        # logger.info("Model config {}".format(config))
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
        model = load_pretrained_weight(model, state_dict)
        return model


class CrossModalTrm(RobertaPreTrainedModel):
    """
    Modification for Joint Frame-Subtitle Encoding
    Includes cross-modality pretraining tasks
    """

    def __init__(self, config, vfeat_dim, max_img_seq_len):
        super().__init__(config)
        self.encoder = BertEncoder(config)
        self.embeddings = SubEmbeddings(config)
        self.img_embeddings = ImageEmbeddings(
            config, vfeat_dim, max_img_seq_len)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)
        self.config = config

        # pretraining
        self.lm_head = BertLMPredictionHead(
            config, self.embeddings.word_embeddings.weight)
        self.vocab_pad = 0
        self.register_buffer('pad', torch.zeros(8, config.hidden_size))

    def pad_vocab(self):
        emb_w = self.embeddings.word_embeddings.weight.data
        padded_emb_w, n_pad = pad_tensor_to_mul(emb_w)
        padded_emb_w = nn.Parameter(padded_emb_w)
        bias, _ = pad_tensor_to_mul(self.lm_head.bias)
        padded_bias = nn.Parameter(bias)
        self.embeddings.word_embeddings.weight = padded_emb_w
        self.lm_head.decoder.weight = padded_emb_w
        self.lm_head.bias = padded_bias
        self.vocab_pad = n_pad

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids,
            token_type_ids=txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_ids,
                                img_type_ids=None, img_masks=None):
        if img_type_ids is None:
            img_type_embeddings = self.embeddings.token_type_embeddings(
                torch.ones(1, 1, dtype=torch.long, device=img_feat.device))
        else:
            img_type_embeddings = self.embeddings.token_type_embeddings(
                img_type_ids)
        output = self.img_embeddings(img_feat, img_type_embeddings,
                                     img_pos_ids, img_masks)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_pos_ids, gather_index,
                                    txt_type_ids=None, img_type_ids=None,
                                    img_masks=None):
        txt_emb, img_emb = None, None
        # embedding layer
        if input_ids is not None:
            # txt only
            txt_emb = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids)
        if img_feat is not None:
            # image only
            img_emb = self._compute_img_embeddings(
                img_feat, img_pos_ids, img_type_ids, img_masks)

        if txt_emb is not None and img_emb is not None:
            assert gather_index is not None
            # align back to most compact input
            gather_index = gather_index.unsqueeze(-1).expand(
                -1, -1, self.config.hidden_size)
            embedding_output = torch.gather(
                torch.cat([img_emb, txt_emb], dim=1),
                dim=1, index=gather_index)
            return embedding_output
        elif txt_emb is not None:
            return txt_emb
        elif img_emb is not None:
            return img_emb
        else:
            raise ValueError("Both img_feat and input_dis are None")

    def init_type_embedding(self):
        new_emb = nn.Embedding(2, self.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0]:
            emb = self.embeddings.token_type_embeddings\
                .weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        new_emb.weight.data[1, :].copy_(emb)
        self.embeddings.token_type_embeddings = new_emb

    def forward(self, batch, task='repr', compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task == 'repr':
            f_sub_input_ids = batch['f_sub_input_ids']
            f_sub_pos_ids = batch['f_sub_pos_ids']
            f_v_feats = batch['f_v_feats']
            f_v_pos_ids = batch['f_v_pos_ids']
            f_attn_masks = batch['f_attn_masks']
            f_gather_index = batch['f_gather_index']
            # handle mfm (frame mask)
            f_v_mask = batch['f_v_masks']
            return self.forward_repr(f_sub_input_ids, f_sub_pos_ids,
                                     f_v_feats, f_v_pos_ids,
                                     f_attn_masks, f_gather_index,
                                     img_masks=f_v_mask)
        elif task == 'txt':
            input_ids = batch['input_ids']
            pos_ids = batch['pos_ids']
            attn_masks = batch['attn_masks']
            return self.forward_repr(
                input_ids=input_ids, position_ids=pos_ids,
                img_feat=None, img_pos_ids=None,
                attention_mask=attn_masks, gather_index=None)
        elif task.startswith('mlm'):
            input_ids = batch['input_ids']
            position_ids = batch['position_ids']
            img_feat = batch['v_feat']
            img_pos_ids = batch['f_pos_ids']
            attention_mask = batch['attn_masks']
            gather_index = batch['gather_index']
            txt_mask_tgt = batch['txt_mask_tgt']
            txt_labels = batch['txt_labels']
            return self.forward_mlm(input_ids, position_ids,
                                    img_feat, img_pos_ids,
                                    attention_mask, gather_index,
                                    txt_mask_tgt, txt_labels, compute_loss)
        else:
            raise ValueError(f'Unrecognized task {task}')

    def forward_repr(self, input_ids, position_ids, img_feat, img_pos_ids,
                     attention_mask, gather_index=None,
                     txt_type_ids=None, img_type_ids=None, img_masks=None):
        # embedding layer
        embedding_output = self._compute_img_txt_embeddings(
            input_ids, position_ids, img_feat, img_pos_ids,
            gather_index, txt_type_ids, img_type_ids, img_masks)

        encoder_outputs = self.encoder(embedding_output, attention_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs

    # MLM
    def forward_mlm(self, input_ids, position_ids, img_feat, img_pos_ids,
                    attention_mask, gather_index, txt_mask_tgt,
                    txt_labels=None, compute_loss=True):
        embedding_output = self._compute_img_txt_embeddings(
            input_ids, position_ids, img_feat, img_pos_ids, gather_index)
        sequence_output = self.encoder(embedding_output, attention_mask)[0]

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_mask_tgt)
        prediction_scores = self._pad_layer_unpad(masked_output, self.lm_head)
        if self.vocab_pad:
            prediction_scores = prediction_scores[:, :-self.vocab_pad]

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores, txt_labels,
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

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


class TemporalTrm(RobertaPreTrainedModel):
    """
    Modification for Cross-frame encoding across the temporal axis
    """

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = FrameEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def forward_encoder(self, embedding_output, attention_mask, pool=False):
        encoder_outputs = self.encoder(embedding_output, attention_mask)

        sequence_output = encoder_outputs[0]
        if pool:
            pooled_output = self.pooler(sequence_output)
            return pooled_output
        return sequence_output

    def forward(
            self,
            clip_level_frame_feat,
            clip_level_pos_ids,
            attention_mask,):
        # embedding layer
        embedding_output = self.embeddings(
            clip_level_frame_feat,
            position_ids=clip_level_pos_ids)
        output = self.forward_encoder(embedding_output, attention_mask)
        return output


class QueryFeatEncoder(nn.Module):
    def __init__(self, config, qfeat_dim, modularized=True):
        super().__init__()
        self.query_input_proj = LinearLayer(
            qfeat_dim, config.hidden_size,
            layer_norm=True, dropout=config.hidden_dropout_prob,
            relu=True)
        self.query_pos_embed = QueryFeatEmbeddings(config)
        self.query_self_attention = BertAttention(config)
        self.modularized = modularized
        if self.modularized:
            self.modular_vector_mapping = nn.Linear(
                in_features=config.hidden_size,
                out_features=1,
                bias=False)

    def get_modularized_queries(self, query, query_mask,
                                return_modular_att=False):
        """
        Args:
            query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        modular_attention_scores = self.modular_vector_mapping(
            query)  # (N, L, 1)

        modular_attention_scores = F.softmax(
            mask_logits(modular_attention_scores,
                        query_mask.unsqueeze(2)), dim=1)
        # TODO check whether it is the same
        modular_queries = torch.einsum(
            "blm,bld->bmd", modular_attention_scores,
            query)  # (N, 1, D)
        if return_modular_att:
            return modular_queries[:, 0], modular_attention_scores
        else:
            return modular_queries[:, 0]

    def forward(self, query_feat, query_attn_mask, query_pos_ids=None):
        # Encode Query Features
        query_feat = self.query_input_proj(query_feat)
        query_embeddings = self.query_pos_embed(query_feat)

        query_attn_mask = query_attn_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = query_attn_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # (N, L, D_hidden)
        attended_query = self.query_self_attention(
            query_embeddings, extended_attention_mask)

        if self.modularized:
            # (N, 1, L), torch.FloatTensor
            modularized_query = self.get_modularized_queries(
                attended_query[0], query_attn_mask)
            return modularized_query
        else:
            return attended_query[0]
