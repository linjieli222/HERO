"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Modified from https://github.com/jayleicn/TVCaption
PyTorch Modules for TVC
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .layers import (BertSelfAttention, BertLayerNorm,
                     BertSelfOutput, BertIntermediate, BertOutput)
from .model import HeroModel


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size,
                 ignore_index=-100, reduction='none'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)

        # count for the ground-truth word
        smoothing_value = label_smoothing / (tgt_vocab_size - 1)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing
        self.reduction = reduction

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size, with indices in
            [-1, tgt_vocab_size-1], `-1` is ignored
        """
        # ignore examples with target value -1
        valid_indices = target != self.ignore_index
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        loss = F.kl_div(output, model_prob, reduction='none').sum(dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError
        return loss


class BertDecEncAttention(BertSelfAttention):
    def forward(self, encoder_outputs, dec_hidden_states, attention_mask=None):
        mixed_query_layer = self.query(dec_hidden_states)
        mixed_key_layer = self.key(encoder_outputs)
        mixed_value_layer = self.value(encoder_outputs)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query"
        # and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is
            # (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs)\
            if self.output_attentions else (context_layer,)
        return outputs


class BertDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attention = BertSelfAttention(config)
        self.add_norm_1 = BertSelfOutput(config)
        self.dec_enc_attention = BertDecEncAttention(config)
        self.add_norm_2 = BertSelfOutput(config)
        self.intermidiate = BertIntermediate(config)
        self.add_norm_3 = BertOutput(config)  # linear + residual + layernorm

    def forward(self, dec_hidden_states, enc_outputs, enc_mask,
                tri_mask=None):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
        Returns:

        """
        if tri_mask is None:
            max_len = dec_hidden_states.size(1)  # Lt
            tri_mask = torch.tril(torch.ones(max_len, max_len), diagonal=0
                                  ).to(dec_hidden_states
                                       ).unsqueeze(0).unsqueeze(1)
            tri_mask = (1.0 - tri_mask) * -10000.0

        # 1, dec self attn + add_norm
        attention_output = self.self_attention(
            dec_hidden_states, tri_mask)[0]  # (N, Lt, D)
        attention_output = self.add_norm_1(attention_output,
                                           dec_hidden_states)  # (N, Lt, D)

        # 2, dec enc attn + add_norm
        # Use the mask associated with key/value, not query. (q, k, v)
        # Additionally, there is no need to do subsequent masking, since each
        # word has the right to see all the video info.
        dec_enc_attention_output = self.dec_enc_attention(
            enc_outputs, attention_output, enc_mask)[0]  # (N, Lt, D)
        dec_enc_attention_output = self.add_norm_2(
            dec_enc_attention_output, attention_output)  # (N, Lt, D)

        # 3, FFN + add_norm
        output = self.intermidiate(dec_enc_attention_output)  # (N, Lt, D)
        output = self.add_norm_3(output,
                                 dec_enc_attention_output)  # (N, Lt, D)
        return output  # (N, Lt, D)


class BertDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertDecoderLayer(config)
                                    for _ in range(config.num_hidden_layers)])
        tri_mask = torch.tril(torch.ones(1024, 1024), diagonal=0)
        tri_mask = (1.0 - tri_mask) * -10000.0
        self.register_buffer('tri_mask', tri_mask)

    def forward(self, dec_hidden_states, enc_outputs, enc_mask,
                output_all_encoded_layers=False):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve
                auto-regressive property
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        len_ = dec_hidden_states.size(1)
        tri_mask = self.tri_mask[:len_, :len_].unsqueeze(0).unsqueeze(1)
        enc_mask = enc_mask.unsqueeze(1).unsqueeze(2).to(dec_hidden_states)
        enc_mask = (1.0 - enc_mask) * -10000.0
        for layer_idx, layer_module in enumerate(self.layer):
            dec_hidden_states = layer_module(
                dec_hidden_states, enc_outputs, enc_mask, tri_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(dec_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(dec_hidden_states)
        return all_encoder_layers


class HeroForTvc(HeroModel):
    def __init__(self, config, vfeat_dim, max_frm_seq_len, lsr=0.1):
        super().__init__(config, vfeat_dim, max_frm_seq_len)
        self.config = config

        self.position_embeddings = nn.Embedding(
            config.d_config.max_position_embeddings,
            config.d_config.hidden_size)
        self.emb_LayerNorm = BertLayerNorm(config.d_config.hidden_size,
                                           eps=1e-5)
        self.decoder = BertDecoder(config.d_config)

        if lsr > 0:
            self.loss_func = LabelSmoothingLoss(lsr,
                                                config.f_config.vocab_size,
                                                ignore_index=-1,
                                                reduction='none')
        else:
            self.loss_func = nn.CrossEntropyLoss(ignore_index=-1,
                                                 reduction='none')

        self.v_encoder.initialize()

    def encode(self, batch):
        frame_embeddings = self.v_encoder(batch, 'repr')
        # pick video segments with associated captions
        segment_embeddings = [frame_embeddings[i, st:ed, :]
                              for i, segs in enumerate(batch['clip_ranges'])
                              for st, ed in segs]

        def pad_tensors(ts):
            """ pad segmet embeddings """
            bs = len(ts)
            max_l = max(t.size(0) for t in ts)
            hid = ts[0].size(1)
            output = torch.zeros(bs, max_l, hid).to(ts[0])
            for i, t in enumerate(ts):
                len_ = t.size(0)
                output[i, :len_, :] = t
            return output

        encoder_outputs = pad_tensors(segment_embeddings)
        return encoder_outputs

    def decode(self, encoder_outputs, encoder_masks,
               caption_ids, pos_ids, label_ids, compute_loss=True):
        """
        Args:
            text_input_ids: (N, Lt)
            text_masks: (N, Lt)  with 1 indicates valid bits
            text_input_labels: (N, Lt)  with `-1` on ignored positions
            encoder_outputs: (N, Lctx, D)
            encoder_masks: (N, Lctx)
        """
        # shared embedding layer
        text_embeddings = self.v_encoder.f_encoder.embeddings.word_embeddings(
            caption_ids)
        pos_embeddings = self.position_embeddings(pos_ids)
        embeddings = self.emb_LayerNorm(text_embeddings + pos_embeddings)
        decoder_outputs = self.decoder(
            embeddings, encoder_outputs, encoder_masks)[-1]  # (N, Lt, D)
        # shared projection layer
        prediction_scores = self.v_encoder.f_encoder.lm_head(
            decoder_outputs)  # (N, Lt, vocab_size)
        if compute_loss:
            caption_loss = self.loss_func(
                prediction_scores.view(-1, self.config.f_config.vocab_size),
                label_ids.view(-1))
            return caption_loss
        else:
            return prediction_scores

    def forward(self, batch, mode='train', compute_loss=True):
        encoder_outputs = self.encode(batch)  # (N, Lv, D)
        attn_mask = batch['cap_attn_mask']
        caption_ids = batch['cap_input_ids']
        pos_ids = batch['cap_pos_ids']
        label_ids = batch['cap_tgt_ids']
        res = self.decode(encoder_outputs, attn_mask,
                          caption_ids, pos_ids, label_ids, compute_loss)
        return res


def _to_fp16(batch):
    if isinstance(batch, torch.Tensor) and 'Float' in batch.type():
        return batch.half()
    elif isinstance(batch, list):
        new_batch = [_to_fp16(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(_to_fp16(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: _to_fp16(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


class TvcGenerator(object):
    def __init__(self, model, max_step, bos, eos, fp16):
        self.model = model
        self.max_step = max_step
        self.bos = bos
        self.eos = eos
        self.fp16 = fp16

    def greedy_decode(self, batch):
        """
        run greedy decoding
        NOTE: Speed can potentially be improved by keeping past
              decoder hidden states and only run `step-wise` forward.
              Also, maybe can add early stop when all sequences reaches eos
              instead of running until max_step.
        """
        if self.fp16:
            batch = _to_fp16(batch)
        encoder_outputs = self.model.encode(batch)  # (N, Lv, D)
        if self.fp16:
            encoder_outputs = encoder_outputs.half()
        enc_mask = batch['cap_attn_mask']
        batch_size = enc_mask.size(0)
        bos = torch.tensor([self.bos]).expand(batch_size).cuda()
        input_ids = torch.zeros(batch_size, self.max_step).to(bos)
        pos_ids = torch.arange(0, self.max_step+1).unsqueeze(0).cuda()
        last_out = bos
        for step in range(self.max_step):
            input_ids[:, step] = last_out
            score = self.model.decode(encoder_outputs, enc_mask,
                                      input_ids[:, :step+1],
                                      pos_ids[:, :step+1],
                                      None, compute_loss=False)
            output_ids = score.max(dim=-1)[1]
            last_out = output_ids[:, -1]

        outputs = [self.cut_eos(ids) for ids in output_ids.tolist()]
        return outputs

    def cut_eos(self, ids):
        out_ids = []
        for i in ids:
            if i == self.eos:
                break
            out_ids.append(i)
        return out_ids
