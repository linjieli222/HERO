"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

HERO for pretraining
"""
from collections import defaultdict

import random
import torch
from torch import nn
from torch.nn import functional as F
import horovod.torch as hvd

from .model import HeroModel
from .modeling_utils import mask_logits
from .encoder import QueryFeatEncoder


class HeroForPretraining(HeroModel):
    def __init__(self, config, vfeat_dim, max_frm_seq_len,
                 conv_stride=1, conv_kernel_size=5,
                 ranking_loss_type="hinge", margin=0.1,
                 lw_neg_ctx=0, lw_neg_q=0, lw_st_ed=0.01, drop_svmr_prob=0,
                 use_hard_negative=False, hard_pool_size=20,
                 hard_neg_weight=10, use_all_neg=True):
        super().__init__(config, vfeat_dim, max_frm_seq_len)

        self.config = config

        # VCMR related configs
        self.lw_st_ed = lw_st_ed
        self.lw_neg_q = lw_neg_q
        self.lw_neg_ctx = lw_neg_ctx
        self.ranking_loss_type = ranking_loss_type
        self.use_hard_negative = use_hard_negative
        self.hard_pool_size = hard_pool_size
        self.hard_neg_weight = hard_neg_weight
        self.margin = margin
        self.use_all_neg = use_all_neg
        self.drop_svmr_prob = drop_svmr_prob
        self.gather_gpus = True  # use this to switch on GPU gathering

        conv_cfg = dict(in_channels=1,
                        out_channels=1,
                        kernel_size=conv_kernel_size,
                        stride=conv_stride,
                        padding=conv_kernel_size // 2,
                        bias=False)
        self.video_query_linear = nn.Linear(
                config.q_config.hidden_size,
                config.c_config.hidden_size)
        self.video_st_predictor = nn.Conv1d(**conv_cfg)
        self.video_ed_predictor = nn.Conv1d(**conv_cfg)

        # Use CrossModalTrm to encode query
        self.qfeat_dim = config.f_config.hidden_size

        self.q_feat_attn = QueryFeatEncoder(
            config.q_config, self.qfeat_dim)

    def forward(self, batch, task='vsm', compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task == 'vsm':
            frame_embeddings = self.v_encoder(batch, 'repr')
            targets = batch['targets']

            modularized_query = self.encode_txt_inputs(
                    batch['query_input_ids'], batch['query_pos_ids'],
                    batch['query_attn_masks'], attn_layer=self.q_feat_attn)

            q2video_scores, st_prob, ed_prob = None, None, None
            if self.lw_st_ed != 0:
                prob = random.random()
                if prob > self.drop_svmr_prob or not self.training:
                    st_prob, ed_prob = self.get_pred_from_mod_query(
                        frame_embeddings, batch['c_attn_masks'],
                        modularized_query)

            if self.lw_neg_ctx != 0 or self.lw_neg_q != 0:
                q2video_scores = self.get_video_level_scores(
                    modularized_query, frame_embeddings,
                    batch['c_attn_masks'])

            if compute_loss:
                dtype = frame_embeddings.dtype
                device = frame_embeddings.device
                loss_st_ed = torch.zeros(1, dtype=dtype, device=device)
                loss_neg_ctx = torch.zeros(1, dtype=dtype, device=device)
                loss_neg_q = torch.zeros(1, dtype=dtype, device=device)
                reduction = 'mean' if self.training else 'sum'
                if st_prob is not None:
                    if len(st_prob.size()) == 3:
                        row_indices = torch.arange(
                            0, len(st_prob), device=device)
                        col_indices = batch["q_vidx"]
                        st_prob = st_prob[row_indices, col_indices]
                        ed_prob = ed_prob[row_indices, col_indices]
                    loss_st = F.cross_entropy(
                        st_prob, targets[:, 0].long(), reduction=reduction,
                        ignore_index=-1)
                    loss_ed = F.cross_entropy(
                        ed_prob, targets[:, 1].long(), reduction=reduction,
                        ignore_index=-1)
                    loss_st_ed = loss_st + loss_ed

                if q2video_scores is not None:
                    loss_neg_ctx, loss_neg_q = self.get_video_level_loss(
                        q2video_scores, reduction)

                loss_st_ed = self.lw_st_ed * loss_st_ed
                loss_neg_ctx = self.lw_neg_ctx * loss_neg_ctx
                loss_neg_q = self.lw_neg_q * loss_neg_q

                return loss_st_ed, loss_neg_ctx, loss_neg_q
            return q2video_scores, st_prob, ed_prob
        elif task.startswith('mlm'):
            return self.v_encoder(batch, task, compute_loss)
        elif task == 'mffr':
            return self.v_encoder(batch, 'mffr', compute_loss)
        elif task == 'mfm-nce':
            return self.v_encoder(batch, 'mfm-nce', compute_loss)
        elif task == 'fom':
            return self.v_encoder(batch, 'fom', compute_loss)
        else:
            raise ValueError(f'Unrecognized task {task}')

    def _get_st_ed_prob(self, modularized_query, context_feat2,
                        context_mask, cross=False):
        """
        Args:
            modularized_query: (N, D)
            context_feat2: (N, L, D),
                output of the first transformer encoder layer
            context_mask: (N, L)
            st_predictor:
            ed_predictor:
            cross: at inference,
                calculate prob for each possible pairs of query and context.
        """
        # (N, D) no need to normalize here.
        query = self.video_query_linear(modularized_query)
        if cross:
            # (Nq, Nv, L)  from query to all videos.
            similarity = torch.einsum(
                "md,nld->mnl", query, context_feat2)
            n_q, n_c, len_ = similarity.shape
            similarity = similarity.view(n_q * n_c, 1, len_)
            st_prob = self.video_st_predictor(
                similarity).view(n_q, n_c, len_)  # (Nq, Nv, L)
            ed_prob = self.video_ed_predictor(
                similarity).view(n_q, n_c, len_)  # (Nq, Nv, L)
            context_mask = context_mask.unsqueeze(0)  # (1, Nv, L)
        else:
            similarity = torch.einsum(
                "bd,bld->bl", query, context_feat2)  # (N, L)
            st_prob = self.video_st_predictor(
                similarity.unsqueeze(1)).squeeze()  # (N, L)
            ed_prob = self.video_ed_predictor(
                similarity.unsqueeze(1)).squeeze()  # (N, L)

        context_mask = context_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        st_prob = mask_logits(st_prob, context_mask)  # (N, L)
        ed_prob = mask_logits(ed_prob, context_mask)
        return st_prob, ed_prob

    def encode_txt_inputs(
            self, input_ids, pos_ids, attn_masks, attn_layer=None,
            normalized=False):
        input_batch = {}
        input_batch['input_ids'] = input_ids
        input_batch['pos_ids'] = pos_ids
        input_batch['attn_masks'] = attn_masks
        outputs = self.v_encoder.f_encoder(input_batch, 'txt')
        # [num_queries, len, 768]
        txt_roberta_feats = outputs[0]
        if normalized:
            txt_roberta_feats = F.normalize(
                txt_roberta_feats, dim=-1, eps=1e-5)
        if attn_layer is not None:
            # [num_queries, 768]
            modularized_txt = attn_layer(
                    txt_roberta_feats, attn_masks)
            return modularized_txt
        return txt_roberta_feats

    def get_pred_from_mod_query(self, frame_embeddings, c_attn_masks,
                                modularized_query, cross=False):
        v_bs = frame_embeddings.shape[0]
        q_bs = modularized_query.shape[0]

        if not cross and v_bs == q_bs:
            cross = False
        else:
            cross = True

        st_prob, ed_prob = self._get_st_ed_prob(
            modularized_query, frame_embeddings,
            c_attn_masks, cross=cross)
        return st_prob, ed_prob

    def get_video_level_loss(self, query_context_scores, reduction="mean"):
        """ ranking loss between (pos. query + pos. video)
            and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (Nq, Nv), cosine similarity [-1, 1],
                Each row contains the scores
                between the query to each of the videos inside the batch.
        """
        bsz_q, bsz_v = query_context_scores.size()  # (Nq, Nv)
        num_q_per_v = int(bsz_q/bsz_v)
        loss_neg_ctx = torch.tensor(0).to(query_context_scores.device)
        loss_neg_q = torch.tensor(0).to(query_context_scores.device)
        if bsz_v == 1:
            return loss_neg_ctx, loss_neg_q

        # (Nq, Nv)
        query_context_scores_masked = query_context_scores.clone()
        pos_video_query_scores = []
        for i, j in zip(range(bsz_v), range(0, bsz_q, num_q_per_v)):
            pos_video_query_scores.append(
                query_context_scores[j: j+num_q_per_v, i])
            # impossibly large for cosine similarity, the copy is created
            # as modifying the original will cause error
            query_context_scores_masked[
                j: j+num_q_per_v, i] = 999
        # (Nv, 5)
        pos_video_query_scores = torch.stack(pos_video_query_scores, dim=0)
        video_query_scores_masked = query_context_scores_masked.transpose(0, 1)
        # (Nq, 1)
        pos_query_video_scores = pos_video_query_scores.view(bsz_q, 1)

        if self.use_all_neg:
            # get negative videos per query
            # (Nq, Nv-1)
            pos_query_neg_context_scores = self.get_all_neg_scores(
                query_context_scores_masked,
                sample_min_idx=1)
            # (Nq, Nv-1)
            loss_neg_ctx = self.get_ranking_loss(
                pos_query_video_scores, pos_query_neg_context_scores)
            if self.use_hard_negative:
                weighting_mat = torch.ones_like(loss_neg_ctx)
                weighting_mat[:, self.hard_pool_size:] = .1
                weighting_mat[:, :self.hard_pool_size] = self.hard_neg_weight
                loss_neg_ctx = weighting_mat * loss_neg_ctx

            # get negative query per video
            # (Nv, Nq-5)
            neg_query_pos_context_scores = self.get_all_neg_scores(
                video_query_scores_masked,
                sample_min_idx=num_q_per_v)
            # (Nv, 1, Nq-5)
            neg_query_pos_context_scores =\
                neg_query_pos_context_scores.unsqueeze(1)
            # (Nv, 5, 1)
            pos_video_query_scores = pos_video_query_scores.unsqueeze(-1)
            # (Nv, 5, Nq-5)
            loss_neg_q = self.get_ranking_loss(
                pos_video_query_scores, neg_query_pos_context_scores)
            # (Nq, Nq-5)
            loss_neg_q = loss_neg_q.view(-1, loss_neg_q.size(2))
            if self.use_hard_negative:
                weighting_mat = torch.ones_like(loss_neg_q)
                weighting_mat[:, self.hard_pool_size:] = .1
                weighting_mat[:, :self.hard_pool_size] = self.hard_neg_weight
                loss_neg_q = weighting_mat * loss_neg_q
        else:
            # (Nq, 1)
            pos_query_neg_context_scores = self.get_sampled_neg_scores(
                query_context_scores_masked,
                sample_min_idx=1).unsqueeze(-1)
            # (Nq, 1)
            loss_neg_ctx = self.get_ranking_loss(
                pos_query_video_scores, pos_query_neg_context_scores)
            # (Nv, 1)
            neg_query_pos_context_scores = self.get_sampled_neg_scores(
                video_query_scores_masked,
                sample_min_idx=num_q_per_v).unsqueeze(-1)
            # (Nv, 5)
            loss_neg_q = self.get_ranking_loss(
                pos_video_query_scores, neg_query_pos_context_scores)

        if reduction == "sum":
            return loss_neg_ctx.mean(1), loss_neg_q.mean(1)
        elif reduction == "mean":
            return loss_neg_ctx.mean(1).mean(0), loss_neg_q.mean(1).mean(0)
        elif reduction is None:
            return loss_neg_ctx, loss_neg_q
        else:
            raise NotImplementedError(f"reduction {reduction} not supported")

    def get_sampled_neg_scores(self, scores_masked, sample_min_idx=1):
        """
        scores_masked: (Nq, Nv)
            except that the diagonal (positive) positions
            are masked with a large value.
        """
        bsz, sample_size = scores_masked.size()
        assert sample_size > sample_min_idx,\
            "Unable to sample negative when bsz==sample_min_idx"
        num_neg = bsz
        pos_indices = torch.arange(bsz).to(scores_masked.device)

        _, sorted_scores_indices = torch.sort(
            scores_masked, descending=True, dim=1)
        # skip the masked positive
        sample_max_idx = min(
            sample_min_idx + self.hard_pool_size, sample_size) \
            if self.use_hard_negative else sample_size
        sampled_neg_score_indices = sorted_scores_indices[
            pos_indices, torch.randint(
                sample_min_idx, sample_max_idx,
                size=(num_neg,)).to(scores_masked.device)]  # (N, )
        sampled_neg_scores = scores_masked[
            pos_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_all_neg_scores(self, scores_masked,
                           pos_indices=None, sample_min_idx=1):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos.
            Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores,
            except that the diagonal (positive) positions
            are masked with a large value.
        """
        bsz, sample_size = scores_masked.size()
        assert sample_size > sample_min_idx,\
            "Unable to sample negative when bsz==sample_min_idx"
        if pos_indices is None:
            pos_indices = torch.arange(bsz).to(scores_masked.device)

        sorted_scores_masked, sorted_scores_indices = torch.sort(
            scores_masked, descending=True, dim=1)
        # skip the masked positive
        # (N, sample_size-sample_min_idx)
        neg_scores = sorted_scores_masked[:, sample_min_idx:]
        return neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger
            than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        if self.ranking_loss_type == "hinge":
            # max(0, m + S_neg - S_pos)
            loss = torch.clamp(
                self.margin + neg_score - pos_score,
                min=0)
        elif self.ranking_loss_type == "lse":
            # log[1 + exp(S_neg - S_pos)]
            loss = torch.log1p(
                torch.exp(neg_score - pos_score))
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")

        return loss

    def get_video_level_scores(self, modularized_query,
                               context_feat1, context_mask,
                               val_gather_gpus=True):
        """ Calculate video2query scores for each pair of video
            and query inside the batch.
        Args:
            modularized_query: (N, D)
            context_feat1: (N, L, D),
                output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)
                score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        modularized_query = F.normalize(modularized_query, dim=-1, eps=1e-5)
        context_feat1 = F.normalize(context_feat1, dim=-1, eps=1e-5)
        # gather all ranks to increase negative examples
        # only do this at training, multi-GPU eval is not supported
        if self.training and self.gather_gpus or\
                not self.training and val_gather_gpus:
            # need to pad video to same length
            bs, vlen, hid = context_feat1.size()
            device = context_feat1.device
            all_vlens = hvd.allgather(torch.tensor([vlen], device=device)
                                      ).view(hvd.size())
            max_vlen = all_vlens.max().item()
            pad_len = max_vlen - all_vlens[hvd.rank()]
            if pad_len != 0:
                pad = torch.zeros(bs, pad_len, hid,
                                  dtype=context_feat1.dtype, device=device)
                context_feat1 = torch.cat([context_feat1, pad], dim=1)
                mask_pad = pad[..., 0].long()
                context_mask = torch.cat([context_mask, mask_pad], dim=1)
            # our backprop compatible allgather
            modularized_query = vsm_allgather(modularized_query).contiguous()
            context_feat1 = vsm_allgather(context_feat1).contiguous()
            context_mask = vsm_allgather(context_mask).contiguous()

        query_context_scores = torch.einsum(
            "md,nld->mln", modularized_query, context_feat1)  # (N, L, N)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        context_mask = context_mask.to(dtype=query_context_scores.dtype
                                       )  # fp16 compatibility
        query_context_scores = mask_logits(
            query_context_scores, context_mask)  # (N, L, N)
        query_context_scores, _ = torch.max(
            query_context_scores,
            dim=1)  # (N, N) diagonal positions are positive pairs.
        return query_context_scores

    def set_hard_negative(self, use_hard_negative, hard_pool_size,
                          hard_neg_weight):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.use_hard_negative = use_hard_negative
        self.hard_pool_size = hard_pool_size
        self.hard_neg_weight = hard_neg_weight

    def set_train_st_ed(self, lw_st_ed):
        """pre-train video retrieval then span prediction"""
        self.lw_st_ed = lw_st_ed


class VsmAllgather(torch.autograd.Function):
    """ our special allgather implementation
        for scaling up TVR VCMR batch size
    """
    @staticmethod
    def forward(ctx, tensor, name):
        ctx.dim = tensor.shape[0]
        # we try to put all sync ops in forward pass
        ctx.all_dims = hvd.allgather(
            torch.tensor([ctx.dim], device=tensor.device)
        ).view(hvd.size())
        handle = hvd.allgather_async(tensor, name)
        return hvd.synchronize(handle)

    @staticmethod
    def backward(ctx, grad_output):
        # every rank is identical after gather, no need to allreduce
        r = hvd.rank()
        offset = (torch.sum(ctx.all_dims.narrow(0, 0, r)).item() if r != 0
                  else 0)
        return grad_output.narrow(0, offset, ctx.dim), None


def vsm_allgather(tensor):
    return VsmAllgather.apply(tensor, None)
