"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Input Embedding Layers
"""
import torch
import torch.nn as nn
from apex.normalization.fused_layer_norm import FusedLayerNorm


class SubEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, position_ids=None,
                token_type_ids=None, inputs_embeds=None):
        device = input_ids.device if input_ids is not None\
            else inputs_embeds.device

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids.
                # Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids).to(device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        if token_type_ids is None:
            token_type_embeddings = self.token_type_embeddings(
                torch.ones(1, 1, dtype=torch.long, device=device))
        else:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (inputs_embeds
                      + position_embeddings
                      + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_input_ids(self, x):
        """ Replace non-padding symbols with their position numbers.
            Position numbers begin at padding_idx+1.
            Padding symbols are ignored.
            This is modified from fairseq's `utils.make_positions`.
        :param torch.Tensor x:
        :return torch.Tensor:
        """
        mask = x.ne(self.padding_idx).long()
        incremental_indicies = torch.cumsum(mask, dim=1) * mask
        return incremental_indicies + self.padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly.
            We cannot infer which are padded so just generate
            sequential position ids.
        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1,
            dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class ImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim, max_img_seq_len):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_LayerNorm = FusedLayerNorm(img_dim, eps=1e-5)
        self.position_embeddings = nn.Embedding(max_img_seq_len,
                                                config.hidden_size)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, type_embeddings, img_pos_ids=None,
                img_masks=None):
        if img_pos_ids is None:
            img_pos_ids = self.create_position_ids_from_inputs_embeds(
                img_feat)

        if img_masks is not None:
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_linear(self.img_LayerNorm(img_feat))
        position_embeddings = self.position_embeddings(img_pos_ids)
        embeddings = transformed_im + position_embeddings + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly.
            We cannot infer which are padded so just generate
            sequential position ids.
        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            0, sequence_length,
            dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class FrameEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, frame_feat, position_ids=None):
        input_shape = frame_feat.size()

        seq_length = input_shape[1]

        if position_ids is None:
            seq_length = frame_feat.shape[1]
            position_ids = torch.arange(seq_length, dtype=torch.long,
                                        device=frame_feat.device).unsqueeze(0)
        # num_videos, num_frames
        position_embeddings = self.position_embeddings(position_ids)
        # num_videos, num_frames, 768
        embeddings = frame_feat + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class QueryFeatEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(QueryFeatEmbeddings, self).__init__()
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_feat, position_ids=None):
        """
        Args:
            input_feat: (N, L, D)
        """
        seq_length = input_feat.shape[1]
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long,
                                        device=input_feat.device).unsqueeze(0)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
