

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import xavier_uniform_, constant_

from models.layers.transformer import MLP
from models.layers.utils import (
    linear_init_,
    bias_init_with_prob,
    _get_clones,
    inverse_sigmoid,
    get_cdn_group,
    multi_scale_deformable_attn_pytorch,
)


class MSDeformAttn(nn.Module):
    """
    Original Multi-Scale Deformable Attention Module.
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}"
            )
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        assert (
            _d_per_head * n_heads == d_model
        ), "`d_model` must be divisible by `n_heads`"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points
        )
        offset_normalizer = torch.as_tensor(
            value_shapes, dtype=query.dtype, device=query.device
        ).flip(-1)
        add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        output = multi_scale_deformable_attn_pytorch(
            value, value_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)
        return output

class DeformableTransformerDecoderLayer(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """
    def __init__(
        self,
        d_model=256,
        n_heads=8,
        d_ffn=1024,
        dropout=0.0,
        act=nn.ReLU(),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        embed,
        refer_bbox,
        feats,
        shapes,
        padding_mask=None,
        attn_mask=None,
        query_pos=None,
    ):
        # self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            embed.transpose(0, 1),
            attn_mask=attn_mask,
        )[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)
        # cross attention
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos),
            refer_bbox.unsqueeze(2),
            feats,
            shapes,
            padding_mask,
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)
        embed = self.forward_ffn(embed)
        return embed


class DeformableTransformerDecoder(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed,  # decoder embeddings
        refer_points,  # anchor
        feats,  # image features
        shapes,  # feature shapes
        centroid_head,
        pos_mlp,
        attn_mask=None,
        padding_mask=None,
        z_head=None,
    ):
        output = embed
        embeddings = []
        dec_centroids = []
        refer_centroids = refer_points[..., :2].sigmoid()
        last_refined_centroid = torch.cat([refer_centroids, refer_points[..., 2:]], -1)
        for i, layer in enumerate(self.layers):
            # last_refined_centroid has to be detached
            output = layer(
               output,
               last_refined_centroid[..., :2],
               feats,
               shapes,
               padding_mask,
               attn_mask,
               pos_mlp(last_refined_centroid[..., :3]),
            )
            centroid = centroid_head[i](output)
            if z_head is not None:
               refined_centroid = torch.sigmoid(
                   centroid + inverse_sigmoid(last_refined_centroid[..., :2])
               )
               refined_z = F.tanh(z_head[i](output)) + last_refined_centroid[..., 2:]
               refined_centroid = torch.cat([refined_centroid, refined_z], -1)
            else:
               refined_centroid = torch.sigmoid(
                   centroid + inverse_sigmoid(last_refined_centroid)
               )
            embeddings.append(output)
            dec_centroids.append(refined_centroid)
            last_refined_centroid = (
               refined_centroid.detach() if self.training else refined_centroid
            )
        return torch.stack(embeddings), torch.stack(dec_centroids)

class DirectDeform(nn.Module):
    def __init__(self):
        super().__init__()
    