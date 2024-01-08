from typing import Union, List, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


from .mvit import _create_mvitv2
from models.layers.transformer import MLP
from models.layers.utils import (
    linear_init_,
    bias_init_with_prob,
    _get_clones,
    inverse_sigmoid,
    get_cdn_group,
    multi_scale_deformable_attn_pytorch,
)
from models.layers.conv import Conv, RepC3, DWConv, HGStem, HGBlock


class HybridEncoder(nn.Module):
    def __init__(self, in_channels: List[int] = [512, 1024, 2048], hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        act = "silu"
        expansion = 1.0
        depth_mult = 1.0

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(Conv(in_channel, hidden_dim, 1, act=True))

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(Conv(hidden_dim, hidden_dim, 1, act=True))

            self.fpn_blocks.append(
                RepC3(hidden_dim * 2, hidden_dim, round(3 * depth_mult), e=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(Conv(hidden_dim, hidden_dim, 3, 2, act=True))
            self.pan_blocks.append(
                RepC3(hidden_dim * 2, hidden_dim, round(3 * depth_mult), e=expansion)
            )

    def forward(self, feats):
        """
        Args:
            feats: feature maps from backbone [P3, P4, P5]
        """
        assert len(feats) == len(self.in_channels)
        # proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        proj_feats = []
        for i, feat in enumerate(feats):
            proj_feats.append(self.input_proj[i](feat))

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], axis=1)
            )
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            a = torch.concat([downsample_feat, feat_height], axis=1)
            out = self.pan_blocks[idx](
                torch.concat([downsample_feat, feat_height], axis=1)
            )
            outs.append(out)

        return outs


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
                bottom-right (1, 1), inum_classesluding padding area
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
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(
                value_shapes, dtype=query.dtype, device=query.device
            ).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = (
                sampling_offsets
                / self.n_points
                * refer_bbox[:, :, None, :, None, 2:]
                * 0.5
            )
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(
                f"Last dim of referenum_classese_points must be 2 or 4, but got {num_points}."
            )
        output = multi_scale_deformable_attn_pytorch(
            value, value_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)


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
        return self.norm3(tgt)

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

        # ffn
        return self.forward_ffn(embed)


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
        refer_bbox,  # anum_classeshor
        feats,  # image features
        shapes,  # feature shapes
        bbox_head,
        score_head,
        pos_mlp,
        attn_mask=None,
        padding_mask=None,
    ):
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                refer_bbox,
                feats,
                shapes,
                padding_mask,
                attn_mask,
                pos_mlp(refer_bbox),
            )

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(
                        torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox))
                    )
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class DeformablePoseViT(nn.Module):
    def __init__(
        self,
        backbone_code: str,
        num_classes: int = 21,
        hidden_dim: int = 256,
        num_queries: int = 2,  # one for rot, one for trans
        in_channels: List[int] = [192, 384, 768],
        channels: List[int] = [256, 256, 256],
        num_decoder_points: int = 8,  # 4
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        learnt_init_query: bool = False,
    ):
        super().__init__()

        self.backbone = _create_mvitv2(backbone_code, pretrained=True, num_token=2)
        embed_dim = self.backbone.embed_dim
        self._setup_backbone()
        self.fpn = HybridEncoder(in_channels, hidden_dim)

        self.bbox_embed = nn.Linear(4, embed_dim)
        self.cls_token = nn.Embedding(num_classes, embed_dim)

        self.hidden_dim = hidden_dim
        self.nhead = num_heads
        self.num_levels = len(channels)  # num level
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers

        # backbone feature projection
        self.input_proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim)
            )
            for x in channels
        )
        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim,
            num_heads,
            dim_feedforward,
            dropout,
            act,
            self.num_levels,
            num_decoder_points,
        )
        self.decoder = DeformableTransformerDecoder(
            hidden_dim, decoder_layer, num_decoder_layers, -1
        )

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(hidden_dim, hidden_dim, 4, num_layers=3)
                for _ in range(num_decoder_layers)
            ]
        )

        self._reset_parameters()

    def forward(self, x, batch=None):
        from ultralytics.models.utils.ops import get_cdn_group

        fpn_feats = self.forward_backbone(x)
        # input projection and embedding
        feats, shapes = self._get_encoder_input(fpn_feats)

        # prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = None, None, None, None
        (
            embed,
            refer_bbox,
            enum_classes_bboxes,
            enum_classes_scores,
        ) = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enum_classes_bboxes, enum_classes_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+num_classes)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def forward_backbone(self, x):
        img = x["roi_img"]
        cls_token = self.cls_token(x["roi_cls"]).unsqueeze(1)
        bbox_token = self.bbox_embed(x["bbox"]).unsqueeze(1)

        feats, feat_size = self.backbone.patch_embed(img)
        feats = torch.cat([cls_token, bbox_token, feats], dim=1)
        # 192, 384, 768
        fpn_feats = []
        for stage in self.backbone.stages:
            feats, feat_size = stage(feats, feat_size)
            fpn_feats.append(
                feats[:, 2:, :].reshape(
                    feats.shape[0],
                    feats.shape[-1],
                    *feat_size[-2:],
                )
            )

        fpn_feats = self.fpn(fpn_feats[-3:])
        return fpn_feats

    def _get_encoder_input(self, x):
        # get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') 
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask


    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        bs = len(feats)
        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(
            shapes, dtype=feats.dtype, device=feats.device
        )
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)
        # query selection
        topk_ind = torch.topk(
            enc_outputs_scores.max(-1).values, self.num_queries, dim=1
        ).indices.view(-1)
        # (bs, num_queries)
        batch_ind = (
            torch.arange(end=bs, dtype=topk_ind.dtype)
            .unsqueeze(-1)
            .repeat(1, self.num_queries)
            .view(-1)
        )

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        # dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features)

        enc_bboxes = refer_bbox.sigmoid()
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(
            bs, self.num_queries, -1
        )

        embeddings = (
            self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            if self.learnt_init_query
            else top_k_features
        )
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _setup_backbone(self):
        del self.backbone.head
        del self.backbone.cls_token
        self.backbone.head = None

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.num_classes
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
