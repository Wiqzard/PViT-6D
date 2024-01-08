from collections import OrderedDict
from typing import Optional


import torch
import torch.nn as nn

# from torch.nn import Sequential
from torch import Tensor
import torch.nn.functional as F
from torch.nn import init



class GeGLU(nn.Module):
    """
    ### GeGLU Activation

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * F.gelu(gate)


class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.positional_encodings = nn.Parameter(
            torch.zeros(max_len, 1, d_model), requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.positional_encodings[: x.shape[0]]


class BoundingBoxEmbeddingSine(nn.Module):
    """
    Positional embedding of bounding box coordinates.
    """

    def __init__(self, num_pos_feats=32):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, bboxes: torch.Tensor):
        # Assuming only the bboxes for a single image get passed
        if bboxes.ndim == 1:
            bboxes = bboxes[None, :]
        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=bboxes.device
        )
        dim_t = 2**dim_t
        x_enc = bboxes[:, 0, None] * dim_t
        y_enc = bboxes[:, 1, None] * dim_t
        w_enc = bboxes[:, 2, None] * dim_t
        h_enc = bboxes[:, 3, None] * dim_t
        x_enc = torch.cat((x_enc.sin(), x_enc.cos()), dim=-1)
        y_enc = torch.cat((y_enc.sin(), y_enc.cos()), dim=-1)
        w_enc = torch.cat((w_enc.sin(), w_enc.cos()), dim=-1)
        h_enc = torch.cat((h_enc.sin(), h_enc.cos()), dim=-1)
        pos_embed = torch.cat((x_enc, y_enc, w_enc, h_enc), dim=-1)
        return pos_embed


class BoundingBoxEmbeddingLearned(nn.Module):
    def __init__(self, img_size: int = 250, out_features: int = 512):
        super().__init__()
        assert out_features % 4 == 0
        out_features = out_features // 4

        self.x_emb = nn.Embedding(img_size, out_features)
        self.y_emb = nn.Embedding(img_size, out_features)
        self.w_emb = nn.Embedding(img_size, out_features)
        self.h_emb = nn.Embedding(img_size, out_features)

    def forward(self, bboxes: torch.Tensor):
        """
        bboxes: [batch_size, 4] (x, y, w, h)
        """
        raise NotImplementedError
        return torch.cat(
            (
                self.x_emb(bboxes[:, 0] / 640),
                self.y_emb(bboxes[:, 1] / 480),
                self.w_emb(bboxes[:, 2] / 640),
                self.h_emb(bboxes[:, 3] / 480),
            ),
            dim=-1,
        )


class BoundingBoxEmbeddingCombined(nn.Module):
    def __init__(self, img_size: int = 250, out_features: int = 512):
        super().__init__()
        self.img_size = img_size
        if out_features % 8 != 0:
            raise ValueError("out_features must be divisible by 8")
        # self.bbox_emb_learned = BoundingBoxEmbeddingLearned(img_size, out_features)
        self.bbox_emb_sine = BoundingBoxEmbeddingSine(out_features // 8)

    # self._init_weights(std=0.02)

    def _init_weights(self, std):
        init.trunc_normal_(self.bbox_emb_learned.x_emb.weight, std=std)
        init.trunc_normal_(self.bbox_emb_learned.y_emb.weight, std=std)
        init.trunc_normal_(self.bbox_emb_learned.w_emb.weight, std=std)
        init.trunc_normal_(self.bbox_emb_learned.h_emb.weight, std=std)

    def forward(self, bboxes: torch.Tensor):
        """
        bboxes: [batch_size, 4, 1] (cx, cy, w, h)
        """
        bbox_emb = self.bbox_emb_sine(
            bboxes  # / self.img_size
        )  # + self.bbox_emb_learned(bboxes)
        return bbox_emb


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes: int, out_features: int = 512):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, out_features)

        init.trunc_normal_(self.class_emb.weight, std=0.02)

    def forward(self, class_ids: torch.Tensor):
        """
        class_ids: [batch_size, 1]
        """
        return self.class_emb(class_ids)


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]
