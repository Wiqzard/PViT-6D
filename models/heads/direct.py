from enum import Enum, auto
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.custom_layers import (
    BoundingBoxEmbeddingSine,
    BoundingBoxEmbeddingLearned,
    BoundingBoxEmbeddingCombined,
    ClassEmbedding,
)
from models.weight_init import trunc_normal_tf_
from models.layers.transformer import MLP

from utils.bbox import xyxy_to_xywh


class PoseRegressionCat(nn.Module):
    """
    Neural Network for pose regression from a single image.
    Takes as input:
           1. a single rgb image and a bounding box.
        OR 2. a single rgb with coord2d concatenated.
    Pools the features from the backbone network and concatenates them with the bounding box embedding.
    Simple MLP for regression.
    """

    def __init__(
        self,
        backbone,
        bbox_emb_dim: int = 256,
        pool_dim: int = 1,
        d_model=768,
        class_emb_dim: int = 0,
        num_classes: int = 1,
        bbox_emb_inp_size: int = 700,
        norm_features: Optional[nn.Module] = None,
        rot_dim: int = 6,
        disentangle: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.pool_dim = pool_dim
        self.d_model = d_model
        self.class_emb_dim = class_emb_dim
        self.num_classes = num_classes
        self.norm_features = norm_features
        self.disentangle = disentangle
        self.bbox_emb_dim = bbox_emb_dim
        self.rot_dim = rot_dim

        self.bbox_emb = None
        if self.pool_dim == 0:
            self.bbox_emb_dim = None
        elif self.pool_dim == 1:
            self.bbox_emb = BoundingBoxEmbeddingCombined(
                bbox_emb_inp_size, self.bbox_emb_dim
            )
            #self.bbox_emb = nn.Linear(4, self.bbox_emb_dim)
        else:
            self.bbox_emb = BoundingBoxEmbeddingCombined(
                bbox_emb_inp_size, self.bbox_emb_dim
            )

        self.cls_emb = None
        if class_emb_dim > 0 and num_classes > 1:
            self.class_emb = ClassEmbedding(num_classes, class_emb_dim)

        if norm_features:
            self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pool = nn.AdaptiveAvgPool1d(pool_dim)
        self.act = nn.LeakyReLU(0.01)
        out_dim = pool_dim * d_model + bbox_emb_dim + class_emb_dim
        self.fc1 = nn.Linear(out_dim, 1024)
        if disentangle:
            self.fc2 = nn.Linear(1024, 256)
            self.fc_r = nn.Linear(256, self.rot_dim)
            self.fc_c = nn.Linear(256, 2)
            self.fc_z = nn.Linear(256, 1)
        else:
            self.fc2 = nn.Linear(1024, 512)
            self.fc_p = nn.Linear(512, self.rot_dim + 3)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        # nn.init.trunc_normal_(m.weight, std=0.02)
        # if isinstance(m, nn.Linear) and m.bias is not None:
        # nn.init.constant_(m.bias, 0.0)
        if hasattr(self, "fc_c"):
            # nn.init.uniform_(self.fc_c.weight, -0.05, 0.05)
            nn.init.zeros_(self.fc_c.bias)
            nn.init.trunc_normal_(self.fc_c.weight, mean=0, std=0.05)
        if hasattr(self, "fc_z"):
            # nn.init.trunc_normal_(self.fc_z.weight, mean=1, std=0.1)
            nn.init.constant_(self.fc_z.bias, 1)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {k for k, _ in self.named_parameters()
    #             if any(n in k for n in ["bbox_emb", "cls_emb"])}

    def forward(self, input_data, visualize=False):
        if self.bbox_emb is not None:
            assert "bbox" in input_data
            bbox = input_data["bbox"].squeeze()# * torch.tensor([640,480,640,480])
            bbox_emb = self.bbox_emb(bbox).unsqueeze(1)

        if self.class_emb_dim > 0:
            assert "roi_cls" in input_data
            cls_emb = self.class_emb(input_data["roi_cls"]).unsqueeze(1)

        assert "roi_img" in input_data
        x = self.backbone(input_data["roi_img"])[-1]  # .permute(0,2,3,1)
        # convnext/ maxxvit
        # x = self.backbone(input_data["roi_img"])[-1].permute(0,2,3,1)
        # resnet
        # x = self.backbone(input_data["roi_img"])[-1].permute(0,2,3,1)

        if self.norm_features:
            x = self.norm(x)
        if x.ndim == 4:
            bs, h, w, feat = x.shape
            assert feat == self.d_model
            x = x.reshape(bs, feat, h * w)
        elif x.ndim == 3:
            bs, feat, seq = x.shape
            x = x.permute(0, 2, 1)

        x = self.pool(x).permute(0, 2, 1).flatten(1)
        if self.bbox_emb is not None:
            x = torch.cat((x, bbox_emb.squeeze(1)), dim=-1)
        if self.class_emb_dim > 0:
            x = torch.cat((x, cls_emb.squeeze(1)), dim=-1)

        # x = x.flatten(1)
        if visualize:
            return x
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        if self.disentangle:
            r = self.fc_r(x)
            c = self.fc_c(x)
            z = self.fc_z(x)
            t = torch.cat((c, z), dim=-1)
        else:
            x = self.fc_p(x)
            r = x[:, 3:]
            t = x[:, :3]
        return r, t


class DirectPoseNet(nn.Module):
    def __init__(
        self,
        backbone,
        head,
        bbox_emb_dim: int = 256,
        pool_dim: int = 1,
        d_model=768,
        class_emb_dim: int = 0,
        num_classes: int = 1,
        bbox_emb_inp_size: int = 700,
        norm_features: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.pool_dim = pool_dim
        self.d_model = d_model
        self.class_emb_dim = class_emb_dim
        self.num_classes = num_classes
        self.norm_features = norm_features
        self.bbox_emb_dim = bbox_emb_dim

        self.bbox_emb = None
        if self.pool_dim == 0:
            self.bbox_emb_dim = None
        elif self.pool_dim == 1:
            self.bbox_emb = BoundingBoxEmbeddingCombined(
                bbox_emb_inp_size, self.bbox_emb_dim
            )
        else:
            self.bbox_emb = BoundingBoxEmbeddingCombined(
                bbox_emb_inp_size, self.bbox_emb_dim
            )

        self.cls_emb = None
        if class_emb_dim > 0 and num_classes > 1:
            self.class_emb = ClassEmbedding(num_classes, class_emb_dim)

        if norm_features:
            self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pool = nn.AdaptiveAvgPool1d(pool_dim)
        out_dim = pool_dim * d_model + bbox_emb_dim + class_emb_dim
        self.fc_out = nn.Linear(out_dim, 1024)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {k for k, _ in self.named_parameters()
    #             if any(n in k for n in ["bbox_emb", "cls_emb"])}

    def forward(self, input_data):
        if self.bbox_emb is not None:
            assert "bbox" in input_data
            bbox = input_data["bbox"].squeeze()
            bbox_emb = self.bbox_emb(bbox).unsqueeze(1)

        if self.class_emb_dim > 0:
            assert "roi_cls" in input_data
            cls_emb = self.class_emb(input_data["roi_cls"]).unsqueeze(1)

        assert "roi_img" in input_data
        # x = self.backbone(input_data["roi_img"])[-1] #.permute(0,2,3,1)
        # convnext/ maxxvit
        # x = self.backbone(input_data["roi_img"])[-1].permute(0,2,3,1)
        # resnet
        x = self.backbone(input_data["roi_img"])[-1].permute(0, 2, 3, 1)

        if self.norm_features:
            x = self.norm(x)
        if x.ndim == 4:
            bs, h, w, feat = x.shape
            assert feat == self.d_model
            x = x.reshape(bs, feat, h * w)
        elif x.ndim == 3:
            bs, feat, seq = x.shape
            x = x.permute(0, 2, 1)

        x = self.pool(x).permute(0, 2, 1).flatten(1)
        if self.bbox_emb is not None:
            x = torch.cat((x, bbox_emb.squeeze(1)), dim=-1)
        if self.class_emb_dim > 0:
            x = torch.cat((x, cls_emb.squeeze(1)), dim=-1)

        x = F.relu(self.fc_out(x))
        r, t = self.head(x)
        return r, t


class PoseHeadDecoupled6d(nn.Module):
    def __init__(self, in_dim: int = 1024, rot_dim: int = 6) -> None:
        super().__init__()
        self.rot_dim = rot_dim
        self.fc = nn.Linear(in_dim, 256)
        self.fc_r = nn.Linear(256, rot_dim)
        self.fc_c = nn.Linear(256, 2)
        self.fc_z = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc(x))
        r = self.fc_r(x)
        if self.rot_dim == 5:
            r[:, -1] = r[:, -1].sigmoid() * 2 * torch.pi
        c = self.fc_c(x)
        z = self.fc_z(x)
        t = torch.cat((c, z), dim=-1)
        return r, t


class DoubleDirect(nn.Module):
    def __init__(
        self,
        trans_backbone,
        rot_backbone,
        bbox_emb_dim: int = 256,
        pool_dim: int = 1,
        d_model=768,
        class_emb_dim: int = 0,
        num_classes: int = 1,
        bbox_emb_inp_size: int = 700,
        norm_features: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.trans_backbone = trans_backbone
        self.rot_backbone = rot_backbone

        self.pool_dim = pool_dim
        self.d_model = d_model
        self.class_emb_dim = class_emb_dim
        self.num_classes = num_classes
        self.norm_features = norm_features
        self.bbox_emb_dim = bbox_emb_dim

        self.bbox_emb = None
        if self.pool_dim == 0:
            self.bbox_emb_dim = None
        elif self.pool_dim == 1:
            self.bbox_emb = BoundingBoxEmbeddingCombined(
                bbox_emb_inp_size, self.bbox_emb_dim
            )
        else:
            self.bbox_emb = BoundingBoxEmbeddingCombined(
                bbox_emb_inp_size, self.bbox_emb_dim
            )

        self.bbox_emb_dim = bbox_emb_dim
        if bbox_emb_dim > 0:
            self.bbox_emb = BoundingBoxEmbeddingCombined(
                bbox_emb_inp_size, self.bbox_emb_dim
            )
        self.cls_emb = None
        if class_emb_dim > 0 and num_classes > 1:
            self.class_emb = ClassEmbedding(num_classes, class_emb_dim)

        #if norm_features:
        #    self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pool = nn.AdaptiveAvgPool1d(pool_dim)
        out_dim = pool_dim * d_model +  class_emb_dim
        self.fc_trans_proj = nn.Linear(out_dim+bbox_emb_dim, 512)
        self.fc_rot_proj = nn.Linear(out_dim, 512)
        self.trans_head = MLP(512, 256, 3, num_layers=3)
        self.rot_head = MLP(512, 256, 6, num_layers=3)
        #nn.init.constant_(self.trans_head[-1].bias, 0.5)

    def forward(self, input_data):
        if self.bbox_emb is not None:
            assert "bbox" in input_data
            bbox = input_data["bbox"].squeeze()
            bbox_emb = self.bbox_emb(bbox).unsqueeze(1)

        if self.class_emb_dim > 0:
            assert "roi_cls" in input_data
            cls_emb = self.class_emb(input_data["roi_cls"]).unsqueeze(1)

        assert "roi_img" in input_data
        trans_features = self.trans_backbone.forward_features(input_data["roi_img"])
        rot_features = self.trans_backbone.forward_features(input_data["roi_img"])
        trans_features = self.pool(trans_features.permute(0, 2, 1)).squeeze(-1)
        rot_features = self.pool(rot_features.permute(0, 2, 1)).squeeze(-1)

        if self.bbox_emb is not None:
            trans_features = torch.cat((trans_features, bbox_emb.squeeze(1)), dim=-1)
        if self.class_emb_dim > 0:
            trans_features = torch.cat((trans_features, cls_emb.squeeze(1)), dim=-1)
            rot_features = torch.cat((rot_features, cls_emb.squeeze(1)), dim=-1)
        trans = self.fc_trans_proj(trans_features)
        trans = self.trans_head(trans)
        rot = self.fc_rot_proj(rot_features)
        rot = self.rot_head(rot)
        return rot, trans


class TransformerOutputMode(Enum):
    AVG = auto()
    FIRST_ONE = auto()
    LAST_ONE = auto()
    FIRST_TWO = auto()
    LAST_TWO = auto()


class TransformerInputMode(Enum):
    CAT = auto()
    CROSS_ATTN = auto()

