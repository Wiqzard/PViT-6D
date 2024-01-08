from typing import Union, List, Tuple, Optional

import torch
import torch.nn as nn
from .mvit import _create_mvitv2
#from .mvit_pose import _create_mvitv2
#from .mvit_pose_old import _create_mvitv2
from models.layers.transformer import MLP
from timm.layers import trunc_normal_tf_


class BackboneWrapper(nn.Module):
    pass

class PoseViT(nn.Module):
    def __init__(
        self,
        backbone_code: str,
        bbox_embed_dim: int,
        cls_embed_dim: int,
        num_classes: int,
        bbox_token: bool = False,
        late_fusion: bool = False,
        pred_cls: bool = False,
        akimbo: bool = False,
    ):
        super().__init__()
        self.akimbo = akimbo
        self.bbox_token = bbox_token
        if akimbo:
            self.rot_backbone = _create_mvitv2(
                backbone_code, pretrained=False, num_token=2
            )
            self.trans_backbone = _create_mvitv2(
                backbone_code, pretrained=False, num_token=3
            )
            embed_dim = self.trans_backbone.embed_dim
            head_dim = self.trans_backbone.num_features
        else:
            num_token = 2 + pred_cls - late_fusion + bbox_token
            self.backbone = _create_mvitv2(
                backbone_code, pretrained=True, num_token=num_token
            )
            embed_dim = self.backbone.embed_dim
            head_dim = self.backbone.num_features
        self._setup_backbone()
        self.late_fusion = late_fusion
        self.pred_cls = pred_cls

        # embeddings
        bbox_embed_dim = embed_dim if bbox_token else bbox_embed_dim
        self.bbox_embed = nn.Linear(4, bbox_embed_dim)

        if pred_cls:
            self.cls_token = nn.Embedding(num_classes, embed_dim)
            self.cls_head = MLP(head_dim, 256, num_classes, num_layers=3)

        in_dim = embed_dim
        if late_fusion:
            head_dim += bbox_embed_dim
            self.rot_token = nn.Embedding(num_classes, cls_embed_dim)
            self.trans_token = nn.Embedding(num_classes, cls_embed_dim)
        else:
            in_dim += cls_embed_dim + bbox_embed_dim
            if bbox_token:
                in_dim = embed_dim
            self.rot_token = nn.Embedding(num_classes, embed_dim)
            self.trans_token = nn.Embedding(num_classes, in_dim)

        # heads
        self.rot_head = MLP(head_dim, 256, 6, num_layers=3)
        self.centroid_head = MLP(head_dim, 256, 2, num_layers=3)
        self.z_head = MLP(head_dim, 256, 1, num_layers=3)

        self._init_weights()

    def _init_weights(self):
        if hasattr(self, "cls_token"):
            trunc_normal_tf_(self.cls_token.weight, std=0.02)
        #trunc_normal_tf_(self.bbox_embed.weight, std=0.02)

        for m in self.rot_token.modules():
            trunc_normal_tf_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        for m in self.trans_token.modules():
            trunc_normal_tf_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _setup_backbone(self):
        if self.akimbo:
            del self.rot_backbone.head
            del self.trans_backbone.head
            del self.rot_backbone.cls_token
            del self.trans_backbone.cls_token
            self.rot_backbone.head = None
            self.trans_backbone.head = None
            self.rot_backbone.float().train()
            self.trans_backbone.float().train()
        else:
            del self.backbone.head
            del self.backbone.cls_token
            self.backbone.head = None
            self.backbone.float().train()

    def forward_tokens(self, x):
        if self.late_fusion:
            rot_token = self.rot_token(x["roi_cls_mapped"]).unsqueeze(1)
            trans_token = self.trans_token(x["roi_cls_mapped"]).unsqueeze(1)
        else:
            #

            bbox_token = self.bbox_embed(x["bbox"]) if self.bbox_token else None
            rot_token = self.rot_token(x["roi_cls_mapped"]).unsqueeze(1)
            #rot_token = torch.cat([rot_token, bbox_token], dim=1).unsqueeze(1)
            trans_token = self.trans_token(x["roi_cls_mapped"]).unsqueeze(1)
            dim = 1 if self.akimbo or self.bbox_token else 2
            #
            trans_token = torch.cat([trans_token, bbox_token], dim=dim) if self.bbox_token else trans_token

        tokens = torch.cat([rot_token, trans_token], dim=1)
        if self.pred_cls:
            cls_token = self.cls_token(x["roi_cls_mapped"]).unsqueeze(1)
            tokens = torch.cat([tokens, cls_token], dim=1)
        return tokens #if not self.late_fusion else None

    def forward_head(self, feats, bbox=None):
        rot_token = feats[:, 0]
        trans_token = feats[:, 1]
        if self.late_fusion:
            bbox_embed = self.bbox_embed(bbox).squeeze(1)
            rot_token = torch.cat([feats[:, 0], bbox_embed], dim=1)
            trans_token = torch.cat([feats[:, 1], bbox_embed], dim=1)

        cls_score = None
        if self.pred_cls:
            cls_token = feats[:, 2]
            cls_score = self.cls_head(cls_token)

        rot = self.rot_head(rot_token)
        centroid = self.centroid_head(trans_token)
        z = self.z_head(trans_token)
        trans = torch.cat([centroid, z], dim=1)
        return rot, trans, cls_score

    def forward_akimbo_head(self, rot_feats, trans_feats, bbox=None):
        rot = self.rot_head(rot_feats[:, 0])
        centroid = self.centroid_head(trans_feats[:, 1])
        z = self.z_head(trans_feats[:, 1])
        trans = torch.cat([centroid, z], dim=1)
        return rot, trans, None

    def forward(self, x):
        tokens = self.forward_tokens(x)
        if self.akimbo:
            rot_feats = self.rot_backbone.forward_features(x["roi_img"], tokens[:, 0:1])
            trans_feats = self.trans_backbone.forward_features(
                x["roi_img"], tokens[:, 1:3]
            )
            rot, trans, score = self.forward_akimbo_head(
                rot_feats, trans_feats, x["bbox"]
            )

        else:
            feats = self.backbone.forward_features(x["roi_img"], tokens)
            rot, trans, score = self.forward_head(feats, x["bbox"])
        return rot, trans, score
