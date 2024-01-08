from typing import Any, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
import timm


from utils import LOGGER, RANK, colorstr
from utils.torch_utils import (
    torch_distributed_zero_first,
    build_dataloader,
    intersect_dicts,
)

from utils.flags import Mode
from utils.pose_ops import pose_from_pred, pose_from_pred_centroid_z, get_rot_mat


from .direct_validator import DirectValidator
from engine.trainer import BaseTrainer

from engine.losses.pm_loss import PyPMLoss
from engine.losses.losses import (
    get_losses_names,
)
from engine.losses.iou_3d import iou3d_loss, get_loss_class

from data_tools.bop_dataset import BOPDataset, DatasetType, ImageType
from data_tools.direct_dataset import DirectDataset
from models.heads.direct import (
    PoseRegressionCat,
)
from models.wrappers import BackboneWrapper


class DirectTrainer(BaseTrainer):
    def __init__(self, args, overrides: Optional[dict] = None):
        super().__init__(args, overrides)
        self.loss_names = get_losses_names(args)
        if RANK in (-1, 0):
            LOGGER.info(f"Selected Losses: {self.loss_names}")
        self.train_dataset, self.eval_dataset, self.test_dataset = None, None, None
        self.transforms = None

    def preprocess_batch(self, batch: Union[tuple[dict], list[dict], dict]) -> Any:
        """Preprocess batch for training/evaluation. Move to device."""
        # bbox, roi_cls_mapped
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        elif isinstance(batch, dict):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        else:
            raise ValueError("Batch type not supported")
        return batch

    def postprocess_batch(
        self,
        batch: Union[tuple[dict], list[dict], dict],
        input_data: dict[torch.Tensor],
        train: bool = False,
    ) -> Any:
        cams = input_data["cams"]
        roi_centers = input_data["roi_centers"]
        resize_ratios = input_data["resize_ratios"]
        roi_whs = input_data["roi_wh"]
        pred_rot_ = batch[0]
        pred_t_ = batch[1]
        rot_type = self.args.rot_type
        if rot_type == "allo_axis_angle":
            pred_rot_[:, -1] = pred_rot_[:, -1].sigmoid() * 2 * torch.pi
        pred_rot_m = get_rot_mat(pred_rot_, rot_type)

        if self.args.trans_type == "centroid_z":
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2].detach()
                if self.args.detach_trans
                else pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3].detach()
                if self.args.detach_trans
                else pred_t_[:, 2:3],
                cams=cams,
                roi_centers=roi_centers,
                resize_ratios=resize_ratios,
                roi_whs=roi_whs,
                eps=1e-4,
                is_allo="allo" in rot_type,
                z_type=self.args.z_type,
            )
        elif self.args.trans_type == "trans":
            pred_ego_rot, pred_trans = pose_from_pred(
                pred_rot_m,
                pred_t_,
                eps=1e-4,
                is_allo="allo" in rot_type,
                is_train=True,
            )

        preds = {"rot": pred_ego_rot, "trans": pred_trans, "pred_t_": pred_t_}
        if self.args.cls_lw > 0:
            preds.update({"scores": batch[2]})
        batch = {k: v for k, v in input_data.items() if k != "roi_img"}
        batch["sym_infos"] = [
            self.trainset.sym_infos[obj_id_.item()].to(self.device)
            for obj_id_ in batch["roi_cls"]
        ]
        return preds, batch

    def get_dataset(
        self, img_path, mode=Mode.TRAIN, use_cache=True, single_object=True
    ):
        if self.args.dataset == "ycbv":
            dataset_type = DatasetType.YCBV
        elif self.args.dataset == "lmo":
            dataset_type = DatasetType.LMO
        elif self.args.dataset == "custom":
            dataset_type = DatasetType.CUSTOM
        set_types = (
            self.args.train_set_types
            if mode in (Mode.TRAIN, Mode.DEBUG)
            else self.args.test_set_types
        )

        dataset = BOPDataset(
            img_path,  #
            mode,  # Mode.DEBUG if self.args.debug else mode,
            dataset_type=dataset_type,
            set_types=set_types,
            use_cache=True,
            single_object=True,
            det=self.args.det_dataset if mode == Mode.TEST else None,
            num_points=self.args.num_points,
            debug=self.args.debug,
            ablation=self.args.ablation,
        )
        self.metadata = dataset.metadata
        return dataset

    def build_dataset(self, dataset_path: Union[str, Path], mode: Mode):
        """Builds the dataset from the dataset path."""
        LOGGER.info(colorstr("bold", "red", f"Setting up {mode.name} dataset..."))

        if mode == Mode.TRAIN:
            if not self.trainset:
                dataset = self.get_dataset(
                    dataset_path, mode=Mode.TRAIN, use_cache=True, single_object=True
                )
                dataset = DirectDataset(
                    bop_dataset=dataset,
                    cfg=self.args,
                    transforms=self.transforms,
                    reduce=self.args.reduce,
                    ensemble=self.args.ensemble,
                    obj_id=self.args.obj_id,
                    pad_to_square=self.args.pad_to_square,
                )
                self.trainset = dataset
            else:
                dataset = self.trainset
        elif mode == Mode.TEST:
            if not self.testset:
                dataset = self.get_dataset(
                    dataset_path, mode=Mode.TEST, use_cache=True, single_object=True
                )
                dataset = DirectDataset(
                    bop_dataset=dataset,
                    cfg=self.args,
                    transforms=self.transforms,
                    reduce=self.args.reduce,
                    ensemble=self.args.ensemble,
                    obj_id=self.args.obj_id,
                    pad_to_square=self.args.pad_to_square,
                )
                self.testset = dataset
            else:
                dataset = self.testset

        self.mapping = {v: k for k, v in self.metadata.mapping.items()}
        return dataset

    def get_dataloader(self, dataset, batch_size=16, rank=0, mode=Mode.TRAIN):
        assert mode in [Mode.TRAIN, Mode.TEST]
        if dataset is None:
            with torch_distributed_zero_first(
                rank
            ):  # init dataset *.cache only once if DDP
                dataset = self.build_dataset(self.dataset_path, mode)
        shuffle = mode == Mode.TRAIN

        workers = self.args.workers if mode == Mode.TRAIN else self.args.workers * 2
        workers = 1 if self.args.debug else workers
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def get_validator(self) -> Any:
        return DirectValidator(
            dataloader=self.test_loader, save_dir=self.save_dir, args=self.args
        )

    def _init_criterion(self) -> Any:
        if self.args.pm_lw > 0 and not hasattr(self, "pm_loss"):
            self.pm_loss = PyPMLoss(
                loss_type="l1",
                beta=1.0,
                reduction="mean",
                loss_weight=self.args.pm_lw,
                norm_by_diameter=self.args.pm_norm_by_diameter,
                symmetric=self.args.pm_loss_sym,
                disentangle_t=self.args.pm_disentangle_t,
                disentangle_z=self.args.pm_disentangle_z,
                t_loss_use_points=self.args.pm_t_use_points,
                r_only=self.args.pm_r_only,
            )
        if self.args.centroid_lw > 0 and not hasattr(self, "centroid_loss"):
            if self.args.centroid_loss_type == "l1":
                self.centroid_loss = nn.L1Loss(reduction="mean")
            elif self.args.centroid_loss_type == "mse":
                self.centroid_loss = nn.MSELoss(reduction="mean")

        if self.args.z_lw > 0 and not hasattr(self, "z_loss"):
            z_loss_type = self.args.z_loss_type
            if z_loss_type == "l1":
                self.z_loss = nn.L1Loss(reduction="mean")
            elif z_loss_type == "mse":
                self.z_loss = nn.MSELoss(reduction="mean")

    def criterion(self, pred: dict, gt: dict) -> dict:
        self._init_criterion()
        out_rot = pred["rot"]
        out_trans = pred["trans"]
        out_centroid = pred["pred_t_"][:, :2]
        out_trans_z = pred["pred_t_"][:, 2]

        gt_trans = gt["gt_pose"][:, :3, 3]
        gt_rot = gt["gt_pose"][:, :3, :3]
        gt_trans_ratio = gt["trans_ratio"]
        gt_points = gt["gt_points"]
        diameter = gt["diameter"]
        sym_infos = gt["sym_infos"]
        mask = ~gt["dn"]
        loss_dict = {}
        if self.args.pm_lw > 0:
            loss_pm = self.pm_loss(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                points=gt_points,
                pred_transes=out_trans,
                gt_transes=gt_trans,
                diameter=diameter,
                sym_infos=sym_infos,
            )
            loss_dict.update({**loss_pm})
        if self.args.centroid_lw > 0 and self.args.trans_type == "centroid_z":
            loss_centroid = (
                self.centroid_loss(out_centroid[mask], gt_trans_ratio[:, :2][mask])
                * self.args.centroid_lw
            )
            loss_dict.update({"loss_centroid": loss_centroid})

        if self.args.z_lw > 0:
            z_type = self.args.z_type
            assert z_type in ["rel", "abs"]
            gt_z = gt_trans_ratio[:, 2] if z_type == "rel" else gt_trans[:, 2]
            loss_z = self.z_loss(out_trans_z[mask], gt_z[mask]) * self.args.z_lw
            loss_dict.update({"loss_z": loss_z})

        if self.args.cls_lw > 0:
            # use predicted cls for bbox
            # if gt["dn"]: ignore pose loss, gt["roi_cls_mapped_dn"]

            gt_cls = gt["dn_roi_cls_mapped"] if self.args.dn else gt["roi_cls_mapped"]

            bbox3d = torch.stack(
                [self.trainset.model_bbox3d[int(cls)] for cls in gt["roi_cls"]]
            ).to(out_trans.device)
            gt_scores = iou3d_loss(
                out_trans.detach(), out_rot.detach(), gt_trans, gt_rot, bbox3d
            )
            loss_cls = (
                get_loss_class(pred["scores"], gt_cls, gt_scores) * self.args.cls_lw
            )
            loss_dict.update({"loss_cls": loss_cls})

        return sum(loss_dict.values()), torch.tensor(list(loss_dict.values()))

    def progress_string(self) -> str:
        return ("\n" + "%11s" * (3 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            "lr",
            *self.loss_names,
        )

    def write_csv(self) -> None:
        pass

    def get_model(self, weights, verbose: bool = False):
        """Build model and load pretrained weights if specified."""
        if isinstance(
            weights, torch.nn.Module
        ):  # if model is loaded beforehand. No setup needed
            model = weights.cpu().float() 
        else: 
            ckpt = None
            from models.mvit.pose_vit import PoseViT

            model = PoseViT(
                "mvitv2_base_cls",  # "mvitv2_large_cls",  # "mvitv2_base_cls",
                # "mvitv2_small_cls",
                bbox_embed_dim=96,
                cls_embed_dim=96,
                num_classes=21, 
                bbox_token=True,
                # late_fusion=True,
                late_fusion=False,
                pred_cls=False,  # True
                akimbo=False,
            ).cpu()

        self.model = model
        for param in model.parameters():
            param.requires_grad = True
        # print number of parameters
        LOGGER.info(
            f"{colorstr('total_parameters:')} {sum(p.numel() for p in model.parameters())}"
        )
        self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ) 
        return model  # ckpt
