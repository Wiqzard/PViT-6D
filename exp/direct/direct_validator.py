from typing import Any, Union

import torch
from torch.utils.data import DataLoader

from utils import LOGGER
from utils.flags import Mode
from engine.metrics import PoseMetrics
from engine.validator import BaseValidator
from utils.pose_ops import pose_from_pred, pose_from_pred_centroid_z, get_rot_mat
from data_tools.direct_dataset import DirectDataset

from data_tools.bop_dataset import BOPDataset, DatasetType, ImageType
from engine.losses.iou_3d import iou3d_loss

# from utils.annotator.annotator import Annotator


class DirectValidator(BaseValidator):
    def __init__(
        self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None
    ):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        if self.testset:
            self.metadata = self.testset.dataset.metadata
            self.metrics = PoseMetrics(
                save_dir=self.save_dir,
                model_pts=self.testset.model_points,
                matcher=None,
                class_names=self.metadata.class_names,
                diameters=self.metadata.diameters,
                binary_symmetries=self.metadata.binary_symmetries,
                pred_cls=self.args.cls_lw > 0,
            )

    def preprocess(self, batch: Union[tuple[dict], list[dict], dict]) -> Any:
        """Preprocess batch for training/evaluation. Move to device."""
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch_in, batch_gt = batch
            batch_in = {
                k: v.to(self.device, non_blocking=True) for k, v in batch_in.items()
            }
            batch_gt = {
                k: v.to(self.device, non_blocking=True) for k, v in batch_gt.items()
            }
            batch = (batch_in, batch_gt)
        elif isinstance(batch, dict):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        else:
            raise ValueError("Batch type not supported")

        return batch

    def postprocess(
        self,
        preds: Union[tuple[dict], list[dict], dict],
        batch: dict[torch.Tensor],
        train: bool = False,
    ) -> Any:
        cams = batch["cams"]
        roi_centers = batch["roi_centers"]
        resize_ratios = batch["resize_ratios"]
        # resize_ratios /= batch["diameter"] *
        roi_whs = batch["roi_wh"]
        pred_rot_ = preds[0]
        pred_t_ = preds[1]
        rot_type = self.args.rot_type
        pred_rot_m = get_rot_mat(pred_rot_, rot_type)
        if self.args.trans_type == "centroid_z":
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],
                cams=cams,  # roi_cams,
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
            )

        preds_dict = {"rot": pred_ego_rot, "trans": pred_trans, "pred_t_": pred_t_}
        if preds[2] is not None:
            preds_dict["pred_scores"] = preds[2].sigmoid()
        # add .cpu() to avoid memory leak
        preds_dict = {k: v.detach() for k, v in preds_dict.items()}
        batch = {k: v.detach() for k, v in batch.items() if k != "roi_img"}
        batch["sym_infos"] = [
            self.testset.sym_infos[obj_id_.item()].to(self.device)
            for obj_id_ in batch["roi_cls"]
        ]
        if self.args.cls_lw > 0:
            gt_trans = batch["gt_pose"][:, :3, 3]
            gt_rot = batch["gt_pose"][:, :3, :3]
            bbox3d = torch.stack(
                [self.testset.model_bbox3d[int(cls)] for cls in batch["roi_cls"]]
            ).to(gt_trans.device)
            gt_scores = iou3d_loss(pred_trans, pred_ego_rot, gt_trans, gt_rot, bbox3d)
            bs, nc = preds[2].shape[:2]
            one_hot = torch.zeros(
                (bs, nc + 1), dtype=torch.int64, device=preds[2].device
            )
            one_hot.scatter_(1, batch["roi_cls_mapped"].unsqueeze(-1), 1)
            one_hot = one_hot[..., :-1]
            gt_scores = gt_scores.view(bs, 1) * one_hot
            batch["gt_scores"] = gt_scores

        else:
            batch["roi_cls_mapped"] = batch["roi_cls"]
        return preds_dict, batch

    def init_metrics(self):
        self.seen = 0
        self.jdict = []
        self.stats = []
        if not hasattr(self, "metrics"):
            self.metrics = PoseMetrics(
                save_dir=self.save_dir,
                model_pts=self.testset.model_points,
                matcher=None,
                class_names=self.metadata.class_names,
                diameters=self.metadata.diameters,
                binary_symmetries=self.metadata.binary_symmetries,
                pred_cls=self.args.cls_lw > 0,
            )
        self.metrics.reset()

    def get_desc(self):
        return ("%22s" + "%11s" * 9) % (
            "Class",
            "Images",
            "Targets",
            "Pose(ADD(-S)",
            "ADD(-S)[m]",
            "Trans",
            "z",
            "Ang",
            "Score-l1",
            "Fitness)",
        )

    def update_metrics(
        self,
        preds,
        input_data,
    ):
        self.metrics.update(
            pred_trans=preds["trans"],
            pred_rot=preds["rot"],
            gt_trans=input_data["gt_pose"][:, :3, 3],
            gt_rot=input_data["gt_pose"][:, :3, :3],
            gt_cls=input_data["roi_cls"],
            pred_scores=preds["pred_scores"] if "pred_scores" in preds else None,
            gt_scores=input_data["gt_scores"] if "gt_scores" in input_data else None,
        )

        bs = input_data["roi_cls"].shape[0]
        self.seen += bs

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed

    def print_results(self):
        pf = "%22s" + "%11i" * 2 + "%11.3g" * (len(self.metrics.keys) + 1)
        LOGGER.info(
            pf
            % (
                "all",
                self.metrics.num_targets,
                self.seen,
                *self.metrics.mean_results,
                self.metrics.fitness,
            )
        )
        # Print results per class
        if self.args.verbose and not self.training:
            for obj_id, metric_list in self.metrics.avg_metrics_cls.items():
                LOGGER.info(
                    pf
                    % (
                        obj_id,
                        self.seen,
                        self.metrics.num[obj_id],
                        *metric_list,
                        self.metrics.fitness,
                    )
                )
            self.metrics.save()
            self.metrics.evaluate_all()

    def get_dataset(self, img_path, mode=Mode.TEST, use_cache=True, single_object=True):
        if self.args.dataset == "ycbv":
            dataset_type = DatasetType.YCBV
        elif self.args.dataset == "lmo":
            dataset_type = DatasetType.LMO
        elif self.args.dataset == "custom":
            dataset_type = DatasetType.CUSTOM
        dataset = BOPDataset(
            img_path,  #
            mode,  # Mode.DEBUG if self.args.debug else mode,
            dataset_type=dataset_type,
            set_types=self.args.test_set_types,
            use_cache=True,
            single_object=True,
            det=self.args.det_dataset if mode == Mode.TEST else None,
            num_points=self.args.num_points,
            debug=self.args.debug,
            ablation=self.args.ablation,
        )
        self.metadata = dataset.metadata
        return dataset

    def get_dataloader(self, mode):
        dataset = self.get_dataset(
            img_path=self.args.dataset_path, use_cache=True, mode=mode
        )
        dataset = DirectDataset(
            bop_dataset=dataset,
            cfg=self.args,
            transforms=self.transforms,
            pad_to_square=self.args.pad_to_square,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch,
            shuffle=False,
            num_workers=8,
        )
        return dataloader

    def save_csv(self, preds, batch, dt):
        with open(self.csv_path, "a") as f:
            scene_ids, img_ids, obj_ids = (
                batch["scene_id"],
                batch["img_id"],
                batch["roi_cls"],
            )
            rotations, translations = preds["rot"], preds["trans"] * 1000
            for scene_id, img_id, obj_id, r, t in zip(
                scene_ids, img_ids, obj_ids, rotations, translations
            ):

                t_values = [f"{t[i]}" for i in range(3)]
                r_values = [f"{r[i][j]}" for i in range(3) for j in range(3)]
                f.write(
                    f"{scene_id},{img_id},{obj_id+1},{1},{' '.join(r_values)},{' '.join(t_values)},0.25\n"
                )
