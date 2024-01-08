# from __future__ import print_function, division


from utils import SimpleClass


from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import torch
from torch import nn
import numpy as np

from utils.flags import MetricType

from engine.losses.rot_loss import angular_distance
from engine.losses.add import (
    calc_translation_error_batch,
    calc_rotation_error_batch,
    add_batch,
    adi_batch,
    add,
    adi,
)


@dataclass
class Metrics:
    """Pose metrics class to store metrics for each eval."""

    args: None
    # poses_pred: dict[int, np.ndarray] = field(default_factory=dict)
    # poses_gt: dict[int, np.ndarray] = field(default_factory=dict)

    losses: list[float] = field(default_factory=list)
    ang_distances_cls: dict[int, list[float]] = field(default_factory=dict)
    trans_distances_cls: dict[int, list[float]] = field(default_factory=dict)
    adds_cls: dict[int, list[float]] = field(default_factory=dict)

    # add_cls: list[float] = field(default_factory=list)
    # adi_cls: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset()

    def update(
        self,
        loss: torch.Tensor,
        preds: dict[str, torch.Tensor],
        input_data: dict[str, torch.Tensor],
    ) -> None:
        """Update the metrics with new data."""

        obj_id = input_data["roi_cls"]
        total_loss = sum(loss) if len(loss) > 1 else loss["loss"]

        pred_rot = preds["rot"].cpu().numpy()
        pred_trans = preds["trans"].cpu().numpy()  # pred_t_ (bs, 3)

        gt_rot = input_data["gt_pose"][:, :3, :3].cpu().numpy()
        gt_trans = input_data["gt_pose"][:, :3, 3].cpu().numpy()

        pts = input_data["gt_points"].cpu().numpy()
        sym_infos = input_data["symmetric"]
        ang_distance = calc_rotation_error_batch(pred_rot, gt_rot)  # mean of batch
        eucl_distance = calc_translation_error_batch(pred_trans, gt_trans)

        self.losses.append(total_loss.item())

        for i, obj in enumerate(obj_id):
            obj = obj.item()
            # if obj not in self.poses_gt:
            #    self.poses_gt[obj] = []
            # if obj not in self.poses_pred:
            #    self.poses_pred[obj] = []
            # self.poses_gt[obj].append(input_data["gt_pose"][i].cpu().numpy())
            if obj not in self.ang_distances_cls:
                self.ang_distances_cls[obj] = []
            if obj not in self.trans_distances_cls:
                self.trans_distances_cls[obj] = []
            if obj not in self.adds_cls:
                self.adds_cls[obj] = []
            self.ang_distances_cls[obj].append(ang_distance[i].item())
            self.trans_distances_cls[obj].append(eucl_distance[i].item())

            if sym_infos[i].item():
                adds = adi(pred_rot[i], pred_trans[i], gt_rot[i], gt_trans[i], pts[i])
            else:
                adds = add(pred_rot[i], pred_trans[i], gt_rot[i], gt_trans[i], pts[i])
            self.adds_cls[obj].append(adds)

    def reset(self) -> None:
        self.losses = []
        self.ang_distances_cls = {}
        self.trans_distances_cls = {}
        self.adds_cls = {}

    @property
    def num_targets(self) -> int:
        return sum(
            len(class_ang_distances)
            for class_ang_distances in self.ang_distances_cls.values()
        )

    @property
    def avg_losses(self) -> dict[str, float]:
        return {
            "avg_loss": self.avg_loss,
            "ang_distance": self.avg_ang_distance,
            "trans_distance": self.avg_trans_distance,
        }

    @property
    def avg_loss(self) -> float:
        return sum(self.losses) / len(self.losses)

    @property
    def avg_ang_distance(self) -> float:
        total_ang_distance = sum(
            sum(ang_list) for ang_list in self.ang_distances_cls.values()
        )
        return total_ang_distance / self.num_targets

    @property
    def avg_trans_distance(self) -> float:
        total_trans_distance = sum(
            sum(trans_list) for trans_list in self.trans_distances_cls.values()
        )
        return total_trans_distance / self.num_targets

    @property
    def avg_adds(self) -> float:
        total_adds = sum(sum(add_list) for add_list in self.adds_cls.values())
        return total_adds / self.num_targets

    @property
    def avg_add(self) -> float:
        total_add = sum(self.add_cls)
        return total_add / self.num_targets

    @property
    def avg_adi(self) -> float:
        total_adi = sum(self.adi_cls)
        return total_adi / self.num_targets

    @property
    def avg_trans_distance_cls(self) -> dict[int, float]:
        return {k: sum(v) / len(v) for k, v in self.trans_distances_cls.items()}

    @property
    def avg_ang_distance_cls(self) -> dict[int, float]:
        return {k: sum(v) / len(v) for k, v in self.ang_distances_cls.items()}

    @property
    def avg_adds_cls(self) -> dict[int, float]:
        return {k: sum(v) / len(v) for k, v in self.adds_cls.items()}

    @property
    def avg_add_cls(self) -> dict[int, float]:
        return {k: sum(v) / len(v) for k, v in self.add_cls.items()}

    @property
    def avg_adi_cls(self) -> dict[int, float]:
        return {k: sum(v) / len(v) for k, v in self.adi_cls.items()}

    @property
    def keys(self) -> list[str]:
        return ["total_loss", "add(-s)", "angular_distance", "translation_distance"]

    @property
    def avg_metrics(self) -> list[float]:
        """Return the metrics."""
        return [
            self.avg_loss,
            self.avg_adds,
            self.avg_ang_distance,
            self.avg_trans_distance,
        ]

    @property
    def avg_metrics_cls(self) -> dict[int, list[float]]:
        avg_metrics_cls = {}
        for k, v in self.avg_ang_distance_cls.items():
            avg_metrics_cls[k] = [
                0,  # loss
                self.avg_adds_cls[k],  # adds
                v,  # ang
                self.avg_trans_distance_cls[k],  # trans
            ]
        return avg_metrics_cls

    @property
    def empty_dict(self) -> list[float]:
        """Return empty metrics."""
        return dict(zip(self.keys, [0.0, 0.0, 0.0]))

    @property
    def results_dict(self) -> dict[str, float]:
        """Return the metrics as a dictionary."""
        results = dict(zip(self.keys, self.avg_metrics))
        results.update({"fitness": self.fitness})
        return results

    @property
    def fitness(self) -> float:
        """Return the fitness of the metrics."""
        w = np.array([0.1, 10, 1, 1])  # weights for angular, translation, loss
        return (1 / np.array(self.avg_metrics).copy() * w).sum() / 100

    @property
    def num_targets_cls(self) -> dict[int, int]:
        return {k: len(v) for k, v in self.ang_distances_cls.items()}

    def __str__(self) -> str:
        """Return the string representation of the metrics."""
        return f" Total Loss: {self.avg_loss:.4f},\n Angular Distance: {self.avg_ang_distance:.4f},\n Translation (L2): {self.avg_trans_distance:.4f}"

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}"
        )
