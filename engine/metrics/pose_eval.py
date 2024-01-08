import os
import shutil
import copy
import json

import torch
import numpy as np
from pathlib import Path

from utils import LOGGER, colorstr
from utils.metrics import calc_add, calc_adi


class PoseEvaluator(object):
    def __init__(self, models, classes, model_info, model_symmetry, depth_scale=0.1):
        """
        Initialization of the Pose Evaluator for LM-O dataset.

        It can calculate the average rotation and translation error, as well as the ADD, ADD-S and ADD-(S) metric.
        Note: The definition of these metrics is slightly different for the LM-O dataset in comparison to the
        YCB-V dataset (see http://www.stefan-hinterstoisser.com/papers/hinterstoisser2012accv.pdf)

        Parameters
            - models: Array containing the points of each object 3D model (Contains the 3D points for each class)
            - classes: Array containing the information about the object classes (mapping between class ids and class names)
            - model_info: Information about the models (diameter and extension)
            - model_symmetry: Indication whether the 3D model of a certain class is symmetric (axis, plane) or not.
        """
        self.models = models
        self.classes = classes
        self.models_info = model_info
        self.model_symmetry = model_symmetry
        self.model_symmetry = {
            k: False if v.shape[0] == 1 else True for k, v in model_symmetry.items()
        }

        self.poses_pred = {}
        self.poses_gt = {}
        self.poses_img = {}
        self.camera_intrinsics = {}
        self.num = {}
        self.depth_scale = depth_scale

        self.reset()  # Initialize

    def reset(self):
        """
        Reset the PoseEvaluator stored poses. Necessary when the same evaluator is used during training
        """
        self.poses_pred = {}
        self.poses_gt = {}
        self.poses_img = {}
        self.camera_intrinsics = {}
        self.num = {}

        for cls in self.classes:
            self.num[cls] = 0
            self.poses_pred[cls] = []
            self.poses_gt[cls] = []
            self.poses_img[cls] = []
            self.camera_intrinsics[cls] = []

    def update(
        self,
        preds: dict[str, torch.Tensor],
        input_data: dict[str, torch.Tensor],
    ) -> None:
        obj_id = input_data["roi_cls"]
        pred_rot = preds["rot"].cpu().numpy()
        pred_trans = preds["trans"].cpu().numpy()

        pred_pose = np.concatenate([pred_rot, pred_trans[..., None]], axis=-1)
        gt_pose = input_data["gt_pose"].cpu().numpy()
        cams = input_data["cams"].cpu().numpy()

        for i, obj in enumerate(obj_id):
            obj = obj.item()
            self.num[obj] += 1
            self.poses_pred[obj].append(pred_pose[i])
            self.poses_gt[obj].append(gt_pose[i])
            self.camera_intrinsics[obj].append(cams[i])

    def write_all(self, output_path):
        """
        Write all the poses to a file
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            colorstr(
                "bold",
                "green",
                "Evaluate pose add(-s) to {}".format(str(output_path) + "/adds"),
            )
        )
        self.evaluate_pose_adds(self.output_path)
        LOGGER.info(
            colorstr(
                "bold",
                "green",
                "Evaluate pose add to {}".format(str(output_path) + "/add"),
            )
        )
        self.evaluate_pose_add(self.output_path)
        LOGGER.info(
            colorstr(
                "bold",
                "green",
                "Evaluate pose adi to {}".format(str(output_path) + "/adi"),
            )
        )
        self.evaluate_pose_adi(self.output_path)
        LOGGER.info(
            colorstr(
                "bold",
                "green",
                "Evaluate translation error to {}".format(
                    str(output_path) + "/avg_t_error"
                ),
            )
        )
        self.calculate_class_avg_translation_error(self.output_path)
        LOGGER.info(
            colorstr(
                "bold",
                "green",
                "Evaluate rotation error to {}".format(
                    str(output_path) + "/avg_rot_error"
                ),
            )
        )
        self.calculate_class_avg_rotation_error(self.output_path)

    def evaluate_pose_adds(self, output_path):
        """
        Evaluate 6D pose by ADD(-S) metric
        Symmetric Object --> ADD-S
        NonSymmetric Objects --> ADD

        For metric definition we refer to PoseCNN: https://arxiv.org/pdf/1711.00199.pdf
        """
        output_path = output_path / "adds/"
        output_path.mkdir(parents=True, exist_ok=True)

        log_file = open(str(output_path / "adds.log"), "w")
        json_file = open(str(output_path / "adds.json"), "w")

        poses_pred = self.poses_pred
        poses_gt = self.poses_gt
        models = self.models
        model_symmetry = self.model_symmetry

        log_file.write(
            "\n* {} *\n {:^}\n* {} *".format("-" * 100, "Metric ADD(-S)", "-" * 100)
        )
        log_file.write("\n")

        n_classes = len(self.classes)
        count_all = np.zeros((n_classes), dtype=np.float32)
        count_correct = {
            k: np.zeros((n_classes), dtype=np.float32) for k in ["0.02", "0.05", "0.10", "0.10d", "mean"]
        }

        threshold_002 = np.zeros((n_classes), dtype=np.float32)
        threshold_005 = np.zeros((n_classes), dtype=np.float32)
        threshold_010 = np.zeros((n_classes), dtype=np.float32)
        threshold_010d = np.zeros((n_classes), dtype=np.float32)
        dx = 0.0001
        threshold_mean = np.tile(
            np.arange(0, 0.1, dx).astype(np.float32), (n_classes, 1)
        )  # (num_class, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct["mean"] = np.zeros((n_classes, num_thresh), dtype=np.float32)

        adds_results = {}
        adds_results["thresholds"] = [0.02, 0.05, 0.10, 0.10]
        sum_error = np.zeros((n_classes), dtype=np.float32)
        self.classes = sorted(self.classes)
        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            threshold_002[i] = 0.02
            threshold_005[i] = 0.05
            threshold_010[i] = 0.10
            threshold_010d[i] = 0.10 * self.models_info[str(cls_name)]["diameter"]/1000 

            symmetry_flag = model_symmetry[cls_name]
            cls_poses_pred = poses_pred[cls_name]
            cls_poses_gt = poses_gt[cls_name]
            model_pts = models[cls_name]

            n_poses = len(cls_poses_gt)
            count_all[i] = n_poses
            for j in range(n_poses):
                pose_pred = cls_poses_pred[j]  # est pose
                pose_gt = cls_poses_gt[j]  # gt pose
                if symmetry_flag:
                    error = calc_adi(model_pts, pose_pred, pose_gt)
                else:
                    error = calc_add(model_pts, pose_pred, pose_gt)

                sum_error[i] += error
                if error < threshold_002[i]:
                    count_correct["0.02"][i] += 1
                if error < threshold_005[i]:
                    count_correct["0.05"][i] += 1
                if error < threshold_010[i]:
                    count_correct["0.10"][i] += 1
                if error < threshold_010d[i]:
                    count_correct["0.10d"][i] += 1
                for thresh_i in range(num_thresh):
                    if error < threshold_mean[i, thresh_i]:
                        count_correct["mean"][i, thresh_i] += 1
            adds_results[cls_name] = {}
            adds_results[cls_name]["threshold"] = {
                "0.02": count_correct["0.02"][i].tolist(),
                "0.05": count_correct["0.05"][i].tolist(),
                "0.10": count_correct["0.10"][i].tolist(),
                "0.10d": count_correct["0.10d"][i].tolist(),
                "mean": count_correct["mean"][i].tolist(),
            }
            avg_error = sum_error[i] / count_all[i] if count_all[i] != 0 else np.array(0)
            adds_results[cls_name]["avg_error"] = avg_error.item() #.tolist()

        plot_data = {}
        sum_acc_mean = np.zeros(1)
        sum_acc_002 = np.zeros(1)
        sum_acc_005 = np.zeros(1)
        sum_acc_010 = np.zeros(1)
        sum_acc_010d = np.zeros(1)
        for i, cls_name in enumerate(self.classes):
            if count_all[i] == 0:
                continue
            plot_data[cls_name] = []
            log_file.write("** {} **".format(cls_name))
            from scipy.integrate import simps

            area = simps(count_correct["mean"][i] / float(count_all[i]), dx=dx) / 0.1
            acc_mean = area * 100
            sum_acc_mean[0] += acc_mean
            acc_002 = 100 * float(count_correct["0.02"][i]) / float(count_all[i])
            sum_acc_002[0] += acc_002
            acc_005 = 100 * float(count_correct["0.05"][i]) / float(count_all[i])
            sum_acc_005[0] += acc_005
            acc_010 = 100 * float(count_correct["0.10"][i]) / float(count_all[i])
            sum_acc_010[0] += acc_010
            acc_010d = 100 * float(count_correct["0.10d"][i]) / float(count_all[i])
            sum_acc_010d[0] += acc_010d

            log_file.write("threshold=[0.0, 0.10], area: {:.2f}".format(acc_mean))
            log_file.write("\n")
            log_file.write(
                "threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}".format(
                    count_correct["0.02"][i], count_all[i], acc_002
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}".format(
                    count_correct["0.05"][i], count_all[i], acc_005
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}".format(
                    count_correct["0.10"][i], count_all[i], acc_010
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.10d, correct poses: {}, all poses: {}, accuracy: {:.2f}".format(
                    count_correct["0.10d"][i], count_all[i], acc_010d
                )
            )
            
            log_file.write("\n")
            log_file.write(
                "average error: {:.4f}".format(
                    float(sum_error[i]) / float(count_all[i])
                )
            )
            log_file.write("\n")
            log_file.write("\n")
            adds_results[cls_name]["accuracy"] = {
                "n_poses": count_all[i].tolist(),
                "0.02": acc_002,
                "0.05": acc_005,
                "0.10": acc_010,
                "0.10d": acc_010d,
                "auc": acc_mean,
            }

        log_file.write("=" * 30)
        log_file.write("\n")

        for iter_i in range(1):
            log_file.write(
                "---------- ADD(-S) performance over {} classes -----------".format(
                    num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write("** iter {} **".format(iter_i + 1))
            log_file.write("\n")
            log_file.write(
                "threshold=[0.0, 0.10], area: {:.2f}".format(
                    sum_acc_mean[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.02, mean accuracy: {:.2f}".format(
                    sum_acc_002[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.05, mean accuracy: {:.2f}".format(
                    sum_acc_005[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.10, mean accuracy: {:.2f}".format(
                    sum_acc_010[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.10d, mean accuracy: {:.2f}".format(
                    sum_acc_010d[iter_i] / num_valid_class
                )
            )

            log_file.write("\n")
            log_file.write(
                "average error: {:.4f}".format(
                    float(sum(sum_error)) / float(sum(count_all))
                )
            )
            log_file.write("\n")
        log_file.write("=" * 30)
        adds_results["accuracy"] = {
            "0.02": sum_acc_002[0].tolist() / num_valid_class,
            "0.05": sum_acc_005[0].tolist() / num_valid_class,
            "0.10": sum_acc_010[0].tolist() / num_valid_class,
            "0.10d": sum_acc_010d[0].tolist() / num_valid_class,
            "auc": sum_acc_mean[0].tolist() / num_valid_class,
        }

        log_file.write("\n")
        log_file.close()
        json.dump(adds_results, json_file)
        json_file.close()
        return

    def evaluate_pose_adi(self, output_path):
        """
        Evaluate 6D pose by ADD-S metric

        For metric definition we refer to PoseCNN: https://arxiv.org/pdf/1711.00199.pdf
        """
        output_path = output_path / "adi"
        output_path.mkdir(parents=True, exist_ok=True)

        log_file = open(str(output_path / "adds.log"), "w")
        json_file = open(str(output_path / "adds.json"), "w")

        poses_pred = copy.deepcopy(self.poses_pred)
        poses_gt = copy.deepcopy(self.poses_gt)
        models = self.models

        log_file.write(
            "\n* {} *\n {:^}\n* {} *".format("-" * 100, "Metric ADD-S", "-" * 100)
        )
        log_file.write("\n")

        eval_method = "adi"
        n_classes = len(self.classes)
        count_all = np.zeros((n_classes), dtype=np.float32)
        count_correct = {
            k: np.zeros((n_classes), dtype=np.float32) for k in ["0.02", "0.05", "0.10", "0.10d", "mean"]
        }

        threshold_002 = np.zeros((n_classes), dtype=np.float32)
        threshold_005 = np.zeros((n_classes), dtype=np.float32)
        threshold_010 = np.zeros((n_classes), dtype=np.float32)
        threshold_010d = np.zeros((n_classes), dtype=np.float32)

        dx = 0.0001
        threshold_mean = np.tile(
            np.arange(0, 0.1, dx).astype(np.float32), (n_classes, 1)
        )  # (num_class, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct["mean"] = np.zeros((n_classes, num_thresh), dtype=np.float32)

        adi_results = {}
        adi_results["thresholds"] = [0.02, 0.05, 0.10]
        sum_error = np.zeros((n_classes), dtype=np.float32)

        self.classes = sorted(self.classes)
        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            threshold_002[i] = 0.02
            threshold_005[i] = 0.05
            threshold_010[i] = 0.10
            threshold_010d[i] = 0.10 * self.models_info[str(cls_name)]["diameter"]/1000 

            cls_poses_pred = poses_pred[cls_name]
            cls_poses_gt = poses_gt[cls_name]
            model_pts = models[cls_name]
            n_poses = len(cls_poses_gt)
            count_all[i] = n_poses
            for j in range(n_poses):
                pose_pred = cls_poses_pred[j]  # est pose
                pose_gt = cls_poses_gt[j]  # gt pose
                error = calc_adi(model_pts, pose_pred, pose_gt)
                sum_error[i] += error
                if error < threshold_002[i]:
                    count_correct["0.02"][i] += 1
                if error < threshold_005[i]:
                    count_correct["0.05"][i] += 1
                if error < threshold_010[i]:
                    count_correct["0.10"][i] += 1
                if error < threshold_010d[i]:
                    count_correct["0.10d"][i] += 1
                for thresh_i in range(num_thresh):
                    if error < threshold_mean[i, thresh_i]:
                        count_correct["mean"][i, thresh_i] += 1
            adi_results[cls_name] = {}
            adi_results[cls_name]["threshold"] = {
                "0.02": count_correct["0.02"][i].tolist(),
                "0.05": count_correct["0.05"][i].tolist(),
                "0.10": count_correct["0.10"][i].tolist(),
                "0.10d": count_correct["0.10d"][i].tolist(),
                "mean": count_correct["mean"][i].tolist(),
            }

            avg_error = sum_error[i] / count_all[i] if count_all[i] != 0 else np.array(0)
            adi_results[cls_name]["avg_error"] =  avg_error.item()

        plot_data = {}
        sum_acc_mean = np.zeros(1)
        sum_acc_002 = np.zeros(1)
        sum_acc_005 = np.zeros(1)
        sum_acc_010 = np.zeros(1)
        sum_acc_010d = np.zeros(1)
        for i, cls_name in enumerate(self.classes):
            if count_all[i] == 0:
                continue
            plot_data[cls_name] = []
            log_file.write("** {} **".format(cls_name))
            from scipy.integrate import simps

            area = simps(count_correct["mean"][i] / float(count_all[i]), dx=dx) / 0.1
            acc_mean = area * 100
            sum_acc_mean[0] += acc_mean
            acc_002 = 100 * float(count_correct["0.02"][i]) / float(count_all[i])
            sum_acc_002[0] += acc_002
            acc_005 = 100 * float(count_correct["0.05"][i]) / float(count_all[i])
            sum_acc_005[0] += acc_005
            acc_010 = 100 * float(count_correct["0.10"][i]) / float(count_all[i])
            sum_acc_010[0] += acc_010
            acc_010d = 100 * float(count_correct["0.10d"][i]) / float(count_all[i])
            sum_acc_010d[0] += acc_010d

            log_file.write("threshold=[0.0, 0.10], area: {:.2f}".format(acc_mean))
            log_file.write("\n")
            log_file.write(
                "threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}".format(
                    count_correct["0.02"][i], count_all[i], acc_002
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}".format(
                    count_correct["0.05"][i], count_all[i], acc_005
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}".format(
                    count_correct["0.10"][i], count_all[i], acc_010
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.10d, correct poses: {}, all poses: {}, accuracy: {:.2f}".format(
                    count_correct["0.10d"][i], count_all[i], acc_010d
                )
            )
            log_file.write("\n")
            log_file.write(
                "average error: {:.4f}".format(
                    float(sum_error[i]) / float(count_all[i])
                )
            )
            log_file.write("\n")
            log_file.write("\n")

            adi_results[cls_name]["accuracy"] = {
                "n_poses": count_all[i].tolist(),
                "0.02": acc_002,
                "0.05": acc_005,
                "0.10": acc_010,
                "0.10d": acc_010d,
                "auc": acc_mean,
            }

        log_file.write("=" * 30)
        log_file.write("\n")

        for iter_i in range(1):
            log_file.write(
                "---------- ADD-S performance over {} classes -----------".format(
                    num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write("** iter {} **".format(iter_i + 1))
            log_file.write("\n")
            log_file.write(
                "average error: {:.4f}".format(
                    float(sum(sum_error)) / float(sum(count_all))
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=[0.0, 0.10], area: {:.2f}".format(
                    sum_acc_mean[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.02, mean accuracy: {:.2f}".format(
                    sum_acc_002[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.05, mean accuracy: {:.2f}".format(
                    sum_acc_005[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.10, mean accuracy: {:.2f}".format(
                    sum_acc_010[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.10d, mean accuracy: {:.2f}".format(
                    sum_acc_010d[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
        log_file.write("=" * 30)
        adi_results["accuracy"] = {
            "0.02": sum_acc_002[0].tolist() / num_valid_class,
            "0.05": sum_acc_005[0].tolist() / num_valid_class,
            "0.10": sum_acc_010[0].tolist() / num_valid_class,
            "0.10d": sum_acc_010d[0].tolist() / num_valid_class,
            "auc": sum_acc_mean[0].tolist() / num_valid_class,
        }

        log_file.write("\n")
        log_file.close()
        json.dump(adi_results, json_file)
        json_file.close()
        return

    def evaluate_pose_add(self, output_path):
        """
        Evaluate 6D pose by ADD Metric

        For metric definition we refer to PoseCNN: https://arxiv.org/pdf/1711.00199.pdf
        """

        output_path = output_path / "add"
        output_path.mkdir(parents=True, exist_ok=True)
        log_file = open(str(output_path / "add.log"), "w")
        json_file = open(str(output_path / "add.json"), "w")

        poses_pred = copy.deepcopy(self.poses_pred)
        poses_gt = copy.deepcopy(self.poses_gt)
        models_info = self.models_info
        models = self.models

        log_file.write(
            "\n* {} *\n {:^}\n* {} *".format("-" * 100, "Metric ADD", "-" * 100)
        )
        log_file.write("\n")

        n_classes = len(self.classes)
        count_all = np.zeros((n_classes), dtype=np.float32)
        count_correct = {
            k: np.zeros((n_classes), dtype=np.float32) for k in ["0.02", "0.05", "0.10", "0.10d"]
        }

        threshold_002 = np.zeros((n_classes), dtype=np.float32)
        threshold_005 = np.zeros((n_classes), dtype=np.float32)
        threshold_010 = np.zeros((n_classes), dtype=np.float32)
        threshold_010d = np.zeros((n_classes), dtype=np.float32)
        dx = 0.0001
        threshold_mean = np.tile(
            np.arange(0, 0.1, dx).astype(np.float32), (n_classes, 1)
        )  # (num_class, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct["mean"] = np.zeros((n_classes, num_thresh), dtype=np.float32)
        add_results = {}
        add_results["thresholds"] = [0.02, 0.05, 0.10]
        sum_error = np.zeros((n_classes), dtype=np.float32)

        self.classes = sorted(self.classes)
        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            threshold_002[i] = 0.02
            threshold_005[i] = 0.05
            threshold_010[i] = 0.10
            threshold_010d[i] = 0.10 * self.models_info[str(cls_name)]["diameter"]/1000 
            # threshold_mean[i, :] *= models_info[cls_name]['diameter']
            cls_poses_pred = poses_pred[cls_name]
            cls_poses_gt = poses_gt[cls_name]
            model_pts = models[cls_name]
            n_poses = len(cls_poses_gt)
            count_all[i] = n_poses
            for j in range(n_poses):
                pose_pred = cls_poses_pred[j]  # est pose
                pose_gt = cls_poses_gt[j]  # gt pose
                error = calc_add(model_pts, pose_pred, pose_gt)
                sum_error[i] += error
                if error < threshold_002[i]:
                    count_correct["0.02"][i] += 1
                if error < threshold_005[i]:
                    count_correct["0.05"][i] += 1
                if error < threshold_010[i]:
                    count_correct["0.10"][i] += 1
                if error < threshold_010d[i]:
                    count_correct["0.10d"][i] += 1
                for thresh_i in range(num_thresh):
                    if error < threshold_mean[i, thresh_i]:
                        count_correct["mean"][i, thresh_i] += 1
            add_results[cls_name] = {}
            add_results[cls_name]["threshold"] = {
                "0.02": count_correct["0.02"][i].tolist(),
                "0.05": count_correct["0.05"][i].tolist(),
                "0.10": count_correct["0.10"][i].tolist(),
                "0.10d": count_correct["0.10d"][i].tolist(),
                "mean": count_correct["mean"][i].tolist(),
            }
            avg_error = sum_error[i] / count_all[i] if count_all[i] != 0 else np.array(0)
            add_results[cls_name]["avg_error"] = avg_error.item()

        plot_data = {}
        sum_acc_mean = np.zeros(1)
        sum_acc_002 = np.zeros(1)
        sum_acc_005 = np.zeros(1)
        sum_acc_010 = np.zeros(1)
        sum_acc_010d = np.zeros(1)
        for i, cls_name in enumerate(self.classes):
            if count_all[i] == 0:
                continue
            plot_data[cls_name] = []
            log_file.write("** {} **".format(cls_name))
            from scipy.integrate import simps

            area = simps(count_correct["mean"][i] / float(count_all[i]), dx=dx) / 0.1
            acc_mean = area * 100
            sum_acc_mean[0] += acc_mean
            acc_002 = 100 * float(count_correct["0.02"][i]) / float(count_all[i])
            sum_acc_002[0] += acc_002
            acc_005 = 100 * float(count_correct["0.05"][i]) / float(count_all[i])
            sum_acc_005[0] += acc_005
            acc_010 = 100 * float(count_correct["0.10"][i]) / float(count_all[i])
            sum_acc_010[0] += acc_010
            acc_010d = 100 * float(count_correct["0.10d"][i]) / float(count_all[i])
            sum_acc_010d[0] += acc_010d

            log_file.write("threshold=[0.0, 0.10], area: {:.2f}".format(acc_mean))
            log_file.write("\n")
            log_file.write(
                "threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}".format(
                    count_correct["0.02"][i], count_all[i], acc_002
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}".format(
                    count_correct["0.05"][i], count_all[i], acc_005
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}".format(
                    count_correct["0.10"][i], count_all[i], acc_010
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.10d, correct poses: {}, all poses: {}, accuracy: {:.2f}".format(
                    count_correct["0.10d"][i], count_all[i], acc_010d
                )
            )
            log_file.write("\n")
            log_file.write(
                "average error: {:.4f}".format(
                    float(sum_error[i]) / float(count_all[i])
                )
            )
            log_file.write("\n")
            log_file.write("\n")
            add_results[cls_name]["accuracy"] = {
                "n_poses": count_all[i].tolist(),
                "0.02": acc_002,
                "0.05": acc_005,
                "0.10": acc_010,
                "auc": acc_mean,
            }

        log_file.write("=" * 30)
        log_file.write("\n")

        for iter_i in range(1):
            log_file.write(
                "---------- ADD performance over {} classes -----------".format(
                    num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write("** iter {} **".format(iter_i + 1))
            log_file.write("\n")
            log_file.write(
                "average error: {:.4f}".format(
                    float(sum(sum_error)) / float(sum(count_all))
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=[0.0, 0.10], area: {:.2f}".format(
                    sum_acc_mean[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.02, mean accuracy: {:.2f}".format(
                    sum_acc_002[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.05, mean accuracy: {:.2f}".format(
                    sum_acc_005[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.10, mean accuracy: {:.2f}".format(
                    sum_acc_010[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
            log_file.write(
                "threshold=0.10d, mean accuracy: {:.2f}".format(
                    sum_acc_010d[iter_i] / num_valid_class
                )
            )
            log_file.write("\n")
        log_file.write("=" * 30)

        add_results["accuracy"] = {
            "0.02": sum_acc_002[0].tolist() / num_valid_class,
            "0.05": sum_acc_005[0].tolist() / num_valid_class,
            "0.10": sum_acc_010[0].tolist() / num_valid_class,
            "0.10d": sum_acc_010d[0].tolist() / num_valid_class,
            "auc": sum_acc_mean[0].tolist() / num_valid_class,
        }

        log_file.write("\n")
        log_file.close()
        json.dump(add_results, json_file)
        json_file.close()
        return

    def calculate_class_avg_translation_error(self, output_path):
        """
        Calculate the average translation error for each class and then the average error across all classes in meters
        """
        output_path = output_path / "avg_t_error"
        output_path.mkdir(parents=True, exist_ok=True)

        log_file = open(str(output_path / "avg_t_error.log"), "w")
        json_file = open(str(output_path / "avg_t_error.json"), "w")

        log_file.write(
            "\n* {} *\n {:^}\n* {} *".format(
                "-" * 100, "Metric Average Translation Error in Meters", "-" * 100
            )
        )
        log_file.write("\n")

        poses_pred = self.poses_pred
        poses_gt = self.poses_gt
        translation_errors = []
        cls_translation_errors = {}
        avg_translation_errors = {}
        accuracy_2cm = {}
        for cls in self.classes:
            cls_translation_errors[cls] = []
            cls_poses_pred = poses_pred[cls]
            cls_poses_gt = poses_gt[cls]
            count_within_threshold = 0
            total_poses = len(cls_poses_gt)
            for pose_est, pose_gt in zip(cls_poses_pred, cls_poses_gt):
                t_est = pose_est[:, 3]
                t_gt = pose_gt[:, 3]
                error = np.sqrt(np.sum(np.square((t_est - t_gt))))
                cls_translation_errors[cls].append(error)
                translation_errors.append(error)
                if error <= 0.02:
                    count_within_threshold += 1
            if len(cls_translation_errors[cls]) != 0:
                avg_error = np.sum(cls_translation_errors[cls]) / len(
                    cls_translation_errors[cls]
                )
                avg_translation_errors[cls] = avg_error
            else:
                avg_translation_errors[cls] = np.nan
            if total_poses != 0:
                accuracy_2cm[cls] = count_within_threshold / total_poses
            else:
                accuracy_2cm[cls] = np.nan
            log_file.write("Class: {} \t\t {}".format(cls, avg_translation_errors[cls]))
            log_file.write("\n")
        total_avg_error = np.sum(translation_errors) / len(translation_errors)
        log_file.write(
            "Class: {} \t\t Avg Error: {} \t\t Accuracy(2cm): {}".format(
                cls, avg_translation_errors[cls], accuracy_2cm[cls]
            )
        )
        log_file.write("\n")
        avg_translation_errors["mean"] = [total_avg_error]
        avg_translation_errors["accuracy_2cm"] = accuracy_2cm

        log_file.write("\n")
        log_file.close()
        json.dump(avg_translation_errors, json_file)
        json_file.close()
        return

    def calculate_class_avg_rotation_error(self, output_path):
        """
        Calculate the average rotation error given by the Geodesic distance for each class and then the average error
        across all classes in degree
        """
        output_path = output_path / "avg_rot_error"
        output_path.mkdir(parents=True, exist_ok=True)

        log_file = open(str(output_path / "avg_rot_error.log"), "w")
        json_file = open(str(output_path / "avg_rot_error.json"), "w")

        log_file.write(
            "\n* {} *\n {:^}\n* {} *".format(
                "-" * 100, "Metric Average Rotation Error in Degrees", "-" * 100
            )
        )
        log_file.write("\n")

        poses_pred = copy.deepcopy(self.poses_pred)
        poses_gt = copy.deepcopy(self.poses_gt)
        rotation_errors = []
        cls_rotation_errors = {}
        avg_rotation_errors = {}
        accuracy_2deg = {}

        for cls in self.classes:
            cls_rotation_errors[cls] = []
            cls_pose_pred = poses_pred[cls]
            cls_pose_gt = poses_gt[cls]
            count_within_threshold = (
                0  # Initialize the count of errors within 2 degrees threshold
            )
            total_poses = len(cls_pose_gt)
            for debug, (pose_est, pose_gt) in enumerate(
                zip(cls_pose_pred, cls_pose_gt)
            ):
                rot_est = pose_est[:3, :3]
                rot_gt = pose_gt[:3, :3]
                rot = np.matmul(rot_est, rot_gt.T)
                trace = np.trace(rot)
                if trace < -1.0:
                    trace = -1
                elif trace > 3.0:
                    trace = 3.0
                angle_diff = np.degrees(np.arccos(0.5 * (trace - 1)))
                cls_rotation_errors[cls].append(angle_diff)
                rotation_errors.append(angle_diff)
                if angle_diff <= 2.0:  # Check if error is within 2 degrees threshold
                    count_within_threshold += 1
            if len(cls_rotation_errors[cls]) != 0:
                avg_error = np.sum(cls_rotation_errors[cls]) / len(
                    cls_rotation_errors[cls]
                )
                avg_rotation_errors[cls] = avg_error
            else:
                avg_rotation_errors[cls] = np.nan
            if total_poses != 0:
                accuracy_2deg[cls] = count_within_threshold / total_poses
            else:
                accuracy_2deg[cls] = np.nan

            log_file.write(
                "Class: {} \t\t Avg Error: {} \t\t Accuracy (2deg): {}".format(
                    cls, avg_rotation_errors[cls], accuracy_2deg[cls]
                )
            )
            log_file.write("\n")
        total_avg_error = np.sum(rotation_errors) / len(rotation_errors)
        avg_rotation_errors["accuracy_2deg"] = accuracy_2deg

        log_file.write("All:\t\t\t\t\t {}".format(total_avg_error))
        avg_rotation_errors["mean"] = [total_avg_error]

        log_file.write("\n")
        log_file.close()
        json.dump(avg_rotation_errors, json_file)
        json_file.close()
        return

