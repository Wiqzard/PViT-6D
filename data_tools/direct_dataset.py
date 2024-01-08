from typing import Optional, Tuple
import random

import cv2
from PIL import Image
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils.bbox import Bbox
from utils.flags import Mode, DistType
from data_tools.bop_dataset import BOPDataset
from data_tools.augmentations.augmentator import Augmentator
from data_tools.utils.data_utils import (
    crop_square_resize,
    crop_pad_resize,
    get_2d_coord_np,
)


class DirectDataset(Dataset):
    def __init__(
        self,
        bop_dataset: BOPDataset,
        cfg=None,
        transforms=None,
        reduce=1,
        img_only=False,
        ensemble=False,
        obj_id: Optional[int] = None,
        pad_to_square=False,
    ) -> None:
        super().__init__()
        # self.reduce = reduce
        self.reduce = reduce
        self.img_only = img_only
        self.dataset = bop_dataset
        self.mode = bop_dataset.mode
        self.transforms = transforms
        self.pad_to_square = pad_to_square
        self.args = cfg
        im_H, im_W = self.dataset.metadata.height, self.dataset.metadata.width
        self.coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)
        self.num_classes = self.dataset.metadata.num_classes
        self.class_names = self.dataset.metadata.class_names
        self.binary_sym_infos = self.dataset.metadata.binary_symmetries
        self.diameters = self.dataset.metadata.diameters
        self.mapping = {v: k for k, v in self.dataset.metadata.mapping.items()}
        self.sym_infos = {
            int(k) - 1: torch.from_numpy(v)
            for k, v in self.dataset.model_symmetries.items()
        }
        self.model_points = {
            int(k) - 1: v / 1000 for k, v in self.dataset.model_points.items()
        }
        self.model_info = {int(k) - 1: v for k, v in self.dataset.models_info.items()}
        self.model_bbox3d = {
            int(k) - 1: torch.tensor(v)[:-1, :] / 1000
            for k, v in self.dataset.model_bbox3d.items()
        }

        self.ensemble = ensemble
        self.obj_id = obj_id
        if ensemble and obj_id:
            self.ensemble_indices = self.dataset.get_ensemble_indices(obj_id)

        self.augmentator = None
        if self.args.aug and self.mode == Mode.TRAIN:
            self._set_augmentator()

    def _set_augmentator(self) -> Augmentator:
        augmentator = Augmentator(imgsz=self.args.imgsz)
        augmentator.add_box_augementator(
            p=self.args.box_aug_prob,
            dist_type=DistType.UNIFORM,
            sigma=self.args.box_aug_sigma,
        )
        augmentator.add_color_augmentator(
            p=self.args.color_aug_prob, aug_code=self.args.color_aug_code
        )
        augmentator.build()
        self.augmentator = augmentator
        return augmentator

    def __len__(self) -> int:
        if self.ensemble:
            return len(self.ensemble_indices) // self.reduce
        return self.dataset.length() // self.reduce

    def dn_group(self, cls, bbox):
        # take cls and bbox, with probability of 0.1 return random cls between 0 and 20 except cls
        dn_bbox = bbox.deepcopy()
        dn_cls = cls
        dn = False
        if random.random() <  0.05:
            classes = [i for i in range(len(self.mapping)) if i != cls]
            dn_cls = random.choice(classes)
            dn = True
        if random.random() < 0.05:
            rand_sign = np.random.choice([-1, 1], size=2)
            rand_part = np.random.rand(2) * 0.1 + 1
            rand_part *= rand_sign
            dn_bbox.shift_center(rand_part[0] * bbox.w, rand_part[1] * bbox.h)
            if (
                dn_bbox.x1 == dn_bbox.x2
                or dn_bbox.y1 == dn_bbox.y2
                or dn_bbox.x2 < 0
                or dn_bbox.y2 < 0
                or dn_bbox.x1 > 1
                or dn_bbox.y1 > 1
            ):
                dn_bbox = bbox.deepcopy()
                dn = False
            dn = True
        return dn, dn_cls, dn_bbox

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        idx *= self.reduce
        idx = self.ensemble_indices[idx] if self.ensemble else idx

        imgsz = self.args.imgsz
        input_res = self.args.inp_size

        img_path = self.dataset.get_img_path(idx)
        img = cv2.imread(str(img_path))
        scene_id = self.dataset.get_scene_id(idx)
        img_id = self.dataset.get_img_id(idx)
        # check if img is loaded

        roi_cls = self.dataset.get_obj_ids(idx)[0]
        if self.ensemble and roi_cls != self.obj_id:
            raise ValueError("Ensemble indices do not match object id")
        roi_cls = roi_cls - 1
        roi_cls_mapped = self.mapping[int(roi_cls)]

        if self.args.use_obj_bbox:
            bboxs = self.dataset.get_bbox_objs(idx)
        else:
            bboxs = self.dataset.get_bbox_visibs(idx)

        # scale bbox to make it bigger or smaller
        bbox = Bbox.from_xywh(bboxs[0])
        bbox.scale_height(self.args.scale_bbox)
        bbox.scale_width(self.args.scale_bbox)

        if self.augmentator:
            _, _, bboxs = self.augmentator(bbox=bbox)
            bbox = bboxs[0]

        dn_roi_cls_mapped = roi_cls_mapped
        dn = False
        if self.args.dn:
            dn, dn_roi_cls_mapped, bbox = self.dn_group(roi_cls, bbox)

        if self.pad_to_square:
            roi_img = crop_pad_resize(
                img, bbox, input_res, interpolation=cv2.INTER_CUBIC
            )
        else:
            roi_img, square_bbox = crop_square_resize(
                img, bbox, input_res, interpolation=cv2.INTER_CUBIC
            )

        roi_img = roi_img.astype(np.uint8)

        if self.augmentator:
            roi_img, _, _ = self.augmentator(img=roi_img)
        if self.transforms is not None:
            roi_img = Image.fromarray(roi_img)
            roi_img = self.transforms(roi_img)
        else:
            raise NotImplementedError
        if self.img_only:
            return {"roi_img": roi_img}

        # roi_coord_2d = self.get_roi_coord_2d(im_res=imgsz, out_res=input_res, bbox=bbox)
        cam = self.dataset.get_cam(idx).squeeze()

        cxcywh = bbox.cxcywh
        cxcywh = [
            cxcywh[0] / imgsz[1],
            cxcywh[1] / imgsz[0],
            cxcywh[2] / imgsz[1],
            cxcywh[3] / imgsz[0],
        ]
        bboxs_tensor = torch.tensor([cxcywh], dtype=torch.float32)
        # bboxs_tensor = torch.clamp(bboxs_tensor, min=0, max=1)

        cam = torch.from_numpy(cam)
        # roi_coord_2d = torch.from_numpy(roi_coord_2d).float()

        # pose representations
        pose = self.dataset.get_poses(idx)[0]
        pose[:, 3] = pose[:, 3] / 1000
        roi_points = self.model_points[roi_cls]
        roi_points = torch.tensor(roi_points, dtype=torch.float32)
        obj_center = self.get_2d_centroid(idx)  # absolute pixels in orig imgsz
        # center objs
        #        if self.args.center_obj:
        #            center = roi_points.mean(dim=0)
        #            roi_points = roi_points - center
        #            pose[:3, 3] = pose[:3, :3] @ center + (pose[:3, 3] - center)
        #            proj = (cam @ pose[:3, 3]).T
        #            obj_center = proj[:2] / proj[2:]

        diameter = torch.tensor(self.diameters[roi_cls], dtype=torch.float32)
        bbox_center = bbox.center  # absolute pixels in orig imgsz
        s_box = max(bbox.w, bbox.h)
        rel_center = obj_center - bbox_center  # absolute pixels in orig imgsz
        norm_rel_center = torch.tensor(
            [rel_center[0] / s_box, rel_center[1] / s_box], dtype=torch.float32
        )  # relative pixels of obj in bbox to bbox
        resize_ratio = self.args.inp_size / s_box  # max(self.args.imgsz) / s_box
        if self.args.z_test:
            focal_length = cam[0, 0] if bbox.w > bbox.h else cam[1, 1]
            resize_ratio = (focal_length * diameter) / s_box
            # z_ratio = pose[-1][-1] * s_box / (focal_length * diameter)
        z_ratio = pose[-1][-1] / resize_ratio
        trans_ratio = torch.tensor(
            [norm_rel_center[0], norm_rel_center[1], z_ratio], dtype=torch.float32
        )
        roi_center = torch.tensor(bbox_center, dtype=torch.float32)
        # roi_wh = torch.tensor([bbox.w, bbox.h], dtype=torch.float32)
        roi_wh = torch.tensor([s_box, s_box], dtype=torch.float32)

        sample = {}
        sample["scene_id"] = scene_id
        sample["img_id"] = img_id
        sample["roi_cls"] = roi_cls
        sample["roi_cls_mapped"] = roi_cls_mapped
        sample["dn_roi_cls_mapped"] = dn_roi_cls_mapped
        sample["dn"] = dn
        sample["roi_img"] = roi_img
        sample["bbox"] = bboxs_tensor
        # sample["roi_coord_2d"] = roi_coord_2d
        sample["cams"] = cam
        sample["roi_centers"] = roi_center
        sample["roi_wh"] = roi_wh
        sample["resize_ratios"] = torch.tensor(resize_ratio, dtype=torch.float32)
        sample["gt_pose"] = torch.tensor(pose, dtype=torch.float32)
        sample["gt_points"] = roi_points
        sample["trans_ratio"] = trans_ratio
        sample["diameter"] = diameter
        sample["symmetric"] = torch.tensor(
            self.binary_sym_infos[roi_cls], dtype=torch.float32
        )
        return sample

    def get_2d_centroid(self, idx: int) -> np.ndarray:
        """
        Get the 2D centroid of an object in an image. Project translation onto the image plane
        and normalize.

        Args:
            idx: The index of the image.

        Returns:
            An array containing the x and y coordinates of the 2D centroid.
        """
        poses = self.dataset.get_poses(idx)[0]
        cam = self.dataset.get_cam(idx)
        trans = poses[:3, 3] / 1000  # convert to meters
        proj = (cam @ trans.T).T
        proj_2d = proj / proj[2:]
        return proj_2d[:-1]

    def get_roi_coord_2d(
        self, im_res: Tuple[int, int], out_res: int, bbox: Bbox
    ) -> np.ndarray:
        """Get 2D coordinates of region-of-interest (ROI) based on the input bounding box.

        Args:
            im_res (Tuple[int, int]): Resolution(Height, Width) of the input image.
            out_res (Tuple[int, int]): Resolution(Width) of the output image.
            bbox (Bbox): Bounding box object corresponding to the ROI.

        Returns:
            np.ndarray: 2D coordinates of the ROI as a numpy array.

        """
        im_H, im_W = im_res
        coord_2d = self.coord_2d
        if coord_2d is None:
            coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)
        self.roi_coord_2d = crop_square_resize(
            coord_2d,
            bbox,
            crop_size=out_res,
        ).transpose(2, 0, 1)
        return self.roi_coord_2d
