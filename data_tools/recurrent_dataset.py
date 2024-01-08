from typing import Optional, Tuple

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
    get_2d_coord_np,
)


class RecurrentDataset(Dataset):
    def __init__(
        self,
        bop_dataset: BOPDataset,
        cfg=None,
        transforms=None,
        reduce=1,
        img_only=False,
    ) -> None:
        super().__init__()
        # self.reduce = reduce
        self.reduce = reduce
        self.img_only = img_only
        self.dataset = bop_dataset
        self.mode = bop_dataset.mode
        self.transforms = transforms
        self.args = cfg
        self.im_h, self.im_w = self.dataset.metadata.height, self.dataset.metadata.width
        self.coord_2d = get_2d_coord_np(self.im_w, self.im_h, low=0, high=1).transpose(
            1, 2, 0
        )
        self.num_classes = self.dataset.metadata.num_classes
        self.class_names = self.dataset.metadata.class_names
        self.binary_sym_infos = self.dataset.metadata.binary_symmetries
        self.diameters = self.dataset.metadata.diameters
        self.sym_infos = {
            int(k) - 1: torch.from_numpy(v)
            for k, v in self.dataset.model_symmetries.items()
        }
        self.model_points = {
            int(k) - 1: v for k, v in self.dataset.model_points.items()
        }
        self.model_info = {int(k) - 1: v for k, v in self.dataset.models_info.items()}

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
        return self.dataset.length() // self.reduce

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        idx *= self.reduce
        imgsz = self.args.imgsz
        input_res = self.args.inp_size

        # get bboxs
        if self.args.use_obj_bbox:
            bboxs = np.asarray(self.dataset.get_bbox_objs(idx), dtype=np.float32)
        else:
            bboxs = np.asarray(self.dataset.get_bbox_visibs(idx), dtype=np.float32)
        xywh = bboxs[:, :4]
        cxywh = xywh.copy()  # convert to center xywh
        cxywh[:, :2] += cxywh[:, 2:] / 2
        cxywh[:, 2:] *= self.args.scale_bbox  # scale bbox to make it bigger or smaller
        if self.augmentator:
            bboxs = [self.augmentator(bbox=bbox)[2] for bbox in bboxs]

        # get roi imgs
        img_paths = self.dataset.get_img_path(idx)
        imgs = [cv2.imread(str(img_path)) for img_path in img_paths]
        roi_imgs = [
            crop_square_resize(
                img, Bbox.from_cxcywh(bbox), input_res, interpolation=cv2.INTER_CUBIC
            )[0].astype(np.uint8)
            for img, bbox in zip(imgs, cxywh)
        ]
        if self.augmentator:
            roi_imgs = [self.augmentator(img=roi_img)[0] for roi_img in roi_imgs]
        if self.transforms is not None:
            roi_imgs = [
                self.transforms(Image.fromarray(roi_img)) for roi_img in roi_imgs
            ]
        else:
            raise NotImplementedError
        roi_imgs = torch.stack(roi_imgs, dim=0)

        # get meta infos
        roi_cls = torch.tensor(self.dataset.get_obj_ids(idx)) - 1
        cam = torch.from_numpy(self.dataset.get_cam(idx).squeeze()).float()

        # pose representations
        cwh_tensor = torch.tensor([cxywh], dtype=torch.float32).squeeze()
        rcwh = torch.cat(
            [
                cwh_tensor[:, 0] / self.im_w,
                cwh_tensor[:, 1] / self.im_h,
                cwh_tensor[:, 2] / self.im_w,
                cwh_tensor[:, 3] / self.im_h,
            ],
            dim=-1,
        )

        poses = torch.tensor(self.dataset.get_poses(idx), dtype=torch.float32)
        poses[:, :, 3] = poses[:, :, 3] / 1000

        trans = poses[:, :3, 3]
        proj = torch.matmul(cam, trans.T).T
        proj_2d = proj / proj[:, 2].unsqueeze(-1)
        centroid_z = torch.cat(
            [
                proj_2d[:, 0, None],
                proj_2d[:, 1, None],
                poses[:, -1, -1, None],
            ],
            dim=1,
        )
        bbox_center = cwh_tensor[:, :2]  # absolute pixels in orig imgsz
        s_box = cwh_tensor[:, 2:].max(dim=-1)[0]
        rel_center = centroid_z[:, :2] - bbox_center  # absolute pixels in orig imgsz
        rel_center = rel_center / s_box.unsqueeze(1).repeat(
            1, 2
        )  # relative pixels of obj in bbox to bbox

        resize_ratio = (
            torch.tensor(self.args.inp_size) / s_box
        )  # max(self.args.imgsz) / s_box
        z_ratio = centroid_z[:, -1] / resize_ratio
        trans_ratio = torch.cat([rel_center[:, 0], rel_center[:, 1], z_ratio], dim=-1)
        roi_center = torch.tensor(bbox_center, dtype=torch.float32)
        # roi_wh = torch.tensor([bbox.w, bbox.h], dtype=torch.float32)
        roi_wh = s_box.unsqueeze(1).repeat(1, 2)
        # roi_wh = torch.tensor([s_box, s_box], dtype=torch.float32)

        # roi_points = [self.model_points[roi_cls_.item()] for roi_cls_ in roi_cls]
        # roi_points = torch.tensor(roi_points, dtype=torch.float32)

        sample = {}
        sample["roi_cls"] = roi_cls
        sample["roi_img"] = roi_imgs
        sample["bbox"] = rcwh
        sample["cams"] = cam
        sample["roi_centers"] = roi_center
        sample["roi_wh"] = roi_wh
        sample["resize_ratios"] = torch.tensor(resize_ratio, dtype=torch.float32)
        sample["gt_pose"] = torch.tensor(poses, dtype=torch.float32)
        # sample["gt_points"] = roi_points
        sample["trans_ratio"] = trans_ratio
        #sample["diameter"] = torch.tensor(self.diameters[roi_cls], dtype=torch.float32)
        #sample["symmetric"] = torch.tensor(
        #    self.binary_sym_infos[roi_cls], dtype=torch.float32
        #)
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
