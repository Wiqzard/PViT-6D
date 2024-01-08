import torch
from torch import Tensor
from torch.utils.data import Dataset
import cv2
from PIL import Image
import numpy as np

from utils.bbox import Bbox, xyxy_to_xywh
from utils.flags import Mode, DistType
from data_tools.bop_dataset import BOPDataset, DatasetType
from data_tools.augmentations.augmentator import Augmentator


class DirectDatasetMult(Dataset):
    def __init__(
        self,
        bop_dataset: BOPDataset,
        cfg=None,
        transforms=None,
        reduce=1,
        nq: int = 30,
        img_only=False,
    ) -> None:
        super().__init__()
        self.reduce = reduce
        self.img_only = img_only
        self.dataset = bop_dataset
        self.mode = bop_dataset.mode
        self.type = bop_dataset.dataset_type
        self.transforms = transforms
        self.args = cfg
        self.im_h, self.im_w = self.dataset.metadata.height, self.dataset.metadata.width

        self.mapping = {v: k for k, v in self.dataset.metadata.mapping.items()}
        self.im_hw_resized = self.args.imgsz
        self.class_names = self.dataset.metadata.class_names
        self.binary_sym_infos = self.dataset.metadata.binary_symmetries
        self.model_points = {
            int(k) - 1: np.asarray(v) / 1000
            for k, v in self.dataset.model_points.items()
        }
        self.model_bbox3d = {
            int(k) - 1: np.asarray(v)[:-1, :] / 1000
            for k, v in self.dataset.model_bbox3d.items()
        }
        assert np.all(self.model_points[0] < 1)
        self.sym_infos = {
            int(k) - 1: torch.from_numpy(v)
            for k, v in self.dataset.model_symmetries.items()
        }
        self.model_info = {int(k) - 1: v for k, v in self.dataset.models_info.items()}
        self.model_keypoints = {
            int(k) - 1: v for k, v in self.dataset.model_keypoints.items()
        }
        self.augmentator = None
        if self.args.aug and self.mode == Mode.TRAIN:
            self._set_augmentator()

    def _set_augmentator(self) -> Augmentator:
        augmentator = Augmentator(imgsz=[self.im_w, self.im_h])
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

        img_path = self.dataset.get_img_path(idx)
        img = cv2.imread(img_path)
        # img = Image.open(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.args.use_obj_bbox:
            bboxs = self.dataset.get_bbox_objs(idx)
        else:
            bboxs = self.dataset.get_bbox_visibs(idx)

        if self.augmentator:
            img, _, _ = self.augmentator(img=img)
        if self.transforms is not None:
            img = Image.fromarray(img)
            img = self.transforms(img)
        else:
            raise NotImplementedError

        obj_cls = self.dataset.get_obj_ids(idx)
        # obj_cls = [int(o) - 1 for o in obj_cls]
        no = len(obj_cls)
        obj_cls = torch.tensor(obj_cls, dtype=torch.long) - 1
        cam = torch.tensor(self.dataset.get_cam(idx))

        bboxs = torch.tensor(bboxs, dtype=torch.float32)
        # from xywh (top-left) to cxchwh
        #        if not bboxs.shape[0] == 0:
        # return {
        # "img": img,
        # "cls": obj_cls,
        # "bboxes": None,

        # }
        # bboxs = None
        # poses = None
        # centroid_z = None

        # ----------------- bboxs ----------------- #
        bboxs = torch.stack(
            [
                (bboxs[:, 0] + bboxs[:, 2] / 2) / self.im_w,
                (bboxs[:, 1] + bboxs[:, 3] / 2) / self.im_h,
                bboxs[:, 2] / self.im_w,
                bboxs[:, 3] / self.im_h,
            ],
            dim=1,
        )
        # ----------------- poses ----------------- #
        poses = torch.tensor(self.dataset.get_poses(idx))
        if not poses.shape[0] == 0:
            poses[:, :, 3] = poses[:, :, 3] / 1000
        trans = poses[:, :3, 3]
        proj = torch.matmul(cam, trans.T).T
        proj_2d = proj / proj[:, 2].unsqueeze(-1)
        centroid_z = torch.cat(
            [
                proj_2d[:, 0, None] / self.im_w,
                proj_2d[:, 1, None] / self.im_h,
                poses[:, -1, -1, None],
            ],
            dim=1,
        )
        # bbox_centroid = bboxs[:, :2] - centroid_z[:, :2]

        # ----------------- keypoints ----------------- #
        # keypoints, projected_keypoints = self._get_keypoints(obj_cls, poses, cam, idx)
        projected_keypoints = torch.zeros((1, 2))

        return {
            "img": img,
            "cls": obj_cls,
            "bboxes": bboxs,
            "poses": poses,
            "centroid_z": centroid_z,
            "keypoints": projected_keypoints,
            "keypoints_cls": torch.zeros(1),  # selected_values,
            "cam": cam,
            "batch_idx": torch.zeros(no).long(),
        }

    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in ["img", "cam"]:
                value = torch.stack(value, 0)
            if k in [
                "masks",
                "keypoints",
                "keypoints_cls",
                "bboxes",
                "poses",
                "centroid_z",
            ]:
                value = torch.cat(value, 0)
            if k in ["cls"]:
                value = torch.cat(value, 0).long()
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

    def pad_tensor(self, tensor, no):
        assert tensor.shape[0] == no
        delta = self.nq - no
        pad = torch.zeros((delta, *tensor.shape[1:]), dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad])
        return tensor

    def _get_keypoints(self, obj_cls, poses, cam, idx):
        n_kpts = self.args.dec_num_points
        keypoints = [
            torch.from_numpy(self.model_keypoints[c.item()][n_kpts]) / 1000
            for c in obj_cls
        ]
        keypoints = torch.stack(keypoints, dim=0)
        transformed_keypoints = torch.bmm(
            poses[:, :, :3], keypoints.transpose(1, 2)
        ).transpose(1, 2) + poses[:, :, 3].unsqueeze(1).repeat(
            1, self.args.dec_num_points, 1
        )
        transformed_keypoints = transformed_keypoints.reshape(-1, 3).T
        projected_keypoints = (cam @ transformed_keypoints).T
        projected_keypoints = projected_keypoints[:, :2] / projected_keypoints[:, 2:]
        projected_keypoints = projected_keypoints.reshape(-1, n_kpts, 2)
        # keypoint visibility
        vis_mask = self.dataset.get_mask_visib_paths(idx)
        vis_mask = [cv2.imread(str(p)) / 255 for p in vis_mask]
        points = projected_keypoints.long()
        mask = torch.stack([torch.from_numpy(m)[..., 0].int() for m in vis_mask], dim=0)
        x_max, y_max = max(mask.shape[1], points[..., 1].max()), max(
            mask.shape[2], points[..., 0].max()
        )
        # extended_mask = torch.zeros((mask.shape[0], mask.shape[1]+500, mask.shape[2]+500))
        extended_mask = torch.zeros((mask.shape[0], x_max + 50, y_max + 50))
        extended_mask[:, : mask.shape[1], : mask.shape[2]] = mask
        bs, num, h, w = mask.size(0), points.size(1), mask.size(1), mask.size(2)
        batch_indices = torch.arange(bs).view(bs, 1).expand(bs, num)
        selected_values = extended_mask[batch_indices, points[..., 1], points[..., 0]]

        projected_keypoints[:, :, 0] /= self.im_w
        projected_keypoints[:, :, 1] /= self.im_h
        return keypoints, projected_keypoints


if __name__ == "__main__":
    pass
