from typing import Any, Tuple, Optional
from pathlib import Path
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmenters import (
    Sequential,
    Sometimes,
    Dropout,
    CoarseDropout,
    GaussianBlur,
    Multiply,
    Add,
    Invert,
    AdditiveGaussianNoise,
)
import imgaug.augmenters.pillike as pillike

from utils.bbox import Bbox
from data_tools.augmentations.augmentations import (
    DistType,
    Distribution,
    BoxAugmentator,
    BgAugmentator,
    MaskAugmentator,
)


class Augmentator:
    def __init__(self, imgsz: Tuple[int]) -> None:
        self.imgsz = imgsz
        self.dist = Distribution(DistType.UNIFORM)
        self.p_augmentators = []

    def build(self) -> None:
        for p, augmentator in self.p_augmentators:
            if p == 0:
                augmentator = None
        if getattr(self, "bg_augmentator", None) and self.truncate_fn:
            self.mask_augmentator = None

    def __call__(
        self,
        img: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        bbox = None,
    ) -> Any:
        p = self.dist(low=0, high=1)

        if img is not None:
            img = img[None, :, :, :]
            if not isinstance(img, np.ndarray) or img.shape[-1] != 3:
                raise ValueError("img must be of type np.ndarray and CHW.")
            if img.ndim != 4:
                img = img[None, ...]
            if getattr(self, "bg_augmentator", None) and self.truncate_fn:
                if mask is None:
                    raise ValueError(
                        "Mask is required if background augmentation is used and truncate_fn=True."
                    )
                if not isinstance(mask, np.ndarray):
                    raise ValueError("mask must be of type np.ndarray.")
                if p < self.p_bg:
                    img, mask = self.bg_augmentator(img, mask)
            if getattr(self, "bg_augmentator", None) and not self.truncate_fn:
                if p < self.p_bg:
                    img = self.bg_augmentator(img)
            if self.color_augmentator:
                if p < self.p_color:
                    img = self.color_augmentator(images=img)
            img = img[0, ...]
        # mask_augmentator is set to None if redundant
        if mask is not None and getattr(self, "mask_augmentator", None):
            if not isinstance(mask, np.ndarray):
                raise ValueError("mask must be of type np.ndarray.")
            if p < self.p_mask:
                mask = self.mask_augmentator(mask)
        aug_bbox = []
        if bbox and self.box_augmentator:
            if isinstance(bbox, Bbox):
                bbox = [bbox]
            for b in bbox:
                if not isinstance(b, Bbox):
                    b = Bbox.from_xywh(b)
                    #raise ValueError("bbox must be of type Bbox.")
                if p < self.p_box:
                    aug_b = self.box_augmentator(b)
                    aug_bbox.append(aug_b)
                else:
                    aug_bbox.append(b)
        return img, mask, aug_bbox

    def add_box_augementator(self, p: float, dist_type: DistType, sigma: float) -> None:
        """Add box augmentator to augmentators."""
        box_aug_dist_shift = Distribution(
            dist_type=dist_type, low=-sigma, high=sigma
        )
        box_aug_dist_scale = Distribution(
            dist_type=dist_type, low=1 - sigma*0.5, high=1 + sigma*0.5
        )
        self.box_augmentator = BoxAugmentator(
            self.imgsz, shift_dist=box_aug_dist_shift, scale_dist=box_aug_dist_scale
        )
        self.p_box = p
        self.p_augmentators.append((self.p_box, self.box_augmentator))

    def add_color_augmentator(self, p: float, aug_code: str) -> None:
        """Add image augmentator to augmentators."""
        augs = [eval(e) for e in aug_code]
        self.color_augmentator = iaa.Sequential(augs, random_order=True)
        self.p_color = p

    def add_bg_augmentator(
        self, p: float, bg_path, im_H: int, im_W: int, truncate_fg: bool
    ) -> None:
        """Add background augmentator to augmentators."""
        # create list of bg image paths if number of images is to large to load into memory
        self.bg_path = bg_path
        self.truncate_fn = truncate_fg
        self.bg_augmentator = BgAugmentator(
            bg_path, im_H=im_H, im_W=im_W, truncate_fg=truncate_fg
        )
        self.p_bg = p
        self.p_augmentators.append((self.p_bg, self.bg_augmentator))

    def add_mask_augmentator(self, p: float) -> None:
        """Add mask augmentator to augmentators."""
        self.mask_augmentator = MaskAugmentator()
        self.p_mask = p
        self.p_augmentators.append((self.p_mask, self.mask_augmentator))
