from typing import Tuple, Any, Optional
from enum import Enum
import random
from pathlib import Path

import numpy as np
import cv2
from utils import RANK
from utils.flags import DistType
from utils.bbox import Bbox


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose"""

    def __init__(
        self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32
    ):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels"""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels




class Distribution:
    def __init__(
        self,
        dist_type: DistType,
        sigma: float = None,
        mean: float = 0,
        low: float = 0,
        high: float = 1,
    ) -> None:
        self.dist_type = dist_type
        self.mean = mean
        self.sigma = sigma
        self.low = low
        self.high = high

    def __call__(self, *args, **kwargs) -> Any:
        if self.dist_type == DistType.NORMAL:
            kwargs["loc"] = self.mean
            kwargs["scale"] = self.sigma
            return np.random.normal(*args, **kwargs)
        elif self.dist_type == DistType.UNIFORM:
            kwargs["low"] = self.low
            kwargs["high"] = self.high
            return np.random.uniform(*args, **kwargs)
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")


class BoxAugmentator:
    def __init__(
        self,
        imgsz: Tuple[int],
        shift_dist: Distribution,
        scale_dist: Distribution,
    ) -> None:
        self.imgsz = imgsz  # (height, width)
        self.shift_dist = shift_dist
        self.scale_dist = scale_dist

    def __call__(self, bbox: Bbox) -> Bbox:
        """Augment bounding box, by shifting center, and scaling width and height randomly."""
        x, y = self.shift_dist(), self.shift_dist()
        #x, y = max(min(self.shift_dist(), 0.15), -0.15), max(
        #    min(self.shift_dist(), 0.15), -0.15
        #)
        x, y = int(x * bbox.w), int(y * bbox.h)
        if (
            bbox.x1 + x <= 0
            or bbox.x2 + x >= self.imgsz[1]
            or bbox.y1 + y <= 0
            or bbox.y2 + y >= self.imgsz[0]
        ):
            x, y = 0, 0
        bbox.shift_center(x, y)

        scale_w, scale_h = max(min(self.scale_dist(), 1 + 0.15), 1 - 0.15), max(
            min(self.scale_dist(), 1 + 0.15), 1 - 0.15
        )
        delta_w = int(bbox.w * scale_w) - bbox.w
        delta_h = int(bbox.h * scale_h) - bbox.h

        if (
            bbox.x1 - delta_w // 2 == bbox.x2 + delta_w // 2
            or bbox.x1 - delta_w // 2 < 0
            or bbox.x2 + delta_w // 2 >= self.imgsz[1]
        ):
            scale_w = 1
        if (
            bbox.y1 - delta_h // 2 == bbox.y2 + delta_h // 2
            or bbox.y1 - delta_h // 2 < 0
            or bbox.y2 + delta_h // 2 >= self.imgsz[0]
        ):
            scale_h = 1
        bbox.scale_width(scale_w)
        bbox.scale_height(scale_h)
        return bbox


class BgAugmentator:
    """Augment background image by replacing it with another image randomly chooses
    from a folder."""

    def __init__(
        self, bg_path, im_H: int, im_W: int, truncate_fg: bool = False
    ) -> None:
        if isinstance(bg_path, str):
            bg_path = Path(bg_path)
        if not bg_path.exists():
            raise ValueError(f"Background path does not exist: {bg_path}")
        self.bg_imgs = list(bg_path.glob("*.jpg"))
        num_bg_imgs = len(self.bg_imgs)
        if num_bg_imgs == 0:
            raise ValueError(f"No background images found in {bg_path}")
        if num_bg_imgs < 100:
            self.bg_imgs = [get_bg_image(p, im_H, im_W) for p in self.bg_imgs]
        self.truncate_fg = truncate_fg

    def __call__(
        self, im: np.ndarray, im_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return replace_bg(im, im_mask, self.bg_imgs, truncate_fg=self.truncate_fg)


class MaskAugmentator:
    """Augment mask by randomly removing a block of pixels from it. Creates
    truncated masks."""

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.copy().astype(np.bool)
        nonzeros = np.nonzero(mask.astype(np.uint8))
        if nonzeros[0].shape[0] == 0:
            return mask
        x1, y1 = np.min(nonzeros, axis=1)
        x2, y2 = np.max(nonzeros, axis=1)
        c_h = 0.5 * (x1 + x2)
        c_w = 0.5 * (y1 + y2)
        rnd = random.random()
        # print(x1, x2, y1, y2, c_h, c_w, rnd, mask.shape)
        if rnd < 0.2:  # block upper
            c_h_ = int(random.uniform(x1, c_h))
            mask[:c_h_, :] = False
        elif rnd < 0.4:  # block bottom
            c_h_ = int(random.uniform(c_h, x2))
            mask[c_h_:, :] = False
        elif rnd < 0.6:  # block left
            c_w_ = int(random.uniform(y1, c_w))
            mask[:, :c_w_] = False
        elif rnd < 0.8:  # block right
            c_w_ = int(random.uniform(c_w, y2))
            mask[:, c_w_:] = False
        else:
            pass
        return mask


def resize_short_edge(
    im,
    target_size,
    max_size,
    stride=0,
    interpolation=cv2.INTER_LINEAR,
    return_scale=False,
):
    """Scale the shorter edge to the given size, with a limit of `max_size` on
    the longer edge. If `max_size` is reached, then downscale so that the
    longer edge does not exceed max_size. only resize input image to target
    size and return scale.
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(
        im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation
    )

    if stride == 0:
        if return_scale:
            return im, im_scale
        else:
            return im
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[: im.shape[0], : im.shape[1], :] = im
        if return_scale:
            return padded_im, im_scale
        else:
            return padded_im


def get_bg_image(bg_image, imH, imW, channel=3):
    """keep aspect ratio of bg during resize target image size:
    imHximWxchannel.
    """
    target_size = min(imH, imW)
    max_size = max(imH, imW)
    real_hw_ratio = float(imH) / float(imW)
    if isinstance(bg_image, str) or isinstance(bg_image, Path):
        bg_image = cv2.imread(str(bg_image))
    if not isinstance(bg_image, np.ndarray) and bg_image.shape[2] != 3:
        raise ValueError("bg_image must be np.ndarray and a 3-channel image.")

    bg_h, bg_w, bg_c = bg_image.shape
    bg_image_resize = np.zeros((imH, imW, channel), dtype="uint8")
    if (float(imH) / float(imW) < 1 and float(bg_h) / float(bg_w) < 1) or (
        float(imH) / float(imW) >= 1 and float(bg_h) / float(bg_w) >= 1
    ):
        if bg_h >= bg_w:
            bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
            if bg_h_new < bg_h:
                bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
            else:
                bg_image_crop = bg_image
        else:
            bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
            if bg_w_new < bg_w:
                bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
            else:
                bg_image_crop = bg_image
    else:
        if bg_h >= bg_w:
            bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
            bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
        else:  # bg_h < bg_w
            bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
            # logger.info(bg_w_new)
            bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
    bg_image_resize_0 = resize_short_edge(bg_image_crop, target_size, max_size)
    h, w, c = bg_image_resize_0.shape
    bg_image_resize[0:h, 0:w, :] = bg_image_resize_0
    return bg_image_resize


def replace_bg(im, im_mask, bg_files, truncate_fg=False):
    H, W = im.shape[:2]

    ind = random.randint(0, len(bg_files) - 1)
    file = bg_files[ind]
    if isinstance(file, str) or isinstance(file, Path):
        bg_img = get_bg_image(file, H, W)
    elif isinstance(file, np.ndarray):
        bg_img = get_bg_image(file, H, W)

    mask = im_mask.copy()
    mask = mask.astype(np.bool)
    if truncate_fg:
        nonzeros = np.nonzero(mask.astype(np.uint8))
        if nonzeros[0].shape[0] == 0:
            return mask
        x1, y1 = np.min(nonzeros, axis=1)
        x2, y2 = np.max(nonzeros, axis=1)
        c_h = 0.5 * (x1 + x2)
        c_w = 0.5 * (y1 + y2)
        rnd = random.random()
        # print(x1, x2, y1, y2, c_h, c_w, rnd, mask.shape)
        if rnd < 0.2:  # block upper
            c_h_ = int(random.uniform(x1, c_h))
            mask[:c_h_, :] = False
        elif rnd < 0.4:  # block bottom
            c_h_ = int(random.uniform(c_h, x2))
            mask[c_h_:, :] = False
        elif rnd < 0.6:  # block left
            c_w_ = int(random.uniform(y1, c_w))
            mask[:, :c_w_] = False
        elif rnd < 0.8:  # block right
            c_w_ = int(random.uniform(c_w, y2))
            mask[:, c_w_:] = False
        else:
            pass
    mask_bg = ~mask
    im[mask_bg] = bg_img[mask_bg]
    im = im.astype(np.uint8)
    return im, mask
