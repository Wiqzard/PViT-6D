from __future__ import annotations
from typing import Tuple, Union, Any
from enum import Enum
from copy import deepcopy
from dataclasses import dataclass, field
import math
import numpy as np
import torch
import cv2


class BboxFormat(Enum):
    XYXY = 1
    XYWH = 2

    @classmethod
    def has_value(cls, value: int) -> bool:
        return any(value == item.value for item in cls)


def xyxy_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x1, y1, x2, y2) to (x_center, y_center, w, h) format.
    bbox: [batch_size, 4]
    """
    if bbox.ndim == 1:
        bbox = bbox.unsqueeze(0)
    elif bbox.ndim != 2:
        raise ValueError("bbox must be 2-dimensional")

    return torch.stack(
        (
            (bbox[:, 0] + bbox[:, 2]) / 2,
            (bbox[:, 1] + bbox[:, 3]) / 2,
            bbox[:, 2] - bbox[:, 0],
            bbox[:, 3] - bbox[:, 1],
        ),
        dim=1,
    )

def xywh_to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (x_center, y_center, w, h) to (x1, y1, x2, y2) format.
    bbox: [batch_size, 4]
    """
    if bbox.ndim == 1:
        bbox = bbox.unsqueeze(0)
    elif bbox.ndim != 2:
        raise ValueError("bbox must be 2-dimensional")

    return torch.stack(
        (
            bbox[:, 0] - bbox[:, 2] / 2,
            bbox[:, 1] - bbox[:, 3] / 2,
            bbox[:, 0] + bbox[:, 2] / 2,
            bbox[:, 1] + bbox[:, 3] / 2,
        ),
        dim=1,
    )


@dataclass
class Bbox:
    """
    Class for representing a bounding box.
    x1, y1, x2, y2: top-left and bottom-right coordinates
    x, y, w, h: left top , center coordinates and width and height
    """

    xyxy: Tuple[int, int, int, int]
    class_id: int = 0
    confidence: float = 1.0
    bbox_format: BboxFormat = BboxFormat.XYXY

   # def __post_init__(self):
   #     if self.bbox_format == BboxFormat.XYXY:
   #         trigger = (
   #             (self.x1 > self.x2)
   #             or (self.y1 > self.y2)
   #             or self.x1 < 0
   #             or self.y1 < 0
   #             or self.x2 < 0
   #             or self.y2 < 0
   #         )
   #     if trigger:
   #         raise ValueError("Invalid bbox coordinates or format")

    @property
    def xywh(self) -> Tuple[int, int, int, int]:
        """Return the bounding box as a tuple of (x, y, w, h), where x and y are the
        coordinates of the top-left corner of the bounding box, and w and h are the
        width and height of the bounding box, respectively.
        """
        return (self.x1, self.y1, self.w, self.h)

    @property
    def cxcywh(self) -> Tuple[int, int, int, int]:
        """Return the bounding box as a tuple of (cx, cy, w, h) where cx and cy are the
        coordinates of the center of the bounding box, and w and h are the width and
        height of the bounding box, respectively."""
        return (self.center[0], self.center[1], self.w, self.h)

    @property
    def x1(self) -> int:
        """Return the x coordinate of the top-left corner of the bounding box."""
        return int(self.xyxy[0])

    @property
    def y1(self) -> int:
        """Return the y coordinate of the top-left corner of the bounding box."""
        return int(self.xyxy[1])

    @property
    def x2(self) -> int:
        """Return the x coordinate of the bottom-right corner of the bounding box."""
        return int(self.xyxy[2])

    @property
    def y2(self) -> int:
        """Return the y coordinate of the bottom-right corner of the bounding box."""
        return int(self.xyxy[3])

    @property
    def w(self) -> int:
        """Return the width of the bounding box."""
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        """Return the height of the bounding box."""
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        """Return the center of the bounding box as a tuple of (x, y)."""
        return (self.x1 + self.w / 2, self.y1 + self.h / 2)

    @property
    def radius(self) -> int:
        """Return the radius of the bounding box."""
        return int(np.sqrt(self.w**2 + self.h**2))  # max(self.w, self.h) // 2
    
    def normalize(self, im_h: int, im_w: int) -> None:
        """Normalize the bounding box coordinates."""
        self.xyxy = (
            self.x1 / im_w,
            self.y1 / im_h,
            self.x2 / im_w,
            self.y2 / im_h,
        )   

    def scale(self, im_h: int, im_w: int) -> float:
        """Return the scale of the bounding box."""
        scale = max(self.y2 - self.y1, self.x2 - self.x1)
        return min(scale, max(im_h, im_w))

    def scale_box(self, scale: float) -> None:
        """Scale the bounding box by a factor of scale."""
        self.xyxy = (int(element * scale) for element in self.xyxy)

    def is_outside_img(self, img_h: int, img_w: int) -> bool:
        """Return True if the bounding box is outside the image."""
        return (
            self.x1 < 0
            or self.x1 >= img_w
            or self.y1 < 0
            or self.y1 >= img_h
            or self.x2 <= 0
            or self.x2 > img_w
            or self.y2 <= 0
            or self.y2 > img_h
        )

    def scale_height(self, scale: float) -> None:
        """Scale the bounding box height by a factor of scale."""
        xyxy = deepcopy(self.xyxy)
        delta_w = int(self.w * scale) - self.w
        delta_h = int(self.h * scale) - self.h
        self.xyxy = (self.x1, self.y1 - delta_h // 2, self.x2, self.y2 + delta_h // 2)
        if self.x1 >= self.x2:
            self.xyxy = (xyxy[0], self.y1, xyxy[2], self.y2)
        if self.y1 >= self.y2:
            self.xyxy = (self.x1, xyxy[1], self.x2, xyxy[3])

    def scale_width(self, scale: float) -> None:
        """Scale the bounding box width by a factor of scale."""
        xyxy = deepcopy(self.xyxy)
        delta_w = int(self.w * scale) - self.w
        self.xyxy = (self.x1 - delta_w // 2, self.y1, self.x2 + delta_w // 2, self.y2)
        if self.x1 >= self.x2:
            self.xyxy = (xyxy[0], self.y1, xyxy[2], self.y2)
        if self.y1 >= self.y2:
            self.xyxy = (self.x1, xyxy[1], self.x2, xyxy[3])

    def shift_center(self, x: int, y: int) -> None:  # Bbox:
        """Shift the bounding box by x and y pixels."""
        xyxy = deepcopy(self.xyxy)
        self.xyxy = (
            max(0, self.x1 + x),
            max(0, self.y1 + y),
            max(0, self.x2 + x),
            max(0, self.y2 + y),
        )
        if self.x1 >= self.x2:
            self.xyxy = (xyxy[0], self.y1, xyxy[2], self.y2)
        if self.y1 >= self.y2:
            self.xyxy = (self.x1, xyxy[1], self.x2, xyxy[3])

    def clip_x(self, x_min: int, x_max: int) -> None:
        """Clip the bounding box x coordinates to x."""
        self.xyxy = (
            min(x_max, (max(x_min, self.x1))),
            self.y1,
            min(x_max, (max(x_min, self.x2))),
            self.y2,
        )

    def clip_y(self, y_min: int, y_max: int) -> None:
        """Clip the bounding box y coordinates to y."""
        self.xyxy = (
            self.x1,
            min(y_max, (max(y_min, self.y1))),
            self.x2,
            min(y_max, (max(y_min, self.y2))),
        )

    @classmethod
    def from_xywh(cls, xywh: Tuple[int, int, int, int]) -> Bbox:
        """Create a Bbox object from a tuple of (x, y, w, h), where x and y are the
        coordinates of the top-left corner of the bounding box, and w and h are the
        width and height of the bounding box, respectively.
        """
        xyxy = (
            int(xywh[0]),
            int(xywh[1]),
            int(xywh[0] + xywh[2]),
            int(xywh[1] + xywh[3]),
        )
        return Bbox(xyxy=xyxy)

    @classmethod
    def from_cxcywh(cls, cxcywh: Tuple[int, int, int, int]) -> Bbox:
        """Create a Bbox object from a tuple of (cx, cy, w, h), where cx and cy are the
        coordinates of the center of the bounding box, and w and h are the
        width and height of the bounding box, respectively.
        """
        xyxy = (
            cxcywh[0] - cxcywh[2] // 2,
            cxcywh[1] - cxcywh[3] // 2,
            cxcywh[0] + cxcywh[2] // 2,
            cxcywh[1] + cxcywh[3] // 2,
        )
        return Bbox(xyxy=xyxy)

    @classmethod
    def from_yolo_result(cls, yolo_result: Any) -> list[Bbox]:
        """Creates a Bbox object from a YOLO result. If the YOLO result is a list of
        bounding boxes, the bounding box with the highest confidence score is
        returned. If the YOLO result comprises multiple batches, a list of Bbox
        objects is returned.
        """
        bboxes = []
        if yolo_result[0].boxes.xyxy.nelement() == 0:
            return []
        for result in yolo_result:
            xyxy = result.boxes.xyxy
            mask = xyxy[:, 0] < 600
            confidence = result.boxes.conf
            # set confidence to 0 where mask is true
            confidence[mask] = 0
            class_id = result.boxes.cls
            if confidence.numel() == 0:
                return []
            max_conf_idx = torch.argmax(confidence)  # , dim=1)

            bbox = Bbox(
                xyxy=xyxy[max_conf_idx].cpu().numpy(),
                class_id=class_id[max_conf_idx].cpu().numpy(),
                confidence=confidence[max_conf_idx].cpu().numpy(),
            )
            bboxes.append(bbox)
        return bboxes

    @classmethod
    def from_xyxy(cls, xyxy: Tuple[int, int, int, int]) -> Bbox:
        return Bbox(xyxy=xyxy)

    @classmethod
    def from_segmentation_mask(cls, mask: np.ndarray) -> Bbox:
        pass

    def deepcopy(self) -> Bbox:
        """Create a deep copy of a Bbox object."""
        return Bbox(xyxy=self.xyxy, class_id=self.class_id, confidence=self.confidence)
    
    def visualize(self, img):
        return cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)


def get_bbox3d_and_center(pts):
    """
    pts: Nx3
    ---
    bb: bb3d+center, 9x3
    """
    bb = []
    minx, maxx = min(pts[:, 0]), max(pts[:, 0])
    miny, maxy = min(pts[:, 1]), max(pts[:, 1])
    minz, maxz = min(pts[:, 2]), max(pts[:, 2])
    avgx = np.average(pts[:, 0])
    avgy = np.average(pts[:, 1])
    avgz = np.average(pts[:, 2])

    """
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    bb = np.array(
        [
            [maxx, maxy, maxz],
            [minx, maxy, maxz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, maxy, minz],
            [minx, maxy, minz],
            [minx, miny, minz],
            [maxx, miny, minz],
            [avgx, avgy, avgz],
        ],
        dtype=np.float32,
    )
    return bb
