from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (384, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> np.ndarray:
    """Resize and pad an image to ``new_shape=(height,width)``."""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def augment_hsv(img: np.ndarray, hgain: float = 0.015, sgain: float = 0.7, vgain: float = 0.4) -> None:
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)


def random_bilateral_blur(img: np.ndarray) -> np.ndarray:
    d = random.choice((5, 7, 9))
    sigma_color = int(random.uniform(25, 90))
    sigma_space = int(random.uniform(25, 90))
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def random_gaussian_blur(img: np.ndarray) -> np.ndarray:
    k = random.choice((3, 5, 7))
    sigma = float(random.uniform(0.0, 1.2))
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)


def random_crop_pair(img: np.ndarray, mask: np.ndarray, crop_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    crop_h, crop_w = int(crop_hw[0]), int(crop_hw[1])
    h, w = img.shape[:2]
    if h <= crop_h or w <= crop_w:
        return img, mask
    y0 = random.randint(0, h - crop_h)
    x0 = random.randint(0, w - crop_w)
    return img[y0:y0 + crop_h, x0:x0 + crop_w], mask[y0:y0 + crop_h, x0:x0 + crop_w]


def random_perspective_pair(
    img: np.ndarray,
    mask: np.ndarray,
    degrees: float = 10,
    translate: float = 0.1,
    scale: float = 0.1,
    shear: float = 10,
    perspective: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = img.shape[:2]
    C = np.eye(3)
    C[0, 2] = -width / 2
    C[1, 2] = -height / 2

    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    R = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    scale_factor = random.uniform(1 - scale, 1.5 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale_factor)

    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    M = T @ S @ R @ P @ C
    if perspective:
        img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        mask = cv2.warpPerspective(mask, M, dsize=(width, height), borderValue=0)
    else:
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
        mask = cv2.warpAffine(mask, M[:2], dsize=(width, height), borderValue=0)
    return img, mask


def _to_one_hot_lane(mask: np.ndarray, gt_hw: Tuple[int, int] = (360, 640), threshold: int = 1) -> torch.Tensor:
    mask = cv2.resize(mask, (gt_hw[1], gt_hw[0]), interpolation=cv2.INTER_LINEAR)
    lane = (mask > threshold).astype(np.float32)
    bg = 1.0 - lane
    return torch.from_numpy(np.stack([bg, lane], axis=0))


def _image_to_chw_rgb_uint8(bgr: np.ndarray) -> torch.Tensor:
    img = bgr[:, :, ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(img))


class BDD100KLaneDataset(Dataset):
    """BDD100K binary lane-mask dataset used by HSCM-Lane."""

    def __init__(self, data_root: str = "BDD100K", split: str = "train", hyp: Optional[Dict[str, Any]] = None, augment: bool = True):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.hyp = hyp or {}
        self.augment = bool(augment and split == "train")
        self.image_dir = self.data_root / "100k" / split
        self.mask_dir = self.data_root / "bdd_lane_gt" / split
        assert self.image_dir.is_dir(), f"Image folder not found: {self.image_dir}"
        assert self.mask_dir.is_dir(), f"Lane-mask folder not found: {self.mask_dir}"
        self.names = sorted([p.name for p in self.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        assert self.names, f"No images found in {self.image_dir}"

    def __len__(self) -> int:
        return len(self.names)

    def _apply_augmentation(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = self.hyp
        if random.random() < float(h.get("prob_perspective", 0.0)):
            img, mask = random_perspective_pair(
                img,
                mask,
                degrees=float(h.get("degrees", 0.0)),
                translate=float(h.get("translate", 0.0)),
                scale=float(h.get("scale", 0.0)),
                shear=float(h.get("shear", 0.0)),
            )
        if random.random() < float(h.get("prob_hsv", 0.0)):
            augment_hsv(img, float(h.get("hgain", 0.015)), float(h.get("sgain", 0.7)), float(h.get("vgain", 0.4)))
        if random.random() < float(h.get("prob_flip", 0.0)):
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if random.random() < float(h.get("prob_bilateral", 0.0)):
            img = random_bilateral_blur(img)
        if random.random() < float(h.get("prob_gaussian", 0.0)):
            img = random_gaussian_blur(img)
        if random.random() < float(h.get("prob_crop", 0.0)):
            img, mask = random_crop_pair(img, mask, (int(h.get("height_crop", 384)), int(h.get("width_crop", 640))))
        return img, mask

    def __getitem__(self, idx: int):
        name = self.names[int(idx)]
        image_path = self.image_dir / name
        mask_path = self.mask_dir / (Path(name).stem + ".png")
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert img is not None, f"Cannot read image: {image_path}"
        assert mask is not None, f"Cannot read lane mask: {mask_path}"

        if self.augment:
            img, mask = self._apply_augmentation(img, mask)

        img = letterbox(img, (384, 640))
        target = _to_one_hot_lane(mask, (360, 640), threshold=1)
        return str(image_path), _image_to_chw_rgb_uint8(img), target


def _read_json_lines(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _resolve_tusimple_label_files(root: Path, label_json: str = "") -> List[Path]:
    if label_json:
        out = []
        for part in label_json.split(","):
            p = Path(part.strip())
            out.append(p if p.is_absolute() else root / p)
        return out
    candidates = [root / "label_data_0601.json", root / "test_label.json", root / "label_data_0531.json", root / "label_data_0313.json"]
    return [p for p in candidates if p.is_file()][:1]


def _split_lane_segments(xs: Sequence[int], ys: Sequence[int]) -> List[List[Tuple[int, int]]]:
    segments: List[List[Tuple[int, int]]] = []
    current: List[Tuple[int, int]] = []
    for x, y in zip(xs, ys):
        xi = int(x)
        if xi < 0:
            if len(current) >= 2:
                segments.append(current)
            current = []
            continue
        current.append((xi, int(y)))
    if len(current) >= 2:
        segments.append(current)
    return segments


def render_tusimple_lane_mask(
    height: int,
    width: int,
    lanes: Sequence[Sequence[int]],
    h_samples: Sequence[int],
    thickness: int = 2,
    line_type: int = cv2.LINE_8,
    split_missing: bool = True,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    ys = [int(y) for y in h_samples]
    for lane in lanes:
        if split_missing:
            segments = _split_lane_segments(lane, ys)
        else:
            segments = [[(int(x), int(y)) for x, y in zip(lane, ys) if int(x) >= 0]]
        for seg in segments:
            pts = []
            for x, y in seg:
                if 0 <= y < height:
                    pts.append((min(max(int(x), 0), width - 1), int(y)))
            if len(pts) >= 2:
                arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(mask, [arr], isClosed=False, color=255, thickness=int(thickness), lineType=int(line_type))
    return mask


class TuSimpleLaneDataset(Dataset):
    """TuSimple row-wise lane annotations converted to binary lane masks."""

    def __init__(
        self,
        root: str = "TuSimple",
        label_json: str = "",
        gt_style: str = "bdd",
        max_samples: int = 0,
    ):
        super().__init__()
        self.root = Path(root)
        self.gt_style = gt_style
        label_files = _resolve_tusimple_label_files(self.root, label_json)
        assert label_files, f"No TuSimple label JSON found under {self.root}"
        items: List[Dict[str, Any]] = []
        for lf in label_files:
            assert lf.is_file(), f"TuSimple label JSON not found: {lf}"
            items.extend(_read_json_lines(lf))
        samples = []
        for it in items:
            raw = it.get("raw_file") or it.get("image")
            if not raw:
                continue
            img_path = Path(raw)
            if not img_path.is_absolute():
                img_path = self.root / img_path
            if img_path.is_file():
                samples.append({"img": img_path, "lanes": it.get("lanes", []), "h_samples": it.get("h_samples", [])})
        if max_samples > 0:
            samples = samples[: int(max_samples)]
        assert samples, f"TuSimple dataset is empty under {self.root}"
        self.samples = samples

        if gt_style == "legacy":
            self.thickness, self.line_type, self.split_missing, self.threshold = 8, cv2.LINE_AA, False, 0
        elif gt_style == "bdd100k":
            self.thickness, self.line_type, self.split_missing, self.threshold = 1, cv2.LINE_8, True, 1
        else:
            self.thickness, self.line_type, self.split_missing, self.threshold = 2, cv2.LINE_8, True, 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[int(idx)]
        img = cv2.imread(str(s["img"]), cv2.IMREAD_COLOR)
        assert img is not None, f"Cannot read image: {s['img']}"
        h0, w0 = img.shape[:2]
        mask = render_tusimple_lane_mask(
            h0,
            w0,
            s["lanes"],
            s["h_samples"],
            thickness=self.thickness,
            line_type=self.line_type,
            split_missing=self.split_missing,
        )
        img = letterbox(img, (384, 640))
        target = _to_one_hot_lane(mask, (360, 640), threshold=self.threshold)
        return str(s["img"]), _image_to_chw_rgb_uint8(img), target
