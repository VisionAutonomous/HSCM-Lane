from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_bdd100k import load_state_dict
from hscm_lane import HSCMLane, letterbox


def iter_images(source: str):
    p = Path(source)
    if p.is_file():
        yield p
    else:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            yield from sorted(p.glob(ext))


def overlay_lane(bgr: np.ndarray, mask: np.ndarray, color=(255, 0, 255), alpha: float = 0.55) -> np.ndarray:
    out = bgr.copy()
    color_img = np.zeros_like(out)
    color_img[:, :] = color
    lane = mask.astype(bool)
    out[lane] = cv2.addWeighted(out, 1.0 - alpha, color_img, alpha, 0)[lane]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run HSCM-Lane inference on one image or an image folder.")
    p.add_argument("--source", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out-dir", default="outputs/inference")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50", "res18", "res34", "res50"])
    p.add_argument("--fusion", default="sum", choices=["sum", "concat", "gated"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--no-pretrained-backbone", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = HSCMLane(backbone=args.backbone, pretrained_backbone=not args.no_pretrained_backbone, fusion=args.fusion)
    model.load_state_dict(load_state_dict(args.weights), strict=True)
    model.to(device).eval()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for image_path in iter_images(args.source):
            bgr0 = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            assert bgr0 is not None, f"Cannot read image: {image_path}"
            h0, w0 = bgr0.shape[:2]
            bgr = letterbox(bgr0, (384, 640))
            rgb_chw = bgr[:, :, ::-1].transpose(2, 0, 1).copy()
            x = torch.from_numpy(rgb_chw).unsqueeze(0).to(device).float() / 255.0
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
            mask_crop = (prob[12:-12, :] >= args.threshold).astype(np.uint8) * 255
            mask_orig = cv2.resize(mask_crop, (w0, h0), interpolation=cv2.INTER_NEAREST)
            overlay = overlay_lane(bgr0, mask_orig > 0)
            cv2.imwrite(str(Path(args.out_dir) / f"{image_path.stem}_mask.png"), mask_orig)
            cv2.imwrite(str(Path(args.out_dir) / f"{image_path.stem}_overlay.png"), overlay)
            print(f"saved: {image_path.stem}_mask.png and {image_path.stem}_overlay.png")


if __name__ == "__main__":
    main()
