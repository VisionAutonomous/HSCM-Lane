from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hscm_lane import HSCMLane, TuSimpleLaneDataset, crop_logits_for_eval, lane_metrics_from_confusion, update_confusion
from evaluate_bdd100k import load_state_dict


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate HSCM-Lane on TuSimple under the derived pixel-level protocol.")
    p.add_argument("--tusimple-root", default="TuSimple")
    p.add_argument("--label-json", default="")
    p.add_argument("--gt-style", default="bdd", choices=["legacy", "bdd", "bdd100k"])
    p.add_argument("--weights", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50", "res18", "res34", "res50"])
    p.add_argument("--fusion", default="sum", choices=["sum", "concat", "gated"])
    p.add_argument("--swin-depth", type=int, default=2)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--no-swin", action="store_true")
    p.add_argument("--no-swin-c1", action="store_true")
    p.add_argument("--no-swin-c2", action="store_true")
    p.add_argument("--no-swin-c3", action="store_true")
    p.add_argument("--no-pretrained-backbone", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dataset = TuSimpleLaneDataset(args.tusimple_root, label_json=args.label_json, gt_style=args.gt_style)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    model = HSCMLane(
        backbone=args.backbone,
        pretrained_backbone=not args.no_pretrained_backbone,
        swin_depth=args.swin_depth,
        window_size=args.window_size,
        use_swin=not args.no_swin,
        use_swin_c1=not args.no_swin_c1,
        use_swin_c2=not args.no_swin_c2,
        use_swin_c3=not args.no_swin_c3,
        fusion=args.fusion,
    )
    model.load_state_dict(load_state_dict(args.weights), strict=True)
    model.to(device).eval()

    conf = torch.zeros((2, 2), device=device, dtype=torch.int64)
    with torch.inference_mode():
        for _, x, target in tqdm(loader, desc="TuSimple derived", dynamic_ncols=True):
            x = x.to(device, non_blocking=True).float() / 255.0
            target = target.to(device, non_blocking=True)
            logits = model(x)
            pred = crop_logits_for_eval(logits).argmax(dim=1)
            gt = target.argmax(dim=1)
            conf = update_confusion(conf, pred, gt)

    metrics = lane_metrics_from_confusion(conf)
    display = {k: (v * 100 if k in {"iou_lane", "recall_lane", "precision_lane", "f1_lane", "balanced_accuracy"} else v) for k, v in metrics.items()}
    print(json.dumps(display, indent=2))


if __name__ == "__main__":
    main()
