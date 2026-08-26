from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from hscm_lane import (
    BDD100KLaneDataset,
    FocalTverskyLoss,
    HSCMLane,
    crop_logits_for_eval,
    lane_metrics_from_confusion,
    update_confusion,
)


def set_seed(seed: int, deterministic: bool = True) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    return gen


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_lr(epoch: int, base_lr: float, max_epochs: int, warmup_epochs: int, warmup_start: float, poly_power: float) -> float:
    if epoch < warmup_epochs:
        t = (epoch + 1) / max(1, warmup_epochs)
        return float(base_lr * (warmup_start + (1.0 - warmup_start) * t))
    t = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
    t = min(max(t, 0.0), 1.0)
    return float(base_lr * (1.0 - t) ** poly_power)


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.ema = deepcopy(model).eval()
        self.updates = 0
        self.decay_base = float(decay)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay_base * (1.0 - np.exp(-self.updates / 2000.0))
        model_state = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(model_state[k].detach(), alpha=1.0 - decay)


def build_model(args: argparse.Namespace) -> HSCMLane:
    return HSCMLane(
        input_hw=(384, 640),
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


def make_loss(hyp: Dict, args: argparse.Namespace) -> FocalTverskyLoss:
    return FocalTverskyLoss(
        tversky_alpha=float(hyp.get("alpha2", 0.9)),
        tversky_gamma=float(hyp.get("gamma2", 1.3333)),
        focal_alpha=float(hyp.get("alpha3", 0.25)),
        focal_gamma=float(hyp.get("gamma3", 2.0)),
        focal_weight=float(args.focal_weight),
        tversky_weight=float(args.tversky_weight),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: FocalTverskyLoss, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    focal_meter = AverageMeter()
    tversky_meter = AverageMeter()
    conf = torch.zeros((2, 2), device=device, dtype=torch.int64)

    for _, x, target in tqdm(loader, desc="val", dynamic_ncols=True):
        x = x.to(device, non_blocking=True).float() / 255.0
        target = target.to(device, non_blocking=True)
        logits = model(x)
        losses = criterion(logits, target)
        bs = x.size(0)
        loss_meter.update(float(losses["loss"].item()), bs)
        focal_meter.update(float(losses["focal"].item()), bs)
        tversky_meter.update(float(losses["tversky"].item()), bs)

        pred = crop_logits_for_eval(logits).argmax(dim=1)
        gt = target.argmax(dim=1)
        conf = update_confusion(conf, pred, gt)

    out = lane_metrics_from_confusion(conf)
    out.update({"loss": loss_meter.avg, "focal": focal_meter.avg, "tversky": tversky_meter.avg})
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: FocalTverskyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema: Optional[ModelEMA],
    epoch: int,
    max_epochs: int,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    focal_meter = AverageMeter()
    tversky_meter = AverageMeter()
    pbar = tqdm(loader, desc=f"train {epoch + 1}/{max_epochs}", dynamic_ncols=True)

    for _, x, target in pbar:
        x = x.to(device, non_blocking=True).float() / 255.0
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        losses = criterion(logits, target)
        losses["loss"].backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)

        bs = x.size(0)
        loss_meter.update(float(losses["loss"].item()), bs)
        focal_meter.update(float(losses["focal"].item()), bs)
        tversky_meter.update(float(losses["tversky"].item()), bs)
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    return {"loss": loss_meter.avg, "focal": focal_meter.avg, "tversky": tversky_meter.avg}


def append_csv(path: Path, row: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HSCM-Lane on BDD100K lane-line segmentation.")
    p.add_argument("--data-root", default="BDD100K")
    p.add_argument("--hyp", default="configs/hyperparameters.yaml")
    p.add_argument("--project", default="runs/hscm_lane")
    p.add_argument("--name", default="hscm_lane_s_resnet18")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--val-batch-size", type=int, default=64)
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50", "res18", "res34", "res50"])
    p.add_argument("--fusion", default="sum", choices=["sum", "concat", "gated"])
    p.add_argument("--swin-depth", type=int, default=2)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--no-swin", action="store_true")
    p.add_argument("--no-swin-c1", action="store_true")
    p.add_argument("--no-swin-c2", action="store_true")
    p.add_argument("--no-swin-c3", action="store_true")
    p.add_argument("--no-pretrained-backbone", action="store_true")
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--warmup-start", type=float, default=0.1)
    p.add_argument("--poly-power", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--non-deterministic", action="store_true")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--focal-weight", type=float, default=1.0)
    p.add_argument("--tversky-weight", type=float, default=1.0)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    generator = set_seed(args.seed, deterministic=not args.non_deterministic)

    with open(args.hyp, "r", encoding="utf-8") as f:
        hyp = yaml.safe_load(f)
    if args.lr is not None:
        hyp["lr"] = float(args.lr)

    train_ds = BDD100KLaneDataset(args.data_root, split="train", hyp=hyp, augment=True)
    val_ds = BDD100KLaneDataset(args.data_root, split="val", hyp=hyp, augment=False)
    if args.debug:
        train_ds.names = train_ds.names[:64]
        val_ds.names = val_ds.names[:32]
        args.max_epochs = min(args.max_epochs, 2)
        args.workers = 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )

    model = build_model(args).to(device)
    criterion = make_loss(hyp, args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hyp.get("lr", 5e-4)),
        betas=(float(hyp.get("momentum", 0.9)), 0.999),
        eps=float(hyp.get("eps", 1e-8)),
        weight_decay=float(hyp.get("weight_decay", 0.01)),
    )
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    run_dir = Path(args.project) / args.name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"args": vars(args), "hyp": hyp}, f, sort_keys=False)

    best_iou = -1.0
    base_lr = float(hyp.get("lr", 5e-4))
    start = time.perf_counter()

    for epoch in range(args.max_epochs):
        lr = compute_lr(epoch, base_lr, args.max_epochs, args.warmup_epochs, args.warmup_start, args.poly_power)
        apply_lr(optimizer, lr)
        train_out = train_one_epoch(model, train_loader, criterion, optimizer, device, ema, epoch, args.max_epochs)
        eval_model = ema.ema if ema is not None else model
        val_out = evaluate(eval_model, val_loader, criterion, device)

        is_best = val_out["iou_lane"] > best_iou
        if is_best:
            best_iou = val_out["iou_lane"]
            torch.save(eval_model.state_dict(), weights_dir / "best.pth")
        torch.save(eval_model.state_dict(), weights_dir / f"epoch_{epoch + 1:03d}.pth")

        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_out["loss"],
            "train_focal": train_out["focal"],
            "train_tversky": train_out["tversky"],
            "val_loss": val_out["loss"],
            "val_focal": val_out["focal"],
            "val_tversky": val_out["tversky"],
            "iou_lane": val_out["iou_lane"],
            "recall_lane": val_out["recall_lane"],
            "balanced_accuracy": val_out["balanced_accuracy"],
            "elapsed_s": time.perf_counter() - start,
        }
        append_csv(run_dir / "metrics.csv", row)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"Epoch {epoch + 1:03d}/{args.max_epochs:03d} | "
            f"IoU={val_out['iou_lane'] * 100:.2f}% | "
            f"Recall={val_out['recall_lane'] * 100:.2f}% | "
            f"BalAcc={val_out['balanced_accuracy'] * 100:.2f}% | "
            f"best={best_iou * 100:.2f}%"
        )

    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
