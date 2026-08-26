from __future__ import annotations

from typing import Dict

import torch


def update_confusion(conf: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    mask = (gt >= 0) & (gt < num_classes)
    idx = num_classes * gt[mask].to(torch.int64) + pred[mask].to(torch.int64)
    return conf + torch.bincount(idx, minlength=num_classes ** 2).view(num_classes, num_classes)


def lane_metrics_from_confusion(conf: torch.Tensor) -> Dict[str, float]:
    c = conf.to(torch.float64)
    tn, fp, fn, tp = c[0, 0], c[0, 1], c[1, 0], c[1, 1]
    iou = tp / (tp + fp + fn + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "iou_lane": float(iou.item()),
        "recall_lane": float(recall.item()),
        "precision_lane": float(precision.item()),
        "f1_lane": float(f1.item()),
        "balanced_accuracy": float(balanced_accuracy.item()),
        "tp": int(tp.item()),
        "fp": int(fp.item()),
        "fn": int(fn.item()),
        "tn": int(tn.item()),
    }


def crop_logits_for_eval(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, :, 12:-12, :].contiguous()
