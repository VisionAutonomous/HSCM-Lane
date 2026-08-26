from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: Optional[float] = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    target = target.type_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    loss = (1.0 - pt).pow(float(gamma)) * bce
    if alpha is not None:
        loss = loss * (float(alpha) * target + (1.0 - float(alpha)) * (1.0 - target))
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def multiclass_focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    num_classes = logits.size(1)
    loss = 0.0
    for cls in range(num_classes):
        cls_target = (target == cls).long()
        cls_logits = logits[:, cls, ...]
        loss = loss + focal_loss_with_logits(cls_logits, cls_target, alpha=alpha, gamma=gamma)
    return loss


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.9,
    beta: float = 0.1,
    gamma: float = 1.3333,
    eps: float = 1e-7,
) -> torch.Tensor:
    num_classes = logits.size(1)
    prob = torch.softmax(logits, dim=1)
    target_flat = target.view(target.size(0), -1)
    prob_flat = prob.view(prob.size(0), num_classes, -1)
    one_hot = F.one_hot(target_flat.long(), num_classes).permute(0, 2, 1).type_as(prob_flat)

    dims = (0, 2)
    tp = torch.sum(prob_flat * one_hot, dim=dims)
    fp = torch.sum(prob_flat * (1.0 - one_hot), dim=dims)
    fn = torch.sum((1.0 - prob_flat) * one_hot, dim=dims)
    score = tp / (tp + float(alpha) * fp + float(beta) * fn + eps)
    loss = 1.0 - score
    present = one_hot.sum(dim=dims) > 0
    loss = loss * present.to(loss.dtype)
    return loss.mean().pow(float(gamma))


class FocalTverskyLoss(nn.Module):
    """Paper-facing HSCM-Lane training objective.

    The model produces logits with height 384. The BDD100K lane mask target has
    height 360. The loss therefore crops 12 pixels from the top and bottom of
    the logits before computing Focal and Tversky terms.
    """

    def __init__(
        self,
        tversky_alpha: float = 0.9,
        tversky_gamma: float = 1.3333,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        focal_weight: float = 1.0,
        tversky_weight: float = 1.0,
    ):
        super().__init__()
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = 1.0 - float(tversky_alpha)
        self.tversky_gamma = float(tversky_gamma)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.focal_weight = float(focal_weight)
        self.tversky_weight = float(tversky_weight)

    def forward(self, logits: torch.Tensor, target_one_hot: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits_crop = logits[:, :, 12:-12, :].contiguous()
        target = target_one_hot.argmax(dim=1).to(logits_crop.device)
        focal = multiclass_focal_loss(logits_crop, target, alpha=self.focal_alpha, gamma=self.focal_gamma)
        tv = tversky_loss(
            logits_crop,
            target,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta,
            gamma=self.tversky_gamma,
        )
        total = self.focal_weight * focal + self.tversky_weight * tv
        return {"loss": total, "focal": focal.detach(), "tversky": tv.detach()}
