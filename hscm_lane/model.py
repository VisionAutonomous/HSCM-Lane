from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)


# =========================
# 0) Utils: Assert contracts
# =========================
def _assert_rank4(x: torch.Tensor, name: str) -> None:
    assert isinstance(x, torch.Tensor), f"{name} must be a torch.Tensor"
    assert x.dim() == 4, f"{name} must be rank-4 BCHW, got {x.dim()} dims"


def _assert_bchw(
    x: torch.Tensor,
    name: str,
    c: Optional[int] = None,
    h: Optional[int] = None,
    w: Optional[int] = None,
) -> None:
    _assert_rank4(x, name)
    B, C, H, W = x.shape
    if c is not None:
        assert C == c, f"{name}.C expected {c}, got {C}"
    if h is not None:
        assert H == h, f"{name}.H expected {h}, got {H}"
    if w is not None:
        assert W == w, f"{name}.W expected {w}, got {W}"


def _assert_same_spatial(a: torch.Tensor, b: torch.Tensor, name_a: str, name_b: str) -> None:
    _assert_rank4(a, name_a)
    _assert_rank4(b, name_b)
    assert a.shape[2:] == b.shape[2:], (
        f"Spatial mismatch: {name_a}.HW={a.shape[2:]} vs {name_b}.HW={b.shape[2:]}"
    )


# =========================
# 1) Basic blocks
# =========================
class ConvBNAct(nn.Module):
    """
    Contract:
      Input : (B, Cin, H, W)
      Output: (B, Cout, H_out, W_out)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        groups: int = 1,
        act: nn.Module = nn.SiLU(inplace=True),
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _assert_bchw(x, "ConvBNAct.x", c=self.in_ch)
        y = self.act(self.bn(self.conv(x)))
        _assert_bchw(y, "ConvBNAct.y", c=self.out_ch)
        return y

# =========================
# 3) Swin-like Window Attention for 2D feature maps
# =========================
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    x: (B, H, W, C)
    return: (num_windows*B, window_size, window_size, C)
    """
    assert x.dim() == 4, "window_partition expects (B,H,W,C)"
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, (
        f"H,W must be divisible by window_size={window_size}, got H={H}, W={W}"
    )
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    windows: (num_windows*B, window_size, window_size, C)
    return: (B, H, W, C)
    """
    assert windows.dim() == 4, "window_reverse expects (nW*B, ws, ws, C)"
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def build_swin_attn_mask(H: int, W: int, window_size: int, shift_size: int, device: torch.device) -> torch.Tensor:
    """
    Standard Swin attention mask for shifted windows.
    Return: (num_windows, window_tokens, window_tokens) with 0 or -100.
    """
    assert 0 <= shift_size < window_size
    if shift_size == 0:
        # no mask needed
        return torch.zeros((0,), device=device)

    img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )

    cnt = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # (nW, ws, ws, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size)  # (nW, L)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, L, L)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class WindowAttention(nn.Module):
    """
    Window-based multi-head self attention.

    Contract:
      Input : (BnW, L, C)
      Output: (BnW, L, C)
    """
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.dim() == 3, f"WindowAttention expects (BnW, L, C), got {x.shape}"
        BnW, L, C = x.shape
        assert C == self.dim, f"Channel mismatch: expected {self.dim}, got {C}"

        qkv = self.qkv(x)  # (BnW, L, 3C)
        qkv = qkv.view(BnW, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (BnW, heads, L, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (BnW, heads, L, L)

        if attn_mask is not None and attn_mask.numel() > 0:
            # attn_mask: (nW, L, L). We need to broadcast per batch.
            nW = attn_mask.shape[0]
            assert (BnW % nW) == 0, "BnW must be multiple of nW for mask broadcasting"
            attn = attn.view(-1, nW, self.num_heads, L, L)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, L, L)

        attn = attn.softmax(dim=-1)
        out = attn @ v  # (BnW, heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(BnW, L, C)
        out = self.proj(out)
        return out


class SwinMLP(nn.Module):
    """
    MLP used inside Transformer block.

    Contract:
      Input : (B, N, C)
      Output: (B, N, C)
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        return self.fc2(self.act(self.fc1(x)))


class SwinBlock2D(nn.Module):
    """
    Minimal Swin block (window attention + optional shift) for 2D feature maps.

    Contract:
      Input : (B, C, H, W) with fixed (H,W) given at init.
      Output: (B, C, H, W)
    """
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        H, W = input_resolution
        assert 0 <= shift_size < window_size
        assert H % window_size == 0 and W % window_size == 0, (
            f"Resolution {(H,W)} must be divisible by window_size={window_size}"
        )

        self.dim = dim
        self.H = H
        self.W = W
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMLP(dim=dim, mlp_ratio=mlp_ratio)

        # register a placeholder buffer; will be overwritten on first forward (device-correct)
        self.register_buffer("attn_mask", torch.zeros((0,)), persistent=False)

    def _get_attn_mask(self, device: torch.device) -> torch.Tensor:
        if self.shift_size == 0:
            return torch.zeros((0,), device=device)
        # if buffer is empty or on different device, rebuild
        if self.attn_mask.numel() == 0 or self.attn_mask.device != device:
            m = build_swin_attn_mask(self.H, self.W, self.window_size, self.shift_size, device=device)
            self.attn_mask = m
        return self.attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _assert_bchw(x, "SwinBlock2D.x", c=self.dim, h=self.H, w=self.W)
        B, C, H, W = x.shape

        shortcut = x
        # (B,C,H,W) -> (B,H,W,C)
        x = x.permute(0, 2, 3, 1).contiguous()

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows
        x_windows = window_partition(x, self.window_size)  # (BnW, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (BnW, L, C)

        # LN + attention
        x_windows = self.norm1(x_windows)
        attn_mask = self._get_attn_mask(x_windows.device)
        attn_out = self.attn(x_windows, attn_mask=attn_mask)
        x_windows = x_windows + attn_out

        # LN + MLP
        x_windows = x_windows + self.mlp(self.norm2(x_windows))

        # merge windows
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_windows, self.window_size, H, W)  # (B,H,W,C)

        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # back to (B,C,H,W)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = x + shortcut
        _assert_bchw(x, "SwinBlock2D.y", c=self.dim, h=self.H, w=self.W)
        return x


class SwinStage2D(nn.Module):
    """
    A stage = multiple Swin blocks with alternating shift (0, ws/2, 0, ws/2, ...)

    Contract:
      Input/Output: (B, C, H, W)
    """
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int = 2,
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert depth >= 1
        blocks = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else (window_size // 2)
            blocks.append(
                SwinBlock2D(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def _resnet_stage_out_channels(layer: nn.Sequential) -> int:
    """Infer output channels of a ResNet stage (layer1/layer2/layer3)."""
    assert isinstance(layer, nn.Sequential) and len(layer) > 0
    blk = layer[-1]
    # Bottleneck has conv3; BasicBlock has conv2
    if hasattr(blk, "conv3"):
        return int(blk.conv3.out_channels)
    if hasattr(blk, "conv2"):
        return int(blk.conv2.out_channels)
    raise AssertionError("Unsupported ResNet block type for channel inference")



def _build_resnet(backbone: str, pretrained: bool) -> nn.Module:
    """Build the ResNet backbone variants used for HSCM-Lane-S/M/L."""
    b = str(backbone).lower().strip()
    if b in ("res18", "resnet18"):
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)
    if b in ("res34", "resnet34"):
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        return resnet34(weights=weights)
    if b in ("res50", "resnet50"):
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        return resnet50(weights=weights)
    raise AssertionError("Unknown backbone. Use one of: resnet18, resnet34, resnet50.")


class ResNetContextPath(nn.Module):
    """
    Context path for the ResNet variants used by HSCM-Lane.

    We always expose 4 tensors for downstream HSCM-Lane:
      - stem: (B, 64, 192, 320)
      - c1  : (B, 64,  96, 160)  (1/4)
      - c2  : (B, 128, 48,  80)  (1/8)
      - c3  : (B, 256, 24,  40)  (1/16)

    Default behavior (use_swin=True for all stages) is identical to the original implementation:
      - ResNet18/34: SwinStage2D runs directly on {c1,c2,c3} and the Swin-updated feature is fed
        into the next ResNet stage.
      - ResNet50: features are projected to {64,128,256}, SwinStage2D runs on projected features,
        then projected features are injected back to raw channels (residual) before feeding the next stage.

    Ablation behavior (use_swin=False or stage-specific disables):
      - Swin compute is removed while keeping tensor shapes for the decoder unchanged.
      - For ResNet50, the residual injection is disabled when the corresponding Swin stage is disabled,
        so the backbone becomes a pure ResNet forward (layer1->layer2->layer3) for the next-stage input.
    """

    def __init__(
        self,
        input_hw: Tuple[int, int] = (384, 640),
        backbone: str = "resnet18",
        swin_depth: int = 2,
        window_size: int = 8,
        heads_c1: int = 4,
        heads_c2: int = 8,
        heads_c3: int = 8,
        pretrained: bool = True,
        out_c1: int = 64,
        out_c2: int = 128,
        out_c3: int = 256,
        # Ablation toggles
        use_swin: bool = True,
        use_swin_c1: bool = True,
        use_swin_c2: bool = True,
        use_swin_c3: bool = True,
    ):
        super().__init__()
        H, W = input_hw
        assert (H, W) == (384, 640), "This implementation is fixed to input (384,640) by contract."

        # build backbone
        net = _build_resnet(backbone=backbone, pretrained=pretrained)

        # shared stem from pretrained ResNet
        self.conv1 = net.conv1  # 7x7 s2
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        # stages (we only use up to layer3 => 1/16)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

        # raw channels inferred from backbone (ResNet18/34: 64/128/256, ResNet50: 256/512/1024)
        self.c1_raw_ch = _resnet_stage_out_channels(self.layer1)
        self.c2_raw_ch = _resnet_stage_out_channels(self.layer2)
        self.c3_raw_ch = _resnet_stage_out_channels(self.layer3)

        # exported channels (fixed to keep decoder unchanged)
        self.c1_ch = int(out_c1)
        self.c2_ch = int(out_c2)
        self.c3_ch = int(out_c3)

        # adapters for ResNet50 (or any backbone whose raw channels != exported channels)
        self.c1_down = self._make_adapter(self.c1_raw_ch, self.c1_ch)
        self.c2_down = self._make_adapter(self.c2_raw_ch, self.c2_ch)
        self.c3_down = self._make_adapter(self.c3_raw_ch, self.c3_ch)

        self.c1_up = self._make_adapter(self.c1_ch, self.c1_raw_ch)
        self.c2_up = self._make_adapter(self.c2_ch, self.c2_raw_ch)
        # c3_up is not needed because we don't feed to layer4 (but keep for completeness)
        self.c3_up = self._make_adapter(self.c3_ch, self.c3_raw_ch)

        # Swin stage enable flags
        self.use_swin = bool(use_swin)
        self.use_swin_c1 = bool(self.use_swin and use_swin_c1)
        self.use_swin_c2 = bool(self.use_swin and use_swin_c2)
        self.use_swin_c3 = bool(self.use_swin and use_swin_c3)

        # resolutions after layers given input 384x640
        # stem: 192x320, c1: 96x160, c2: 48x80, c3: 24x40
        self.swin_c1 = (
            SwinStage2D(dim=self.c1_ch, input_resolution=(96, 160), depth=swin_depth, num_heads=heads_c1, window_size=window_size)
            if self.use_swin_c1 else nn.Identity()
        )
        self.swin_c2 = (
            SwinStage2D(dim=self.c2_ch, input_resolution=(48, 80), depth=swin_depth, num_heads=heads_c2, window_size=window_size)
            if self.use_swin_c2 else nn.Identity()
        )
        self.swin_c3 = (
            SwinStage2D(dim=self.c3_ch, input_resolution=(24, 40), depth=swin_depth, num_heads=heads_c3, window_size=window_size)
            if self.use_swin_c3 else nn.Identity()
        )

    @staticmethod
    def _make_adapter(in_ch: int, out_ch: int) -> nn.Module:
        if int(in_ch) == int(out_ch):
            return nn.Identity()
        return nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(out_ch)),
        )

    def forward(self, x: torch.Tensor):
        _assert_bchw(x, "ContextPath.x", c=3, h=384, w=640)

        stem = self.relu(self.bn1(self.conv1(x)))  # (B,64,192,320)
        _assert_bchw(stem, "ContextPath.stem", c=64, h=192, w=320)

        y = self.maxpool(stem)  # (B,64,96,160)
        _assert_bchw(y, "ContextPath.after_maxpool", c=64, h=96, w=160)

        # ---------- stage c1 (1/4) ----------
        c1_raw = self.layer1(y)  # (B,c1_raw_ch,96,160)
        _assert_bchw(c1_raw, "ContextPath.c1_raw", c=self.c1_raw_ch, h=96, w=160)

        if self.c1_raw_ch == self.c1_ch:
            c1_proj = c1_raw
            c1 = self.swin_c1(c1_proj) if self.use_swin_c1 else c1_proj
            # ResNet18/34: feed Swin-updated feature into next stage (same as original)
            c1_for_next = c1
        else:
            # ResNet50: always export projected feature to decoder
            c1_proj = self.c1_down(c1_raw)  # -> (B,64,96,160)
            _assert_bchw(c1_proj, "ContextPath.c1_down", c=self.c1_ch, h=96, w=160)

            if self.use_swin_c1:
                c1 = self.swin_c1(c1_proj)
                # inject Swin-updated projected feature back to raw channels for next stage
                c1_for_next = c1_raw + self.c1_up(c1)
            else:
                c1 = c1_proj
                # no injection if Swin is disabled (pure ResNet forward for next stage)
                c1_for_next = c1_raw

        _assert_bchw(c1, "ContextPath.c1", c=self.c1_ch, h=96, w=160)

        # ---------- stage c2 (1/8) ----------
        c2_raw = self.layer2(c1_for_next)  # (B,c2_raw_ch,48,80)
        _assert_bchw(c2_raw, "ContextPath.c2_raw", c=self.c2_raw_ch, h=48, w=80)

        if self.c2_raw_ch == self.c2_ch:
            c2_proj = c2_raw
            c2 = self.swin_c2(c2_proj) if self.use_swin_c2 else c2_proj
            c2_for_next = c2
        else:
            c2_proj = self.c2_down(c2_raw)  # -> (B,128,48,80)
            _assert_bchw(c2_proj, "ContextPath.c2_down", c=self.c2_ch, h=48, w=80)

            if self.use_swin_c2:
                c2 = self.swin_c2(c2_proj)
                c2_for_next = c2_raw + self.c2_up(c2)
            else:
                c2 = c2_proj
                c2_for_next = c2_raw

        _assert_bchw(c2, "ContextPath.c2", c=self.c2_ch, h=48, w=80)

        # ---------- stage c3 (1/16) ----------
        c3_raw = self.layer3(c2_for_next)  # (B,c3_raw_ch,24,40)
        _assert_bchw(c3_raw, "ContextPath.c3_raw", c=self.c3_raw_ch, h=24, w=40)

        c3_proj = c3_raw if (self.c3_raw_ch == self.c3_ch) else self.c3_down(c3_raw)
        if self.c3_raw_ch != self.c3_ch:
            _assert_bchw(c3_proj, "ContextPath.c3_down", c=self.c3_ch, h=24, w=40)

        c3 = self.swin_c3(c3_proj) if self.use_swin_c3 else c3_proj
        _assert_bchw(c3, "ContextPath.c3", c=self.c3_ch, h=24, w=40)

        return stem, c1, c2, c3

class GatedFusion(nn.Module):
    """
    Fuse two same-resolution features (A,B) into one.

    Contract:
      A: (B, Ca, H, W)
      B: (B, Cb, H, W)
      Out: (B, Cout, H, W)
    """
    def __init__(self, ca: int, cb: int, out_ch: int):
        super().__init__()
        self.ca = ca
        self.cb = cb
        self.out_ch = out_ch

        self.proj_a = nn.Conv2d(ca, out_ch, kernel_size=1, bias=False)
        self.proj_b = nn.Conv2d(cb, out_ch, kernel_size=1, bias=False)

        # spatial gate (position-wise)
        self.gate = nn.Conv2d(out_ch * 2, 1, kernel_size=1, bias=True)

        self.refine = ConvBNAct(out_ch, out_ch, k=3, s=1, p=1, act=nn.SiLU(inplace=True))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        _assert_bchw(a, "GatedFusion.a", c=self.ca)
        _assert_bchw(b, "GatedFusion.b", c=self.cb)
        _assert_same_spatial(a, b, "GatedFusion.a", "GatedFusion.b")

        a1 = self.proj_a(a)
        b1 = self.proj_b(b)

        gate = torch.sigmoid(self.gate(torch.cat([a1, b1], dim=1)))  # (B,1,H,W)
        y = gate * a1 + (1.0 - gate) * b1
        y = self.refine(y)
        _assert_bchw(y, "GatedFusion.y", c=self.out_ch, h=a.shape[2], w=a.shape[3])
        return y


class SumFusion(nn.Module):
    """
    Fixed (static) fusion of two same-resolution features.

    This is an ablation variant of GatedFusion where the spatial gate is removed and replaced
    by a constant mixing ratio (default: 0.5). Shapes & decoder topology remain unchanged.

    Contract:
      A: (B, Ca, H, W)
      B: (B, Cb, H, W)
      Out: (B, Cout, H, W)
    """
    def __init__(self, ca: int, cb: int, out_ch: int, alpha: float = 0.5):
        super().__init__()
        self.ca = ca
        self.cb = cb
        self.out_ch = out_ch
        self.alpha = float(alpha)

        self.proj_a = nn.Conv2d(ca, out_ch, kernel_size=1, bias=False)
        self.proj_b = nn.Conv2d(cb, out_ch, kernel_size=1, bias=False)

        # keep same refine block as GatedFusion for fair comparison
        self.refine = ConvBNAct(out_ch, out_ch, k=3, s=1, p=1, act=nn.SiLU(inplace=True))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        _assert_bchw(a, "SumFusion.a", c=self.ca)
        _assert_bchw(b, "SumFusion.b", c=self.cb)
        _assert_same_spatial(a, b, "SumFusion.a", "SumFusion.b")

        a1 = self.proj_a(a)
        b1 = self.proj_b(b)

        # Constant convex combination (alpha*a1 + (1-alpha)*b1)
        y = self.alpha * a1 + (1.0 - self.alpha) * b1
        y = self.refine(y)
        _assert_bchw(y, "SumFusion.y", c=self.out_ch, h=a.shape[2], w=a.shape[3])
        return y


class ConcatFusion(nn.Module):
    """
    Standard concatenation-based fusion.

    Contract:
      A:   (B, Ca, H, W)
      B:   (B, Cb, H, W)
      Out: (B, Cout, H, W)
    """
    def __init__(self, ca: int, cb: int, out_ch: int):
        super().__init__()
        self.ca = ca
        self.cb = cb
        self.out_ch = out_ch

        self.proj = ConvBNAct(
            ca + cb,
            out_ch,
            k=1,
            s=1,
            p=0,
            act=nn.SiLU(inplace=True),
        )

        self.refine = ConvBNAct(
            out_ch,
            out_ch,
            k=3,
            s=1,
            p=1,
            act=nn.SiLU(inplace=True),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        _assert_bchw(a, "ConcatFusion.a", c=self.ca)
        _assert_bchw(b, "ConcatFusion.b", c=self.cb)
        _assert_same_spatial(a, b, "ConcatFusion.a", "ConcatFusion.b")

        y = torch.cat([a, b], dim=1)
        y = self.proj(y)
        y = self.refine(y)

        _assert_bchw(y, "ConcatFusion.y", c=self.out_ch, h=a.shape[2], w=a.shape[3])
        return y


class UpFuseBlock(nn.Module):
    """
    Upsample low-res feature by 2x then fuse with skip feature at higher resolution.

    Contract:
      x   : (B, Cin, H, W)
      skip: (B, Cskip, 2H, 2W)
      out : (B, Cout, 2H, 2W)
    """
    def __init__(self, cin: int, cskip: int, cout: int, fusion: str = "gated"):
        super().__init__()
        self.cin = cin
        self.cskip = cskip
        self.cout = cout

        f = str(fusion).lower().strip()

        if f == "gated":
            self.fuse = GatedFusion(ca=cin, cb=cskip, out_ch=cout)
        elif f in ("sum", "avg", "static"):
            # "sum" is implemented as a fixed convex combination (default alpha=0.5)
            self.fuse = SumFusion(ca=cin, cb=cskip, out_ch=cout, alpha=0.5)
        elif f in ("concat", "cat"):
            self.fuse = ConcatFusion(ca=cin, cb=cskip, out_ch=cout)
        else:
            raise AssertionError(f"Unknown fusion type: {fusion}. Choose from: gated | sum | concat")

        # if f == "gated":
        #     self.fuse = GatedFusion(ca=cin, cb=cskip, out_ch=cout)
        # elif f in ("sum", "avg", "static"):
        #     # "sum" is implemented as a fixed convex combination (default alpha=0.5)
        #     self.fuse = SumFusion(ca=cin, cb=cskip, out_ch=cout, alpha=0.5)
        # else:
        #     raise AssertionError(f"Unknown fusion type: {fusion}. Choose from: gated | sum")

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        _assert_bchw(x, "UpFuseBlock.x", c=self.cin)
        _assert_bchw(skip, "UpFuseBlock.skip", c=self.cskip)

        x_up = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        _assert_same_spatial(x_up, skip, "UpFuseBlock.x_up", "UpFuseBlock.skip")

        y = self.fuse(x_up, skip)
        _assert_bchw(y, "UpFuseBlock.y", c=self.cout, h=skip.shape[2], w=skip.shape[3])
        return y

class SegmentationHead(nn.Module):
    """
    Small segmentation head.

    Contract:
      Input : (B, Cin, H, W)
      Output: (B, num_classes, H, W)
    """
    def __init__(self, cin: int, num_classes: int = 2, mid_ch: int = 32):
        super().__init__()
        self.cin = cin
        self.num_classes = num_classes
        self.conv = ConvBNAct(cin, mid_ch, k=3, s=1, p=1, act=nn.SiLU(inplace=True))
        self.logits = nn.Conv2d(mid_ch, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _assert_bchw(x, "SegHead.x", c=self.cin)
        y = self.logits(self.conv(x))
        _assert_bchw(y, "SegHead.y", c=self.num_classes, h=x.shape[2], w=x.shape[3])
        return y


# =========================
# 7) Full Model: HSCM-Lane
# =========================



class HSCMLane(nn.Module):
    """HSCM-Lane for pixel-wise binary lane segmentation.

    This class matches the paper-facing model: a ResNet encoder, HSCM at the
    1/4, 1/8 and 1/16 encoder levels, a lightweight multi-level decoder, and a
    two-class segmentation head.

    Input:
        Tensor of shape (B, 3, 384, 640).
    Output:
        Logits of shape (B, 2, 384, 640), where channel 0 is background and
        channel 1 is lane marking.
    """

    def __init__(
        self,
        input_hw: Tuple[int, int] = (384, 640),
        backbone: str = "resnet18",
        pretrained_backbone: bool = True,
        swin_depth: int = 2,
        window_size: int = 8,
        use_swin: bool = True,
        use_swin_c1: bool = True,
        use_swin_c2: bool = True,
        use_swin_c3: bool = True,
        fusion: str = "sum",
    ):
        super().__init__()
        H, W = input_hw
        assert (H, W) == (384, 640), "Contract: input must be (384,640)."
        self.in_h = H
        self.in_w = W

        b = str(backbone).lower().strip()
        if b in ("res18", "resnet18"):
            b = "resnet18"
        elif b in ("res34", "resnet34"):
            b = "resnet34"
        elif b in ("res50", "resnet50"):
            b = "resnet50"
        else:
            raise AssertionError("Unknown backbone. Use one of: resnet18, resnet34, resnet50.")

        self.context = ResNetContextPath(
            input_hw=input_hw,
            backbone=b,
            swin_depth=int(swin_depth),
            window_size=int(window_size),
            heads_c1=4,
            heads_c2=8,
            heads_c3=8,
            pretrained=bool(pretrained_backbone),
            use_swin=bool(use_swin),
            use_swin_c1=bool(use_swin and use_swin_c1),
            use_swin_c2=bool(use_swin and use_swin_c2),
            use_swin_c3=bool(use_swin and use_swin_c3),
        )

        self.up_c3_to_c2 = UpFuseBlock(cin=256, cskip=128, cout=128, fusion=fusion)
        self.up_c2_to_c1 = UpFuseBlock(cin=128, cskip=64, cout=64, fusion=fusion)
        self.up_c1_to_d2 = UpFuseBlock(cin=64, cskip=64, cout=64, fusion=fusion)
        self.head = SegmentationHead(cin=64, num_classes=2, mid_ch=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _assert_bchw(x, "HSCMLane.x", c=3, h=self.in_h, w=self.in_w)
        stem, c1, c2, c3 = self.context(x)
        p2 = self.up_c3_to_c2(c3, c2)
        p1 = self.up_c2_to_c1(p2, c1)
        p0 = self.up_c1_to_d2(p1, stem)
        logits_half = self.head(p0)
        logits_full = F.interpolate(logits_half, scale_factor=2.0, mode="bilinear", align_corners=False)
        _assert_bchw(logits_full, "HSCMLane.logits_full", c=2, h=self.in_h, w=self.in_w)
        return logits_full

