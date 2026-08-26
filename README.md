# HSCM-Lane

Minimal paper-facing source-code release for **HSCM-Lane: A ResNet-Based Encoder-Decoder Architecture with Multi-Level Shifted-Window Context Modeling for Pixel-Wise Lane Segmentation**.

This repository contains the author-generated code needed to inspect, train, evaluate, and run inference for the HSCM-Lane binary lane-line segmentation model. It intentionally excludes datasets, trained checkpoints, generated figures, video demos, virtual environments, IDE metadata, GUI-only code, and exploratory backbone variants not used by the paper-facing HSCM-Lane-S/M/L configurations.

## Paper-facing model names

The public model class is named `HSCMLane` to match the manuscript terminology. The three reported configurations are:

| Paper name | Backbone | HSCM levels | Window | Depth | Decoder fusion |
|---|---|---:|---:|---:|---|
| HSCM-Lane-S | ResNet18 | 1/4, 1/8, 1/16 | 8 | 2 | static sum |
| HSCM-Lane-M | ResNet34 | 1/4, 1/8, 1/16 | 8 | 2 | static sum |
| HSCM-Lane-L | ResNet50 | 1/4, 1/8, 1/16 | 8 | 2 | static sum |


## Repository layout

```text
HSCM-Lane/
├── hscm_lane/
│   ├── __init__.py
│   ├── model.py
│   ├── datasets.py
│   ├── losses.py
│   └── metrics.py
├── configs/
│   ├── hyperparameters.yaml
│   ├── hscm_lane_s_resnet18.yaml
│   ├── hscm_lane_m_resnet34.yaml
│   └── hscm_lane_l_resnet50.yaml
├── train.py
├── evaluate_bdd100k.py
├── evaluate_tusimple.py
├── infer_image.py
├── CODE_AVAILABILITY.md
├── requirements.txt
├── LICENSE
└── README.md
```

## What was intentionally removed

The release is restricted to the code required for the manuscript model and reproducibility. The following materials are not included:

```text
.idea/
.venv/
__pycache__/
old/
outputs/
videos/
BDD100K/
TuSimple/
weights/
GUI demo files
RegNet, SegFormer, DINO, DINOv2, and FlashInternImage exploratory backbones
Detail-path and LAM experimental modules
OHEM and boundary-loss experimental training options
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the PyTorch build appropriate for your CUDA version if the generic requirement does not match your system.

## Dataset placement

Datasets are **not redistributed** in this repository. Prepare BDD100K in the following form:

```text
BDD100K/
├── 100k/
│   ├── train/*.jpg
│   └── val/*.jpg
└── bdd_lane_gt/
    ├── train/*.png
    └── val/*.png
```

For TuSimple derived pixel-level evaluation:

```text
TuSimple/
├── clips/...
├── label_data_0313.json
├── label_data_0531.json
└── label_data_0601.json
```

The TuSimple evaluation script converts row-wise lane annotations into binary masks. These results are not official TuSimple benchmark scores.

## Quick sanity check

```bash
python -m py_compile train.py evaluate_bdd100k.py evaluate_tusimple.py infer_image.py hscm_lane/model.py
python - <<'PY'
import torch
from hscm_lane import HSCMLane
model = HSCMLane(backbone="resnet18", pretrained_backbone=False)
x = torch.randn(1, 3, 384, 640)
y = model(x)
print(y.shape)
PY
```

Expected output:

```text
torch.Size([1, 2, 384, 640])
```

## Training

HSCM-Lane-S:

```bash
python train.py \
  --device cuda:0 \
  --data-root /path/to/BDD100K \
  --project runs/hscm_lane \
  --name hscm_lane_s_resnet18 \
  --backbone resnet18 \
  --batch-size 12 \
  --val-batch-size 64 \
  --workers 12 \
  --max-epochs 80 \
  --ema \
  --fusion sum \
  --swin-depth 2 \
  --window-size 8
```

Use `--backbone resnet34` for HSCM-Lane-M and `--backbone resnet50` for HSCM-Lane-L. Checkpoints are saved under `runs/hscm_lane/<name>/weights/`. The selected checkpoint is `best.pth`, chosen by validation lane IoU.

## BDD100K evaluation

```bash
python evaluate_bdd100k.py \
  --device cuda:0 \
  --data-root /path/to/BDD100K \
  --weights runs/hscm_lane/hscm_lane_s_resnet18/weights/best.pth \
  --backbone resnet18 \
  --batch-size 64 \
  --workers 12
```

The script reports lane IoU, lane recall, lane precision, lane F1, balanced accuracy, and the confusion-matrix counts. The model outputs logits of shape `[B, 2, 384, 640]`. Evaluation crops 12 pixels from the top and bottom before comparing against the `[B, 2, 360, 640]` target mask.

## TuSimple derived pixel-level evaluation

```bash
python evaluate_tusimple.py \
  --device cuda:0 \
  --tusimple-root /path/to/TuSimple \
  --weights runs/hscm_lane/hscm_lane_s_resnet18/weights/best.pth \
  --backbone resnet18 \
  --gt-style bdd \
  --batch-size 64 \
  --workers 12
```

## Inference on images

```bash
python infer_image.py \
  --device cuda:0 \
  --source /path/to/images_or_one_image \
  --weights runs/hscm_lane/hscm_lane_s_resnet18/weights/best.pth \
  --backbone resnet18 \
  --out-dir outputs/inference
```

## License

The source code is released under the MIT License. Dataset images and annotations are not included and remain subject to their original licenses and terms.
