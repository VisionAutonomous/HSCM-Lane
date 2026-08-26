# HSCM-Lane

Official implementation of **HSCM-Lane: A ResNet-Based Encoder-Decoder Architecture with Multi-Level Shifted-Window Context Modeling for Pixel-Wise Lane Segmentation**.

**Authors:** Toai Ton Quang, Hien Vu Thanh, Nguyen Vu Thanh, and Tuong Le

## Overview

HSCM-Lane is a compact deep-learning framework for pixel-wise binary lane segmentation in road-scene images. The model is designed to improve lane-mask segmentation under challenging conditions such as sparse lane markings, thin structures, low contrast, occlusion, discontinuous markings, shadows, road-surface reflections, and adverse illumination.

The architecture follows a ResNet-based encoder-decoder design enhanced by a Hierarchical Swin Context Module (HSCM). The HSCM module performs multi-level shifted-window context modeling at intermediate encoder feature levels, allowing the network to combine the local inductive bias of convolutional features with attention-based spatial context modeling.

This repository provides the core implementation required to inspect, train, validate, and reproduce the main HSCM-Lane experimental workflow.

## Method Summary

HSCM-Lane consists of three main components:

1. **ResNet-based encoder**

   The encoder extracts hierarchical visual features from the input road-scene image. The released implementation focuses on the manuscript model and its ResNet-based variants.

2. **Hierarchical Swin Context Module**

   HSCM is inserted into intermediate encoder feature levels to enrich spatial representations through shifted-window attention. This design allows context-enhanced features to be propagated through subsequent residual stages and reused by the decoder.

3. **Lightweight multi-level decoder**

   The decoder progressively recovers spatial resolution and reconstructs the final binary lane mask. Multi-level feature fusion is used to combine deep semantic information with shallower geometric detail.

## Model Variants

The same HSCM-Lane architecture can be evaluated with different ResNet backbone capacities:

| Variant | Backbone | Purpose |
|---|---|---|
| HSCM-Lane-S | ResNet18 | Main compact model and component-ablation reference |
| HSCM-Lane-M | ResNet34 | Medium-capacity backbone comparison |
| HSCM-Lane-L | ResNet50 | Large-capacity backbone comparison |

The variants differ in backbone capacity, while the HSCM placement, shifted-window context design, decoder strategy, and training objective are kept consistent.

## Repository Scope

This repository is a focused research-code release. It contains the implementation files needed for the proposed model, dataset loading, training, validation, loss computation, metric evaluationn.

The release does not include large datasets, trained checkpoints, generated outputs, temporary development files, local environment folders, or exploratory research variants. These materials are intentionally excluded to keep the repository concise, inspectable, and reproducible.

Users should download the required datasets from their official sources and configure the local dataset paths before running training or validation.

## Repository Structure

```text
.
├── laneformer.py
├── train.py
├── val.py
├── run_with_5_interface_functions.py
├── BDD100K.py
├── TuSimple.py
├── loss.py
├── IOUEval.py
└── README.md
```

## Main Files

| File | Description |
|---|---|
| `laneformer.py` | Core implementation of the HSCM-Lane model |
| `train.py` | Training script for the lane-segmentation model |
| `val.py` | Validation and metric-evaluation script |
| `run_with_5_interface_functions.py` | Minimal example demonstrating the main model interface functions |
| `BDD100K.py` | Dataset interface for BDD100K-based experiments |
| `TuSimple.py` | Dataset interface for TuSimple-based evaluation or qualitative testing |
| `loss.py` | Loss functions used for model optimization |
| `IOUEval.py` | Intersection-over-union and segmentation-metric utilities |

## Requirements

The code requires a Python environment with PyTorch and common computer-vision packages.

A typical environment includes:

```text
python >= 3.8
torch
torchvision
numpy
opencv-python
pillow
tqdm
```

The exact PyTorch and CUDA versions should be selected according to the user’s GPU and operating system.

## Dataset Preparation

The datasets are not redistributed in this repository. Please obtain them from their official sources and follow the corresponding dataset licenses and terms of use.

Expected datasets include:

```text
BDD100K/
TuSimple/
```

Before running the scripts, update the dataset root paths in the relevant dataset-interface files or training and validation scripts.

A typical local structure may follow:

```text
datasets/
├── BDD100K/
│   ├── images/
│   └── labels/
└── TuSimple/
    ├── clips/
    └── label_data/
```

The exact structure may be adjusted according to the local preprocessing pipeline, provided that the dataset interface files are updated consistently.

## Training

After preparing the environment and dataset paths, run:

```bash
python train.py
```

The training script initializes the model, loads the dataset, computes the training objective, and updates the network parameters.

Before training, check the following settings:

- dataset root path
- input image size
- batch size
- learning rate
- number of epochs
- checkpoint output path
- GPU or CPU device setting

## Validation

To evaluate a trained model, run:

```bash
python val.py
```

Before validation, check the following settings:

- validation dataset path
- checkpoint path
- input image size
- number of classes
- metric configuration
- output directory, if prediction masks or logs are saved

The validation script computes the segmentation metrics used to evaluate lane-mask prediction
