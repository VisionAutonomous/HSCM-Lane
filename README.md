# HSCM-Lane

This repository provides the core implementation of **HSCM-Lane**, the lane-line segmentation model described in the accompanying manuscript. The code is intended to support model inspection, training, validation, and reviewer-side reproducibility of the main experimental setting.

## Overview

HSCM-Lane is a compact deep-learning framework for pixel-wise lane-line segmentation in road-scene images. The repository focuses on the final manuscript model and the minimum set of scripts required to reproduce the principal experiments.

The release includes the model implementation, dataset interfaces, loss functions, evaluation utilities, training and validation scripts, and a minimal usage example. It is designed as a clean research-code release rather than a full development workspace.

## Repository Scope

This repository contains the code required for the manuscript model and its reproducibility workflow. Temporary development files, local environment folders, generated outputs, large datasets, trained checkpoints, and exploratory research variants are not included in order to keep the public release concise and reproducible.

Users should prepare the required datasets separately according to the official dataset instructions and update the dataset paths before running training or validation.

## Main Files

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

The main files serve the following purposes:

* `laneformer.py`: implementation of the proposed HSCM-Lane model.
* `train.py`: training entry point for the manuscript model.
* `val.py`: validation and evaluation entry point.
* `run_with_5_interface_functions.py`: minimal example showing how to call the model through the main interface functions.
* `BDD100K.py`: dataset interface for BDD100K-based experiments.
* `TuSimple.py`: dataset interface for TuSimple-based evaluation or qualitative testing.
* `loss.py`: loss functions used during model training.
* `IOUEval.py`: evaluation utilities for segmentation metrics.

## Environment

The code requires a Python deep-learning environment with PyTorch and common computer-vision libraries. A typical environment includes:

```text
python
torch
torchvision
numpy
opencv-python
pillow
tqdm
```

Users may install the required packages manually or create their own environment according to their CUDA, PyTorch, and GPU configuration.

## Dataset Preparation

The datasets are not redistributed in this repository. Please download and prepare each dataset from its official source and ensure that the local directory structure matches the paths used by the dataset interface files.

Before running the code, update the dataset paths in the corresponding scripts or dataset-interface files.

Expected datasets:

```text
BDD100K/
TuSimple/
```

The repository provides dataset-loading interfaces, but the image files and annotations must be prepared locally by the user.

## Training

After preparing the dataset paths and environment, run:

```bash
python train.py
```

The training script loads the model, prepares the dataset, computes the loss, and updates the network parameters according to the manuscript setting.

If necessary, users should adjust batch size, learning rate, dataset root, checkpoint path, and device settings according to their hardware environment.

## Validation

To evaluate a trained model, run:

```bash
python val.py
```

The validation script computes the segmentation metrics used to assess lane-line prediction quality. Users should verify that the checkpoint path and validation dataset path are correctly configured before running the script.

## Checkpoints

Large trained checkpoint files are not bundled with this lightweight code release. If checkpoints are required for review or reproduction, they can be provided separately through an external storage link or supplementary material.

When using a checkpoint, please update the checkpoint path in the validation or inference script before execution.

## Reproducibility Notes

To reproduce the manuscript experiments, users should keep the model configuration, input resolution, dataset split, training protocol, and evaluation script consistent with the manuscript. Differences in library versions, CUDA versions, GPU hardware, random seeds, and data preprocessing may lead to small numerical variations.

The repository is therefore intended to provide transparent implementation details and a practical reproduction path, while the exact experimental protocol should be interpreted together with the manuscript.

## Citation

If this repository is useful for your research, please cite the accompanying manuscript.

```bibtex
@article{hscmlane,
  title   = {HSCM-Lane: [Manuscript Title]},
  author  = {[Author Names]},
  journal = {[Journal Name]},
  year    = {[Year]}
}
```

## License

Please use this repository for academic research and manuscript-review purposes. Dataset usage must follow the licenses and terms of the original dataset providers.

## Contact

For questions about the implementation or reproducibility workflow, please contact the corresponding author of the manuscript.
