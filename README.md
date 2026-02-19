<div align="center">

# Bridging Heterogeneous Medical Datasets via Mixture-of-Specialists Adapters

[![Paper](https://img.shields.io/badge/Paper-MICCAI_2026-blue.svg)](#)
[![Code License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Official PyTorch implementation for unifying 2D and 3D medical imaging tasks via structural disentanglement.**

[Overview](#overview) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#training) 

</div>

---

> **MOSAIC** (Mixture-of-Specialists Adapter for Imaging Classification): a parameter-efficient framework that unifies 18 MedMNIST datasets spanning 6 imaging modalities using a frozen ViT backbone.

## Overview

Medical imaging data is inherently fragmented across modalities (pathology, radiology, CT, ultrasound) and dimensions (2D / 3D), making unified model training challenging. Naïvely mixing heterogeneous datasets triggers **negative transfer** due to feature entanglement between conflicting visual patterns (e.g., texture-dominant pathology vs. shape-dominant radiology).

MOSAIC addresses this via a **Mixture-of-Specialists (MoS)** mechanism with explicit hard routing, which structurally disentangles texture-oriented, shape-oriented, and volumetric feature subspaces within a frozen ViT backbone. Combined with cyclic training and an expert-aware EMA teacher, MOSAIC achieves **84.16% average accuracy** across 18 datasets, matching or exceeding single-task specialists with only **7.9% trainable parameters** (7.40M / 93M).

![MOSAIC Framework](assets/framework.png)
*(If the image does not load, please refer to the [framework.pdf](framework.pdf) included in this repository.)*

## Architecture

```text
Input (2D / 3D)
      │
      ▼
┌──────────────┐
│Modality-Aware│   2D: Conv2d  (224×224 → 196 tokens)
│  Tokenizer   │   3D: Conv3d  (64³     → 512 tokens)
└──────┬───────┘
       ▼
┌──────────────┐
│  Frozen      │   12-layer ViT-B/16 (ImageNet-pretrained)
│  ViT Backbone│   + Parallel MoS Adapter per layer
│              │
│  ┌─────────┐ │   Specialist A (d=64):  Bio-Medical (RGB texture)
│  │   MoS   │ │   Specialist B (d=96):  Radiology (grayscale shape)
│  │ Adapter │ │   Specialist C (d=192): Volumetric (3D spatial)
│  └─────────┘ │
└──────┬───────┘
       ▼
┌──────────────┐
│  Multi-task  │   18 independent classification heads
│    Heads     │
└──────────────┘
```

## Installation
```bash
git clone [https://github.com/xxx/mosaic-medical.git](https://github.com/xxx/mosaic-medical.git)
cd mosaic-medical
pip install -r requirements.txt
```

## Data Preparation

We use the [MedMNIST v2](https://medmnist.com/) benchmark (12 × 2D + 6 × 3D datasets). The datasets are automatically downloaded via the `medmnist` library:

```bash
pip install medmnist
```

## Pre-trained Weights

Download ViT-B/16 ImageNet-21k weights:

```bash
wget [https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) -O vit_base_patch16_224.npz
```

## Training
To train the unified MOSAIC model across all 18 datasets:
```bash
python main.py \
    --data_root ./data \
    --pretrained ./vit_base_patch16_224.npz \
    --adapter_mode v2_moe \
    --freeze_backbone \
    --num_rounds 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --adapter_bottleneck_a 64 \
    --adapter_bottleneck_b 96 \
    --adapter_bottleneck_c 192 \
    --ema_momentum 0.9 \
    --ema_momentum_3d 0.95 \
    --consist_weight 0.1 \
    --early_stopping_patience 5 \
    --seed 42
```

## Evaluation
To evaluate a trained checkpoint on all tasks:
```bash
python testing.py \
    --checkpoint ./output/best_model.pth \
    --data_root ./data \
    --adapter_mode v2_moe
```

## Specialist Routing
MOSAIC automatically routes diverse modalities to corresponding specialists based on intrinsic dimensionality:

| Specialist | Modality              | Bottleneck | Datasets                                                                                    |
| ---------- | --------------------- | ---------- | ------------------------------------------------------------------------------------------- |
| A          | Bio-Medical (RGB)     | 64         | PathMNIST, BloodMNIST, TissueMNIST, DermaMNIST, RetinaMNIST                                 |
| B          | Radiology (Grayscale) | 96         | ChestMNIST, PneumoniaMNIST, BreastMNIST, OCTMNIST, OrganA/C/SMNIST                          |
| C          | Volumetric (3D)       | 192        | OrganMNIST3D, NoduleMNIST3D, AdrenalMNIST3D, VesselMNIST3D, FractureMNIST3D, SynapseMNIST3D |


## Project Structure

```
mosaic-medical/
├── main.py                  # Training entry point
├── testing.py               # Evaluation entry point
├── config/
│   └── datasets.py          # Dataset registry & specialist routing
├── dataloader/
│   ├── medmnist_loader.py   # MedMNIST data loading
│   └── transforms.py        # 2D / 3D augmentation pipelines
├── engine/
│   ├── trainer.py           # Cyclic training with EMA teacher
│   └── evaluator.py         # Multi-task evaluation
├── model/
│   ├── adapter.py           # MoS adapter (core contribution)
│   ├── patch_embed.py       # Modality-aware tokenizer
│   ├── transformer_block.py # Transformer with parallel adapter
│   └── unified_model.py     # Full model, teacher & weight loading
└── utils/
    └── logger.py            # Experiment logging
```

## Acknowledgements

- [MedMNIST](https://medmnist.com/) benchmark
- [AdaptFormer](https://arxiv.org/abs/2205.13535) for adapter design
- [Ark](https://arxiv.org/abs/2110.05006) for cyclic training
- [MedCoSS](https://arxiv.org/abs/2305.12850) for unified tokenizer

## License

Apache License 2.0
