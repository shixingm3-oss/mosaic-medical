#!/usr/bin/env python
"""
MOSAIC: Mixture-of-Specialists Adapter for Imaging Classification

Usage:
    python main.py \
        --data_root /path/to/medmnist \
        --pretrained /path/to/vit_base_patch16_224.npz \
        --adapter_mode v2_moe \
        --freeze_backbone \
        --num_rounds 100 \
        --seed 42
"""

import os
import json
import argparse
import random
import torch
import numpy as np

from config.datasets import (
    ALL_DATASETS, INTERLEAVED_DATASETS, DATASETS_2D, DATASETS_3D,
    get_num_classes_list,
)
from dataloader.medmnist_loader import create_all_dataloaders
from model.unified_model import create_model_and_teacher
from engine.trainer import Trainer, create_optimizer, create_scheduler
from utils.logger import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(description="MOSAIC Training")

    # Data
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset_list", nargs="+", default=None)
    parser.add_argument("--use_interleaved", action="store_true")
    parser.add_argument("--only_2d", action="store_true")
    parser.add_argument("--only_3d", action="store_true")

    # Model
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--use_adapter", action="store_true", default=True)
    parser.add_argument("--no_adapter", action="store_true")
    parser.add_argument("--adapter_mode", type=str, default="v2_moe",
                        choices=["v1", "v2_moe"])
    parser.add_argument("--adapter_bottleneck", type=int, default=64)
    parser.add_argument("--adapter_bottleneck_a", type=int, default=64,
                        help="Expert A bottleneck dim (Bio-Medical)")
    parser.add_argument("--adapter_bottleneck_b", type=int, default=96,
                        help="Expert B bottleneck dim (Radiology)")
    parser.add_argument("--adapter_bottleneck_c", type=int, default=192,
                        help="Expert C bottleneck dim (Volumetric 3D)")
    parser.add_argument("--adapter_scalar", type=float, default=0.1)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")

    # Training
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_rounds", type=int, default=5)
    parser.add_argument("--consist_weight", type=float, default=0.1)
    parser.add_argument("--ema_momentum", type=float, default=0.9)
    parser.add_argument("--ema_momentum_3d", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Misc
    parser.add_argument("--exp_name", type=str, default="mosaic")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    if args.no_adapter:
        args.use_adapter = False
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve dataset list
    if args.dataset_list is not None:
        dataset_list = args.dataset_list
    elif args.only_2d:
        dataset_list = DATASETS_2D
    elif args.only_3d:
        dataset_list = DATASETS_3D
    elif args.use_interleaved:
        dataset_list = INTERLEAVED_DATASETS
    else:
        dataset_list = ALL_DATASETS

    print(f"\nDatasets ({len(dataset_list)}):")
    for i, name in enumerate(dataset_list):
        print(f"  [{i}] {name}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 1. DataLoaders
    print("\n[1/6] Creating DataLoaders ...")
    dataloaders = create_all_dataloaders(
        dataset_list=dataset_list,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dual_transform=True,
    )

    # 2. Model
    print("\n[2/6] Creating Models ...")
    num_classes_list = get_num_classes_list(dataset_list)
    student, teacher = create_model_and_teacher(
        num_classes_list=num_classes_list,
        use_adapter=args.use_adapter,
        adapter_mode=args.adapter_mode,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_bottleneck_a=args.adapter_bottleneck_a,
        adapter_bottleneck_b=args.adapter_bottleneck_b,
        adapter_bottleneck_c=args.adapter_bottleneck_c,
        adapter_scalar=args.adapter_scalar,
        pretrained_path=args.pretrained,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    )

    if args.freeze_backbone:
        student.freeze_backbone()
        print("Backbone frozen. Only training adapters and classification heads.")

    total = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    adapter = sum(p.numel() for n, p in student.named_parameters()
                  if "adapter" in n or "expert" in n)
    print(f"  Total params:     {total:,} ({total / 1e6:.2f}M)")
    print(f"  Trainable params: {trainable:,} ({trainable / 1e6:.2f}M)")
    print(f"  Adapter params:   {adapter:,} ({adapter / 1e6:.2f}M)")
    print(f"  Adapter ratio:    {adapter / total * 100:.2f}%")

    # 3. Optimizer & Scheduler
    print("\n[3/6] Creating Optimizer & Scheduler ...")
    optimizer = create_optimizer(student, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, args.num_rounds, args.warmup_rounds)

    # 4. Logger
    print("\n[4/6] Creating Logger ...")
    logger = ExperimentLogger(
        exp_name=args.exp_name,
        dataset_list=dataset_list,
        output_dir=args.output_dir,
        config=vars(args),
    )

    # 5. Trainer
    trainer = Trainer(
        student=student,
        teacher=teacher,
        dataloaders=dataloaders,
        dataset_list=dataset_list,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
        logger=logger,
    )

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # 6. Train
    print("\n[5/6] Starting Training ...")
    trainer.train(
        num_rounds=args.num_rounds,
        eval_every=args.eval_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    print(f"\n[6/6] Done! Best ACC: {trainer.best_acc * 100:.2f}%")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()