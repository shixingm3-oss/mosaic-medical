#!/usr/bin/env python
"""
Test script — load a checkpoint and evaluate on MedMNIST.

Usage:
    python testing.py --checkpoint /path/to/best_model.pth --data_root /path/to/data
"""

import os
import argparse
import torch

from config.datasets import (
    ALL_DATASETS, DATASETS_2D, DATASETS_3D, get_num_classes_list,
)
from dataloader.medmnist_loader import create_all_dataloaders
from model.unified_model import create_model_and_teacher
from engine.evaluator import evaluate_all_datasets, print_evaluation_results


def parse_args():
    parser = argparse.ArgumentParser(description="MOSAIC Testing")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_list", nargs="+", default=None)
    parser.add_argument("--only_2d", action="store_true")
    parser.add_argument("--only_3d", action="store_true")
    parser.add_argument("--use_adapter", action="store_true", default=True)
    parser.add_argument("--no_adapter", action="store_true")
    parser.add_argument("--adapter_mode", type=str, default="v2_moe")
    parser.add_argument("--adapter_bottleneck", type=int, default=64)
    parser.add_argument("--adapter_bottleneck_a", type=int, default=64)
    parser.add_argument("--adapter_bottleneck_b", type=int, default=96)
    parser.add_argument("--adapter_bottleneck_c", type=int, default=192)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])

    args = parser.parse_args()
    if args.no_adapter:
        args.use_adapter = False
    return args


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve dataset list
    if args.dataset_list is not None:
        dataset_list = args.dataset_list
    elif args.only_2d:
        dataset_list = DATASETS_2D
    elif args.only_3d:
        dataset_list = DATASETS_3D
    else:
        dataset_list = ALL_DATASETS

    print(f"\nDatasets to test: {len(dataset_list)}")

    # Data
    dataloaders = create_all_dataloaders(
        dataset_list=dataset_list,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dual_transform=False,
    )

    # Model
    num_classes_list = get_num_classes_list(dataset_list)
    student, teacher = create_model_and_teacher(
        num_classes_list=num_classes_list,
        use_adapter=args.use_adapter,
        adapter_mode=args.adapter_mode,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_bottleneck_a=args.adapter_bottleneck_a,
        adapter_bottleneck_b=args.adapter_bottleneck_b,
        adapter_bottleneck_c=args.adapter_bottleneck_c,
    )

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "round" in checkpoint:
        print(f"  Round: {checkpoint['round'] + 1}")
    if "best_acc" in checkpoint:
        print(f"  Best ACC: {checkpoint['best_acc'] * 100:.2f}%")

    student.load_state_dict(checkpoint["student_state_dict"])
    if "teacher_state_dict" in checkpoint:
        teacher.model.load_state_dict(checkpoint["teacher_state_dict"])

    student.to(device)
    teacher.to(device)

    # Evaluate
    for tag, model in [("Student", student), ("Teacher", teacher)]:
        results = evaluate_all_datasets(
            model=model,
            dataloaders=dataloaders,
            dataset_list=dataset_list,
            device=device,
            split=args.split,
        )
        print_evaluation_results(results, dataset_list, title=f"{tag} {args.split.capitalize()} Results")

    print("Testing completed.")


if __name__ == "__main__":
    main()