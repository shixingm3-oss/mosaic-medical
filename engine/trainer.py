"""
Ark+-style cyclic trainer with Teacher-Student consistency.

Each *round* iterates over every dataset once.  The teacher is updated
via expert-aware EMA after every training step, and an early-stopping
mechanism monitors the mean validation accuracy.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader

from config.datasets import get_dataset_config, get_expert_id
from .evaluator import evaluate_all_datasets, print_evaluation_results


class Trainer:

    def __init__(self, student, teacher, dataloaders, dataset_list,
                 optimizer, scheduler=None, device=None, args=None,
                 logger=None):
        self.student = student
        self.teacher = teacher
        self.dataloaders = dataloaders
        self.dataset_list = dataset_list
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.logger = logger

        self.student.to(self.device)
        self.teacher.to(self.device)

        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        self.current_round = 0
        self.global_step = 0
        self.best_acc = 0.0
        self.best_round = 0
        self.history: Dict = {"train_loss": [], "val_results": [], "test_results": []}

    # ------------------------------------------------------------------
    def _loss_fn(self, task_type):
        return self.bce_loss if task_type == "multi-label" else self.ce_loss

    # ------------------------------------------------------------------
    def train_one_epoch(self, dataset_name, task_id, epoch):
        self.student.train()
        self.teacher.eval()

        cfg = get_dataset_config(dataset_name)
        expert_id = get_expert_id(dataset_name)
        adapter_mode = getattr(self.student, "adapter_mode", "v1")
        loader = self.dataloaders[dataset_name]["train"]
        cls_fn = self._loss_fn(cfg.task_type)

        total_loss = total_cls = total_con = 0.0
        n_batches = 0

        for images, labels in loader:
            # Dual augmentation for teacher-student
            if isinstance(images, (tuple, list)) and len(images) == 2:
                imgs_s, imgs_t = images[0].to(self.device), images[1].to(self.device)
            else:
                imgs_s = imgs_t = images.to(self.device)

            labels = labels.to(self.device)
            labels = labels.float() if cfg.task_type == "multi-label" else labels.long().squeeze()

            feat_s, logits_s = self.student(imgs_s, task_id=task_id, expert_id=expert_id)
            with torch.no_grad():
                feat_t = self.teacher(imgs_t, task_id=task_id, return_features=True, expert_id=expert_id)

            loss_cls = cls_fn(logits_s, labels)
            loss_con = self.mse_loss(feat_s, feat_t.detach())
            cw = getattr(self.args, "consist_weight", 0.1)
            loss = loss_cls + cw * loss_con

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(),
                                     getattr(self.args, "max_grad_norm", 1.0))
            self.optimizer.step()

            # Expert-aware EMA
            mom = getattr(self.args, "ema_momentum", 0.9)
            mom_3d = getattr(self.args, "ema_momentum_3d", 0.95)
            if adapter_mode == "v2_moe":
                m = mom_3d if expert_id == "C" else mom
                self.teacher.ema_update(self.student, momentum=m, expert_id=expert_id)
            else:
                m = mom_3d if cfg.is_3d else mom
                self.teacher.ema_update(self.student, momentum=m, is_3d=cfg.is_3d)

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_con += loss_con.item()
            n_batches += 1
            self.global_step += 1

        return {"loss": total_loss / n_batches,
                "loss_cls": total_cls / n_batches,
                "loss_consist": total_con / n_batches}

    # ------------------------------------------------------------------
    def train_one_round(self, round_idx):
        self.current_round = round_idx
        losses = {}
        print(f"\n{'=' * 60}\n Round {round_idx + 1}\n{'=' * 60}")

        for tid, name in enumerate(self.dataset_list):
            res = self.train_one_epoch(name, tid, round_idx)
            losses[name] = res["loss"]
            print(f"  {name}: loss={res['loss']:.4f}  cls={res['loss_cls']:.4f}"
                  f"  consist={res['loss_consist']:.4f}")

        if self.scheduler is not None:
            self.scheduler.step()
        losses["mean"] = sum(losses.values()) / len(losses)
        return losses

    # ------------------------------------------------------------------
    def train(self, num_rounds, eval_every=1, save_every=10,
              output_dir="./output", early_stopping_patience=5,
              early_stopping_min_delta=0.001):
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        no_improve = 0

        for r in range(num_rounds):
            rl = self.train_one_round(r)
            self.history["train_loss"].append(rl)

            if (r + 1) % eval_every == 0:
                print(f"\n--- Evaluation (Round {r + 1}) ---")
                for tag, mdl in [("Student", self.student), ("Teacher", self.teacher)]:
                    res = evaluate_all_datasets(mdl, self.dataloaders,
                                                self.dataset_list, self.device, "val")
                    print_evaluation_results(res, self.dataset_list,
                                             f"{tag} Validation (Round {r + 1})")
                    if tag == "Student":
                        val_s = res
                    else:
                        val_t = res

                if self.logger:
                    self.logger.log_round(r, val_s, val_t)
                self.history["val_results"].append({"round": r + 1, "student": val_s, "teacher": val_t})

                cur = val_s.get("mean_acc", 0)
                if cur > self.best_acc:
                    self.best_acc, self.best_round = cur, r
                    self.save_checkpoint(os.path.join(output_dir, "best_model.pth"), r)
                    print(f"  * New best! ACC: {cur * 100:.2f}%")
                    no_improve = 0
                elif cur > self.best_acc - early_stopping_min_delta:
                    no_improve = 0
                else:
                    no_improve += 1
                    print(f"  No improvement ({no_improve}/{early_stopping_patience})")

                if no_improve >= early_stopping_patience:
                    print(f"\nEarly stopping at round {r + 1}.")
                    break

            if (r + 1) % save_every == 0:
                self.save_checkpoint(os.path.join(output_dir, f"checkpoint_round{r + 1}.pth"), r)

        # Final test with best model
        best_path = os.path.join(output_dir, "best_model.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device)
            self.student.load_state_dict(ckpt["student_state_dict"])
            self.teacher.model.load_state_dict(ckpt["teacher_state_dict"])

        for tag, mdl in [("Student", self.student), ("Teacher", self.teacher)]:
            res = evaluate_all_datasets(mdl, self.dataloaders,
                                        self.dataset_list, self.device, "test")
            print_evaluation_results(res, self.dataset_list, f"{tag} Test Results")
            self.history["test_results"].append(res)

        if self.logger:
            self.logger.log_test_results(
                self.history["test_results"][0], self.history["test_results"][1],
                self.best_round, self.best_acc)
            self.logger.finalize()

        print(f"\nTraining completed in {(time.time() - start) / 3600:.2f} h")
        self._save_history(os.path.join(output_dir, "history.json"))
        return self.history

    # ------------------------------------------------------------------
    def save_checkpoint(self, path, round_idx):
        ckpt = {
            "round": round_idx,
            "student_state_dict": self.student.state_dict(),
            "teacher_state_dict": self.teacher.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
            "global_step": self.global_step,
        }
        if self.scheduler:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(ckpt, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.student.load_state_dict(ckpt["student_state_dict"])
        self.teacher.model.load_state_dict(ckpt["teacher_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_acc = ckpt.get("best_acc", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.current_round = ckpt.get("round", 0)
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"Resumed from round {self.current_round + 1}")

    def _save_history(self, path):
        def _convert(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, dict):
                return {k: _convert(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_convert(v) for v in o]
            return o
        with open(path, "w") as f:
            json.dump(_convert(self.history), f, indent=2)


# ---- Helpers -----------------------------------------------------------------

def create_optimizer(model, lr=1e-4, weight_decay=0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr,
                             weight_decay=weight_decay)


def create_scheduler(optimizer, num_rounds, warmup_rounds=5, min_lr=1e-6):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_rounds)
    cosine = CosineAnnealingLR(optimizer, T_max=num_rounds - warmup_rounds,
                               eta_min=min_lr)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_rounds])