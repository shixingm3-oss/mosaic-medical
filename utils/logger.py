"""
Experiment logger — tracks per-round metrics, forgetting, and generates
CSV / JSON / PNG outputs.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class ExperimentLogger:

    def __init__(self, exp_name, dataset_list, output_dir="./results",
                 config=None):
        self.exp_name = exp_name
        self.dataset_list = dataset_list
        self.datasets_2d = [d for d in dataset_list if "3D" not in d]
        self.datasets_3d = [d for d in dataset_list if "3D" in d]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(output_dir) / f"{exp_name}_{ts}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.data = {
            "config": {"exp_name": exp_name, "datasets": dataset_list,
                       "timestamp": ts, **(config or {})},
            "round_results": [], "forgetting_history": [],
        }
        self.best_acc = {d: 0.0 for d in dataset_list}
        print(f"Logger: {self.save_dir}")

    # ------------------------------------------------------------------
    def log_round(self, round_idx, student_results, teacher_results=None):
        s_acc = {d: student_results.get(d, {}).get("acc", 0) * 100
                 for d in self.dataset_list}
        fgt = self._forgetting(s_acc)
        for d, a in s_acc.items():
            self.best_acc[d] = max(self.best_acc[d], a)

        rec = {
            "round": round_idx + 1,
            "student_mean_acc": student_results.get("mean_acc", 0) * 100,
            "student_mean_auc": student_results.get("mean_auc", 0),
            "avg_forgetting": fgt["avg"], "max_forgetting": fgt["max"],
        }
        if teacher_results:
            rec["teacher_mean_acc"] = teacher_results.get("mean_acc", 0) * 100
        for d in self.dataset_list:
            rec[f"{d}_acc"] = s_acc.get(d, 0)
        self.data["round_results"].append(rec)
        self._save_json()

    def log_test_results(self, student_results, teacher_results=None,
                         best_round=None, best_val_acc=None):
        self.data["test_results"] = {
            "student": student_results, "teacher": teacher_results,
            "best_round": (best_round + 1) if best_round is not None else None,
            "best_val_acc": best_val_acc,
        }
        self._save_json()

    # ------------------------------------------------------------------
    def _forgetting(self, cur):
        vals = [max(0, self.best_acc[d] - cur.get(d, 0))
                for d in self.dataset_list]
        return {"avg": float(np.mean(vals)), "max": float(np.max(vals))}

    def _save_json(self):
        with open(self.save_dir / "results.json", "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    def finalize(self):
        self._save_json()
        if pd is not None and self.data["round_results"]:
            pd.DataFrame(self.data["round_results"]).to_csv(
                self.save_dir / "round_results.csv", index=False)
        if plt is not None and self.data["round_results"]:
            self._plot()
        self._summary()
        print(f"Results saved to {self.save_dir}")

    def _plot(self):
        rr = self.data["round_results"]
        rounds = [r["round"] for r in rr]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(rounds, [r["student_mean_acc"] for r in rr], "o-", label="Student")
        if "teacher_mean_acc" in rr[0]:
            ax1.plot(rounds, [r["teacher_mean_acc"] for r in rr], "s--", label="Teacher")
        ax1.set(xlabel="Round", ylabel="Mean ACC (%)", title="Accuracy")
        ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(rounds, [r["avg_forgetting"] for r in rr], "o-", color="red")
        ax2.set(xlabel="Round", ylabel="Forgetting (%)", title="Forgetting")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.save_dir / "plot.png", dpi=150)
        plt.close(fig)

    def _summary(self):
        with open(self.save_dir / "summary.txt", "w") as f:
            f.write(f"Experiment: {self.exp_name}\n")
            f.write(f"Datasets: {len(self.dataset_list)} "
                    f"({len(self.datasets_2d)} 2D, {len(self.datasets_3d)} 3D)\n")
            if self.data["round_results"]:
                last = self.data["round_results"][-1]
                f.write(f"Final student ACC: {last['student_mean_acc']:.2f}%\n")


def create_logger(args, dataset_list):
    return ExperimentLogger(
        getattr(args, "exp_name", "exp"), dataset_list,
        getattr(args, "output_dir", "./results"), vars(args))