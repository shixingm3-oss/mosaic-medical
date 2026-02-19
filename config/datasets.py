"""
MedMNIST dataset registry and MoE expert routing table.

Covers all 18 MedMNIST datasets (12 x 2D, 6 x 3D) with per-dataset
metadata: number of classes, task type, evaluation metric, and expert
assignment for the three-specialist MoE adapter.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    task_type: str          # multi-class | binary-class | multi-label | ordinal-regression
    is_3d: bool
    metric: str             # primary metric: acc | auc
    in_channels: int
    medmnist_name: Optional[str] = None

    def __post_init__(self):
        if self.medmnist_name is None:
            self.medmnist_name = self.name.lower()


# ---- Dataset configs --------------------------------------------------------

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    # 2D datasets
    "PathMNIST":      DatasetConfig("PathMNIST",      9,  "multi-class",        False, "acc", 3, "pathmnist"),
    "DermaMNIST":     DatasetConfig("DermaMNIST",      7,  "multi-class",        False, "acc", 3, "dermamnist"),
    "OCTMNIST":       DatasetConfig("OCTMNIST",        4,  "multi-class",        False, "acc", 1, "octmnist"),
    "PneumoniaMNIST": DatasetConfig("PneumoniaMNIST",  2,  "binary-class",       False, "acc", 1, "pneumoniamnist"),
    "ChestMNIST":     DatasetConfig("ChestMNIST",      14, "multi-label",        False, "auc", 1, "chestmnist"),
    "BreastMNIST":    DatasetConfig("BreastMNIST",     2,  "binary-class",       False, "acc", 1, "breastmnist"),
    "BloodMNIST":     DatasetConfig("BloodMNIST",      8,  "multi-class",        False, "acc", 3, "bloodmnist"),
    "TissueMNIST":    DatasetConfig("TissueMNIST",     8,  "multi-class",        False, "acc", 1, "tissuemnist"),
    "RetinaMNIST":    DatasetConfig("RetinaMNIST",     5,  "ordinal-regression", False, "acc", 3, "retinamnist"),
    "OrganAMNIST":    DatasetConfig("OrganAMNIST",     11, "multi-class",        False, "acc", 1, "organamnist"),
    "OrganCMNIST":    DatasetConfig("OrganCMNIST",     11, "multi-class",        False, "acc", 1, "organcmnist"),
    "OrganSMNIST":    DatasetConfig("OrganSMNIST",     11, "multi-class",        False, "acc", 1, "organsmnist"),
    # 3D datasets
    "OrganMNIST3D":   DatasetConfig("OrganMNIST3D",    11, "multi-class",        True,  "acc", 1, "organmnist3d"),
    "NoduleMNIST3D":  DatasetConfig("NoduleMNIST3D",   2,  "binary-class",       True,  "acc", 1, "nodulemnist3d"),
    "AdrenalMNIST3D": DatasetConfig("AdrenalMNIST3D",  2,  "binary-class",       True,  "acc", 1, "adrenalmnist3d"),
    "VesselMNIST3D":  DatasetConfig("VesselMNIST3D",   2,  "binary-class",       True,  "acc", 1, "vesselmnist3d"),
    "FractureMNIST3D":DatasetConfig("FractureMNIST3D", 3,  "multi-class",        True,  "acc", 1, "fracturemnist3d"),
    "SynapseMNIST3D": DatasetConfig("SynapseMNIST3D",  2,  "binary-class",       True,  "acc", 1, "synapsemnist3d"),
}


# ---- Dataset lists ----------------------------------------------------------

DATASETS_2D: List[str] = [n for n, c in DATASET_CONFIGS.items() if not c.is_3d]
DATASETS_3D: List[str] = [n for n, c in DATASET_CONFIGS.items() if c.is_3d]
ALL_DATASETS: List[str] = DATASETS_2D + DATASETS_3D

INTERLEAVED_DATASETS: List[str] = [
    "PathMNIST",      "OrganMNIST3D",
    "DermaMNIST",      "NoduleMNIST3D",
    "OCTMNIST",        "AdrenalMNIST3D",
    "PneumoniaMNIST",  "VesselMNIST3D",
    "ChestMNIST",      "FractureMNIST3D",
    "BreastMNIST",     "SynapseMNIST3D",
    "BloodMNIST",      "TissueMNIST",
    "RetinaMNIST",     "OrganAMNIST",
    "OrganCMNIST",     "OrganSMNIST",
]


# ---- Expert routing (hard) --------------------------------------------------
# A = Bio-Medical (RGB, microscopic)   — colour is a key diagnostic cue
# B = Radiology   (grayscale, macro)   — shape / contour is the key cue
# C = Volumetric  (3D voxels)          — spatial continuity is the key cue

EXPERT_ROUTING: Dict[str, str] = {
    "PathMNIST": "A",  "BloodMNIST": "A",  "TissueMNIST": "A",
    "DermaMNIST": "A", "RetinaMNIST": "A",
    "ChestMNIST": "B", "PneumoniaMNIST": "B", "BreastMNIST": "B",
    "OCTMNIST": "B",   "OrganAMNIST": "B",    "OrganCMNIST": "B",
    "OrganSMNIST": "B",
    "OrganMNIST3D": "C",  "NoduleMNIST3D": "C",  "AdrenalMNIST3D": "C",
    "VesselMNIST3D": "C", "FractureMNIST3D": "C", "SynapseMNIST3D": "C",
}

EXPERT_BOTTLENECK: Dict[str, int] = {"A": 64, "B": 96, "C": 192}


# ---- Utility functions -------------------------------------------------------

def get_expert_id(name: str) -> str:
    return EXPERT_ROUTING[name]

def get_dataset_config(name: str) -> DatasetConfig:
    return DATASET_CONFIGS[name]

def get_num_classes_list(dataset_list: List[str]) -> List[int]:
    return [DATASET_CONFIGS[n].num_classes for n in dataset_list]

def get_task_id(name: str, dataset_list: List[str]) -> int:
    return dataset_list.index(name)

def is_3d_dataset(name: str) -> bool:
    return DATASET_CONFIGS[name].is_3d