from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class SubjectSplit:
    train_subjects: List[str]
    val_subjects: List[str]
    test_subjects: List[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_subject_split(
    all_subjects: Iterable[str],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> SubjectSplit:
    subjects = sorted(set(all_subjects))
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train : n_train + n_val]
    test_subjects = subjects[n_train + n_val :]
    return SubjectSplit(train_subjects, val_subjects, test_subjects)


def _tensor_from_any(x, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


def load_athletic_mm_samples(data_path: str) -> List[Dict]:
    payload = torch.load(data_path, map_location="cpu")
    if not isinstance(payload, list):
        raise TypeError("Expected .pt payload to be a list of samples.")
    return payload


def compute_train_normalizer(samples: List[Dict], train_subjects: Iterable[str]) -> Dict[str, torch.Tensor]:
    train_subjects = set(train_subjects)
    kin_chunks = []
    phy_chunks = []
    for sample in samples:
        if sample["subject_id"] not in train_subjects:
            continue
        kin_chunks.append(_tensor_from_any(sample["inputs"]["kinematic"], torch.float32))
        phy_chunks.append(_tensor_from_any(sample["inputs"]["physiological"], torch.float32))

    if not kin_chunks or not phy_chunks:
        raise ValueError("No training samples found for normalizer computation.")

    kin = torch.cat(kin_chunks, dim=0)
    phy = torch.cat(phy_chunks, dim=0)
    kin_mean = kin.mean(dim=0)
    kin_std = kin.std(dim=0).clamp_min(1e-6)
    phy_mean = phy.mean(dim=0)
    phy_std = phy.std(dim=0).clamp_min(1e-6)

    return {
        "kin_mean": kin_mean,
        "kin_std": kin_std,
        "phy_mean": phy_mean,
        "phy_std": phy_std,
    }


class AthleticMMDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        subject_ids: Iterable[str],
        normalizer: Dict[str, torch.Tensor],
    ) -> None:
        subject_ids = set(subject_ids)
        self.samples = [s for s in samples if s["subject_id"] in subject_ids]
        self.normalizer = normalizer
        if not self.samples:
            raise ValueError("No samples in dataset after subject filtering.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        kin = _tensor_from_any(s["inputs"]["kinematic"], torch.float32)
        phy = _tensor_from_any(s["inputs"]["physiological"], torch.float32)
        grf = _tensor_from_any(s["labels"]["grf"], torch.float32)
        fatigue = _tensor_from_any(s["labels"]["fatigue"], torch.long).view(())
        technique = _tensor_from_any(s["labels"]["technique"], torch.long).view(())

        kin = (kin - self.normalizer["kin_mean"]) / self.normalizer["kin_std"]
        phy = (phy - self.normalizer["phy_mean"]) / self.normalizer["phy_std"]

        return {
            "kinematic": kin,
            "physiological": phy,
            "grf": grf,
            "fatigue": fatigue,
            "technique": technique,
            "subject_id": s["subject_id"],
        }


def infer_num_classes(samples: List[Dict]) -> Tuple[int, int]:
    fatigue_max = -1
    technique_max = -1
    for s in samples:
        fatigue = int(_tensor_from_any(s["labels"]["fatigue"], torch.long).item())
        technique = int(_tensor_from_any(s["labels"]["technique"], torch.long).item())
        fatigue_max = max(fatigue_max, fatigue)
        technique_max = max(technique_max, technique)
    return fatigue_max + 1, technique_max + 1
