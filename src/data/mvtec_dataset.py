"""
PyTorch Dataset for MVTec AD.

Design: one MVTecDataset instance always corresponds to exactly ONE
category (e.g. "bottle") and ONE split ("train" or "test"). Multi-category
comparisons are handled by instantiating this class once per category at
the script/orchestration level, not inside this class.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    """
    Dataset for a single MVTec AD category and split.

    train split:
        Only loads images from train/good/. Every sample has label=0.
        This enforces the "train on normal-only data" requirement by
        construction -- it is structurally impossible for this class
        to load a defective image when split="train".

    test split:
        Loads images from test/good/ AND every defect-type subfolder.
        label=0 for good, label=1 for any defect type.
        Also loads the corresponding ground-truth mask for defective
        images, or an all-zero mask for normal images.
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            root: Path to the MVTec AD root, e.g. "data/raw/mvtec_ad"
            category: e.g. "bottle"
            split: "train" or "test"
            transform: optional callable applied to the PIL image,
                       expected to return a torch.Tensor. If None,
                       a minimal default (PIL -> tensor, no resize) is used.
        """
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform

        self.category_root = self.root / category
        if not self.category_root.exists():
            raise FileNotFoundError(f"Category folder not found: {self.category_root}")

        self.samples: list[dict] = self._build_sample_list()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for category='{category}', split='{split}'. "
                f"Check that the dataset is correctly placed under {self.category_root}"
            )

    def _build_sample_list(self) -> list[dict]:
        """Scan disk and build a list of sample descriptors (paths + labels)."""
        samples = []

        if self.split == "train":
            good_dir = self.category_root / "train" / "good"
            if not good_dir.exists():
                raise FileNotFoundError(f"Expected folder not found: {good_dir}")

            for image_path in sorted(good_dir.glob("*.png")):
                samples.append({
                    "image_path": image_path,
                    "mask_path": None,
                    "label": 0,
                    "defect_type": "good",
                })

        else:  # split == "test"
            test_root = self.category_root / "test"
            if not test_root.exists():
                raise FileNotFoundError(f"Expected folder not found: {test_root}")

            defect_dirs = sorted(d for d in test_root.iterdir() if d.is_dir())

            for defect_dir in defect_dirs:
                defect_type = defect_dir.name
                label = 0 if defect_type == "good" else 1

                for image_path in sorted(defect_dir.glob("*.png")):
                    mask_path = None
                    if label == 1:
                        mask_path = (
                            self.category_root
                            / "ground_truth"
                            / defect_type
                            / f"{image_path.stem}_mask.png"
                        )
                        if not mask_path.exists():
                            raise FileNotFoundError(
                                f"Expected mask not found: {mask_path} "
                                f"for image {image_path}"
                            )

                    samples.append({
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "label": label,
                        "defect_type": defect_type,
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        image = Image.open(sample["image_path"]).convert("RGB")

        if sample["mask_path"] is not None:
            mask = Image.open(sample["mask_path"]).convert("L")  # single channel
        else:
            # No defect -> all-zero mask, same spatial size as the image.
            mask = Image.new("L", image.size, color=0)

        if self.transform is not None:
            image = self.transform(image)
            mask_tensor = self.transform(mask)
        else:
            # Minimal default: PIL -> float tensor in [0, 1], no resizing.
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0

        return {
            "image": image,
            "mask": mask_tensor,
            "label": sample["label"],
            "defect_type": sample["defect_type"],
            "image_path": str(sample["image_path"]),
        }