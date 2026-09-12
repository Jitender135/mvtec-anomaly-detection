"""
Dataset inspection utility for MVTec AD.

Run this BEFORE writing or trusting any Dataset/DataLoader class.
It verifies the on-disk structure matches our assumptions and
reports counts/dimensions so we understand the data before modeling it.
"""

from pathlib import Path
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _list_images(folder: Path) -> list[Path]:
    """Return all image files directly inside a folder (non-recursive)."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_category(category_root: Path) -> dict:
    """
    Inspect a single MVTec AD category folder and return a structured report.

    Args:
        category_root: Path to e.g. data/raw/mvtec_ad/bottle

    Returns:
        A dictionary summarizing counts, defect types, and any problems found.
    """
    report: dict = {
        "category": category_root.name,
        "train": {},
        "test": {},
        "ground_truth": {},
        "problems": [],
    }

    # ---- TRAIN SPLIT ----
    train_root = category_root / "train"
    if not train_root.exists():
        report["problems"].append(f"Missing train/ folder at {train_root}")
    else:
        train_subfolders = [d for d in train_root.iterdir() if d.is_dir()]
        for sub in train_subfolders:
            images = _list_images(sub)
            report["train"][sub.name] = len(images)

        # THE CRITICAL LEAKAGE CHECK:
        # train/ must contain ONLY a "good" subfolder. Anything else means
        # defective images could leak into training.
        non_good = [name for name in report["train"] if name != "good"]
        if non_good:
            report["problems"].append(
                f"LEAKAGE RISK: train/ contains non-'good' folders: {non_good}"
            )
        if "good" not in report["train"]:
            report["problems"].append("train/good folder is missing entirely")

    # ---- TEST SPLIT ----
    test_root = category_root / "test"
    if not test_root.exists():
        report["problems"].append(f"Missing test/ folder at {test_root}")
    else:
        for sub in sorted(d for d in test_root.iterdir() if d.is_dir()):
            images = _list_images(sub)
            report["test"][sub.name] = len(images)

    # ---- GROUND TRUTH ----
    gt_root = category_root / "ground_truth"
    if gt_root.exists():
        for sub in sorted(d for d in gt_root.iterdir() if d.is_dir()):
            masks = _list_images(sub)
            report["ground_truth"][sub.name] = len(masks)

        # Every defective test folder should have a matching ground_truth folder.
        defect_types_in_test = [k for k in report["test"] if k != "good"]
        for defect_type in defect_types_in_test:
            if defect_type not in report["ground_truth"]:
                report["problems"].append(
                    f"No ground_truth mask folder for test defect type '{defect_type}'"
                )
    else:
        report["problems"].append(f"Missing ground_truth/ folder at {gt_root}")

    # ---- SAMPLE IMAGE DIMENSIONS ----
    sample_images = _list_images(train_root / "good") if train_root.exists() else []
    if sample_images:
        with Image.open(sample_images[0]) as img:
            report["sample_image_size"] = img.size  # (width, height)
            report["sample_image_mode"] = img.mode  # e.g. "RGB"
    else:
        report["problems"].append("No sample image found in train/good to inspect")

    return report


def print_report(report: dict) -> None:
    print("=" * 60)
    print(f"CATEGORY: {report['category']}")
    print("=" * 60)

    print("\nTrain split:")
    for name, count in report["train"].items():
        print(f"  {name}: {count} images")

    print("\nTest split:")
    total_test = 0
    for name, count in report["test"].items():
        print(f"  {name}: {count} images")
        total_test += count
    print(f"  TOTAL test images: {total_test}")

    print("\nGround truth masks:")
    for name, count in report["ground_truth"].items():
        print(f"  {name}: {count} masks")

    if "sample_image_size" in report:
        print(f"\nSample image size (W, H): {report['sample_image_size']}")
        print(f"Sample image mode: {report['sample_image_mode']}")

    print("\nProblems found:")
    if report["problems"]:
        for problem in report["problems"]:
            print(f"  ⚠ {problem}")
    else:
        print("  None. Structure looks correct.")

    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.data.inspect <category_name>")
        print("Example: python -m src.data.inspect bottle")
        sys.exit(1)

    category_name = sys.argv[1]
    category_path = Path("data/raw/mvtec_ad") / category_name

    if not category_path.exists():
        print(f"Category folder not found: {category_path.resolve()}")
        sys.exit(1)

    report = inspect_category(category_path)
    print_report(report)