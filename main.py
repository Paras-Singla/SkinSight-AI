"""
Four-Class Skin Lesion Classification using EfficientNetV2B0 (ISIC 2019).

This module contains well-organized functions for:
1. Dataset download via KaggleHub
2. Dataset structure discovery, label verification, and four-class filtering
3. Image integrity verification, duplicate/near-duplicate detection, and EDA
4. Lesion/duplicate-aware stratified splitting, preprocessing, model creation,
   two-stage transfer learning, controlled experiments, and evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
# Project Paths and Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
ORIGINAL_DIR = DATASET_DIR / "Original"
PROCESSED_DIR = DATASET_DIR / "Processed"
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_DIR = PROJECT_ROOT / "Results"

KAGGLE_HANDLE = "salviohexia/isic-2019-skin-lesion-images-for-classification"

# Exact class mapping required by specification (alphabetical: 0=BCC, 1=BKL, 2=MEL, 3=NV)
CLASS_TO_INDEX: dict[str, int] = {
    "BCC": 0,
    "BKL": 1,
    "MEL": 2,
    "NV": 3,
}
INDEX_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_INDEX.items()}
SELECTED_CLASSES: list[str] = list(CLASS_TO_INDEX.keys())

CLASS_FULL_NAMES: dict[str, str] = {
    "BCC": "Basal Cell Carcinoma",
    "BKL": "Benign Keratosis",
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevi",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RANDOM_SEED = 42

ADV_ROOT = PROJECT_ROOT / "Advanced_Experiments"
ADV_MODELS_DIR = ADV_ROOT / "Models"
ADV_RESULTS_DIR = ADV_ROOT / "Results"
PHASE6_DIR = ADV_ROOT / "Phase6"
PHASE6_MODELS_DIR = PHASE6_DIR / "Models"

PHASE5_ARCHITECTURES = [
    "DenseNet121",
    "ResNet50V2",
    "MobileNetV3Large",
    "ConvNeXtTiny",
]


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def check_environment() -> dict[str, Any]:
    """Inspect TensorFlow version and GPU availability, and enable memory growth."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    info = {
        "tensorflow_version": tf.__version__,
        "gpu_count": len(gpus),
        "gpu_devices": [gpu.name for gpu in gpus],
    }
    print(f"TensorFlow version: {info['tensorflow_version']}")
    print(f"GPUs detected by TensorFlow: {info['gpu_count']} {info['gpu_devices']}")
    if not gpus:
        print(
            "NOTE: No TensorFlow GPU detected (TensorFlow >= 2.11 does not support "
            "native Windows GPU without WSL2 or DirectML)."
        )
    return info


def download_dataset(original_dir: Path = ORIGINAL_DIR) -> Path:
    """
    Download the ISIC 2019 dataset into Dataset/Original using KaggleHub.

    Scoping KAGGLEHUB_CACHE to Dataset/Original avoids downloading into an
    external global cache or duplicating files across directories.
    """
    original_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(original_dir)

    import kagglehub

    try:
        # Official KaggleHub download snippet
        path = kagglehub.dataset_download(KAGGLE_HANDLE)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download dataset via KaggleHub. If authentication is required, "
            "create a Kaggle API token at https://www.kaggle.com/settings and place "
            "kaggle.json in ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME and KAGGLE_KEY "
            "environment variables). Never hardcode credentials in source files.\n"
            f"Original error: {exc}"
        ) from exc

    print("Path to dataset files:", path)
    dataset_root = Path(path)

    image_files = [
        p for p in dataset_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    print(f"Final dataset path: {dataset_root}")
    print(f"Total image files found: {len(image_files)}")
    return dataset_root


def discover_dataset_files(dataset_root: Path) -> dict[str, Any]:
    """
    Inspect the downloaded directory structure without assuming hardcoded filenames.
    Identifies CSV files and image subdirectories dynamically.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    subdirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    csv_files = sorted(dataset_root.rglob("*.csv"))

    gt_csv: Path | None = None
    meta_csv: Path | None = None
    for csv_path in csv_files:
        cols = set(pd.read_csv(csv_path, nrows=1).columns)
        if {"image", "MEL", "NV", "BCC", "BKL"}.issubset(cols):
            gt_csv = csv_path
        elif {"image", "lesion_id"}.issubset(cols) or "age_approx" in cols:
            meta_csv = csv_path

    if gt_csv is None:
        raise FileNotFoundError(f"Could not locate ground-truth CSV under {dataset_root}")

    folder_counts: dict[str, int] = {}
    disk_records: list[dict[str, Any]] = []
    for subdir in subdirs:
        imgs = [
            p for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if imgs:
            folder_counts[subdir.name] = len(imgs)
            for p in imgs:
                disk_records.append(
                    {
                        "image": p.stem,
                        "folder_label": subdir.name,
                        "filepath": str(p),
                        "ext": p.suffix.lower(),
                        "file_size_bytes": p.stat().st_size,
                    }
                )

    return {
        "dataset_root": dataset_root,
        "gt_csv": gt_csv,
        "meta_csv": meta_csv,
        "folder_counts": folder_counts,
        "disk_df": pd.DataFrame(disk_records),
    }


def inspect_and_filter_dataset(discovery: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Verify official ground-truth labels against folder structure and metadata,
    record all original class counts, and filter strictly to the 4 target classes.
    """
    gt_df = pd.read_csv(discovery["gt_csv"])
    disk_df = discovery["disk_df"]
    meta_df = pd.read_csv(discovery["meta_csv"]) if discovery["meta_csv"] else None

    label_cols = [c for c in gt_df.columns if c != "image"]
    row_sums = gt_df[label_cols].sum(axis=1)
    multi_or_empty_rows = int((row_sums != 1.0).sum())

    gt_df["label"] = gt_df[label_cols].idxmax(axis=1)
    original_gt_counts = {c: int(gt_df[c].sum()) for c in label_cols}

    merged = gt_df[["image", "label"]].merge(disk_df, on="image", how="outer")
    missing_on_disk = int(merged["filepath"].isna().sum())
    missing_in_gt = int(merged["label"].isna().sum())
    folder_gt_mismatches = int(
        (merged["label"].notna() & merged["folder_label"].notna() & (merged["label"] != merged["folder_label"])).sum()
    )

    if meta_df is not None:
        merged = merged.merge(meta_df, on="image", how="left")
        available_identifiers = list(meta_df.columns)
    else:
        available_identifiers = ["image"]

    # Filter strictly to the four selected classes without remapping other diagnoses
    filtered_df = merged[merged["label"].isin(SELECTED_CLASSES)].copy().reset_index(drop=True)
    filtered_df["target"] = filtered_df["label"].map(CLASS_TO_INDEX)

    # Check if any lesion_id maps to multiple diagnoses
    lesion_multi_class = 0
    if "lesion_id" in filtered_df.columns:
        non_null_les = filtered_df.dropna(subset=["lesion_id"])
        lesion_multi_class = int((non_null_les.groupby("lesion_id")["label"].nunique() > 1).sum())

    summary = {
        "total_gt_rows": len(gt_df),
        "total_disk_images": len(disk_df),
        "original_gt_counts": original_gt_counts,
        "folder_counts": discovery["folder_counts"],
        "multi_or_empty_gt_rows": multi_or_empty_rows,
        "missing_on_disk": missing_on_disk,
        "missing_in_gt": missing_in_gt,
        "folder_gt_mismatches": folder_gt_mismatches,
        "available_metadata_columns": available_identifiers,
        "filtered_total": len(filtered_df),
        "excluded_total": len(gt_df) - len(filtered_df),
        "lesion_multi_class_conflicts": lesion_multi_class,
    }
    return filtered_df, summary


def _inspect_single_image(row_tuple: tuple[int, str]) -> dict[str, Any]:
    """Verify readability/integrity, dimensions, MD5 hash, 256-bit dHash, and 32x32 RGB thumbnail."""
    idx, filepath = row_tuple
    try:
        with open(filepath, "rb") as f:
            raw_bytes = f.read()
        file_md5 = hashlib.md5(raw_bytes).hexdigest()

        with Image.open(filepath) as img:
            img.verify()

        with Image.open(filepath) as img:
            mode = img.mode
            width, height = img.size
            rgb = img.convert("RGB")
            gray17 = rgb.convert("L").resize((17, 16), Image.Resampling.BILINEAR)
            arr17 = np.asarray(gray17, dtype=np.int16)
            dh256 = (arr17[:, 1:] > arr17[:, :-1]).flatten()
            dhash_bytes = np.packbits(dh256).tobytes()
            thumb32 = np.asarray(rgb.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.uint8)

        return {
            "idx": idx,
            "corrupted": False,
            "error": "",
            "width": int(width),
            "height": int(height),
            "mode": mode,
            "file_md5": file_md5,
            "dhash_bytes": dhash_bytes,
            "thumb32": thumb32,
        }
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return {
            "idx": idx,
            "corrupted": True,
            "error": str(exc),
            "width": -1,
            "height": -1,
            "mode": "ERROR",
            "file_md5": "",
            "dhash_bytes": b"\x00" * 32,
            "thumb32": np.zeros((32, 32, 3), dtype=np.uint8),
        }


def verify_images_and_find_duplicates(
    df: pd.DataFrame,
    near_dup_max_hamming: int = 12,
    near_dup_max_mae: float = 10.0,
    max_workers: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Check every image for corruption, summarize image resolutions/modes,
    and detect exact MD5 duplicates and verified near-duplicates
    (256-bit dHash Hamming <= 12 AND 32x32 RGB MAE <= 10.0).
    """
    tasks = list(zip(df.index.tolist(), df["filepath"].tolist()))
    print(f"Verifying integrity and computing hashes for {len(tasks)} images...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_inspect_single_image, tasks))

    res_df = pd.DataFrame(results).set_index("idx").sort_index()
    enriched = df.copy()
    for col in ["corrupted", "error", "width", "height", "mode", "file_md5"]:
        enriched[col] = res_df[col]

    corrupted_count = int(enriched["corrupted"].sum())
    valid_mask = ~enriched["corrupted"]
    valid_df = enriched[valid_mask].copy()
    valid_indices = valid_df.index.to_numpy()

    # Exact file MD5 duplicate groups
    md5_counts = valid_df["file_md5"].value_counts()
    exact_dup_hashes = md5_counts[md5_counts > 1]
    exact_dup_images = int(exact_dup_hashes.sum() - len(exact_dup_hashes))
    md5_dup_df = valid_df[valid_df["file_md5"].isin(exact_dup_hashes.index)]
    exact_md5_cross_class_groups = int((md5_dup_df.groupby("file_md5")["label"].nunique() > 1).sum())

    # 256-bit dHash + 32x32 RGB MAE verification on valid images
    valid_res = res_df.loc[valid_indices]
    dh_arr = np.frombuffer(b"".join(valid_res["dhash_bytes"].tolist()), dtype=np.uint64).reshape(-1, 4)
    thumbs = np.stack(valid_res["thumb32"].tolist()).astype(np.int16)

    n = len(dh_arr)
    candidate_pairs: list[tuple[int, int, int]] = []
    chunk_size = 256
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        xor_block = dh_arr[start:end, None, :] ^ dh_arr[None, :, :]
        dist_block = np.bitwise_count(xor_block).sum(axis=2)
        rows, cols = np.where(dist_block <= near_dup_max_hamming)
        for r, c in zip(rows, cols):
            i = start + int(r)
            j = int(c)
            if j > i:
                candidate_pairs.append((i, j, int(dist_block[r, c])))

    near_dup_pairs: list[tuple[int, int, int, float]] = []
    exact_dhash_pairs = 0
    for i, j, hd in candidate_pairs:
        mae = float(np.mean(np.abs(thumbs[i] - thumbs[j])))
        if mae <= near_dup_max_mae:
            if hd == 0:
                exact_dhash_pairs += 1
            near_dup_pairs.append((int(valid_indices[i]), int(valid_indices[j]), hd, mae))

    cross_class_pairs = 0
    cross_lesion_pairs = 0
    involving_missing_lesion = 0
    for idx_a, idx_b, _, _ in near_dup_pairs:
        row_a = enriched.loc[idx_a]
        row_b = enriched.loc[idx_b]
        if row_a["label"] != row_b["label"]:
            cross_class_pairs += 1
        les_a = row_a.get("lesion_id")
        les_b = row_b.get("lesion_id")
        if pd.isna(les_a) or pd.isna(les_b):
            involving_missing_lesion += 1
        elif les_a != les_b:
            cross_lesion_pairs += 1

    # Build composite group IDs uniting lesion_id, exact MD5, and verified near-duplicates
    parent = {int(idx): int(idx) for idx in valid_indices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if "lesion_id" in valid_df.columns:
        for _, grp in valid_df.dropna(subset=["lesion_id"]).groupby("lesion_id"):
            idxs = grp.index.tolist()
            for k in range(1, len(idxs)):
                union(int(idxs[0]), int(idxs[k]))

    for _, grp in valid_df.groupby("file_md5"):
        idxs = grp.index.tolist()
        for k in range(1, len(idxs)):
            union(int(idxs[0]), int(idxs[k]))

    for idx_a, idx_b, _, _ in near_dup_pairs:
        union(idx_a, idx_b)

    enriched["split_group"] = [
        f"grp_{find(int(idx))}" if not enriched.loc[idx, "corrupted"] else "corrupted"
        for idx in enriched.index
    ]

    valid_groups = enriched.loc[valid_indices, "split_group"]
    dup_summary = {
        "corrupted_count": corrupted_count,
        "image_modes": valid_df["mode"].value_counts().to_dict(),
        "width_min": int(valid_df["width"].min()),
        "width_median": int(valid_df["width"].median()),
        "width_max": int(valid_df["width"].max()),
        "height_min": int(valid_df["height"].min()),
        "height_median": int(valid_df["height"].median()),
        "height_max": int(valid_df["height"].max()),
        "top_resolutions": (
            valid_df["width"].astype(str) + "x" + valid_df["height"].astype(str)
        ).value_counts().head(8).to_dict(),
        "exact_md5_duplicate_groups": int(len(exact_dup_hashes)),
        "exact_md5_extra_copies": exact_dup_images,
        "exact_md5_cross_class_groups": exact_md5_cross_class_groups,
        "near_dup_max_hamming": near_dup_max_hamming,
        "near_dup_max_mae": near_dup_max_mae,
        "exact_dhash_pairs": exact_dhash_pairs,
        "total_near_dup_pairs": len(near_dup_pairs),
        "near_dup_cross_class_pairs": cross_class_pairs,
        "near_dup_cross_lesion_pairs": cross_lesion_pairs,
        "near_dup_involving_missing_lesion": involving_missing_lesion,
        "unique_composite_groups": int(valid_groups.nunique()),
        "largest_composite_group_size": int(valid_groups.value_counts().max()),
        "near_dup_pairs": near_dup_pairs,
    }
    return enriched, dup_summary


def save_representative_samples(
    df: pd.DataFrame, output_path: Path = RESULTS_DIR / "representative_samples.png", samples_per_class: int = 4
) -> Path:
    """Save a visual grid of representative images from each of the 4 classes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_df = df[~df["corrupted"]].copy()

    fig, axes = plt.subplots(
        len(SELECTED_CLASSES),
        samples_per_class,
        figsize=(samples_per_class * 3.2, len(SELECTED_CLASSES) * 3.2),
    )
    for row_idx, cls in enumerate(SELECTED_CLASSES):
        cls_df = valid_df[valid_df["label"] == cls].sample(
            n=min(samples_per_class, len(valid_df[valid_df["label"] == cls])),
            random_state=RANDOM_SEED,
        )
        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < len(cls_df):
                rec = cls_df.iloc[col_idx]
                with Image.open(rec["filepath"]) as img:
                    ax.imshow(img.convert("RGB"))
                ax.set_title(
                    f"{cls} ({CLASS_FULL_NAMES[cls]})\n{rec['image']} ({rec['width']}x{rec['height']})",
                    fontsize=8,
                )

    plt.suptitle("Representative ISIC 2019 Images Across Selected Four Classes", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved representative class image grid to: {output_path}")
    return output_path


def generate_eda_report(
    discovery: dict[str, Any],
    summary: dict[str, Any],
    df: pd.DataFrame,
    dup_summary: dict[str, Any],
    report_path: Path = DATASET_DIR / "dataset_report.txt",
) -> Path:
    """Write comprehensive initial EDA and dataset quality report to Dataset/dataset_report.txt."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    valid_df = df[~df["corrupted"]]

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("ISIC 2019 FOUR-CLASS SKIN LESION DATASET - INITIAL EDA REPORT")
    lines.append("=" * 78)
    lines.append(f"Dataset root       : {discovery['dataset_root']}")
    lines.append(f"Ground truth CSV   : {discovery['gt_csv'].name}")
    lines.append(f"Metadata CSV       : {discovery['meta_csv'].name if discovery['meta_csv'] else 'None'}")
    lines.append(f"Metadata columns   : {', '.join(summary['available_metadata_columns'])}")
    lines.append("")

    lines.append("1. ORIGINAL DATASET VERIFICATION (ALL 8 DIAGNOSTIC CLASSES)")
    lines.append("-" * 78)
    lines.append(f"Total rows in GroundTruth CSV : {summary['total_gt_rows']}")
    lines.append(f"Total image files on disk     : {summary['total_disk_images']}")
    lines.append(f"Non-single-label GT rows      : {summary['multi_or_empty_gt_rows']}")
    lines.append(f"Missing on disk               : {summary['missing_on_disk']}")
    lines.append(f"Missing in GroundTruth CSV    : {summary['missing_in_gt']}")
    lines.append(f"Folder vs GroundTruth mismatch: {summary['folder_gt_mismatches']}")
    lines.append("")
    lines.append("Original Class Counts (Official GroundTruth):")
    for cls, cnt in sorted(summary["original_gt_counts"].items(), key=lambda x: -x[1]):
        status = "INCLUDED" if cls in SELECTED_CLASSES else "EXCLUDED"
        lines.append(f"  {cls:<5} : {cnt:>6}  [{status}]")
    lines.append("")

    lines.append("2. FOUR-CLASS FILTERED SUBSET & CLASS MAPPING")
    lines.append("-" * 78)
    lines.append(f"Selected 4-class total : {summary['filtered_total']}")
    lines.append(f"Excluded images total  : {summary['excluded_total']}")
    lines.append("Class Mapping & Distribution:")
    for cls in SELECTED_CLASSES:
        idx = CLASS_TO_INDEX[cls]
        cnt = int((df["label"] == cls).sum())
        pct = cnt / len(df) * 100.0
        lines.append(f"  {idx} = {cls} ({CLASS_FULL_NAMES[cls]:<20}): {cnt:>6} ({pct:5.2f}%)")
    lines.append("")

    lines.append("3. METADATA & LESION IDENTIFIER ANALYSIS")
    lines.append("-" * 78)
    for col in ["lesion_id", "age_approx", "anatom_site_general", "sex"]:
        if col in df.columns:
            nn = int(df[col].notna().sum())
            miss = int(df[col].isna().sum())
            lines.append(f"  {col:<22}: {nn:>6} present ({nn/len(df)*100:5.2f}%), {miss:>5} missing ({miss/len(df)*100:5.2f}%)")
    lines.append(f"Lesion IDs spanning >1 class : {summary['lesion_multi_class_conflicts']}")
    lines.append("Per-Class lesion_id Coverage:")
    for cls in SELECTED_CLASSES:
        sub = df[df["label"] == cls]
        nn = int(sub["lesion_id"].notna().sum())
        uniq = int(sub["lesion_id"].dropna().nunique())
        avg_imgs = nn / uniq if uniq else 0.0
        lines.append(
            f"  {cls}: {nn:>5}/{len(sub):<5} ({nn/len(sub)*100:6.2f}%) with lesion_id | "
            f"{uniq:>4} unique lesions (avg {avg_imgs:.2f} imgs/lesion)"
        )
    lines.append("")

    lines.append("4. IMAGE INTEGRITY, DIMENSIONS & DUPLICATE VERIFICATION")
    lines.append("-" * 78)
    lines.append(f"Corrupted / unreadable images : {dup_summary['corrupted_count']}")
    lines.append(f"Color modes                   : {dup_summary['image_modes']}")
    lines.append(
        f"Width  (min / median / max)   : "
        f"{dup_summary['width_min']} / {dup_summary['width_median']} / {dup_summary['width_max']}"
    )
    lines.append(
        f"Height (min / median / max)   : "
        f"{dup_summary['height_min']} / {dup_summary['height_median']} / {dup_summary['height_max']}"
    )
    lines.append(f"Top image dimensions          : {dup_summary['top_resolutions']}")
    lines.append(f"Exact MD5 duplicate groups    : {dup_summary['exact_md5_duplicate_groups']}")
    lines.append(f"Exact MD5 extra copies        : {dup_summary['exact_md5_extra_copies']}")
    lines.append(f"Exact MD5 cross-class groups  : {dup_summary['exact_md5_cross_class_groups']}")
    lines.append(
        f"256-bit dHash (Hamming <= {dup_summary['near_dup_max_hamming']}) AND 32x32 RGB MAE <= {dup_summary['near_dup_max_mae']}:"
    )
    lines.append(
        f"  exact dHash-0 pairs: {dup_summary['exact_dhash_pairs']}, "
        f"total verified near-dup pairs: {dup_summary['total_near_dup_pairs']}"
    )
    lines.append(f"  Near-dup pairs across classes       : {dup_summary['near_dup_cross_class_pairs']}")
    lines.append(f"  Near-dup pairs across lesion_ids    : {dup_summary['near_dup_cross_lesion_pairs']}")
    lines.append(f"  Near-dup pairs with missing lesion  : {dup_summary['near_dup_involving_missing_lesion']}")
    lines.append(f"Composite split groups (lesion+dup)   : {dup_summary['unique_composite_groups']}")
    lines.append(f"Largest composite group size          : {dup_summary['largest_composite_group_size']}")
    lines.append("")

    lines.append("5. LIMITATIONS & LEAKAGE NOTES")
    lines.append("-" * 78)
    missing_lesion_cnt = int(valid_df["lesion_id"].isna().sum())
    lines.append(
        f"- {missing_lesion_cnt} of {len(valid_df)} images ({missing_lesion_cnt/len(valid_df)*100:.2f}%) "
        "do not have a lesion_id in ISIC_2019_Training_Metadata.csv (primarily older ISIC_000xxxx "
        "archive entries from NV, MEL, and BKL)."
    )
    lines.append(
        "- Therefore, splitting uses composite group IDs combining official lesion_id (where present), "
        "exact MD5 clusters, and verified 256-bit dHash + 32x32 RGB MAE near-duplicate clusters. "
        "While this prevents known lesion-level and visual duplicate leakage, absolute lesion-level "
        "independence cannot be guaranteed for the 8.93% of images lacking lesion_id if visually "
        "distinct views of the same unlabeled lesion exist."
    )
    lines.append(
        "- 2 exact MD5 duplicate pairs exist between BCN_0003560 (MEL) and BCN_0000237 (NV) in the "
        "official ISIC 2019 metadata; composite grouping keeps them in the same partition rather than "
        "silently altering official labels, preventing train/test leakage."
    )
    lines.append(
        f"- Class imbalance is substantial: NV ({int((valid_df['label']=='NV').sum())}) vs "
        f"BKL ({int((valid_df['label']=='BKL').sum())}) is ~{int((valid_df['label']=='NV').sum())/int((valid_df['label']=='BKL').sum()):.2f}:1."
    )

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Saved EDA report to: {report_path}")
    print("\n" + report_text)
    return report_path


def create_stratified_group_split(
    df: pd.DataFrame,
    processed_dir: Path = PROCESSED_DIR,
    report_path: Path = DATASET_DIR / "dataset_report.txt",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Create a leak-free 70% / 15% / 15% stratified group-aware split over composite
    groups (lesion_id + exact MD5 + verified near-duplicates) using 20-fold
    StratifiedGroupKFold, materialize unaugmented JPEGs into Dataset/Processed/,
    verify zero partition overlap, and append the split report.
    """
    valid_df = df[~df["corrupted"]].copy().reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    folds = np.zeros(len(valid_df), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(
        sgkf.split(valid_df, valid_df["target"], valid_df["split_group"])
    ):
        folds[test_idx] = fold_idx

    valid_df["split"] = np.where(
        folds < 14, "train", np.where(folds < 17, "val", "test")
    )

    # Verify zero group, lesion_id, and exact MD5 overlap across partitions
    partitions = ["train", "val", "test"]
    for i, p1 in enumerate(partitions):
        for p2 in partitions[i + 1 :]:
            s1 = valid_df[valid_df["split"] == p1]
            s2 = valid_df[valid_df["split"] == p2]
            grp_overlap = len(set(s1["split_group"]) & set(s2["split_group"]))
            md5_overlap = len(set(s1["file_md5"]) & set(s2["file_md5"]))
            les_overlap = 0
            if "lesion_id" in valid_df.columns:
                les_overlap = len(
                    set(s1["lesion_id"].dropna()) & set(s2["lesion_id"].dropna())
                )
            if grp_overlap or md5_overlap or les_overlap:
                raise RuntimeError(
                    f"Data leakage detected between {p1} and {p2}: "
                    f"group={grp_overlap}, md5={md5_overlap}, lesion_id={les_overlap}"
                )

    # Materialize Dataset/Processed/{train,val,test}/{BCC,BKL,MEL,NV} cleanly
    for split in partitions:
        for cls in SELECTED_CLASSES:
            target_dir = processed_dir / split / cls
            target_dir.mkdir(parents=True, exist_ok=True)
            for existing in target_dir.iterdir():
                if existing.is_file():
                    existing.unlink()

    copy_tasks: list[tuple[str, str]] = []
    for row in valid_df.itertuples():
        dst = processed_dir / row.split / row.label / f"{row.image}{row.ext}"
        copy_tasks.append((row.filepath, str(dst)))

    print(f"Copying {len(copy_tasks)} original unaugmented images to {processed_dir}...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda pair: shutil.copy2(pair[0], pair[1]), copy_tasks))

    # Verify materialized disk counts
    disk_counts: dict[str, dict[str, int]] = {s: {} for s in partitions}
    for split in partitions:
        for cls in SELECTED_CLASSES:
            cls_dir = processed_dir / split / cls
            cnt = sum(1 for p in cls_dir.iterdir() if p.is_file())
            expected = int(((valid_df["split"] == split) & (valid_df["label"] == cls)).sum())
            if cnt != expected:
                raise RuntimeError(
                    f"Split verification mismatch at {cls_dir}: found {cnt}, expected {expected}"
                )
            disk_counts[split][cls] = cnt

    # Append/update Section 6 in Dataset/dataset_report.txt cleanly
    split_lines = [
        "",
        "6. DATA SPLIT & LEAKAGE VERIFICATION (70% / 15% / 15% STRATIFIED GROUP SPLIT)",
        "-" * 78,
        f"Random seed : {seed}",
        "Method      : StratifiedGroupKFold(n_splits=20) over composite (lesion_id + MD5 + near-dup) groups",
        "Overlap checks (train∩val / train∩test / val∩test): 0 / 0 / 0 groups, lesions, and MD5 hashes",
        "",
        "Materialized Split Counts:",
    ]
    for split in partitions:
        row_total = sum(disk_counts[split].values())
        pct = row_total / len(valid_df) * 100.0
        detail = ", ".join(f"{cls}={disk_counts[split][cls]}" for cls in SELECTED_CLASSES)
        split_lines.append(f"  {split:<5}: {row_total:>5} ({pct:5.2f}%)  [{detail}]")

    if report_path.exists():
        existing_text = report_path.read_text(encoding="utf-8")
        marker = "\n6. DATA SPLIT & LEAKAGE VERIFICATION"
        if marker in existing_text:
            existing_text = existing_text[: existing_text.index(marker)].rstrip() + "\n"
        report_path.write_text(existing_text + "\n".join(split_lines) + "\n", encoding="utf-8")
    else:
        report_path.write_text("\n".join(split_lines) + "\n", encoding="utf-8")

    print("\n".join(split_lines))
    return valid_df


def compute_class_weights(train_dir: Path) -> tuple[dict[int, float], list[float]]:
    """Compute inverse-frequency balanced class weights strictly from the training split."""
    counts: dict[int, int] = {}
    for idx, cls in enumerate(SELECTED_CLASSES):
        cls_dir = train_dir / cls
        counts[idx] = sum(1 for p in cls_dir.iterdir() if p.is_file())
    total = sum(counts.values())
    n_classes = len(SELECTED_CLASSES)
    class_weight = {idx: float(total / (n_classes * cnt)) for idx, cnt in counts.items()}
    # Normalized alpha weights summing to 1.0 for CategoricalFocalCrossentropy
    raw = [1.0 / counts[idx] for idx in range(n_classes)]
    raw_sum = sum(raw)
    focal_alpha = [float(r / raw_sum) for r in raw]
    return class_weight, focal_alpha


def build_augmentation_layer(seed: int = RANDOM_SEED):
    """Moderate medically-preserving training-only augmentation layer (0-255 pixel range)."""
    import keras

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            keras.layers.RandomRotation(0.06, fill_mode="reflect", seed=seed),
            keras.layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", seed=seed),
            keras.layers.RandomZoom(0.10, 0.10, fill_mode="reflect", seed=seed),
            keras.layers.RandomBrightness(0.08, value_range=(0, 255), seed=seed),
            keras.layers.RandomContrast(0.08, value_range=(0, 255), seed=seed),
        ],
        name="moderate_lesion_augmentation",
    )


def create_tf_dataset(
    split_dir: Path,
    image_size: int,
    batch_size: int,
    shuffle: bool = False,
    augment: bool = False,
    seed: int = RANDOM_SEED,
):
    """
    Stream RGB images from Dataset/Processed/{split}/{class} without loading all
    images into RAM or saving resized duplicates. Pixel values stay in [0, 255]
    for EfficientNetV2B0(include_preprocessing=True).
    """
    import tensorflow as tf

    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=SELECTED_CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=shuffle,
        seed=seed,
        interpolation="bilinear",
    )

    if augment:
        aug_layer = build_augmentation_layer(seed=seed)
        ds = ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.prefetch(2)


def build_efficientnetv2b0_model(
    image_size: int,
    dropout_rate: float = 0.25,
    dense_units: int = 512,
    num_classes: int = 4,
):
    """
    Build modified EfficientNetV2B0 model:
    Input -> EfficientNetV2B0 (ImageNet, include_preprocessing=True) -> GAP ->
    Dense(512, relu) -> Dropout(0.25) -> Dense(4, softmax).
    """
    import keras

    inputs = keras.Input(shape=(image_size, image_size, 3), name="input_image")
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        include_preprocessing=True,
    )
    backbone.trainable = False

    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(backbone.output)
    x = keras.layers.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="predictions"
    )(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"EfficientNetV2B0_{image_size}")
    return model, backbone


def unfreeze_top_backbone_layers(backbone, unfreeze_top_n: int = 40) -> int:
    """
    Unfreeze the top N layers of the EfficientNetV2B0 backbone while keeping all
    BatchNormalization layers frozen to preserve ImageNet running statistics.
    """
    import keras

    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - unfreeze_top_n)
    unfrozen = 0
    for idx, layer in enumerate(backbone.layers):
        if idx < cutoff or isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen += 1
    return unfrozen


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Compute full multi-class and per-class evaluation metrics."""
    y_pred = np.argmax(y_prob, axis=1)
    y_true_bin = label_binarize(y_true, classes=list(range(len(SELECTED_CLASSES))))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_auc = float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=SELECTED_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    mel_recall = float(report_dict["MEL"]["recall"])

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_roc_auc": macro_auc,
        "mel_recall": mel_recall,
        "per_class": {cls: report_dict[cls] for cls in SELECTED_CLASSES},
        "confusion_matrix": cm.tolist(),
    }


def _collect_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a tf.data.Dataset and return integer labels and probabilities."""
    y_true_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []
    for batch_x, batch_y in dataset:
        probs = model.predict(batch_x, verbose=0)
        y_prob_list.append(probs)
        y_true_list.append(np.argmax(batch_y.numpy(), axis=1))
    return np.concatenate(y_true_list, axis=0), np.concatenate(y_prob_list, axis=0)


def _merge_histories(h1: dict[str, list[float]], h2: dict[str, list[float]]) -> dict[str, list[float]]:
    """Concatenate Stage 1 and Stage 2 Keras history dictionaries."""
    merged: dict[str, list[float]] = {}
    for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        merged[key] = list(h1.get(key, [])) + list(h2.get(key, []))
    merged["stage1_epochs"] = [len(h1.get("loss", []))]
    return merged


def run_single_experiment(
    exp_cfg: dict[str, Any],
    class_weights: dict[int, float],
    focal_alpha: list[float],
    models_dir: Path = MODELS_DIR,
    seed: int = RANDOM_SEED,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """
    Execute a single controlled two-stage transfer-learning experiment on train/val
    (never inspecting the test set). Saves its best validation checkpoint and JSON summary.
    """
    import keras

    name = exp_cfg["name"]
    ckpt_dir = models_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}.keras"
    result_json_path = ckpt_dir / f"{name}_result.json"

    if not force_retrain and ckpt_path.exists() and result_json_path.exists():
        cached = json.loads(result_json_path.read_text(encoding="utf-8"))
        vm = cached["val_metrics"]
        print(
            f"\n[RESUME CACHE] Skipping completed experiment {name}: "
            f"Val Acc={vm['accuracy']:.4f} | Macro F1={vm['macro_f1']:.4f} | "
            f"Bal Acc={vm['balanced_accuracy']:.4f} | MEL Recall={vm['mel_recall']:.4f}"
        )
        return cached

    set_global_seed(seed)
    keras.backend.clear_session()

    img_size = int(exp_cfg["image_size"])
    batch_size = int(exp_cfg["batch_size"])
    use_class_weight = bool(exp_cfg.get("use_class_weight", False))
    use_focal_loss = bool(exp_cfg.get("use_focal_loss", False))
    use_augment = bool(exp_cfg.get("use_augment", False))
    stage1_epochs = int(exp_cfg.get("stage1_epochs", 6))
    stage2_epochs = int(exp_cfg.get("stage2_epochs", 14))
    stage1_lr = float(exp_cfg.get("stage1_lr", 1e-4))
    stage2_lr = float(exp_cfg.get("stage2_lr", 1e-5))
    unfreeze_layers = int(exp_cfg.get("unfreeze_layers", 80))

    print(f"\n{'='*78}\nRUNNING EXPERIMENT: {name}\nConfig: {exp_cfg}\n{'='*78}")

    train_ds = create_tf_dataset(
        PROCESSED_DIR / "train", img_size, batch_size, shuffle=True, augment=use_augment, seed=seed
    )
    val_ds = create_tf_dataset(
        PROCESSED_DIR / "val", img_size, batch_size, shuffle=False, augment=False, seed=seed
    )

    model, backbone = build_efficientnetv2b0_model(
        image_size=img_size,
        dropout_rate=float(exp_cfg.get("dropout", 0.25)),
        dense_units=int(exp_cfg.get("dense_units", 512)),
    )

    if use_focal_loss:
        loss_fn = keras.losses.CategoricalFocalCrossentropy(
            alpha=focal_alpha, gamma=float(exp_cfg.get("focal_gamma", 2.0))
        )
    else:
        loss_fn = keras.losses.CategoricalCrossentropy()

    fit_class_weight = class_weights if use_class_weight else None
    start_time = time.time()

    # Stage 1: Frozen backbone head training
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=stage1_lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    cb_stage1 = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
    ]
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        class_weight=fit_class_weight,
        callbacks=cb_stage1,
        shuffle=False,
        verbose=2,
    )

    # Stage 2: Fine-tune upper layers with frozen BatchNormalization
    unfrozen = unfreeze_top_backbone_layers(backbone, unfreeze_top_n=unfreeze_layers)
    print(f"Stage 2 unfrozen non-BN backbone layers: {unfrozen}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=stage2_lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    cb_stage2 = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=int(exp_cfg.get("stage2_lr_patience", 2)),
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(exp_cfg.get("stage2_es_patience", 4)),
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        class_weight=fit_class_weight,
        callbacks=cb_stage2,
        shuffle=False,
        verbose=2,
    )
    elapsed_sec = time.time() - start_time

    # Reload best Stage 2 validation checkpoint for unbiased validation scoring
    best_model = keras.models.load_model(ckpt_path, compile=False)
    y_val_true, y_val_prob = _collect_predictions(best_model, val_ds)
    val_metrics = compute_classification_metrics(y_val_true, y_val_prob)
    history = _merge_histories(h1.history, h2.history)

    print(
        f"[{name}] Val Acc={val_metrics['accuracy']:.4f} | "
        f"Macro F1={val_metrics['macro_f1']:.4f} | "
        f"Bal Acc={val_metrics['balanced_accuracy']:.4f} | "
        f"MEL Recall={val_metrics['mel_recall']:.4f} | "
        f"Time={elapsed_sec:.1f}s"
    )
    result_payload = {
        "name": name,
        "config": exp_cfg,
        "checkpoint_path": str(ckpt_path),
        "training_time_sec": round(elapsed_sec, 2),
        "val_metrics": val_metrics,
        "history": history,
    }
    result_json_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    return result_payload


def _selection_score(val_metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    """Model selection priority: Macro F1, Balanced Accuracy, Melanoma Recall, Accuracy."""
    return (
        val_metrics["macro_f1"],
        val_metrics["balanced_accuracy"],
        val_metrics["mel_recall"],
        val_metrics["accuracy"],
    )


def _save_experiment_table(all_results: list[dict[str, Any]]) -> None:
    """Write incremental experiment comparison CSV to Results/experiment_results.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_rows = []
    for r in all_results:
        vm = r["val_metrics"]
        cfg = r["config"]
        exp_rows.append(
            {
                "experiment": r["name"],
                "image_size": cfg["image_size"],
                "batch_size": cfg["batch_size"],
                "class_weighted": bool(cfg.get("use_class_weight", False)),
                "focal_loss": bool(cfg.get("use_focal_loss", False)),
                "augmentation": bool(cfg.get("use_augment", False)),
                "unfreeze_layers": int(cfg.get("unfreeze_layers", 80)),
                "stage2_lr": float(cfg.get("stage2_lr", 1e-5)),
                "training_time_sec": r["training_time_sec"],
                "val_accuracy": vm["accuracy"],
                "val_balanced_accuracy": vm["balanced_accuracy"],
                "val_macro_precision": vm["macro_precision"],
                "val_macro_recall": vm["macro_recall"],
                "val_macro_f1": vm["macro_f1"],
                "val_weighted_f1": vm["weighted_f1"],
                "val_macro_roc_auc": vm["macro_roc_auc"],
                "val_mel_recall": vm["mel_recall"],
            }
        )
    pd.DataFrame(exp_rows).to_csv(RESULTS_DIR / "experiment_results.csv", index=False)


def _run_experiment_isolated(
    exp_cfg: dict[str, Any],
    enable_mixed_precision: bool = False,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """
    Run a single experiment in an isolated Python subprocess (or load from cache)
    so 100% of host RAM, XLA buffers, and GPU VRAM are reclaimed after each run.
    """
    name = exp_cfg["name"]
    ckpt_dir = MODELS_DIR / "checkpoints"
    ckpt_path = ckpt_dir / f"{name}.keras"
    result_json_path = ckpt_dir / f"{name}_result.json"

    if not force_retrain and ckpt_path.exists() and result_json_path.exists():
        cached = json.loads(result_json_path.read_text(encoding="utf-8"))
        vm = cached["val_metrics"]
        print(
            f"\n[RESUME CACHE] Skipping completed experiment {name}: "
            f"Val Acc={vm['accuracy']:.4f} | Macro F1={vm['macro_f1']:.4f} | "
            f"Bal Acc={vm['balanced_accuracy']:.4f} | MEL Recall={vm['mel_recall']:.4f}"
        )
        return cached

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--stage",
        "train-single",
        "--exp-json",
        json.dumps(exp_cfg),
    ]
    if enable_mixed_precision:
        cmd.append("--mixed-precision")
    if force_retrain:
        cmd.append("--force-retrain")

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0 or not result_json_path.exists():
        raise RuntimeError(
            f"Isolated training subprocess failed for {name} (exit code {proc.returncode})."
        )
    return json.loads(result_json_path.read_text(encoding="utf-8"))


def run_controlled_experiments(
    batch_size: int = 20,
    max_epochs: int = 20,
    enable_mixed_precision: bool = False,
    force_retrain: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run controlled experiments A through G, select the best model strictly on
    validation results, and save best_model.keras + class_mapping.json + model_config.json.
    """
    class_weights, focal_alpha = compute_class_weights(PROCESSED_DIR / "train")
    print(f"Training class weights: {class_weights}")
    print(f"Focal loss alpha: {focal_alpha}")

    s1_ep = min(6, max(1, max_epochs // 3))
    s2_ep = max(1, max_epochs - s1_ep)

    # Experiments A, B, C: Resolution sweep
    res_experiments = [
        {"name": "A_Res224_Baseline", "image_size": 224, "batch_size": batch_size, "stage1_epochs": s1_ep, "stage2_epochs": s2_ep, "unfreeze_layers": 80},
        {"name": "B_Res384_Baseline", "image_size": 384, "batch_size": batch_size, "stage1_epochs": s1_ep, "stage2_epochs": s2_ep, "unfreeze_layers": 80},
        {"name": "C_Res448_Baseline", "image_size": 448, "batch_size": min(batch_size, 16) if enable_mixed_precision else batch_size, "stage1_epochs": s1_ep, "stage2_epochs": s2_ep, "unfreeze_layers": 80},
    ]

    all_results: list[dict[str, Any]] = []
    for cfg in res_experiments:
        all_results.append(_run_experiment_isolated(cfg, enable_mixed_precision=enable_mixed_precision, force_retrain=force_retrain))
        _save_experiment_table(all_results)

    best_res_run = max(all_results, key=lambda r: _selection_score(r["val_metrics"]))
    best_res = int(best_res_run["config"]["image_size"])
    best_bs = int(best_res_run["config"]["batch_size"])
    print(f"\nSelected best resolution from A-C on validation: {best_res}x{best_res}")

    # Experiments D, E, F at best resolution
    intervention_experiments = [
        {"name": f"D_Res{best_res}_ClassWeighted", "image_size": best_res, "batch_size": best_bs, "use_class_weight": True, "stage1_epochs": s1_ep, "stage2_epochs": s2_ep, "unfreeze_layers": 80},
        {"name": f"E_Res{best_res}_FocalLoss", "image_size": best_res, "batch_size": best_bs, "use_focal_loss": True, "stage1_epochs": s1_ep, "stage2_epochs": s2_ep, "unfreeze_layers": 80},
        {"name": f"F_Res{best_res}_Augmentation", "image_size": best_res, "batch_size": best_bs, "use_augment": True, "stage1_epochs": s1_ep, "stage2_epochs": s2_ep, "unfreeze_layers": 80},
    ]

    intervention_runs: list[dict[str, Any]] = []
    for cfg in intervention_experiments:
        res = _run_experiment_isolated(cfg, enable_mixed_precision=enable_mixed_precision, force_retrain=force_retrain)
        intervention_runs.append(res)
        all_results.append(res)
        _save_experiment_table(all_results)

    # Determine winning loss/weighting between D, E, and baseline
    d_run, e_run, f_run = intervention_runs
    best_loss_run = max([best_res_run, d_run, e_run], key=lambda r: _selection_score(r["val_metrics"]))
    use_cw_g = bool(best_loss_run["config"].get("use_class_weight", False))
    use_fl_g = bool(best_loss_run["config"].get("use_focal_loss", False))
    use_aug_g = _selection_score(f_run["val_metrics"]) >= _selection_score(best_res_run["val_metrics"])

    # Experiment G: Best configuration + improved fine-tuning (deeper unfreeze + tuned Stage 2 LR)
    g_cfg = {
        "name": f"G_Res{best_res}_ImprovedFineTune",
        "image_size": best_res,
        "batch_size": best_bs,
        "use_class_weight": use_cw_g,
        "use_focal_loss": use_fl_g,
        "use_augment": use_aug_g,
        "stage1_epochs": s1_ep,
        "stage2_epochs": s2_ep + 8,
        "stage1_lr": 1e-4,
        "stage2_lr": 3e-5,
        "unfreeze_layers": 220,
        "stage2_lr_patience": 3,
        "stage2_es_patience": 5,
    }
    all_results.append(_run_experiment_isolated(g_cfg, enable_mixed_precision=enable_mixed_precision, force_retrain=force_retrain))
    _save_experiment_table(all_results)

    # Select overall best model strictly using validation metrics
    best_run = max(all_results, key=lambda r: _selection_score(r["val_metrics"]))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best_model_dst = MODELS_DIR / "best_model.keras"
    shutil.copy2(best_run["checkpoint_path"], best_model_dst)

    mapping_payload = {
        "class_to_index": CLASS_TO_INDEX,
        "index_to_class": {str(k): v for k, v in INDEX_TO_CLASS.items()},
        "full_names": CLASS_FULL_NAMES,
    }
    (MODELS_DIR / "class_mapping.json").write_text(
        json.dumps(mapping_payload, indent=2), encoding="utf-8"
    )

    config_payload = {
        "experiment_name": best_run["name"],
        "backbone": "EfficientNetV2B0",
        "weights": "imagenet",
        "image_size": best_run["config"]["image_size"],
        "preprocessing": "RGB, bilinear resize, float32 [0, 255] with EfficientNetV2B0(include_preprocessing=True)",
        "training_config": best_run["config"],
        "best_val_metrics": best_run["val_metrics"],
    }
    (MODELS_DIR / "model_config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    print(f"\nBest validation model: {best_run['name']} -> saved to {best_model_dst}")
    return all_results, best_run


def evaluate_best_on_test_and_plot(
    all_results: list[dict[str, Any]], best_run: dict[str, Any]
) -> dict[str, Any]:
    """
    Evaluate the validation-selected best model ONCE on the untouched test set,
    save classification_report.csv, update model_config.json, and generate all graphs.
    """
    import keras

    img_size = int(best_run["config"]["image_size"])
    batch_size = int(best_run["config"]["batch_size"])

    test_ds = create_tf_dataset(
        PROCESSED_DIR / "test", img_size, batch_size, shuffle=False, augment=False
    )
    best_model = keras.models.load_model(MODELS_DIR / "best_model.keras", compile=False)
    y_test_true, y_test_prob = _collect_predictions(best_model, test_ds)
    test_metrics = compute_classification_metrics(y_test_true, y_test_prob)

    # Update model_config.json with untouched test metrics
    cfg_path = MODELS_DIR / "model_config.json"
    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_data["test_metrics"] = test_metrics
    cfg_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")

    # Save classification_report.csv
    report_rows = []
    for cls in SELECTED_CLASSES:
        m = test_metrics["per_class"][cls]
        report_rows.append(
            {
                "class": cls,
                "full_name": CLASS_FULL_NAMES[cls],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1-score"],
                "support": int(m["support"]),
            }
        )
    report_rows.append(
        {
            "class": "MACRO_AVG",
            "full_name": "Macro Average",
            "precision": test_metrics["macro_precision"],
            "recall": test_metrics["macro_recall"],
            "f1_score": test_metrics["macro_f1"],
            "support": int(len(y_test_true)),
        }
    )
    report_rows.append(
        {
            "class": "OVERALL",
            "full_name": f"Accuracy={test_metrics['accuracy']:.4f}, BalAcc={test_metrics['balanced_accuracy']:.4f}, AUC={test_metrics['macro_roc_auc']:.4f}",
            "precision": test_metrics["accuracy"],
            "recall": test_metrics["balanced_accuracy"],
            "f1_score": test_metrics["weighted_f1"],
            "support": int(len(y_test_true)),
        }
    )
    pd.DataFrame(report_rows).to_csv(RESULTS_DIR / "classification_report.csv", index=False)

    exp_df = pd.read_csv(RESULTS_DIR / "experiment_results.csv")
    names = exp_df["experiment"].tolist()

    def _bar_plot(col: str, title: str, ylabel: str, filename: str, color: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(names, exp_df[col], color=color, edgecolor="black", alpha=0.88)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=25)
        for bar, val in zip(bars, exp_df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / filename, dpi=150)
        plt.close(fig)

    # 1. Accuracy comparison graph
    _bar_plot("val_accuracy", "Validation Accuracy Across Controlled Experiments", "Accuracy", "model_accuracy.png", "#2b6cb0")
    # 2. Macro F1 comparison graph
    _bar_plot("val_macro_f1", "Validation Macro F1-Score Across Experiments", "Macro F1", "macro_f1_comparison.png", "#2f855a")
    # 3. Balanced accuracy comparison graph
    _bar_plot("val_balanced_accuracy", "Validation Balanced Accuracy Across Experiments", "Balanced Accuracy", "balanced_accuracy_comparison.png", "#6b46c1")
    # 4. Melanoma recall comparison graph
    _bar_plot("val_mel_recall", "Validation Melanoma (MEL) Recall Across Experiments", "MEL Recall", "melanoma_recall_comparison.png", "#c53030")

    # Combined multi-metric model_comparison.png
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    w = 0.2
    ax.bar(x - 1.5 * w, exp_df["val_accuracy"], w, label="Accuracy")
    ax.bar(x - 0.5 * w, exp_df["val_macro_f1"], w, label="Macro F1")
    ax.bar(x + 0.5 * w, exp_df["val_balanced_accuracy"], w, label="Balanced Acc")
    ax.bar(x + 1.5 * w, exp_df["val_mel_recall"], w, label="MEL Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Controlled Experiments Comparison (Validation Metrics)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    # 5. Best model confusion matrix (untouched test set)
    cm = np.array(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SELECTED_CLASSES,
        yticklabels=SELECTED_CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(
        f"Test Confusion Matrix — {best_run['name']} (Acc={test_metrics['accuracy']:.4f})",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 6 & 7. Training vs Validation Accuracy & Loss for best model
    hist = best_run["history"]
    epochs_range = range(1, len(hist["accuracy"]) + 1)
    s1_end = hist["stage1_epochs"][0]

    for metric_key, title, ylabel, fname in [
        ("accuracy", "Training vs Validation Accuracy", "Accuracy", "training_validation_accuracy.png"),
        ("loss", "Training vs Validation Loss", "Loss", "training_validation_loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs_range, hist[metric_key], marker="o", label=f"Train {ylabel}")
        ax.plot(epochs_range, hist[f"val_{metric_key}"], marker="s", label=f"Val {ylabel}")
        if 0 < s1_end < len(epochs_range):
            ax.axvline(s1_end + 0.5, color="gray", linestyle="--", label="Stage 2 Fine-Tuning Start")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {best_run['name']}", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / fname, dpi=150)
        plt.close(fig)

    # 8. One-vs-Rest ROC curves on untouched test set
    y_test_bin = label_binarize(y_test_true, classes=list(range(len(SELECTED_CLASSES))))
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, cls in enumerate(SELECTED_CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, idx], y_test_prob[:, idx])
        cls_auc = roc_auc_score(y_test_bin[:, idx], y_test_prob[:, idx])
        ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {cls_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"One-vs-Rest Test ROC Curves (Macro AUC = {test_metrics['macro_roc_auc']:.4f})",
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    # 9. Per-class performance graph on untouched test set
    per_cls_df = pd.DataFrame(
        [
            {
                "Class": cls,
                "Precision": test_metrics["per_class"][cls]["precision"],
                "Recall": test_metrics["per_class"][cls]["recall"],
                "F1-Score": test_metrics["per_class"][cls]["f1-score"],
            }
            for cls in SELECTED_CLASSES
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SELECTED_CLASSES))
    w = 0.25
    ax.bar(x - w, per_cls_df["Precision"], w, label="Precision")
    ax.bar(x, per_cls_df["Recall"], w, label="Recall")
    ax.bar(x + w, per_cls_df["F1-Score"], w, label="F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(SELECTED_CLASSES)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"Test Per-Class Precision / Recall / F1 — {best_run['name']}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_class_performance.png", dpi=150)
    plt.close(fig)

    print("\n" + "=" * 78)
    print("FINAL UNTOUCHED TEST SET EVALUATION")
    print("=" * 78)
    print(f"Selected Best Model : {best_run['name']}")
    print(f"Test Accuracy       : {test_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy   : {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro Precision     : {test_metrics['macro_precision']:.4f}")
    print(f"Macro Recall        : {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score      : {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score   : {test_metrics['weighted_f1']:.4f}")
    print(f"Macro ROC-AUC       : {test_metrics['macro_roc_auc']:.4f}")
    print(f"Melanoma Recall     : {test_metrics['mel_recall']:.4f}")
    return test_metrics


def run_smoke_check() -> None:
    """Verify tf.data pipeline, [0, 255] preprocessing, model architecture, and Stage 2 unfreezing."""
    import keras

    set_global_seed(RANDOM_SEED)
    check_environment()
    ds = create_tf_dataset(PROCESSED_DIR / "train", image_size=224, batch_size=4, shuffle=False, augment=True)
    batch_x, batch_y = next(iter(ds))
    x_min, x_max = float(batch_x.numpy().min()), float(batch_x.numpy().max())
    print(f"Smoke batch shape: {batch_x.shape}, label shape: {batch_y.shape}, pixel range: [{x_min:.2f}, {x_max:.2f}]")
    model, backbone = build_efficientnetv2b0_model(image_size=224)
    preds = model(batch_x, training=False)
    print(f"Forward pass output shape: {preds.shape}, row sums: {preds.numpy().sum(axis=1)}")
    unfrozen = unfreeze_top_backbone_layers(backbone, unfreeze_top_n=40)
    bn_trainable = sum(
        1 for l in backbone.layers if isinstance(l, keras.layers.BatchNormalization) and l.trainable
    )
    print(f"Stage 2 unfrozen non-BN layers: {unfrozen}, trainable BN layers: {bn_trainable}")


def run_eda_stage() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute download verification, dataset inspection, duplicate checks, and EDA."""
    set_global_seed(RANDOM_SEED)
    check_environment()
    dataset_root = download_dataset(ORIGINAL_DIR)
    discovery = discover_dataset_files(dataset_root)
    filtered_df, summary = inspect_and_filter_dataset(discovery)
    enriched_df, dup_summary = verify_images_and_find_duplicates(filtered_df)
    save_representative_samples(enriched_df, RESULTS_DIR / "representative_samples.png")
    generate_eda_report(discovery, summary, enriched_df, dup_summary, DATASET_DIR / "dataset_report.txt")
    return enriched_df, dup_summary


# ===========================================================================
# PHASE 5 — MULTI-ARCHITECTURE COMPARISON, IMBALANCE HANDLING & ENSEMBLE
# ===========================================================================


def require_gpu_environment() -> dict[str, Any]:
    """
    Enforce GPU availability inside WSL2/Linux, enable memory growth,
    verify an actual TensorFlow operation executes on /GPU:0, and return details.
    Stops immediately if no GPU is detected.
    """
    import tensorflow as tf

    info = check_environment()
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "GPU detection failed (0 GPUs visible to TensorFlow). "
            "Stopping Phase 5 training rather than silently falling back to CPU."
        )
    details = tf.config.experimental.get_device_details(gpus[0])
    mem_info = tf.config.experimental.get_memory_info("GPU:0")
    with tf.device("/GPU:0"):
        a = tf.random.normal([512, 512], dtype=tf.float16)
        b = tf.random.normal([512, 512], dtype=tf.float16)
        c = tf.matmul(a, b)
        _ = c.numpy()
    print(
        f"Verified GPU op on {c.device} | Device: {details.get('device_name')} "
        f"(CC {details.get('compute_capability')}) | TF MemInfo: {mem_info}"
    )
    info["device_details"] = details
    return info


def build_phase5_model(
    arch_name: str,
    image_size: int,
    dropout_rate: float = 0.25,
    dense_units: int = 512,
    num_classes: int = 4,
):
    """
    Build one of the four Phase 5 pretrained architectures with serializable
    architecture-specific preprocessing (no Lambda layers) and unified head:
    GAP -> Dense(512, relu) -> Dropout(0.25) -> Dense(4, softmax).
    """
    import keras

    inputs = keras.Input(shape=(image_size, image_size, 3), name="input_image")

    if arch_name == "DenseNet121":
        # Exact match to keras.applications.densenet.preprocess_input (mode='torch')
        x = keras.layers.Rescaling(1.0 / 255.0, name="densenet_rescale")(inputs)
        x = keras.layers.Normalization(
            mean=[0.485, 0.456, 0.406],
            variance=[0.229**2, 0.224**2, 0.225**2],
            name="densenet_norm",
        )(x)
        backbone = keras.applications.DenseNet121(
            include_top=False, weights="imagenet", input_tensor=x
        )
    elif arch_name == "ResNet50V2":
        # Exact match to keras.applications.resnet_v2.preprocess_input (mode='tf')
        x = keras.layers.Rescaling(
            scale=1.0 / 127.5, offset=-1.0, name="resnetv2_rescale"
        )(inputs)
        backbone = keras.applications.ResNet50V2(
            include_top=False, weights="imagenet", input_tensor=x
        )
    elif arch_name == "MobileNetV3Large":
        backbone = keras.applications.MobileNetV3Large(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            include_preprocessing=True,
        )
    elif arch_name == "ConvNeXtTiny":
        backbone = keras.applications.ConvNeXtTiny(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            include_preprocessing=True,
        )
    else:
        raise ValueError(f"Unsupported Phase 5 architecture: {arch_name}")

    backbone.trainable = False
    feat = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(backbone.output)
    feat = keras.layers.Dense(dense_units, activation="relu", name="head_dense")(feat)
    feat = keras.layers.Dropout(dropout_rate, name="head_dropout")(feat)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="predictions"
    )(feat)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name=f"{arch_name}_{image_size}"
    )
    return model, backbone


def unfreeze_phase5_backbone(backbone, unfreeze_fraction: float = 0.35) -> tuple[int, int]:
    """
    Unfreeze top `unfreeze_fraction` of backbone layers while keeping all
    BatchNormalization layers frozen to preserve transfer-learning statistics.
    Returns (unfrozen_non_bn_layers, frozen_bn_layers).
    """
    import keras

    backbone.trainable = True
    n_layers = len(backbone.layers)
    cutoff = max(0, int(n_layers * (1.0 - unfreeze_fraction)))
    unfrozen = 0
    frozen_bn = 0
    for idx, layer in enumerate(backbone.layers):
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1
        elif idx < cutoff:
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen += 1
    return unfrozen, frozen_bn


def create_weighted_sampled_tf_dataset(
    train_dir: Path,
    image_size: int,
    batch_size: int,
    augment: bool = False,
    seed: int = RANDOM_SEED,
) -> tuple[Any, int]:
    """
    Build a class-balanced training stream via tf.data.Dataset.sample_from_datasets
    over the 4 individual class folders (uniform 0.25 probability per class).
    Returns (dataset, steps_per_epoch).
    """
    import tensorflow as tf

    per_class_ds = []
    total_images = 0
    for cls_idx, cls_name in enumerate(SELECTED_CLASSES):
        cls_dir = train_dir / cls_name
        file_paths = sorted(str(p) for p in cls_dir.iterdir() if p.is_file())
        total_images += len(file_paths)
        one_hot = tf.one_hot(cls_idx, depth=len(SELECTED_CLASSES), dtype=tf.float32)

        path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
        path_ds = path_ds.shuffle(len(file_paths), seed=seed + cls_idx, reshuffle_each_iteration=True)

        def _load_image(fp, lbl=one_hot):
            raw = tf.io.read_file(fp)
            img = tf.io.decode_jpeg(raw, channels=3)
            img = tf.image.resize(img, (image_size, image_size), method=tf.image.ResizeMethod.BILINEAR)
            img = tf.cast(img, tf.float32)
            return img, lbl

        img_ds = path_ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE).repeat()
        per_class_ds.append(img_ds)

    sampled = tf.data.Dataset.sample_from_datasets(
        per_class_ds, weights=[0.25, 0.25, 0.25, 0.25], seed=seed
    ).batch(batch_size, drop_remainder=False)

    if augment:
        aug_layer = build_augmentation_layer(seed=seed)
        sampled = sampled.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    steps_per_epoch = math.ceil(total_images / batch_size)
    return sampled.prefetch(2), steps_per_epoch


def _collect_predictions_timed(
    model, dataset, num_images: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Stream timed validation inference (excluding batch 0 warmup) without loading dataset into RAM."""
    y_true_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []
    t0: float | None = None
    for idx, (batch_x, batch_y) in enumerate(dataset):
        probs = model.predict(batch_x, verbose=0)
        if idx == 0:
            t0 = time.perf_counter()
        y_prob_list.append(probs)
        y_true_list.append(np.argmax(batch_y.numpy(), axis=1))
    elapsed = (time.perf_counter() - t0) if t0 is not None else 0.0
    measured_images = max(1, num_images - len(y_true_list[0])) if len(y_true_list) > 1 else max(1, num_images)
    ms_per_img = (elapsed / measured_images) * 1000.0
    est_total_sec = (ms_per_img / 1000.0) * num_images
    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    return y_true, y_prob, float(est_total_sec), float(ms_per_img)


def run_phase5_single_experiment(
    exp_cfg: dict[str, Any],
    class_weights: dict[int, float],
    focal_alpha: list[float],
    force_retrain: bool = False,
) -> dict[str, Any]:
    """
    Execute a single Phase 5 two-stage transfer-learning experiment on GPU,
    evaluate strictly on validation, measure parameters/file-size/inference-time,
    and save checkpoint + result JSON + val_probs.npy under Advanced_Experiments/Models/<arch>/.
    """
    import keras
    import tensorflow as tf

    name = exp_cfg["name"]
    arch = exp_cfg["architecture"]
    arch_dir = ADV_MODELS_DIR / arch
    arch_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = arch_dir / f"{name}.keras"
    result_json_path = arch_dir / f"{name}_result.json"
    val_probs_path = arch_dir / f"{name}_val_probs.npy"

    if not force_retrain and ckpt_path.exists() and result_json_path.exists() and val_probs_path.exists():
        cached = json.loads(result_json_path.read_text(encoding="utf-8"))
        vm = cached["val_metrics"]
        print(
            f"\n[RESUME CACHE] Skipping {name}: "
            f"Val Acc={vm['accuracy']:.4f} | Macro F1={vm['macro_f1']:.4f} | "
            f"Bal Acc={vm['balanced_accuracy']:.4f} | MEL Recall={vm['mel_recall']:.4f}"
        )
        return cached

    require_gpu_environment()
    set_global_seed(RANDOM_SEED)
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy("mixed_float16")

    img_size = int(exp_cfg.get("image_size", 384))
    batch_size = int(exp_cfg.get("batch_size", 8))
    strategy = exp_cfg.get("imbalance_strategy", "standard")
    use_augment = bool(exp_cfg.get("use_augment", False))
    stage1_epochs = int(exp_cfg.get("stage1_epochs", 6))
    stage2_epochs = int(exp_cfg.get("stage2_epochs", 14))
    stage1_lr = float(exp_cfg.get("stage1_lr", 1e-4))
    stage2_lr = float(exp_cfg.get("stage2_lr", 1e-5))
    unfreeze_fraction = float(exp_cfg.get("unfreeze_fraction", 0.35))

    print(f"\n{'='*78}\nPHASE 5 EXPERIMENT: {name}\nConfig: {exp_cfg}\n{'='*78}")

    steps_per_epoch = None
    if strategy == "weighted_sampling":
        train_ds, steps_per_epoch = create_weighted_sampled_tf_dataset(
            PROCESSED_DIR / "train", img_size, batch_size, augment=use_augment, seed=RANDOM_SEED
        )
    else:
        train_ds = create_tf_dataset(
            PROCESSED_DIR / "train", img_size, batch_size, shuffle=True, augment=use_augment, seed=RANDOM_SEED
        )

    val_ds = create_tf_dataset(
        PROCESSED_DIR / "val", img_size, batch_size, shuffle=False, augment=False, seed=RANDOM_SEED
    )

    model, backbone = build_phase5_model(
        arch_name=arch,
        image_size=img_size,
        dropout_rate=float(exp_cfg.get("dropout", 0.25)),
        dense_units=int(exp_cfg.get("dense_units", 512)),
    )

    if strategy == "focal_loss":
        loss_fn = keras.losses.CategoricalFocalCrossentropy(
            alpha=focal_alpha, gamma=float(exp_cfg.get("focal_gamma", 2.0))
        )
    else:
        loss_fn = keras.losses.CategoricalCrossentropy()

    fit_class_weight = class_weights if strategy == "class_weight" else None

    tf.config.experimental.reset_memory_stats("GPU:0")
    t0 = time.time()

    # Stage 1: Frozen backbone head training
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=stage1_lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    cb_stage1 = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
    ]
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        steps_per_epoch=steps_per_epoch,
        class_weight=fit_class_weight,
        callbacks=cb_stage1,
        shuffle=False,
        verbose=2,
    )

    # Stage 2: Fine-tuning upper layers
    unfrozen, frozen_bn = unfreeze_phase5_backbone(backbone, unfreeze_fraction=unfreeze_fraction)
    print(f"Stage 2 unfrozen non-BN layers: {unfrozen} | Frozen BN layers: {frozen_bn}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=stage2_lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    cb_stage2 = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
    ]
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        steps_per_epoch=steps_per_epoch,
        class_weight=fit_class_weight,
        callbacks=cb_stage2,
        shuffle=False,
        verbose=2,
    )
    elapsed_sec = time.time() - t0
    gpu_mem = tf.config.experimental.get_memory_info("GPU:0")
    peak_vram_mb = round(gpu_mem["peak"] / (1024**2), 2)

    # Evaluate strictly on validation using saved best checkpoint
    best_model = keras.models.load_model(ckpt_path, compile=False)
    y_val_true, y_val_prob, inf_sec, inf_ms_img = _collect_predictions_timed(
        best_model, val_ds, num_images=3501
    )
    val_metrics = compute_classification_metrics(y_val_true, y_val_prob)
    history = _merge_histories(h1.history, h2.history)
    param_count = int(best_model.count_params())
    model_size_mb = round(ckpt_path.stat().st_size / (1024**2), 2)

    np.save(val_probs_path, y_val_prob.astype(np.float32))
    ADV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    val_targets_path = ADV_RESULTS_DIR / "val_targets.npy"
    if not val_targets_path.exists():
        np.save(val_targets_path, y_val_true.astype(np.int32))

    result_payload = {
        "name": name,
        "architecture": arch,
        "config": exp_cfg,
        "checkpoint_path": str(ckpt_path),
        "val_probs_path": str(val_probs_path),
        "training_time_sec": round(elapsed_sec, 2),
        "inference_time_val_sec": round(inf_sec, 3),
        "inference_ms_per_image": round(inf_ms_img, 3),
        "param_count": param_count,
        "model_size_mb": model_size_mb,
        "peak_tf_vram_mb": peak_vram_mb,
        "val_metrics": val_metrics,
        "history": history,
    }
    result_json_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(
        f"[{name}] Val Acc={val_metrics['accuracy']:.4f} | "
        f"Macro F1={val_metrics['macro_f1']:.4f} | "
        f"Bal Acc={val_metrics['balanced_accuracy']:.4f} | "
        f"MEL Recall={val_metrics['mel_recall']:.4f} | "
        f"AUC={val_metrics['macro_roc_auc']:.4f} | "
        f"Inf={inf_ms_img:.2f} ms/img | Peak TF VRAM={peak_vram_mb:.1f} MB | Time={elapsed_sec:.1f}s"
    )
    return result_payload


def _run_phase5_isolated(exp_cfg: dict[str, Any], force_retrain: bool = False) -> dict[str, Any]:
    """Run a Phase 5 experiment in a fresh subprocess with automatic batch-size fallback on OOM."""
    name = exp_cfg["name"]
    arch = exp_cfg["architecture"]
    arch_dir = ADV_MODELS_DIR / arch
    ckpt_path = arch_dir / f"{name}.keras"
    result_json_path = arch_dir / f"{name}_result.json"
    val_probs_path = arch_dir / f"{name}_val_probs.npy"

    if not force_retrain and ckpt_path.exists() and result_json_path.exists() and val_probs_path.exists():
        cached = json.loads(result_json_path.read_text(encoding="utf-8"))
        vm = cached["val_metrics"]
        print(
            f"\n[RESUME CACHE] Skipping {name}: "
            f"Val Acc={vm['accuracy']:.4f} | Macro F1={vm['macro_f1']:.4f} | "
            f"Bal Acc={vm['balanced_accuracy']:.4f} | MEL Recall={vm['mel_recall']:.4f}"
        )
        return cached

    cfg_try = dict(exp_cfg)
    while True:
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--stage",
            "phase5-single",
            "--exp-json",
            json.dumps(cfg_try),
        ]
        if force_retrain:
            cmd.append("--force-retrain")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0 and result_json_path.exists():
            return json.loads(result_json_path.read_text(encoding="utf-8"))

        cur_bs = int(cfg_try.get("batch_size", 8))
        if cur_bs > 2:
            next_bs = max(2, cur_bs // 2)
            print(f"WARNING: {name} failed at batch_size={cur_bs}; retrying with batch_size={next_bs}...")
            cfg_try["batch_size"] = next_bs
        else:
            raise RuntimeError(f"Phase 5 experiment {name} failed even at batch_size={cur_bs}.")


def evaluate_existing_effnet_on_val(force_recompute: bool = False) -> dict[str, Any]:
    """
    Evaluate the existing untouched EfficientNetV2B0 model (Models/best_model.keras)
    on the validation split so all 5 architectures have apples-to-apples validation
    metrics, inference time, parameter count, and file size. Never retrains A-G.
    """
    ADV_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ADV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ref_json = ADV_MODELS_DIR / "EfficientNetV2B0_ref_result.json"
    ref_probs = ADV_MODELS_DIR / "EfficientNetV2B0_ref_val_probs.npy"

    if not force_recompute and ref_json.exists() and ref_probs.exists():
        return json.loads(ref_json.read_text(encoding="utf-8"))

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--stage",
        "phase5-eval-effnet",
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0 or not ref_json.exists():
        raise RuntimeError("Failed to evaluate existing EfficientNetV2B0 reference on validation.")
    return json.loads(ref_json.read_text(encoding="utf-8"))


def _eval_effnet_ref_worker() -> None:
    """Subprocess worker that evaluates existing Models/best_model.keras on val."""
    import keras

    require_gpu_environment()
    best_model_path = MODELS_DIR / "best_model.keras"
    orig_cfg = json.loads((MODELS_DIR / "model_config.json").read_text(encoding="utf-8"))
    orig_ckpt_res_path = MODELS_DIR / "checkpoints" / f"{orig_cfg['experiment_name']}_result.json"
    orig_ckpt_res = (
        json.loads(orig_ckpt_res_path.read_text(encoding="utf-8"))
        if orig_ckpt_res_path.exists()
        else {}
    )

    img_size = int(orig_cfg.get("image_size", 384))
    val_ds = create_tf_dataset(
        PROCESSED_DIR / "val", img_size, batch_size=8, shuffle=False, augment=False, seed=RANDOM_SEED
    )
    model = keras.models.load_model(best_model_path, compile=False)
    y_val_true, y_val_prob, inf_sec, inf_ms_img = _collect_predictions_timed(
        model, val_ds, num_images=3501
    )
    val_metrics = compute_classification_metrics(y_val_true, y_val_prob)

    np.save(ADV_MODELS_DIR / "EfficientNetV2B0_ref_val_probs.npy", y_val_prob.astype(np.float32))
    val_targets_path = ADV_RESULTS_DIR / "val_targets.npy"
    if not val_targets_path.exists():
        np.save(val_targets_path, y_val_true.astype(np.int32))

    payload = {
        "name": f"Ref_{orig_cfg['experiment_name']}",
        "architecture": "EfficientNetV2B0",
        "config": orig_cfg.get("training_config", {}),
        "checkpoint_path": str(best_model_path),
        "val_probs_path": str(ADV_MODELS_DIR / "EfficientNetV2B0_ref_val_probs.npy"),
        "training_time_sec": float(orig_ckpt_res.get("training_time_sec", 0.0)),
        "inference_time_val_sec": round(inf_sec, 3),
        "inference_ms_per_image": round(inf_ms_img, 3),
        "param_count": int(model.count_params()),
        "model_size_mb": round(best_model_path.stat().st_size / (1024**2), 2),
        "peak_tf_vram_mb": 0.0,
        "val_metrics": val_metrics,
        "history": orig_ckpt_res.get("history", {}),
    }
    (ADV_MODELS_DIR / "EfficientNetV2B0_ref_result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        f"[EfficientNetV2B0 Reference] Val Acc={val_metrics['accuracy']:.4f} | "
        f"Macro F1={val_metrics['macro_f1']:.4f} | Inf={inf_ms_img:.2f} ms/img"
    )


def _resolve_project_path(raw_path: str) -> Path:
    """Resolve a path saved under WSL2 (/mnt/<drive>/...) or Windows to the local filesystem."""
    p = Path(raw_path)
    if p.exists():
        return p
    norm = raw_path.replace("\\", "/")
    marker = f"/{PROJECT_ROOT.name}/"
    if marker in norm:
        rel = norm.split(marker, 1)[1]
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return candidate
    return p


def run_phase5_ensemble(
    best_per_arch: dict[str, dict[str, Any]], val_targets: np.ndarray
) -> dict[str, Any]:
    """
    Evaluate soft-voting ensembles of top 2 and top 3 architectures strictly on
    validation probability predictions. Retain ensemble only if it outperforms
    the strongest single validation model on Macro F1 / Balanced Accuracy / MEL Recall.
    """
    ens_dir = ADV_MODELS_DIR / "Ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)

    ranked = sorted(
        best_per_arch.values(),
        key=lambda r: _selection_score(r["val_metrics"]),
        reverse=True,
    )
    best_single = ranked[0]
    best_single_score = _selection_score(best_single["val_metrics"])

    probs_by_name = {
        r["name"]: np.load(_resolve_project_path(r["val_probs_path"])).astype(np.float64)
        for r in ranked
    }

    candidates: list[dict[str, Any]] = []

    # 1. Top-2 equal soft voting
    if len(ranked) >= 2:
        top2 = ranked[:2]
        p_eq2 = 0.5 * probs_by_name[top2[0]["name"]] + 0.5 * probs_by_name[top2[1]["name"]]
        candidates.append(
            {
                "ensemble_name": "Ensemble_Top2_Equal",
                "members": [m["name"] for m in top2],
                "architectures": [m["architecture"] for m in top2],
                "checkpoints": [m["checkpoint_path"] for m in top2],
                "image_sizes": [int(m["config"].get("image_size", 384)) for m in top2],
                "weights": [0.5, 0.5],
                "probs": p_eq2,
                "val_metrics": compute_classification_metrics(val_targets, p_eq2),
            }
        )

    # 2. Top-3 equal & grid-weighted soft voting
    if len(ranked) >= 3:
        top3 = ranked[:3]
        p0 = probs_by_name[top3[0]["name"]]
        p1 = probs_by_name[top3[1]["name"]]
        p2 = probs_by_name[top3[2]["name"]]

        p_eq3 = (p0 + p1 + p2) / 3.0
        candidates.append(
            {
                "ensemble_name": "Ensemble_Top3_Equal",
                "members": [m["name"] for m in top3],
                "architectures": [m["architecture"] for m in top3],
                "checkpoints": [m["checkpoint_path"] for m in top3],
                "image_sizes": [int(m["config"].get("image_size", 384)) for m in top3],
                "weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                "probs": p_eq3,
                "val_metrics": compute_classification_metrics(val_targets, p_eq3),
            }
        )

        # Simplex grid search (step 0.05) strictly on validation probabilities
        best_w = (1.0, 0.0, 0.0)
        best_grid_metrics = None
        best_grid_score = (-1.0, -1.0, -1.0, -1.0)
        best_grid_probs = p_eq3
        steps = 20
        for i in range(1, steps):
            for j in range(1, steps - i):
                k = steps - i - j
                if k <= 0:
                    continue
                w0, w1, w2 = i / steps, j / steps, k / steps
                p_mix = w0 * p0 + w1 * p1 + w2 * p2
                m_mix = compute_classification_metrics(val_targets, p_mix)
                sc = _selection_score(m_mix)
                if sc > best_grid_score:
                    best_grid_score = sc
                    best_w = (w0, w1, w2)
                    best_grid_metrics = m_mix
                    best_grid_probs = p_mix

        if best_grid_metrics is not None:
            candidates.append(
                {
                    "ensemble_name": "Ensemble_Top3_Weighted",
                    "members": [m["name"] for m in top3],
                    "architectures": [m["architecture"] for m in top3],
                    "checkpoints": [m["checkpoint_path"] for m in top3],
                    "image_sizes": [int(m["config"].get("image_size", 384)) for m in top3],
                    "weights": [round(best_w[0], 4), round(best_w[1], 4), round(best_w[2], 4)],
                    "probs": best_grid_probs,
                    "val_metrics": best_grid_metrics,
                }
            )

    best_ens = max(candidates, key=lambda c: _selection_score(c["val_metrics"]))
    ens_improved = _selection_score(best_ens["val_metrics"]) > best_single_score

    np.save(ens_dir / "ensemble_val_probs.npy", best_ens["probs"].astype(np.float32))
    ens_payload = {
        "retained": bool(ens_improved),
        "best_single_model": best_single["name"],
        "best_single_val_metrics": best_single["val_metrics"],
        "selected_ensemble": {
            k: v for k, v in best_ens.items() if k != "probs"
        },
        "all_candidates": [
            {k: v for k, v in c.items() if k != "probs"} for c in candidates
        ],
    }
    (ens_dir / "ensemble_config.json").write_text(
        json.dumps(ens_payload, indent=2), encoding="utf-8"
    )
    print(
        f"\n[ENSEMBLE] Best={best_ens['ensemble_name']} weights={best_ens['weights']} | "
        f"Val Macro F1={best_ens['val_metrics']['macro_f1']:.4f} vs Best Single={best_single['val_metrics']['macro_f1']:.4f} | "
        f"Retained={ens_improved}"
    )
    return ens_payload


def generate_phase5_plots_and_table(
    all_runs: list[dict[str, Any]],
    best_per_arch: dict[str, dict[str, Any]],
    ens_payload: dict[str, Any],
) -> None:
    """
    Save Advanced_Experiments/Results/model_comparison.csv and all 10 comparison
    visualizations across all 5 architectures (and Ensemble if evaluated).
    All metrics are explicitly labeled as Validation Results.
    """
    ADV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Full experiment comparison CSV (all Phase 5 runs + EfficientNetV2B0 ref + best Ensemble)
    rows = []
    for r in all_runs:
        vm = r["val_metrics"]
        cfg = r.get("config", {})
        rows.append(
            {
                "experiment": r["name"],
                "architecture": r["architecture"],
                "image_size": int(cfg.get("image_size", 384)),
                "batch_size": int(cfg.get("batch_size", 8)),
                "imbalance_strategy": cfg.get("imbalance_strategy", "standard"),
                "augmentation": bool(cfg.get("use_augment", False)),
                "training_time_sec": r["training_time_sec"],
                "inference_time_val_sec": r["inference_time_val_sec"],
                "inference_ms_per_image": r["inference_ms_per_image"],
                "param_count": r["param_count"],
                "model_size_mb": r["model_size_mb"],
                "peak_tf_vram_mb": r.get("peak_tf_vram_mb", 0.0),
                "val_accuracy": vm["accuracy"],
                "val_balanced_accuracy": vm["balanced_accuracy"],
                "val_macro_precision": vm["macro_precision"],
                "val_macro_recall": vm["macro_recall"],
                "val_macro_f1": vm["macro_f1"],
                "val_weighted_f1": vm["weighted_f1"],
                "val_macro_roc_auc": vm["macro_roc_auc"],
                "val_mel_recall": vm["mel_recall"],
            }
        )

    sel_ens = ens_payload.get("selected_ensemble", {})
    if sel_ens:
        evm = sel_ens["val_metrics"]
        rows.append(
            {
                "experiment": sel_ens["ensemble_name"],
                "architecture": "+".join(sel_ens["architectures"]),
                "image_size": max(sel_ens["image_sizes"]),
                "batch_size": 8,
                "imbalance_strategy": f"soft_vote_{sel_ens['weights']}",
                "augmentation": False,
                "training_time_sec": 0.0,
                "inference_time_val_sec": sum(
                    best_per_arch[a]["inference_time_val_sec"] for a in sel_ens["architectures"]
                ),
                "inference_ms_per_image": sum(
                    best_per_arch[a]["inference_ms_per_image"] for a in sel_ens["architectures"]
                ),
                "param_count": sum(
                    best_per_arch[a]["param_count"] for a in sel_ens["architectures"]
                ),
                "model_size_mb": round(
                    sum(best_per_arch[a]["model_size_mb"] for a in sel_ens["architectures"]), 2
                ),
                "peak_tf_vram_mb": 0.0,
                "val_accuracy": evm["accuracy"],
                "val_balanced_accuracy": evm["balanced_accuracy"],
                "val_macro_precision": evm["macro_precision"],
                "val_macro_recall": evm["macro_recall"],
                "val_macro_f1": evm["macro_f1"],
                "val_weighted_f1": evm["weighted_f1"],
                "val_macro_roc_auc": evm["macro_roc_auc"],
                "val_mel_recall": evm["mel_recall"],
            }
        )

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(ADV_RESULTS_DIR / "model_comparison.csv", index=False)

    # 5-architecture + Ensemble comparison subset for clean bar plots
    arch_order = ["EfficientNetV2B0"] + PHASE5_ARCHITECTURES
    five_rows = []
    for a in arch_order:
        r = best_per_arch[a]
        vm = r["val_metrics"]
        five_rows.append(
            {
                "label": f"{a}\n({r['name']})",
                "short": a,
                "val_accuracy": vm["accuracy"],
                "val_macro_f1": vm["macro_f1"],
                "val_balanced_accuracy": vm["balanced_accuracy"],
                "val_mel_recall": vm["mel_recall"],
                "val_macro_roc_auc": vm["macro_roc_auc"],
                "inference_ms_per_image": r["inference_ms_per_image"],
            }
        )
    if sel_ens and ens_payload.get("retained", False):
        evm = sel_ens["val_metrics"]
        five_rows.append(
            {
                "label": f"Ensemble\n({sel_ens['ensemble_name']})",
                "short": "Ensemble",
                "val_accuracy": evm["accuracy"],
                "val_macro_f1": evm["macro_f1"],
                "val_balanced_accuracy": evm["balanced_accuracy"],
                "val_mel_recall": evm["mel_recall"],
                "val_macro_roc_auc": evm["macro_roc_auc"],
                "inference_ms_per_image": sum(
                    best_per_arch[a]["inference_ms_per_image"] for a in sel_ens["architectures"]
                ),
            }
        )
    five_df = pd.DataFrame(five_rows)

    def _bar(col: str, title: str, ylabel: str, fname: str, color: str, ylim_top: float = 1.05) -> None:
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        bars = ax.bar(five_df["short"], five_df[col], color=color, edgecolor="black", alpha=0.88)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        if ylim_top is not None:
            ax.set_ylim(0, ylim_top)
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, five_df[col]):
            fmt = f"{val:.3f}" if ylim_top else f"{val:.2f}"
            offset = 0.015 if ylim_top else max(five_df[col]) * 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset, fmt, ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(ADV_RESULTS_DIR / fname, dpi=150)
        plt.close(fig)

    _bar("val_accuracy", "Five-Architecture Validation Accuracy Comparison", "Validation Accuracy", "accuracy_comparison.png", "#2563eb")
    _bar("val_macro_f1", "Five-Architecture Validation Macro F1-Score Comparison", "Validation Macro F1", "macro_f1_comparison.png", "#16a34a")
    _bar("val_balanced_accuracy", "Five-Architecture Validation Balanced Accuracy Comparison", "Validation Balanced Accuracy", "balanced_accuracy_comparison.png", "#7c3aed")
    _bar("val_mel_recall", "Five-Architecture Validation Melanoma (MEL) Recall Comparison", "Validation MEL Recall", "melanoma_recall_comparison.png", "#dc2626")
    _bar("val_macro_roc_auc", "Five-Architecture Validation Macro ROC-AUC Comparison", "Validation Macro ROC-AUC", "roc_auc_comparison.png", "#0891b2")
    _bar("inference_ms_per_image", "Validation Inference Latency Comparison (ms / image)", "Latency (ms/image)", "inference_time_comparison.png", "#d97706", ylim_top=max(five_df["inference_ms_per_image"]) * 1.18)

    # Identify overall Phase 5 validation winner (single vs retained ensemble)
    best_single = max(best_per_arch.values(), key=lambda r: _selection_score(r["val_metrics"]))
    if ens_payload.get("retained", False):
        winner_name = sel_ens["ensemble_name"]
        winner_vm = sel_ens["val_metrics"]
    else:
        winner_name = best_single["name"]
        winner_vm = best_single["val_metrics"]

    # Confusion matrix of best Phase 5 model (Validation)
    cm = np.array(winner_vm["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SELECTED_CLASSES,
        yticklabels=SELECTED_CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(
        f"Validation Confusion Matrix — {winner_name} (Acc={winner_vm['accuracy']:.4f})",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(ADV_RESULTS_DIR / "confusion_matrix_best_model.png", dpi=150)
    plt.close(fig)

    # Per-class performance graph of best Phase 5 model (Validation)
    pcls = winner_vm["per_class"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(SELECTED_CLASSES))
    w = 0.25
    ax.bar(x - w, [pcls[c]["precision"] for c in SELECTED_CLASSES], w, label="Precision")
    ax.bar(x, [pcls[c]["recall"] for c in SELECTED_CLASSES], w, label="Recall")
    ax.bar(x + w, [pcls[c]["f1-score"] for c in SELECTED_CLASSES], w, label="F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(SELECTED_CLASSES)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Validation Score")
    ax.set_title(f"Validation Per-Class Performance — {winner_name}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ADV_RESULTS_DIR / "per_class_performance.png", dpi=150)
    plt.close(fig)

    # Training vs Validation Accuracy & Loss curves across best run per architecture
    for metric_key, title, fname, ylabel in [
        ("accuracy", "Training vs Validation Accuracy Across Architectures", "training_validation_accuracy.png", "Accuracy"),
        ("loss", "Training vs Validation Loss Across Architectures", "training_validation_loss.png", "Loss"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        axes_flat = axes.flatten()
        for idx, arch in enumerate(arch_order):
            ax = axes_flat[idx]
            r = best_per_arch[arch]
            hist = r.get("history", {})
            tr_vals = hist.get(metric_key, [])
            va_vals = hist.get(f"val_{metric_key}", [])
            if tr_vals:
                ep = range(1, len(tr_vals) + 1)
                ax.plot(ep, tr_vals, marker="o", markersize=3, label="Train")
                ax.plot(ep, va_vals, marker="s", markersize=3, label="Val")
                s1 = hist.get("stage1_epochs", [0])[0]
                if 0 < s1 < len(tr_vals):
                    ax.axvline(s1 + 0.5, color="gray", linestyle="--", linewidth=1)
            ax.set_title(f"{arch}\n({r['name']})", fontsize=9.5, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes_flat[-1].axis("off")
        plt.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(ADV_RESULTS_DIR / fname, dpi=150)
        plt.close(fig)

    # Save Phase 5 summary JSON for Streamlit app
    summary_payload = {
        "best_single_model": {k: v for k, v in best_single.items() if k != "history"},
        "best_per_architecture": {
            arch: {k: v for k, v in r.items() if k != "history"}
            for arch, r in best_per_arch.items()
        },
        "ensemble": ens_payload,
        "overall_winner_name": winner_name,
        "overall_winner_is_ensemble": bool(ens_payload.get("retained", False)),
    }
    (ADV_RESULTS_DIR / "phase5_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    print(f"Saved Phase 5 comparison table and 10 graphs to {ADV_RESULTS_DIR}")


def run_phase5_pipeline(force_retrain: bool = False) -> None:
    """
    Orchestrate Phase 5:
    1. Evaluate untouched existing EfficientNetV2B0 on validation (for fair 5-model table).
    2. Train H-K (DenseNet121, ResNet50V2, MobileNetV3Large, ConvNeXtTiny) at 384x384.
    3. Compare 4 imbalance/augmentation strategies independently on top H-K candidate,
       plus test 448x448 and top strategy on #2 candidate.
    4. Run validation-only soft-voting Ensemble experiment.
    5. Generate Advanced_Experiments/Results/model_comparison.csv and all 10 plots.
    """
    all_runs: list[dict[str, Any]] = []

    eff_ref = evaluate_existing_effnet_on_val(force_recompute=force_retrain)
    all_runs.append(eff_ref)

    # Controlled baseline + fine-tuning H-K (384x384, bs=8, max 20 epochs)
    hk_configs = [
        {"name": "H_DenseNet121_384", "architecture": "DenseNet121", "image_size": 384, "batch_size": 8, "imbalance_strategy": "standard", "use_augment": False, "stage1_epochs": 6, "stage2_epochs": 14, "stage1_lr": 1e-4, "stage2_lr": 1e-5, "unfreeze_fraction": 0.35},
        {"name": "I_ResNet50V2_384", "architecture": "ResNet50V2", "image_size": 384, "batch_size": 8, "imbalance_strategy": "standard", "use_augment": False, "stage1_epochs": 6, "stage2_epochs": 14, "stage1_lr": 1e-4, "stage2_lr": 1e-5, "unfreeze_fraction": 0.35},
        {"name": "J_MobileNetV3Large_384", "architecture": "MobileNetV3Large", "image_size": 384, "batch_size": 8, "imbalance_strategy": "standard", "use_augment": False, "stage1_epochs": 6, "stage2_epochs": 14, "stage1_lr": 1e-4, "stage2_lr": 1e-5, "unfreeze_fraction": 0.35},
        {"name": "K_ConvNeXtTiny_384", "architecture": "ConvNeXtTiny", "image_size": 384, "batch_size": 8, "imbalance_strategy": "standard", "use_augment": False, "stage1_epochs": 6, "stage2_epochs": 14, "stage1_lr": 1e-4, "stage2_lr": 1e-5, "unfreeze_fraction": 0.35},
    ]

    hk_runs: list[dict[str, Any]] = []
    for cfg in hk_configs:
        res = _run_phase5_isolated(cfg, force_retrain=force_retrain)
        hk_runs.append(res)
        all_runs.append(res)

    hk_ranked = sorted(hk_runs, key=lambda r: _selection_score(r["val_metrics"]), reverse=True)
    top1_arch = hk_ranked[0]["architecture"]
    top2_arch = hk_ranked[1]["architecture"]
    print(f"\nTop H-K candidates on validation: #1={top1_arch}, #2={top2_arch}")

    # Independent comparison of 4 imbalance & augmentation strategies on strongest H-K candidate
    imbalance_configs = [
        {"name": f"L1_{top1_arch}_384_ClassWeight", "architecture": top1_arch, "image_size": 384, "batch_size": 8, "imbalance_strategy": "class_weight", "use_augment": False, "stage1_epochs": 6, "stage2_epochs": 14, "stage1_lr": 1e-4, "stage2_lr": 1e-5, "unfreeze_fraction": 0.35},
        {"name": f"L2_{top1_arch}_384_FocalLoss", "architecture": top1_arch, "image_size": 384, "batch_size": 8, "imbalance_strategy": "focal_loss", "use_augment": False, "stage1_epochs": 6, "stage2_epochs": 14, "stage1_lr": 1e-4, "stage2_lr": 1e-5, "unfreeze_fraction": 0.35},
        {"name": f"L3_{top1_arch}_384_WeightedSampling", "architecture": top1_arch, "image_size": 384, "batch_size": 8, "imbalance_strategy": "weighted_sampling", "use_augment": False, "stage1_epochs": 6, "stage2_epochs": 14, "stage1_lr": 1e-4, "stage2_lr": 1e-5, "unfreeze_fraction": 0.35},
        {"name": f"L4_{top1_arch}_384_Augmentation", "architecture": top1_arch, "image_size": 384, "batch_size": 8, "imbalance_strategy": "standard", "use_augment": True, "stage1_epochs": 6, "stage2_epochs": 14, "stage1_lr": 1e-4, "stage2_lr": 1e-5, "unfreeze_fraction": 0.35},
    ]

    imb_runs: list[dict[str, Any]] = []
    for cfg in imbalance_configs:
        res = _run_phase5_isolated(cfg, force_retrain=force_retrain)
        imb_runs.append(res)
        all_runs.append(res)

    best_imb_run = max([hk_ranked[0]] + imb_runs, key=lambda r: _selection_score(r["val_metrics"]))
    best_strategy = best_imb_run["config"].get("imbalance_strategy", "standard")
    best_aug = bool(best_imb_run["config"].get("use_augment", False))

    # 448x448 test on strongest H-K candidate + best strategy on #2 candidate
    extra_configs = [
        {"name": f"M_{top1_arch}_448", "architecture": top1_arch, "image_size": 448, "batch_size": 8, "imbalance_strategy": best_strategy, "use_augment": best_aug, "stage1_epochs": 6, "stage2_epochs": 14, "stage1_lr": 1e-4, "stage2_lr": 1e-5, "unfreeze_fraction": 0.35},
        {"name": f"N_{top2_arch}_384_{best_strategy}{'_Aug' if best_aug else ''}", "architecture": top2_arch, "image_size": 384, "batch_size": 8, "imbalance_strategy": best_strategy, "use_augment": best_aug, "stage1_epochs": 6, "stage2_epochs": 14, "stage1_lr": 1e-4, "stage2_lr": 1e-5, "unfreeze_fraction": 0.35},
    ]
    for cfg in extra_configs:
        all_runs.append(_run_phase5_isolated(cfg, force_retrain=force_retrain))

    # Pick single strongest validation run per architecture
    best_per_arch: dict[str, dict[str, Any]] = {"EfficientNetV2B0": eff_ref}
    for arch in PHASE5_ARCHITECTURES:
        arch_runs = [r for r in all_runs if r["architecture"] == arch]
        best_per_arch[arch] = max(arch_runs, key=lambda r: _selection_score(r["val_metrics"]))

    val_targets = np.load(ADV_RESULTS_DIR / "val_targets.npy")
    ens_payload = run_phase5_ensemble(best_per_arch, val_targets)
    generate_phase5_plots_and_table(all_runs, best_per_arch, ens_payload)


def run_phase5_results_only() -> None:
    """
    Run Ensemble evaluation, model_comparison.csv, and all 10 visualizations
    over all completed Phase 5 experiments cached under Advanced_Experiments/Models/.
    """
    eff_ref = evaluate_existing_effnet_on_val(force_recompute=False)
    all_runs: list[dict[str, Any]] = [eff_ref]

    for arch in PHASE5_ARCHITECTURES:
        arch_dir = ADV_MODELS_DIR / arch
        if arch_dir.exists():
            for res_path in sorted(arch_dir.glob("*_result.json")):
                all_runs.append(json.loads(res_path.read_text(encoding="utf-8")))

    best_per_arch: dict[str, dict[str, Any]] = {"EfficientNetV2B0": eff_ref}
    for arch in PHASE5_ARCHITECTURES:
        arch_runs = [r for r in all_runs if r["architecture"] == arch]
        if arch_runs:
            best_per_arch[arch] = max(arch_runs, key=lambda r: _selection_score(r["val_metrics"]))

    val_targets = np.load(ADV_RESULTS_DIR / "val_targets.npy")
    ens_payload = run_phase5_ensemble(best_per_arch, val_targets)
    generate_phase5_plots_and_table(all_runs, best_per_arch, ens_payload)


def run_phase5_smoke_check() -> None:
    """Verify GPU ops, weighted sampling dataset, and safe .keras save/load across all 4 models."""
    import keras, tempfile

    require_gpu_environment()
    keras.mixed_precision.set_global_policy("mixed_float16")
    ws_ds, steps = create_weighted_sampled_tf_dataset(
        PROCESSED_DIR / "train", image_size=384, batch_size=8, augment=True
    )
    bx, by = next(iter(ws_ds))
    print(f"Weighted sampled batch: x={bx.shape}, y={by.shape}, steps/epoch={steps}, class_sum={by.numpy().sum(axis=0)}")

    for arch in PHASE5_ARCHITECTURES:
        keras.backend.clear_session()
        model, bb = build_phase5_model(arch, 384)
        unfrozen, frozen_bn = unfreeze_phase5_backbone(bb, 0.35)
        preds = model(bx[:2], training=False).numpy()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / f"{arch}.keras"
            model.save(p)
            m2 = keras.models.load_model(p, compile=False)
            p2 = m2(bx[:2], training=False).numpy()
        diff = float(np.max(np.abs(preds - p2)))
        print(f"Smoke {arch:<17}: params={model.count_params():>8} unfrozen={unfrozen:>3} frozen_bn={frozen_bn:>3} save/load_diff={diff:.2e}")


# ===========================================================================
# PHASE 6 — TARGETED OPTIMIZATION (BKL/MEL, REGULARISATION, CALIBRATION)
# ===========================================================================


def run_phase6_single_experiment(
    exp_cfg: dict[str, Any], force_retrain: bool = False
) -> dict[str, Any]:
    """
    Execute a single Phase 6 regularized/targeted experiment on GPU, saving
    artifacts strictly under Advanced_Experiments/Phase6/Models/<name>/.
    """
    import keras
    import tensorflow as tf

    name = exp_cfg["name"]
    arch = exp_cfg["architecture"]
    exp_dir = PHASE6_MODELS_DIR / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = exp_dir / f"{name}.keras"
    result_json_path = exp_dir / f"{name}_result.json"
    val_probs_path = exp_dir / f"{name}_val_probs.npy"

    if not force_retrain and ckpt_path.exists() and result_json_path.exists() and val_probs_path.exists():
        cached = json.loads(result_json_path.read_text(encoding="utf-8"))
        vm = cached["val_metrics"]
        print(
            f"\n[RESUME CACHE] Skipping Phase 6 {name}: "
            f"Val Acc={vm['accuracy']:.4f} | Macro F1={vm['macro_f1']:.4f} | "
            f"BKL Rec={vm['per_class']['BKL']['recall']:.4f} | MEL Rec={vm['mel_recall']:.4f}"
        )
        return cached

    require_gpu_environment()
    set_global_seed(RANDOM_SEED)
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy("mixed_float16")

    img_size = int(exp_cfg.get("image_size", 384))
    batch_size = int(exp_cfg.get("batch_size", 8))
    use_augment = bool(exp_cfg.get("use_augment", True))
    dropout_rate = float(exp_cfg.get("dropout", 0.35))
    weight_decay = float(exp_cfg.get("weight_decay", 1e-4))
    label_smoothing = float(exp_cfg.get("label_smoothing", 0.0))
    stage1_epochs = int(exp_cfg.get("stage1_epochs", 6))
    stage2_epochs = int(exp_cfg.get("stage2_epochs", 14))
    stage1_lr = float(exp_cfg.get("stage1_lr", 1e-4))
    stage2_lr = float(exp_cfg.get("stage2_lr", 5e-6))
    unfreeze_fraction = float(exp_cfg.get("unfreeze_fraction", 0.30))
    raw_cw = exp_cfg.get("class_weight")
    fit_class_weight = {int(k): float(v) for k, v in raw_cw.items()} if raw_cw else None

    print(f"\n{'='*78}\nPHASE 6 EXPERIMENT: {name}\nConfig: {exp_cfg}\n{'='*78}")

    train_ds = create_tf_dataset(
        PROCESSED_DIR / "train", img_size, batch_size, shuffle=True, augment=use_augment, seed=RANDOM_SEED
    )
    val_ds = create_tf_dataset(
        PROCESSED_DIR / "val", img_size, batch_size, shuffle=False, augment=False, seed=RANDOM_SEED
    )

    model, backbone = build_phase5_model(
        arch_name=arch,
        image_size=img_size,
        dropout_rate=dropout_rate,
        dense_units=int(exp_cfg.get("dense_units", 512)),
    )
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    tf.config.experimental.reset_memory_stats("GPU:0")
    t0 = time.time()

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=stage1_lr, weight_decay=weight_decay),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    cb_stage1 = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
    ]
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        class_weight=fit_class_weight,
        callbacks=cb_stage1,
        shuffle=False,
        verbose=2,
    )

    unfrozen, frozen_bn = unfreeze_phase5_backbone(backbone, unfreeze_fraction=unfreeze_fraction)
    print(f"Stage 2 unfrozen non-BN layers: {unfrozen} | Frozen BN layers: {frozen_bn}")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=stage2_lr, weight_decay=weight_decay),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    cb_stage2 = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
    ]
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        class_weight=fit_class_weight,
        callbacks=cb_stage2,
        shuffle=False,
        verbose=2,
    )
    elapsed_sec = time.time() - t0
    gpu_mem = tf.config.experimental.get_memory_info("GPU:0")
    peak_vram_mb = round(gpu_mem["peak"] / (1024**2), 2)

    best_model = keras.models.load_model(ckpt_path, compile=False)
    y_val_true, y_val_prob, inf_sec, inf_ms_img = _collect_predictions_timed(
        best_model, val_ds, num_images=3501
    )
    val_metrics = compute_classification_metrics(y_val_true, y_val_prob)
    history = _merge_histories(h1.history, h2.history)
    param_count = int(best_model.count_params())
    model_size_mb = round(ckpt_path.stat().st_size / (1024**2), 2)

    np.save(val_probs_path, y_val_prob.astype(np.float32))
    result_payload = {
        "name": name,
        "architecture": arch,
        "config": exp_cfg,
        "checkpoint_path": str(ckpt_path),
        "val_probs_path": str(val_probs_path),
        "training_time_sec": round(elapsed_sec, 2),
        "inference_time_val_sec": round(inf_sec, 3),
        "inference_ms_per_image": round(inf_ms_img, 3),
        "param_count": param_count,
        "model_size_mb": model_size_mb,
        "peak_tf_vram_mb": peak_vram_mb,
        "val_metrics": val_metrics,
        "history": history,
    }
    result_json_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(
        f"[{name}] Val Acc={val_metrics['accuracy']:.4f} | Macro F1={val_metrics['macro_f1']:.4f} | "
        f"Bal Acc={val_metrics['balanced_accuracy']:.4f} | "
        f"BKL Rec={val_metrics['per_class']['BKL']['recall']:.4f} | "
        f"MEL Rec={val_metrics['mel_recall']:.4f} | Time={elapsed_sec:.1f}s"
    )
    return result_payload


def _run_phase6_isolated(exp_cfg: dict[str, Any], force_retrain: bool = False) -> dict[str, Any]:
    """Run a Phase 6 experiment in an isolated subprocess with automatic batch-size fallback."""
    name = exp_cfg["name"]
    exp_dir = PHASE6_MODELS_DIR / name
    ckpt_path = exp_dir / f"{name}.keras"
    result_json_path = exp_dir / f"{name}_result.json"
    val_probs_path = exp_dir / f"{name}_val_probs.npy"

    if not force_retrain and ckpt_path.exists() and result_json_path.exists() and val_probs_path.exists():
        cached = json.loads(result_json_path.read_text(encoding="utf-8"))
        vm = cached["val_metrics"]
        print(
            f"\n[RESUME CACHE] Skipping Phase 6 {name}: "
            f"Val Acc={vm['accuracy']:.4f} | Macro F1={vm['macro_f1']:.4f} | "
            f"BKL Rec={vm['per_class']['BKL']['recall']:.4f} | MEL Rec={vm['mel_recall']:.4f}"
        )
        return cached

    cfg_try = dict(exp_cfg)
    while True:
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--stage",
            "phase6-single",
            "--exp-json",
            json.dumps(cfg_try),
        ]
        if force_retrain:
            cmd.append("--force-retrain")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0 and result_json_path.exists():
            return json.loads(result_json_path.read_text(encoding="utf-8"))

        cur_bs = int(cfg_try.get("batch_size", 8))
        if cur_bs > 2:
            next_bs = max(2, cur_bs // 2)
            print(f"WARNING: Phase 6 {name} failed at bs={cur_bs}; retrying bs={next_bs}...")
            cfg_try["batch_size"] = next_bs
        else:
            raise RuntimeError(f"Phase 6 experiment {name} failed even at batch_size={cur_bs}.")


def _eval_predictions_full(y_true: np.ndarray, probs: np.ndarray, y_pred: np.ndarray | None = None) -> dict[str, Any]:
    """Evaluate classification metrics for custom threshold/multiplier rules on probabilities."""
    if y_pred is None:
        y_pred = np.argmax(probs, axis=1)
    y_true_bin = label_binarize(y_true, classes=list(range(len(SELECTED_CLASSES))))
    rep = classification_report(y_true, y_pred, target_names=SELECTED_CLASSES, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_roc_auc": float(roc_auc_score(y_true_bin, probs, average="macro", multi_class="ovr")),
        "bkl_recall": float(rep["BKL"]["recall"]),
        "bkl_precision": float(rep["BKL"]["precision"]),
        "bkl_f1": float(rep["BKL"]["f1-score"]),
        "mel_recall": float(rep["MEL"]["recall"]),
        "mel_precision": float(rep["MEL"]["precision"]),
        "mel_f1": float(rep["MEL"]["f1-score"]),
        "per_class": {c: rep[c] for c in SELECTED_CLASSES},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_phase6_analysis_and_deliverables(p6_runs: list[dict[str, Any]]) -> None:
    """
    Perform Phase 6 validation-only ensemble weight optimization, probability
    calibration, and out-of-fold threshold optimization; generate all Phase 6
    CSVs, PNGs, optimized_ensemble.json, and phase6_report.md under Advanced_Experiments/Phase6/.
    """
    from scipy.optimize import minimize
    from sklearn.model_selection import StratifiedKFold

    PHASE6_DIR.mkdir(parents=True, exist_ok=True)
    y_true = np.load(ADV_RESULTS_DIR / "val_targets.npy")

    # Load all Phase 5 + Phase 6 saved results and validation probabilities
    all_records: dict[str, dict[str, Any]] = {}
    prob_map: dict[str, np.ndarray] = {}
    for rp in list((ADV_MODELS_DIR).rglob("*_result.json")) + list(PHASE6_MODELS_DIR.rglob("*_result.json")):
        d = json.loads(rp.read_text(encoding="utf-8"))
        all_records[d["name"]] = d
        prob_map[d["name"]] = np.load(_resolve_project_path(d["val_probs_path"])).astype(np.float64)

    p_l1 = prob_map["L1_ConvNeXtTiny_384_ClassWeight"]
    p_l3 = prob_map["L3_ConvNeXtTiny_384_WeightedSampling"]
    p_eff = prob_map["Ref_G_Res384_ImprovedFineTune"]
    p_mob = prob_map["J_MobileNetV3Large_384"]
    p_den = prob_map["H_DenseNet121_384"]

    # Pick strongest ConvNeXt & MobileNet variants across Phase 5 + Phase 6
    conv_candidates = [
        all_records[n] for n in all_records
        if all_records[n]["architecture"] == "ConvNeXtTiny"
    ]
    best_conv = max(conv_candidates, key=lambda r: _selection_score(r["val_metrics"]))
    mob_candidates = [
        all_records[n] for n in all_records
        if all_records[n]["architecture"] == "MobileNetV3Large"
    ]
    best_mob = max(mob_candidates, key=lambda r: _selection_score(r["val_metrics"]))

    p_best_conv = prob_map[best_conv["name"]]
    p_best_mob = prob_map[best_mob["name"]]

    # Section 3: Ensemble weight comparisons (avoiding excessive search via principled coarse candidates)
    ensemble_configs = [
        ("Phase5_Top3_Weighted", ["L1_ConvNeXtTiny_384_ClassWeight", "Ref_G_Res384_ImprovedFineTune", "J_MobileNetV3Large_384"], [0.45, 0.15, 0.40]),
        ("Top3_EqualWeight", ["L1_ConvNeXtTiny_384_ClassWeight", "Ref_G_Res384_ImprovedFineTune", "J_MobileNetV3Large_384"], [1 / 3, 1 / 3, 1 / 3]),
        ("TwoModel_ConvNeXt_EffNet", ["L1_ConvNeXtTiny_384_ClassWeight", "Ref_G_Res384_ImprovedFineTune"], [0.5, 0.5]),
        ("TwoModel_ConvNeXt_MobileNet", ["L1_ConvNeXtTiny_384_ClassWeight", "J_MobileNetV3Large_384"], [0.5, 0.5]),
        ("Phase6_Top3_BestArchs_Equal", [best_conv["name"], "Ref_G_Res384_ImprovedFineTune", best_mob["name"]], [1 / 3, 1 / 3, 1 / 3]),
        ("Phase6_Top3_BestArchs_Weighted", [best_conv["name"], "Ref_G_Res384_ImprovedFineTune", best_mob["name"]], [0.45, 0.20, 0.35]),
        ("Phase6_Top4_Diverse_Equal", [best_conv["name"], "Ref_G_Res384_ImprovedFineTune", best_mob["name"], "H_DenseNet121_384"], [0.25, 0.25, 0.25, 0.25]),
        ("Phase6_Top4_Diverse_Weighted", [best_conv["name"], "Ref_G_Res384_ImprovedFineTune", best_mob["name"], "H_DenseNet121_384"], [0.35, 0.20, 0.30, 0.15]),
        ("Phase6_MultiDiverse_5Models", ["L1_ConvNeXtTiny_384_ClassWeight", best_conv["name"], "Ref_G_Res384_ImprovedFineTune", best_mob["name"], "H_DenseNet121_384"], [0.20, 0.25, 0.20, 0.25, 0.10]),
    ]

    # Deduplicate if best_conv == L1 and best_mob == J
    comp_rows = []
    for name in [
        "Ref_G_Res384_ImprovedFineTune",
        "H_DenseNet121_384",
        "I_ResNet50V2_384",
        "J_MobileNetV3Large_384",
        "K_ConvNeXtTiny_384",
        "L1_ConvNeXtTiny_384_ClassWeight",
        "L3_ConvNeXtTiny_384_WeightedSampling",
    ] + [r["name"] for r in p6_runs]:
        if name not in all_records:
            continue
        rec = all_records[name]
        m = _eval_predictions_full(y_true, prob_map[name])
        comp_rows.append(
            {
                "config_name": name,
                "kind": "single",
                "details": rec["architecture"],
                "inference_ms_per_image": rec["inference_ms_per_image"],
                "val_accuracy": m["accuracy"],
                "val_balanced_accuracy": m["balanced_accuracy"],
                "val_macro_precision": m["macro_precision"],
                "val_macro_recall": m["macro_recall"],
                "val_macro_f1": m["macro_f1"],
                "val_weighted_f1": m["weighted_f1"],
                "val_macro_roc_auc": m["macro_roc_auc"],
                "val_bkl_recall": m["bkl_recall"],
                "val_bkl_precision": m["bkl_precision"],
                "val_mel_recall": m["mel_recall"],
                "val_mel_precision": m["mel_precision"],
            }
        )

    ens_results: dict[str, dict[str, Any]] = {}
    for ename, members, weights in ensemble_configs:
        p_mix = np.zeros((len(y_true), 4), dtype=np.float64)
        for mname, w in zip(members, weights):
            p_mix += w * prob_map[mname]
        p_mix /= p_mix.sum(axis=1, keepdims=True)
        m = _eval_predictions_full(y_true, p_mix)
        inf_lat = sum(all_records[mname]["inference_ms_per_image"] for mname in set(members))
        ens_results[ename] = {
            "name": ename,
            "members": members,
            "weights": [round(float(w), 4) for w in weights],
            "probs": p_mix,
            "metrics": m,
            "inference_ms_per_image": round(inf_lat, 3),
        }
        comp_rows.append(
            {
                "config_name": ename,
                "kind": "ensemble",
                "details": "+".join(members),
                "inference_ms_per_image": round(inf_lat, 3),
                "val_accuracy": m["accuracy"],
                "val_balanced_accuracy": m["balanced_accuracy"],
                "val_macro_precision": m["macro_precision"],
                "val_macro_recall": m["macro_recall"],
                "val_macro_f1": m["macro_f1"],
                "val_weighted_f1": m["weighted_f1"],
                "val_macro_roc_auc": m["macro_roc_auc"],
                "val_bkl_recall": m["bkl_recall"],
                "val_bkl_precision": m["bkl_precision"],
                "val_mel_recall": m["mel_recall"],
                "val_mel_precision": m["mel_precision"],
            }
        )

    # Select best ensemble cleanly by validation Macro F1 + Bal Acc + BKL/MEL recall
    def _p6_score(m: dict[str, Any]) -> tuple[float, float, float, float]:
        return (m["macro_f1"], m["balanced_accuracy"], 0.5 * (m["bkl_recall"] + m["mel_recall"]), m["accuracy"])

    best_ens_key = max(ens_results.keys(), key=lambda k: _p6_score(ens_results[k]["metrics"]))
    best_ens = ens_results[best_ens_key]
    p_base_ens = best_ens["probs"]

    # Section 2: Probability calibration & threshold optimization on best ensemble (5-fold CV OOF)
    eps = 1e-7
    logits = np.log(np.clip(p_base_ens, eps, 1.0))

    def _nll(t_arr):
        t = t_arr[0]
        sc = logits / t
        sc -= sc.max(axis=1, keepdims=True)
        ex = np.exp(sc)
        pr = ex / ex.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(np.clip(pr[np.arange(len(y_true)), y_true], eps, 1.0))))

    t_opt = float(minimize(_nll, [1.0], bounds=[(0.3, 5.0)]).x[0])
    sc = logits / t_opt
    sc -= sc.max(axis=1, keepdims=True)
    p_temp = np.exp(sc) / np.exp(sc).sum(axis=1, keepdims=True)

    true_prev = np.array([(y_true == i).mean() for i in range(4)])
    mean_pred = p_base_ens.mean(axis=0)
    prior_ratio = true_prev / np.clip(mean_pred, 1e-6, 1.0)
    p_prior = (p_base_ens * prior_ratio)
    p_prior /= p_prior.sum(axis=1, keepdims=True)

    # 5-fold CV out-of-fold class multipliers with precision guard (>= 0.60 per class)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_mult_preds = np.zeros_like(y_true)
    fold_mults: list[list[float]] = []
    for tr_idx, va_idx in skf.split(p_base_ens, y_true):
        ptr, ytr = p_base_ens[tr_idx], y_true[tr_idx]
        best_sc = -1.0
        best_m = [1.0, 1.0, 1.0, 1.0]
        for m_bcc in [0.95, 1.0, 1.05]:
            for m_bkl in [1.0, 1.15, 1.30, 1.45]:
                for m_mel in [0.90, 1.0, 1.10, 1.20]:
                    mult = np.array([m_bcc, m_bkl, m_mel, 1.0])
                    yp = np.argmax(ptr * mult, axis=1)
                    precs = precision_score(ytr, yp, average=None, zero_division=0)
                    if precs.min() < 0.60:
                        continue
                    mf1 = f1_score(ytr, yp, average="macro")
                    bal = balanced_accuracy_score(ytr, yp)
                    bkl_r = recall_score(ytr, yp, average=None, zero_division=0)[1]
                    mel_r = recall_score(ytr, yp, average=None, zero_division=0)[2]
                    sc_val = 0.45 * mf1 + 0.30 * bal + 0.15 * bkl_r + 0.10 * mel_r
                    if sc_val > best_sc:
                        best_sc = sc_val
                        best_m = [m_bcc, m_bkl, m_mel, 1.0]
        fold_mults.append(best_m)
        oof_mult_preds[va_idx] = np.argmax(p_base_ens[va_idx] * np.array(best_m), axis=1)

    median_mult = [float(np.median([fm[i] for fm in fold_mults])) for i in range(4)]

    # Explicit BKL / MEL threshold decision rule (OOF validated)
    oof_thresh_preds = np.zeros_like(y_true)
    fold_taus: list[tuple[float, float]] = []
    for tr_idx, va_idx in skf.split(p_base_ens, y_true):
        ptr, ytr = p_base_ens[tr_idx], y_true[tr_idx]
        best_sc = -1.0
        best_tau = (0.35, 0.35)
        for tau_bkl in [0.28, 0.32, 0.36, 0.40]:
            for tau_mel in [0.32, 0.36, 0.40, 0.44]:
                yp = np.argmax(ptr, axis=1)
                # Override NV when BKL or MEL exceeds targeted threshold
                nv_mask = yp == 3
                bkl_override = nv_mask & (ptr[:, 1] >= tau_bkl) & (ptr[:, 1] >= ptr[:, 2])
                mel_override = nv_mask & (ptr[:, 2] >= tau_mel) & (ptr[:, 2] > ptr[:, 1])
                yp[bkl_override] = 1
                yp[mel_override] = 2
                precs = precision_score(ytr, yp, average=None, zero_division=0)
                if precs.min() < 0.60:
                    continue
                mf1 = f1_score(ytr, yp, average="macro")
                bal = balanced_accuracy_score(ytr, yp)
                sc_val = 0.55 * mf1 + 0.45 * bal
                if sc_val > best_sc:
                    best_sc = sc_val
                    best_tau = (tau_bkl, tau_mel)
        fold_taus.append(best_tau)
        pva = p_base_ens[va_idx]
        yp_va = np.argmax(pva, axis=1)
        nv_m = yp_va == 3
        yp_va[nv_m & (pva[:, 1] >= best_tau[0]) & (pva[:, 1] >= pva[:, 2])] = 1
        yp_va[nv_m & (pva[:, 2] >= best_tau[1]) & (pva[:, 2] > pva[:, 1])] = 2
        oof_thresh_preds[va_idx] = yp_va

    median_tau = (float(np.median([t[0] for t in fold_taus])), float(np.median([t[1] for t in fold_taus])))

    threshold_rules = [
        ("Argmax_Baseline", p_base_ens, np.argmax(p_base_ens, axis=1), f"Standard argmax on {best_ens_key}"),
        ("Temperature_Scaled_Argmax", p_temp, np.argmax(p_temp, axis=1), f"T={t_opt:.3f}"),
        ("Prior_Aligned_Reweighting", p_prior, np.argmax(p_prior, axis=1), f"Ratio={[round(float(x), 3) for x in prior_ratio]}"),
        ("OOF_Class_Multipliers", p_base_ens, oof_mult_preds, f"5-Fold OOF median mult={[round(x, 2) for x in median_mult]} (P>=0.60)"),
        ("OOF_BKL_MEL_Threshold_Rule", p_base_ens, oof_thresh_preds, f"5-Fold OOF median tau_BKL={median_tau[0]:.2f}, tau_MEL={median_tau[1]:.2f}"),
    ]

    thresh_rows = []
    thresh_metrics_map: dict[str, dict[str, Any]] = {}
    for rname, rprobs, rpred, rdesc in threshold_rules:
        tm = _eval_predictions_full(y_true, rprobs, rpred)
        thresh_metrics_map[rname] = tm
        thresh_rows.append(
            {
                "decision_rule": rname,
                "parameters": rdesc,
                "val_accuracy": tm["accuracy"],
                "val_balanced_accuracy": tm["balanced_accuracy"],
                "val_macro_precision": tm["macro_precision"],
                "val_macro_recall": tm["macro_recall"],
                "val_macro_f1": tm["macro_f1"],
                "val_weighted_f1": tm["weighted_f1"],
                "val_macro_roc_auc": tm["macro_roc_auc"],
                "val_bkl_precision": tm["bkl_precision"],
                "val_bkl_recall": tm["bkl_recall"],
                "val_bkl_f1": tm["bkl_f1"],
                "val_mel_precision": tm["mel_precision"],
                "val_mel_recall": tm["mel_recall"],
                "val_mel_f1": tm["mel_f1"],
            }
        )
        if rname != "Argmax_Baseline":
            comp_rows.append(
                {
                    "config_name": f"{best_ens_key}__{rname}",
                    "kind": "threshold_calibrated",
                    "details": rdesc,
                    "inference_ms_per_image": best_ens["inference_ms_per_image"],
                    "val_accuracy": tm["accuracy"],
                    "val_balanced_accuracy": tm["balanced_accuracy"],
                    "val_macro_precision": tm["macro_precision"],
                    "val_macro_recall": tm["macro_recall"],
                    "val_macro_f1": tm["macro_f1"],
                    "val_weighted_f1": tm["weighted_f1"],
                    "val_macro_roc_auc": tm["macro_roc_auc"],
                    "val_bkl_recall": tm["bkl_recall"],
                    "val_bkl_precision": tm["bkl_precision"],
                    "val_mel_recall": tm["mel_recall"],
                    "val_mel_precision": tm["mel_precision"],
                }
            )

    best_rule_name = max(thresh_metrics_map.keys(), key=lambda k: _p6_score(thresh_metrics_map[k]))
    best_rule_metrics = thresh_metrics_map[best_rule_name]

    # Save CSVs
    pd.DataFrame(comp_rows).to_csv(PHASE6_DIR / "phase6_comparison.csv", index=False)
    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(PHASE6_DIR / "threshold_comparison.csv", index=False)

    opt_rep_rows = []
    for c in SELECTED_CLASSES:
        pc = best_rule_metrics["per_class"][c]
        opt_rep_rows.append(
            {
                "class": c,
                "full_name": CLASS_FULL_NAMES[c],
                "precision": pc["precision"],
                "recall": pc["recall"],
                "f1_score": pc["f1-score"],
                "support": int(pc["support"]),
            }
        )
    opt_rep_rows.append(
        {
            "class": "MACRO_AVG",
            "full_name": "Validation Macro Average",
            "precision": best_rule_metrics["macro_precision"],
            "recall": best_rule_metrics["macro_recall"],
            "f1_score": best_rule_metrics["macro_f1"],
            "support": int(len(y_true)),
        }
    )
    pd.DataFrame(opt_rep_rows).to_csv(PHASE6_DIR / "optimized_classification_report.csv", index=False)

    # Save optimized_ensemble.json
    opt_ens_payload = {
        "metrics_scope": "validation_only",
        "validation_samples": int(len(y_true)),
        "selected_ensemble": {
            "name": best_ens["name"],
            "members": best_ens["members"],
            "architectures": [all_records[m]["architecture"] for m in best_ens["members"]],
            "checkpoints": [all_records[m]["checkpoint_path"] for m in best_ens["members"]],
            "image_sizes": [int(all_records[m]["config"].get("image_size", 384)) for m in best_ens["members"]],
            "weights": best_ens["weights"],
            "inference_ms_per_image": best_ens["inference_ms_per_image"],
        },
        "selected_decision_rule": {
            "rule_name": best_rule_name,
            "temperature": round(t_opt, 4),
            "prior_ratio": [round(float(x), 4) for x in prior_ratio],
            "oof_median_multipliers": median_mult,
            "oof_median_thresholds_bkl_mel": list(median_tau),
        },
        "validation_metrics": best_rule_metrics,
        "phase5_baseline_metrics": ens_results["Phase5_Top3_Weighted"]["metrics"],
    }
    (PHASE6_DIR / "optimized_ensemble.json").write_text(
        json.dumps(opt_ens_payload, indent=2), encoding="utf-8"
    )

    # Generate required Phase 6 plots
    # 1. threshold_performance.png & class_threshold_comparison.png
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(thresh_df))
    w = 0.18
    ax.bar(x - 1.5 * w, thresh_df["val_accuracy"], w, label="Accuracy")
    ax.bar(x - 0.5 * w, thresh_df["val_macro_f1"], w, label="Macro F1")
    ax.bar(x + 0.5 * w, thresh_df["val_bkl_recall"], w, label="BKL Recall")
    ax.bar(x + 1.5 * w, thresh_df["val_mel_recall"], w, label="MEL Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(thresh_df["decision_rule"], rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation Score")
    ax.set_title(f"Threshold & Calibration Comparison ({best_ens_key}, Val n=3,501)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PHASE6_DIR / "threshold_performance.png", dpi=150)
    plt.savefig(PHASE6_DIR / "class_threshold_comparison.png", dpi=150)
    plt.close(fig)

    # 2. optimized_confusion_matrix.png
    cm = np.array(best_rule_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SELECTED_CLASSES,
        yticklabels=SELECTED_CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(
        f"Phase 6 Optimized Validation Confusion Matrix\n{best_ens_key} ({best_rule_name}, Acc={best_rule_metrics['accuracy']:.4f})",
        fontsize=10.5,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(PHASE6_DIR / "optimized_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 3. class_performance_comparison.png (Phase 5 Ensemble vs Phase 6 Optimized)
    p5_m = ens_results["Phase5_Top3_Weighted"]["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x_cls = np.arange(len(SELECTED_CLASSES))
    w2 = 0.35
    axes[0].bar(x_cls - w2 / 2, [p5_m["per_class"][c]["recall"] for c in SELECTED_CLASSES], w2, label="Phase 5 Ensemble")
    axes[0].bar(x_cls + w2 / 2, [best_rule_metrics["per_class"][c]["recall"] for c in SELECTED_CLASSES], w2, label=f"Phase 6 ({best_rule_name})")
    axes[0].set_xticks(x_cls)
    axes[0].set_xticklabels(SELECTED_CLASSES)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Validation Recall")
    axes[0].set_title("Per-Class Recall: Phase 5 vs Phase 6", fontweight="bold")
    axes[0].legend()

    axes[1].bar(x_cls - w2 / 2, [p5_m["per_class"][c]["f1-score"] for c in SELECTED_CLASSES], w2, label="Phase 5 Ensemble")
    axes[1].bar(x_cls + w2 / 2, [best_rule_metrics["per_class"][c]["f1-score"] for c in SELECTED_CLASSES], w2, label=f"Phase 6 ({best_rule_name})")
    axes[1].set_xticks(x_cls)
    axes[1].set_xticklabels(SELECTED_CLASSES)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Validation F1-Score")
    axes[1].set_title("Per-Class F1-Score: Phase 5 vs Phase 6", fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(PHASE6_DIR / "class_performance_comparison.png", dpi=150)
    plt.close(fig)

    # 4. training_history_comparison.png (Phase 5 architectures & imbalance strategies)
    hist_names = [
        "Ref_G_Res384_ImprovedFineTune",
        "H_DenseNet121_384",
        "J_MobileNetV3Large_384",
        "K_ConvNeXtTiny_384",
        "L1_ConvNeXtTiny_384_ClassWeight",
        "L3_ConvNeXtTiny_384_WeightedSampling",
    ] + [r["name"] for r in p6_runs if r["name"] not in {"K_ConvNeXtTiny_384", "L1_ConvNeXtTiny_384_ClassWeight"}]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for hname in hist_names:
        if hname not in all_records:
            continue
        hist = all_records[hname].get("history", {})
        if not hist.get("val_accuracy"):
            continue
        ep = range(1, len(hist["val_accuracy"]) + 1)
        axes[0].plot(ep, hist["val_accuracy"], marker="o", markersize=3, label=hname)
        axes[1].plot(ep, hist["val_loss"], marker="s", markersize=3, label=hname)
    axes[0].set_title("Validation Accuracy Across Training Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7.5)
    axes[1].set_title("Validation Loss Across Training Epochs", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(PHASE6_DIR / "training_history_comparison.png", dpi=150)
    plt.close(fig)

    # 5. Write phase6_report.md
    p6_lines = [
        "# Phase 6 — Targeted Optimization Report (Validation Results Only)",
        "",
        "> **Scope & Disclaimer:** All Phase 6 tuning, threshold calibration, and ensemble comparisons were performed strictly on the `3,501`-image validation split (`Dataset/Processed/val`). The test split was **not** used for optimization. This software is an educational research prototype and is not a clinically validated diagnostic system.",
        "",
        "## 1. Phase 5 Audit Findings",
        "- Verified 1:1 class alignment across all checkpoints and `val_probs.npy` (`0=BCC (498), 1=BKL (394), 2=MEL (678), 3=NV (1,931)`).",
        "- Phase 5 `Ensemble_Top3_Weighted` errors concentrated on: `BKL -> MEL` (`72/394`, `18.3%`), `BKL -> BCC` (`65/394`, `16.5%`), `MEL -> NV` (`179/678`, `26.4%`), and `NV -> MEL` (`177/1931`, `9.2%`).",
        "- Mean predicted probabilities showed `BKL` under-prediction (`0.0956` vs `0.1125` prevalence) and `MEL` over-prediction (`0.2357` vs `0.1937` prevalence), while Stage 2 training histories peaked around Epoch 7–10 with `+0.155` to `+0.245` train-val accuracy gaps.",
        "",
        "## 2. Targeted Phase 6 Training Experiments (Overfitting Reduction & BKL/MEL Focus)",
        "| Experiment | Architecture | Val Acc | Bal Acc | Macro F1 | BKL Recall | MEL Recall | Train-Val Gap | Time (s) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in p6_runs:
        vm = r["val_metrics"]
        h = r.get("history", {})
        gap = (h["accuracy"][-1] - h["val_accuracy"][-1]) if h.get("accuracy") else 0.0
        p6_lines.append(
            f"| `{r['name']}` | `{r['architecture']}` | {vm['accuracy']*100:.2f}% | "
            f"{vm['balanced_accuracy']*100:.2f}% | {vm['macro_f1']*100:.2f}% | "
            f"{vm['per_class']['BKL']['recall']*100:.2f}% | {vm['mel_recall']*100:.2f}% | "
            f"{gap:+.3f} | {r['training_time_sec']:.1f} |"
        )

    p6_lines += [
        "",
        "## 3. Selected Phase 6 Configuration & Comparison vs Phase 5 Baseline (Validation)",
        f"- **Selected Ensemble:** `{best_ens_key}` (`members={best_ens['members']}`, `weights={best_ens['weights']}`)",
        f"- **Selected Decision Rule:** `{best_rule_name}`",
        "",
        "| Metric (Validation, n=3,501) | Phase 5 Ensemble | Phase 6 Optimized | Delta |",
        "| :--- | ---: | ---: | ---: |",
        f"| Overall Accuracy | {p5_m['accuracy']*100:.2f}% | {best_rule_metrics['accuracy']*100:.2f}% | {(best_rule_metrics['accuracy']-p5_m['accuracy'])*100:+.2f}% |",
        f"| Balanced Accuracy | {p5_m['balanced_accuracy']*100:.2f}% | {best_rule_metrics['balanced_accuracy']*100:.2f}% | {(best_rule_metrics['balanced_accuracy']-p5_m['balanced_accuracy'])*100:+.2f}% |",
        f"| Macro F1-Score | {p5_m['macro_f1']*100:.2f}% | {best_rule_metrics['macro_f1']*100:.2f}% | {(best_rule_metrics['macro_f1']-p5_m['macro_f1'])*100:+.2f}% |",
        f"| Macro ROC-AUC | {p5_m['macro_roc_auc']*100:.2f}% | {best_rule_metrics['macro_roc_auc']*100:.2f}% | {(best_rule_metrics['macro_roc_auc']-p5_m['macro_roc_auc'])*100:+.2f}% |",
        f"| BKL Recall | {p5_m['per_class']['BKL']['recall']*100:.2f}% | {best_rule_metrics['bkl_recall']*100:.2f}% | {(best_rule_metrics['bkl_recall']-p5_m['per_class']['BKL']['recall'])*100:+.2f}% |",
        f"| MEL Recall | {p5_m['mel_recall']*100:.2f}% | {best_rule_metrics['mel_recall']*100:.2f}% | {(best_rule_metrics['mel_recall']-p5_m['mel_recall'])*100:+.2f}% |",
        "",
    ]
    (PHASE6_DIR / "phase6_report.md").write_text("\n".join(p6_lines), encoding="utf-8")
    print(f"Saved all Phase 6 deliverables to {PHASE6_DIR}")


def run_phase6_pipeline(force_retrain: bool = False) -> None:
    """Run controlled Phase 6 experiments (P6_1..P6_3) and full Phase 6 analysis."""
    # Sqrt-tempered class weights boosting BKL & MEL without over-penalizing NV
    tempered_cw = {"0": 1.29, "1": 1.62, "2": 1.23, "3": 0.64}

    p6_configs = [
        {
            "name": "P6_1_ConvNeXtTiny_Reg_Aug",
            "architecture": "ConvNeXtTiny",
            "image_size": 384,
            "batch_size": 8,
            "use_augment": True,
            "dropout": 0.35,
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
            "stage1_epochs": 6,
            "stage2_epochs": 14,
            "stage1_lr": 1e-4,
            "stage2_lr": 5e-6,
            "unfreeze_fraction": 0.30,
        },
        {
            "name": "P6_2_ConvNeXtTiny_Reg_Aug_TemperedCW",
            "architecture": "ConvNeXtTiny",
            "image_size": 384,
            "batch_size": 8,
            "use_augment": True,
            "dropout": 0.35,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "class_weight": tempered_cw,
            "stage1_epochs": 6,
            "stage2_epochs": 14,
            "stage1_lr": 1e-4,
            "stage2_lr": 5e-6,
            "unfreeze_fraction": 0.30,
        },
        {
            "name": "P6_3_MobileNetV3_Reg_Aug_TemperedCW",
            "architecture": "MobileNetV3Large",
            "image_size": 384,
            "batch_size": 8,
            "use_augment": True,
            "dropout": 0.35,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "class_weight": tempered_cw,
            "stage1_epochs": 6,
            "stage2_epochs": 14,
            "stage1_lr": 1e-4,
            "stage2_lr": 5e-6,
            "unfreeze_fraction": 0.30,
        },
    ]

    p6_runs: list[dict[str, Any]] = []
    for cfg in p6_configs:
        p6_runs.append(_run_phase6_isolated(cfg, force_retrain=force_retrain))

    run_phase6_analysis_and_deliverables(p6_runs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Four-Class Skin Lesion Classification (ISIC 2019)"
    )
    parser.add_argument(
        "--stage",
        choices=[
            "eda",
            "split",
            "smoke",
            "train",
            "train-single",
            "all",
            "phase5-smoke",
            "phase5-single",
            "phase5-eval-effnet",
            "phase5-results",
            "phase5",
            "phase6-single",
            "phase6-analyze",
            "phase6",
        ],
        default="eda",
        help="Pipeline stage to execute.",
    )
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for training/eval.")
    parser.add_argument("--max-epochs", type=int, default=20, help="Max epochs for initial experiments.")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed_float16 (recommended on 6 GB NVIDIA GPUs).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore cached per-experiment checkpoints and retrain from scratch.",
    )
    parser.add_argument(
        "--exp-json",
        type=str,
        default="",
        help="Internal JSON config for isolated subprocess execution.",
    )
    args = parser.parse_args()

    if args.stage == "eda":
        run_eda_stage()
    elif args.stage == "split":
        enriched_df, _ = run_eda_stage()
        create_stratified_group_split(enriched_df)
    elif args.stage == "smoke":
        run_smoke_check()
    elif args.stage == "train-single":
        import keras

        set_global_seed(RANDOM_SEED)
        check_environment()
        if args.mixed_precision:
            keras.mixed_precision.set_global_policy("mixed_float16")
            print("Enabled mixed_float16 policy.")
        exp_cfg = json.loads(args.exp_json)
        class_weights, focal_alpha = compute_class_weights(PROCESSED_DIR / "train")
        run_single_experiment(
            exp_cfg, class_weights, focal_alpha, force_retrain=args.force_retrain
        )
    elif args.stage == "train":
        set_global_seed(RANDOM_SEED)
        all_results, best_run = run_controlled_experiments(
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            enable_mixed_precision=args.mixed_precision,
            force_retrain=args.force_retrain,
        )
        check_environment()
        evaluate_best_on_test_and_plot(all_results, best_run)
    elif args.stage == "all":
        enriched_df, _ = run_eda_stage()
        create_stratified_group_split(enriched_df)
        all_results, best_run = run_controlled_experiments(
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            enable_mixed_precision=args.mixed_precision,
            force_retrain=args.force_retrain,
        )
        check_environment()
        evaluate_best_on_test_and_plot(all_results, best_run)
    elif args.stage == "phase5-smoke":
        set_global_seed(RANDOM_SEED)
        run_phase5_smoke_check()
    elif args.stage == "phase5-single":
        exp_cfg = json.loads(args.exp_json)
        class_weights, focal_alpha = compute_class_weights(PROCESSED_DIR / "train")
        run_phase5_single_experiment(
            exp_cfg, class_weights, focal_alpha, force_retrain=args.force_retrain
        )
    elif args.stage == "phase5-eval-effnet":
        set_global_seed(RANDOM_SEED)
        _eval_effnet_ref_worker()
    elif args.stage == "phase5-results":
        set_global_seed(RANDOM_SEED)
        run_phase5_results_only()
    elif args.stage == "phase5":
        set_global_seed(RANDOM_SEED)
        run_phase5_pipeline(force_retrain=args.force_retrain)
    elif args.stage == "phase6-single":
        exp_cfg = json.loads(args.exp_json)
        run_phase6_single_experiment(exp_cfg, force_retrain=args.force_retrain)
    elif args.stage == "phase6-analyze":
        set_global_seed(RANDOM_SEED)
        p6_runs = [
            json.loads(rp.read_text(encoding="utf-8"))
            for rp in sorted(PHASE6_MODELS_DIR.rglob("*_result.json"))
        ]
        run_phase6_analysis_and_deliverables(p6_runs)
    elif args.stage == "phase6":
        set_global_seed(RANDOM_SEED)
        run_phase6_pipeline(force_retrain=args.force_retrain)


if __name__ == "__main__":
    main()
