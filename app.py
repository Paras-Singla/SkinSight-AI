"""
SkinSight AI — AI-Powered Skin Lesion Classification Web Application.

Runs inference using the pre-trained Ensemble_Top3_Weighted model
(ConvNeXtTiny + EfficientNetV2B0 + MobileNetV3Large) defined in
Models/ensemble_config.json, with automatic single-model fallback
if only one checkpoint is available in the deployment environment.

Never trains models, loads datasets, or exposes internal filesystem paths.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Paths & Constants (Strictly relative to project root — no local drive paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "Models"
ASSETS_DIR = PROJECT_ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
CLASS_MAPPING_PATH = MODELS_DIR / "class_mapping.json"
ENSEMBLE_CONFIG_PATH = MODELS_DIR / "ensemble_config.json"

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}
MAX_IMAGE_PIXELS = 40_000_000  # Guard against decompression bombs

MEDICAL_DISCLAIMER = (
    "This application is an educational and research prototype. Its predictions are "
    "not a medical diagnosis and must not replace examination by a qualified "
    "dermatologist. If you have a suspicious or changing skin lesion, consult a "
    "healthcare professional."
)

CLASS_INFO: dict[str, dict[str, str]] = {
    "BCC": {
        "name": "Basal Cell Carcinoma",
        "short": (
            "A slow-growing, localized skin lesion arising from basal skin cells, "
            "typically associated with sun-exposed areas."
        ),
    },
    "BKL": {
        "name": "Benign Keratosis",
        "short": (
            "A non-cancerous skin growth category including seborrheic keratoses "
            "and solar lentigines."
        ),
    },
    "MEL": {
        "name": "Melanoma",
        "short": (
            "A serious pigment-cell skin cancer requiring prompt medical evaluation "
            "when changes in color, border, or size appear."
        ),
    },
    "NV": {
        "name": "Melanocytic Nevi",
        "short": (
            "Common benign moles formed by clusters of pigment-producing melanocytes "
            "with generally uniform appearance."
        ),
    },
}


# ---------------------------------------------------------------------------
# Runtime Configuration & Remote Weight Handling
# ---------------------------------------------------------------------------
def _configure_tensorflow_runtime() -> None:
    """Enable GPU memory growth if present; operate cleanly on CPU otherwise."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        # CPU-only containers and restricted cloud runtimes run cleanly on host CPU.
        pass


def _resolve_secret_or_env(env_var: str, secret_key: str, default_url: str) -> str:
    """Resolve optional HTTPS model download URL without hardcoding credentials."""
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        return env_val
    try:
        if "model_urls" in st.secrets and secret_key in st.secrets["model_urls"]:
            return str(st.secrets["model_urls"][secret_key]).strip()
        if secret_key in st.secrets:
            return str(st.secrets[secret_key]).strip()
    except Exception:
        pass
    return default_url.strip()


def _verify_sha256(file_path: Path, expected_sha256: str) -> bool:
    """Verify file digest when a reference SHA-256 checksum is configured."""
    if not expected_sha256:
        return True
    digest = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected_sha256.strip().lower()


def _download_model_if_configured(member: dict[str, Any]) -> Path | None:
    """
    Download a missing checkpoint from a configured HTTPS URL into target_path.
    Returns the downloaded Path on success, or None if no URL is configured or
    the download fails.
    """
    url = _resolve_secret_or_env(
        member.get("env_url_var", ""),
        member.get("secret_key", ""),
        member.get("download_url", ""),
    )
    if not url:
        return None

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() != "https":
        return None

    target_rel = member.get("target_path", "")
    if not target_rel:
        return None
    target_path = (PROJECT_ROOT / target_rel).resolve()
    if PROJECT_ROOT not in target_path.parents and target_path != PROJECT_ROOT:
        return None

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".keras.tmp")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SkinSight-AI-ModelFetcher/1.0"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        expected_sha = str(member.get("sha256", "")).strip()
        if not _verify_sha256(tmp_path, expected_sha):
            tmp_path.unlink(missing_ok=True)
            return None
        os.replace(tmp_path, target_path)
        return target_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        return None


def _resolve_member_checkpoint(member: dict[str, Any]) -> Path | None:
    """Locate an ensemble member checkpoint across candidate project paths."""
    for rel_path in member.get("candidate_paths", []):
        candidate = PROJECT_ROOT / rel_path
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return _download_model_if_configured(member)


@st.cache_resource(show_spinner=False)
def load_class_mapping() -> dict[int, str]:
    """Load verified class index-to-abbreviation mapping (0=BCC, 1=BKL, 2=MEL, 3=NV)."""
    if CLASS_MAPPING_PATH.is_file():
        raw = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
        idx_map = raw.get("index_to_class", raw)
        return {int(k): str(v) for k, v in idx_map.items()}
    return {0: "BCC", 1: "BKL", 2: "MEL", 3: "NV"}


@st.cache_resource(show_spinner=False)
def load_inference_engine() -> dict[str, Any]:
    """
    Load the Ensemble_Top3_Weighted models once per server process.

    Falls back to the highest-priority individual model if any ensemble member
    checkpoint cannot be located or loaded in the current deployment environment.
    """
    _configure_tensorflow_runtime()

    if not ENSEMBLE_CONFIG_PATH.is_file():
        return {"ready": False, "reason": "missing_config"}

    cfg = json.loads(ENSEMBLE_CONFIG_PATH.read_text(encoding="utf-8"))
    members = cfg.get("members", [])
    fallback_order = cfg.get(
        "fallback_priority",
        ["ConvNeXtTiny", "EfficientNetV2B0", "MobileNetV3Large"],
    )

    resolved_members: list[dict[str, Any]] = []
    for member in members:
        ckpt = _resolve_member_checkpoint(member)
        if ckpt is not None:
            resolved_members.append({**member, "resolved_path": ckpt})

    # Preferred path: load full Ensemble_Top3_Weighted when all members are available
    if len(resolved_members) == len(members) and len(members) > 0:
        loaded: list[dict[str, Any]] = []
        ensemble_load_ok = True
        for item in resolved_members:
            try:
                model = tf.keras.models.load_model(item["resolved_path"], compile=False)
                loaded.append(
                    {
                        "architecture": item["architecture"],
                        "weight": float(item["weight"]),
                        "image_size": int(item.get("image_size", 384)),
                        "model": model,
                    }
                )
            except Exception:
                ensemble_load_ok = False
                loaded.clear()
                break

        if ensemble_load_ok and len(loaded) == len(members):
            return {
                "ready": True,
                "mode": "ensemble",
                "models": loaded,
            }

    # Fallback path: load the single strongest available architecture checkpoint
    by_arch = {item["architecture"]: item for item in resolved_members}
    for arch in fallback_order:
        item = by_arch.get(arch)
        if item is None:
            continue
        try:
            model = tf.keras.models.load_model(item["resolved_path"], compile=False)
            return {
                "ready": True,
                "mode": "single_fallback",
                "models": [
                    {
                        "architecture": item["architecture"],
                        "weight": 1.0,
                        "image_size": int(item.get("image_size", 384)),
                        "model": model,
                    }
                ],
            }
        except Exception:
            continue

    return {"ready": False, "reason": "no_models_available"}


# ---------------------------------------------------------------------------
# Image Validation, Preprocessing & Inference Pipeline
# ---------------------------------------------------------------------------
def validate_and_decode_image(file_bytes: bytes) -> tuple[Image.Image, str]:
    """Validate uploaded bytes and decode into an oriented RGB PIL Image."""
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    with Image.open(io.BytesIO(file_bytes)) as check_img:
        img_format = (check_img.format or "").upper()
        width, height = check_img.size
        if img_format not in ALLOWED_PIL_FORMATS:
            raise ValueError("Unsupported format. Please upload a JPG, JPEG, or PNG image.")
        if width <= 0 or height <= 0 or (width * height) > MAX_IMAGE_PIXELS:
            raise ValueError("Image dimensions are invalid or exceed safe limits.")
        check_img.verify()

    with Image.open(io.BytesIO(file_bytes)) as img:
        oriented = ImageOps.exif_transpose(img)
        rgb_img = oriented.convert("RGB")
        return rgb_img, img_format


def preprocess_image(image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    """
    Match validation-time preprocessing across all ensemble backbones:
    - RGB channel order
    - Bilinear resize to (height, width)
    - float32 values in [0, 255]; saved .keras graphs contain architecture-specific
      Rescaling and Normalization layers internally.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    resized = tf.image.resize(arr, size=target_size, method=tf.image.ResizeMethod.BILINEAR)
    return np.expand_dims(resized.numpy(), axis=0)


def run_prediction(
    engine: dict[str, Any],
    image: Image.Image,
    index_to_class: dict[int, str],
) -> list[dict[str, Any]]:
    """Run forward inference and return sorted 4-class probabilities summing to 1."""
    num_classes = len(index_to_class)
    accum = np.zeros((num_classes,), dtype=np.float64)
    total_weight = 0.0
    batch_cache: dict[int, np.ndarray] = {}

    for entry in engine["models"]:
        img_sz = int(entry["image_size"])
        if img_sz not in batch_cache:
            batch_cache[img_sz] = preprocess_image(image, (img_sz, img_sz))
        preds = entry["model"](batch_cache[img_sz], training=False).numpy()[0].astype(np.float64)
        w = float(entry["weight"])
        accum += w * preds
        total_weight += w

    if total_weight <= 0.0 or not np.all(np.isfinite(accum)):
        raise RuntimeError("Invalid model output during inference.")

    probs = np.clip(accum / total_weight, 0.0, None)
    prob_sum = float(probs.sum())
    if prob_sum <= 0.0:
        raise RuntimeError("Model produced zero probability mass.")
    probs = probs / prob_sum

    results: list[dict[str, Any]] = []
    for idx in sorted(index_to_class.keys()):
        code = index_to_class[idx]
        p = float(probs[idx])
        meta = CLASS_INFO.get(code, {"name": code, "short": ""})
        results.append(
            {
                "index": idx,
                "code": code,
                "name": meta["name"],
                "short": meta["short"],
                "probability": p,
                "percentage": p * 100.0,
            }
        )

    results.sort(key=lambda item: item["probability"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Styling & UI Components
# ---------------------------------------------------------------------------
def _logo_data_uri() -> str:
    """Return embedded base64 URI for assets/logo.png if present."""
    if not LOGO_PATH.is_file():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def inject_custom_css() -> None:
    """Apply dark healthcare SaaS theme and responsive layout styles."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-base: #070b14;
    --bg-card: rgba(15, 23, 42, 0.86);
    --bg-card-elevated: rgba(20, 30, 52, 0.94);
    --border-subtle: rgba(148, 163, 184, 0.16);
    --border-accent: rgba(56, 189, 248, 0.42);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent-cyan: #38bdf8;
    --accent-indigo: #6366f1;
    --accent-violet: #8b5cf6;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background:
        radial-gradient(900px 420px at 8% -5%, rgba(56, 189, 248, 0.11), transparent 60%),
        radial-gradient(850px 420px at 92% -5%, rgba(139, 92, 246, 0.11), transparent 60%),
        var(--bg-base);
}

/* Remove generic Streamlit chrome clutter */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stHeader"] {
    display: none !important;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* Header bar */
.ss-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.ss-logo {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.22);
    flex-shrink: 0;
}

.ss-brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.15;
    background: linear-gradient(95deg, #f8fafc 10%, #38bdf8 60%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.ss-brand-sub {
    font-size: 0.84rem;
    color: var(--text-secondary);
    font-weight: 500;
    margin-top: 2px;
}

.ss-header-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.28), rgba(139,92,246,0.24), rgba(148,163,184,0.06));
    margin: 0.9rem 0 1.65rem 0;
}

/* Hero Section */
.ss-hero {
    padding: 0.35rem 0 1.1rem 0;
}

.ss-hero h1 {
    font-size: clamp(1.75rem, 3.2vw, 2.55rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.18;
    margin: 0 0 0.65rem 0;
    color: var(--text-primary);
}

.ss-hero p {
    font-size: 1.02rem;
    line-height: 1.6;
    color: var(--text-secondary);
    max-width: 720px;
    margin: 0;
}

/* Disclaimer Banner */
.ss-disclaimer {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 0.9rem 1.15rem;
    border-radius: 12px;
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.28);
    color: #fde68a;
    font-size: 0.87rem;
    line-height: 1.5;
    margin: 0.9rem 0 1.5rem 0;
}

.ss-disclaimer strong {
    color: #fcd34d;
}

/* Dark Cards */
.ss-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.35rem 1.4rem;
    box-shadow: 0 12px 32px rgba(2, 6, 23, 0.45);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.ss-card:hover {
    border-color: rgba(148, 163, 184, 0.28);
}

.ss-card-title {
    font-size: 1.02rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text-primary);
    margin: 0 0 0.35rem 0;
}

.ss-card-sub {
    font-size: 0.86rem;
    color: var(--text-secondary);
    margin: 0 0 1rem 0;
    line-height: 1.45;
}

/* Image metadata strip */
.ss-meta-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0.75rem 0 0.95rem 0;
}

.ss-meta-pill {
    font-size: 0.78rem;
    font-weight: 500;
    color: #cbd5e1;
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 999px;
    padding: 4px 11px;
    font-variant-numeric: tabular-nums;
}

/* Result Hero Card */
.ss-result-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.96), rgba(30, 27, 75, 0.62));
    border: 1px solid var(--border-accent);
    border-radius: 16px;
    padding: 1.4rem 1.45rem;
    box-shadow: 0 14px 36px rgba(56, 189, 248, 0.12);
    margin-bottom: 1.15rem;
}

.ss-result-kicker {
    display: inline-block;
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.32);
    border-radius: 999px;
    padding: 4px 10px;
    margin-bottom: 0.75rem;
}

.ss-result-grid {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 16px;
    flex-wrap: wrap;
}

.ss-result-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    margin-bottom: 4px;
}

.ss-result-class {
    font-size: clamp(1.3rem, 2.2vw, 1.65rem);
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.ss-result-prob {
    font-size: clamp(1.75rem, 3vw, 2.3rem);
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}

.ss-result-desc {
    margin-top: 0.85rem;
    padding-top: 0.85rem;
    border-top: 1px solid rgba(148, 163, 184, 0.14);
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.5;
}

/* Probability Progress Bars */
.ss-prob-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.3rem 1.4rem;
}

.ss-prob-row {
    padding: 0.65rem 0.75rem;
    border-radius: 12px;
    margin-bottom: 0.55rem;
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid transparent;
    transition: border-color 0.2s ease, background 0.2s ease;
}

.ss-prob-row:last-child {
    margin-bottom: 0;
}

.ss-prob-row.top {
    background: rgba(56, 189, 248, 0.08);
    border-color: rgba(56, 189, 248, 0.34);
}

.ss-prob-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
}

.ss-prob-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.ss-code-badge {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 2px 8px;
    border-radius: 6px;
    background: rgba(148, 163, 184, 0.14);
    color: #cbd5e1;
}

.ss-prob-row.top .ss-code-badge {
    background: rgba(56, 189, 248, 0.2);
    color: #7dd3fc;
}

.ss-prob-val {
    font-size: 0.94rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: #cbd5e1;
}

.ss-prob-row.top .ss-prob-val {
    color: #38bdf8;
}

.ss-bar-track {
    width: 100%;
    height: 9px;
    border-radius: 999px;
    background: rgba(30, 41, 59, 0.95);
    overflow: hidden;
}

.ss-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #475569, #64748b);
    transition: width 0.35s ease;
}

.ss-prob-row.top .ss-bar-fill {
    background: linear-gradient(90deg, #06b6d4 0%, #6366f1 58%, #8b5cf6 100%);
}

/* Placeholder right panel */
.ss-empty-result {
    background: var(--bg-card);
    border: 1px dashed rgba(148, 163, 184, 0.24);
    border-radius: 16px;
    padding: 2.3rem 1.5rem;
    text-align: center;
}

.ss-empty-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.45rem;
}

.ss-empty-sub {
    font-size: 0.89rem;
    color: var(--text-secondary);
    line-height: 1.55;
    max-width: 420px;
    margin: 0 auto;
}

/* Info Cards Grid */
.ss-section-heading {
    font-size: 1.12rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text-primary);
    margin: 2rem 0 0.85rem 0;
}

.ss-info-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
}

.ss-info-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.05rem 1.1rem;
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.ss-info-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
    transform: translateY(-2px);
}

.ss-info-code {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #38bdf8;
    margin-bottom: 4px;
}

.ss-info-name {
    font-size: 0.96rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 6px;
}

.ss-info-desc {
    font-size: 0.83rem;
    line-height: 1.48;
    color: var(--text-secondary);
}

/* Streamlit Buttons & File Uploader Polish */
div.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    min-height: 42px !important;
    transition: all 0.2s ease !important;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(95deg, #0ea5e9 0%, #6366f1 55%, #8b5cf6 100%) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 8px 22px rgba(99, 102, 241, 0.28) !important;
}

div.stButton > button[kind="primary"]:hover {
    filter: brightness(1.08);
    box-shadow: 0 10px 26px rgba(56, 189, 248, 0.36) !important;
}

div.stButton > button[kind="secondary"] {
    background: rgba(30, 41, 59, 0.85) !important;
    border: 1px solid rgba(148, 163, 184, 0.22) !important;
    color: #e2e8f0 !important;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: rgba(56, 189, 248, 0.45) !important;
    color: #ffffff !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 23, 42, 0.65) !important;
    border: 1.5px dashed rgba(56, 189, 248, 0.35) !important;
    border-radius: 14px !important;
    padding: 1.15rem !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(56, 189, 248, 0.68) !important;
    background: rgba(15, 23, 42, 0.88) !important;
}

[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.16);
}

.ss-footer {
    margin-top: 2.2rem;
    padding-top: 1.1rem;
    border-top: 1px solid rgba(148, 163, 184, 0.12);
    font-size: 0.8rem;
    color: #64748b;
    line-height: 1.5;
}

/* Responsive Breakpoints */
@media (max-width: 960px) {
    .ss-info-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
    }
    .ss-brand {
        margin-bottom: 0.45rem;
    }
    .ss-info-grid {
        grid-template-columns: 1fr;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Home"
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    if "prediction_results" not in st.session_state:
        st.session_state["prediction_results"] = None
    if "prediction_sig" not in st.session_state:
        st.session_state["prediction_sig"] = None


def _clear_image_and_prediction() -> None:
    st.session_state["prediction_results"] = None
    st.session_state["prediction_sig"] = None
    st.session_state["uploader_key"] += 1


def render_header() -> str:
    """Render minimal SkinSight AI brand header with Home / About navigation."""
    logo_uri = _logo_data_uri()
    logo_html = f'<img src="{logo_uri}" class="ss-logo" alt="SkinSight AI logo">' if logo_uri else ""

    brand_col, nav_col = st.columns([3.2, 1.3], vertical_alignment="center")
    with brand_col:
        st.markdown(
            f"""
<div class="ss-brand">
{logo_html}
<div>
<div class="ss-brand-title">SkinSight AI</div>
<div class="ss-brand-sub">AI-Powered Skin Lesion Classification</div>
</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with nav_col:
        b1, b2 = st.columns(2, gap="small")
        active = st.session_state["active_page"]
        with b1:
            if st.button(
                "Home",
                key="nav_home",
                type="primary" if active == "Home" else "secondary",
                width="stretch",
            ):
                st.session_state["active_page"] = "Home"
                st.rerun()
        with b2:
            if st.button(
                "About",
                key="nav_about",
                type="primary" if active == "About" else "secondary",
                width="stretch",
            ):
                st.session_state["active_page"] = "About"
                st.rerun()

    st.markdown('<div class="ss-header-divider"></div>', unsafe_allow_html=True)
    return str(st.session_state["active_page"])


def render_disclaimer() -> None:
    st.markdown(
        f"""
<div class="ss-disclaimer">
<div><strong>Medical Disclaimer:</strong> {html.escape(MEDICAL_DISCLAIMER)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_lesion_categories_section() -> None:
    """Render concise educational overview of the 4 skin lesion categories."""
    cards_html = []
    for code in ("BCC", "BKL", "MEL", "NV"):
        info = CLASS_INFO[code]
        cards_html.append(
            f'<div class="ss-info-card">'
            f'<div class="ss-info-code">{html.escape(code)}</div>'
            f'<div class="ss-info-name">{html.escape(info["name"])}</div>'
            f'<div class="ss-info-desc">{html.escape(info["short"])}</div>'
            f'</div>'
        )
    st.markdown('<div class="ss-section-heading">Lesion Categories</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ss-info-grid">{"".join(cards_html)}</div>', unsafe_allow_html=True)


def render_prediction_panel(results: list[dict[str, Any]]) -> None:
    """Render the highlighted top prediction card and horizontal probability bars."""
    top = results[0]
    top_title = f"{html.escape(top['name'])} ({html.escape(top['code'])})"
    top_pct = f"{top['percentage']:.1f}%"
    top_desc = html.escape(top.get("short", ""))

    st.markdown(
        f"""
<div class="ss-result-card">
<div class="ss-result-kicker">AI Prediction Result</div>
<div class="ss-result-grid">
<div>
<div class="ss-result-label">Predicted Class</div>
<div class="ss-result-class">{top_title}</div>
</div>
<div style="text-align: right;">
<div class="ss-result-label">Probability</div>
<div class="ss-result-prob">{top_pct}</div>
</div>
</div>
<div class="ss-result-desc">{top_desc}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    rows_html = []
    for idx, item in enumerate(results):
        row_cls = "ss-prob-row top" if idx == 0 else "ss-prob-row"
        bar_w = max(0.5, min(100.0, float(item["percentage"])))
        pct_str = f"{item['percentage']:.1f}%"
        rows_html.append(
            f'<div class="{row_cls}">'
            f'<div class="ss-prob-header">'
            f'<div class="ss-prob-name">'
            f'<span>{html.escape(item["name"])}</span>'
            f'<span class="ss-code-badge">{html.escape(item["code"])}</span>'
            f'</div>'
            f'<div class="ss-prob-val">{pct_str}</div>'
            f'</div>'
            f'<div class="ss-bar-track">'
            f'<div class="ss-bar-fill" style="width: {bar_w:.2f}%;"></div>'
            f'</div>'
            f'</div>'
        )

    st.markdown(
        f"""
<div class="ss-prob-card">
<div class="ss-card-title">Probability Distribution</div>
<div class="ss-card-sub">Estimated likelihood across all four lesion classes</div>
{"".join(rows_html)}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_home_page(engine: dict[str, Any], index_to_class: dict[int, str]) -> None:
    """Render hero, disclaimer, image upload/prediction workspace, and category guide."""
    st.markdown(
        """
<div class="ss-hero">
<h1>Understand Your Skin Lesion Image with AI</h1>
<p>Upload a skin lesion image to receive an AI-generated classification with probability estimates across four categories.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    render_disclaimer()

    if not engine.get("ready"):
        st.error(
            "Model weights are not currently available in this deployment environment. "
            "Please configure the model checkpoint URLs or files as documented in README.md."
        )
        render_lesion_categories_section()
        return

    left_col, right_col = st.columns([1.0, 1.12], gap="large")

    with left_col:
        st.markdown(
            """
<div class="ss-card">
<div class="ss-card-title">Upload Lesion Image</div>
<div class="ss-card-sub">Drag and drop or browse a clear JPG, JPEG, or PNG skin lesion image.</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload skin lesion image",
            type=ALLOWED_EXTENSIONS,
            key=f"skinsight_uploader_{st.session_state['uploader_key']}",
            label_visibility="collapsed",
        )

        decoded_image: Image.Image | None = None
        image_sig: str | None = None
        predict_clicked = False

        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            try:
                decoded_image, img_format = validate_and_decode_image(raw_bytes)
            except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
                st.error("Unable to read this file. Please upload a valid JPG, JPEG, or PNG image.")
                decoded_image = None
            else:
                image_sig = f"{uploaded_file.name}:{len(raw_bytes)}"
                if st.session_state["prediction_sig"] != image_sig:
                    st.session_state["prediction_results"] = None

                st.image(decoded_image, width="stretch")
                w, h = decoded_image.size
                size_kb = len(raw_bytes) / 1024.0
                st.markdown(
                    f"""
<div class="ss-meta-strip">
<span class="ss-meta-pill">Dimensions: {w} &times; {h} px</span>
<span class="ss-meta-pill">Format: {html.escape(img_format)}</span>
<span class="ss-meta-pill">Size: {size_kb:.1f} KB</span>
</div>
                    """,
                    unsafe_allow_html=True,
                )

                act1, act2 = st.columns([1.4, 1.0], gap="small")
                with act1:
                    predict_clicked = st.button("Predict", type="primary", width="stretch")
                with act2:
                    st.button(
                        "Clear Image",
                        type="secondary",
                        on_click=_clear_image_and_prediction,
                        width="stretch",
                    )

        if predict_clicked and decoded_image is not None and image_sig is not None:
            try:
                with st.spinner("Analyzing lesion image..."):
                    results = run_prediction(engine, decoded_image, index_to_class)
                st.session_state["prediction_results"] = results
                st.session_state["prediction_sig"] = image_sig
            except Exception:
                st.error("Prediction could not be completed for this image. Please try another file.")

    with right_col:
        active_results = st.session_state.get("prediction_results")
        if active_results:
            render_prediction_panel(active_results)
        else:
            st.markdown(
                """
<div class="ss-empty-result">
<div class="ss-empty-title">AI Prediction Result</div>
<div class="ss-empty-sub">
Upload a skin lesion image on the left and click <strong>Predict</strong> to view the predicted category and four-class probability distribution.
</div>
</div>
                """,
                unsafe_allow_html=True,
            )

    render_lesion_categories_section()


def render_about_page(engine: dict[str, Any]) -> None:
    """Render minimal, product-focused About page without evaluation dashboards."""
    st.markdown(
        """
<div class="ss-hero">
<h1>About SkinSight AI</h1>
<p>An educational AI prototype designed to analyze dermoscopic and skin lesion images across four clinical categories.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    if engine.get("mode") == "ensemble":
        arch_summary = (
            "Weighted three-architecture neural network ensemble combining "
            "ConvNeXtTiny, EfficientNetV2B0, and MobileNetV3Large at 384 &times; 384 RGB resolution."
        )
    elif engine.get("mode") == "single_fallback" and engine.get("models"):
        active_arch = html.escape(str(engine["models"][0]["architecture"]))
        arch_summary = (
            f"Running single-architecture fallback inference ({active_arch} at "
            "384 &times; 384 RGB resolution) because one or more ensemble weight files "
            "are not present in this environment."
        )
    else:
        arch_summary = "Multi-architecture 384 &times; 384 RGB neural network inference pipeline."

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            f"""
<div class="ss-card">
<div class="ss-card-title">How It Works</div>
<div class="ss-card-sub">End-to-end image analysis pipeline</div>
<p style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6; margin: 0;">
Uploaded JPG or PNG images are validated, oriented, resized to a standardized
384 &times; 384 RGB representation, and evaluated using deep convolutional neural networks.
The model produces normalized probability estimates summing to 100% across four target classes.
</p>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
<div class="ss-card">
<div class="ss-card-title">Active Model Architecture</div>
<div class="ss-card-sub">Multi-backbone visual representation</div>
<p style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6; margin: 0;">
{arch_summary}
</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    render_lesion_categories_section()
    render_disclaimer()


def main() -> None:
    page_icon: Any = str(LOGO_PATH) if LOGO_PATH.is_file() else "🔬"
    st.set_page_config(
        page_title="SkinSight AI — Skin Lesion Classification",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _init_state()
    inject_custom_css()

    active_page = render_header()
    index_to_class = load_class_mapping()
    engine = load_inference_engine()

    if active_page == "About":
        render_about_page(engine)
    else:
        render_home_page(engine, index_to_class)

    st.markdown(
        f'<div class="ss-footer">{html.escape(MEDICAL_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
