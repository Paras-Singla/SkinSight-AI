# SkinSight AI — AI-Powered Skin Lesion Classification

**SkinSight AI** is a dark-themed Streamlit web application for classifying dermoscopic skin lesion images across four categories (`BCC`, `BKL`, `MEL`, `NV`) using a weighted deep-learning ensemble (`ConvNeXtTiny`, `EfficientNetV2B0`, and `MobileNetV3Large`).

> **Medical Disclaimer:** This application is an educational and research prototype. Its predictions are not a medical diagnosis and must not replace examination by a qualified dermatologist. If you have a suspicious or changing skin lesion, consult a healthcare professional.

---

## 1. Project Overview

SkinSight AI takes a user-uploaded skin lesion image (`JPG`, `JPEG`, or `PNG`), validates and resizes it to `384 × 384` RGB, runs cached CPU-compatible TensorFlow/Keras inference via a weighted three-architecture ensemble (`Ensemble_Top3_Weighted`), and outputs:

- The highest-probability lesion class and probability percentage.
- Probability estimates across all four target classes (`0 = BCC`, `1 = BKL`, `2 = MEL`, `3 = NV`).
- Concise educational descriptions for each category.

### Target Classes
| Index | Code | Category |
| :---: | :--- | :--- |
| `0` | `BCC` | Basal Cell Carcinoma |
| `1` | `BKL` | Benign Keratosis |
| `2` | `MEL` | Melanoma |
| `3` | `NV` | Melanocytic Nevi |

---

## 2. Technologies Used

- **Frontend / Web Framework:** Streamlit (custom dark SaaS interface & responsive CSS)
- **Deep Learning Runtime:** TensorFlow / Keras (`ConvNeXtTiny`, `EfficientNetV2B0`, `MobileNetV3Large` with embedded architecture preprocessing layers)
- **Image Processing & Numerics:** Pillow (`PIL`), NumPy

---

## 3. Required Dependencies

Listed in `requirements.txt`:

- `streamlit>=1.38.0`
- `tensorflow>=2.18.0`
- `numpy>=1.26.0`
- `pillow>=10.0.0`

Install locally:

```bash
pip install -r requirements.txt
```

*(If running locally on Python 3.14 where TensorFlow uses a pre-release build, run `pip install --pre -r requirements.txt`.)*

---

## 4. Deployment Structure

```text
Skin_Lesion_Classification/
├── .streamlit/
│   └── config.toml                 # Dark theme and upload settings
├── assets/
│   └── logo.png                    # SkinSight AI icon
├── Models/
│   ├── class_mapping.json          # Exact index-to-class mapping (0=BCC, 1=BKL, 2=MEL, 3=NV)
│   └── ensemble_config.json        # Ensemble weights, input sizes, checkpoint paths & fallback order
├── app.py                          # Streamlit web application entry point
├── main.py                         # Offline training & experiment pipeline (not called by app.py)
├── requirements.txt                # Runtime dependencies
├── .gitignore                      # Excludes datasets, caches, logs, and oversized weights
└── README.md
```

*(Local training folders such as `Dataset/`, `Results/`, and `Advanced_Experiments/` are preserved on disk for offline research and excluded from GitHub via `.gitignore`.)*

---

## 5. Model Files & Setup

### Ensemble Configuration (`Models/ensemble_config.json`)

The preferred inference engine is **`Ensemble_Top3_Weighted`**:

| Architecture | Ensemble Weight | Input Size | Checkpoint Filename | Size on Disk |
| :--- | :---: | :---: | :--- | :---: |
| `ConvNeXtTiny` | `0.45` | `384 × 384` | `L1_ConvNeXtTiny_384_ClassWeight.keras` | `414.39 MB` |
| `MobileNetV3Large` | `0.40` | `384 × 384` | `J_MobileNetV3Large_384.keras` | `61.29 MB` |
| `EfficientNetV2B0` | `0.15` | `384 × 384` | `best_model.keras` | `73.30 MB` |

### Why `.keras` Checkpoints Are Excluded from Standard Git Pushes
GitHub rejects files larger than **100 MB** (`L1_ConvNeXtTiny_384_ClassWeight.keras` is `414.39 MB`). To prevent broken pushes and keep the Git repository lightweight, `.gitignore` excludes `*.keras` checkpoints, `Dataset/`, `Results/`, and `Advanced_Experiments/`.

### Fresh-Environment Setup Options

#### Option A — Remote HTTPS Model Storage (Recommended for Streamlit Community Cloud)
1. Upload the three `.keras` checkpoints to a public or signed HTTPS model host (such as **Hugging Face Hub** or **GitHub Releases**).
2. Configure download URLs via **environment variables** or **Streamlit Secrets** (`.streamlit/secrets.toml`, never committed to Git):
   ```toml
   [model_urls]
   convnext_url = "https://<your-host>/L1_ConvNeXtTiny_384_ClassWeight.keras"
   effnet_url   = "https://<your-host>/best_model.keras"
   mobilenet_url = "https://<your-host>/J_MobileNetV3Large_384.keras"
   ```
   Equivalent environment variables:
   - `SKINSIGHT_CONVNEXT_URL`
   - `SKINSIGHT_EFFNET_URL`
   - `SKINSIGHT_MOBILENET_URL`
3. On startup, `app.py` checks local candidate paths (`Models/<arch>/...` and `Advanced_Experiments/Models/...`). Any missing checkpoint with a configured `https://` URL is downloaded securely once and cached.

#### Option B — Local Checkpoints
If running in the existing local project folder, `app.py` automatically detects the checkpoints already saved under `Models/best_model.keras` and `Advanced_Experiments/Models/...` with no file duplication required. Or place them under:
- `Models/ConvNeXtTiny/L1_ConvNeXtTiny_384_ClassWeight.keras`
- `Models/EfficientNetV2B0/best_model.keras` (or `Models/best_model.keras`)
- `Models/MobileNetV3Large/J_MobileNetV3Large_384.keras`

#### Option C — Git LFS (Self-Hosted Git Storage)
If your GitHub account has sufficient Git LFS quota, remove `*.keras` from `.gitignore`, initialize LFS before committing:
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
```

### Automatic Single-Model Fallback
If the full three-model ensemble cannot be loaded in a constrained or partial environment, `app.py` automatically falls back to the highest-priority individual checkpoint available (`ConvNeXtTiny` -> `EfficientNetV2B0` -> `MobileNetV3Large`) without crashing.

---

## 6. How to Run Locally

From the project directory (`Skin_Lesion_Classification/`):

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal (default: `http://localhost:8501`). Inference runs on CPU automatically when no NVIDIA GPU is available.

---

## 7. How to Upload to GitHub

1. Initialize Git inside `Skin_Lesion_Classification/` (if not already initialized):
   ```bash
   git init
   git branch -M main
   ```
2. Verify that large checkpoints and datasets are ignored:
   ```bash
   git status
   ```
   Confirm `Dataset/`, `Results/`, `Advanced_Experiments/`, `.streamlit/secrets.toml`, and `*.keras` do **not** appear in untracked files (unless intentionally tracked with Git LFS).
3. Commit and push to your GitHub repository:
   ```bash
   git add .
   git commit -m "Deploy SkinSight AI dark-theme Streamlit web application"
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

---

## 8. How to Deploy on Streamlit Community Cloud

1. Push the cleaned repository to GitHub (Section 7).
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.
3. Click **Create app** -> **Deploy a public app from GitHub** (or private repo).
4. Set:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Open **Advanced settings -> Secrets** and paste your `[model_urls]` TOML block (from Section 5, Option A) if loading checkpoints from Hugging Face Hub or GitHub Releases. Never embed private tokens directly inside `app.py` or `Models/ensemble_config.json`.
6. Click **Deploy** and monitor the build logs until Streamlit installs `requirements.txt` and starts `app.py`.
