## ADNI-based Alzheimer’s Disease Decision Support Backend

This repository contains the backend for our senior design project: a **Decision Support System for Alzheimer’s disease** based on structural MRI. The service exposes a **FastAPI** endpoint that accepts 3D NIfTI brain scans, performs preprocessing (orientation and registration to MNI space), extracts a representative slice, and uses a **deep learning classifier (InceptionV3 in PyTorch)** to predict one of three labels:

- **AD** – Alzheimer’s Disease  
- **MCI** – Mild Cognitive Impairment  
- **CN** – Cognitively Normal  

> **Important**: This project is **for research and educational purposes only** and **not approved for clinical use**.

---

## Project Structure

- `code/api-script.py` – FastAPI application defining the MRI classification API (`/predict/`).
- `code/inference_only/` – Standalone inference scripts for different architectures (InceptionV3, ViT, ResNet18, EfficientNet-B3/B7, AlexNet, VGG16).
- `code/training_without_balancing/`, `code/training_with_undersampling/`, `code/training_with_augmentation_updated/` – Training scripts and notebooks exploring various CNN/ViT models and data augmentation strategies.
- `code/preprocessing_nifti/` – Scripts for NIfTI preprocessing (FSL orientation, FLIRT registration, etc.).
- `code/skull-stripping/` – Skull-stripping scripts (e.g. SynthStrip, BET).
- `code/augmentation_scripts/` – Data augmentation utilities for MRI slices.
- `code/nii_to_png.py`, `code/view_nii.py`, `code/select-image.py`, `code/check_middle_slice.py` – Utilities for converting and visualizing NIfTI/MRI data.
- `code/model_outputs.md` – Summary of training and inference performance across different architectures and augmentation settings.
- `requirements.txt` – Python backend dependencies.

The **production-facing backend** is centered on `code/api-script.py`; the rest of the code documents the model development and experimentation process.

---

## Backend Architecture

- **Framework**: FastAPI
- **Server**: Uvicorn
- **Deep Learning Library**: PyTorch + Torchvision
- **Primary Deployed Model**: InceptionV3
  - Pretrained on ImageNet, fine-tuned for 3-way AD/MCI/CN classification
  - Model weights expected at `../InceptionV3_model.pth` (relative to `code/api-script.py`)
- **Input format**: 3D NIfTI MRI volume (`.nii` or `.nii.gz`)
- **Preprocessing pipeline** (in `api-script.py`):
  1. Save uploaded NIfTI to `./uploaded_nifti/`.
  2. **Reorient to standard** using FSL `fslreorient2std` → `./reoriented/`.
  3. **Linear registration to MNI152 template** using FSL `flirt` → `./registered/`.
  4. Load registered NIfTI with NiBabel and extract the **middle axial slice**.
  5. Intensity clipping (1st–99th percentile) and rescaling to \([0, 255]\) with `skimage.exposure`.
  6. Convert to a grayscale `PIL.Image`, then to a 3‑channel tensor.
  7. Resize to \(299 \times 299\), normalize with ImageNet mean/std.
  8. Run a forward pass through InceptionV3 and apply softmax.

The `/predict/` endpoint returns the predicted class and associated probability.

---

## Requirements

### Python & Libraries

- **Python**: 3.8+ (tested with modern 3.x)
- **Python dependencies** (from `requirements.txt`):
  - `fastapi`
  - `uvicorn[standard]`
  - `torch`
  - `torchvision`
  - `Pillow`
  - `python-multipart`

Install them with:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

### System Dependencies (FSL)

The backend relies on **FSL** command-line tools for orientation and registration:

- `fslreorient2std`
- `flirt`
- MNI template: `MNI152_T1_1mm_brain`

In `code/api-script.py` the template is referenced as:

```python
MNI_TEMPLATE = "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain"
```

On your system, you may need to:

- Install FSL and ensure its binaries are in your `PATH`.
- Update the `MNI_TEMPLATE` path to match your local installation.

### Model Weights

Place the trained InceptionV3 weights file at:

- `InceptionV3_model.pth` in the **project root**, so that from `code/api-script.py` it is reachable as:

```python
model.load_state_dict(
    torch.load("../InceptionV3_model.pth", map_location=device, weights_only=False)
)
```

---

## Running the API Server

From the project root:

```bash
.venv\Scripts\activate  # if not already active
python -m uvicorn code.api_script:app --reload --host 0.0.0.0 --port 8000
```

---

## API Usage

### Endpoint

- **Method**: `POST`  
- **URL**: `/predict/`  
- **Content type**: `multipart/form-data`
- **Field**: `file` – uploaded NIfTI file (`.nii` or `.nii.gz`)

The backend checks the `content_type` of the uploaded file and expects one of:

- `application/nii`
- `application/nii.gz`
- `application/octet-stream`

### Example Request (Python)

```python
import requests

url = "http://localhost:8000/predict/"
files = {"file": open("example_subject.nii.gz", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

### Example Response

```json
{
  "predicted_class": "AD",
  "probability": 0.82
}
```

### Error Responses

- `400 Bad Request` – unsupported file type or invalid NIfTI file.
- `500 Internal Server Error` – failures in FSL preprocessing (reorient/FLIRT) or inference.

---

## Model Development & Performance

We experimented with a range of image classification architectures on MRI slices derived from ADNI-like T1-weighted images, including:

- **AlexNet**
- **VGG16**
- **ResNet18 / ResNet50**
- **InceptionV3**
- **EfficientNet-B3 / B7**
- **Vision Transformer (ViT)**
- **MobileNet_V2**
- **DenseNet-121**

Key details and metrics are summarized in `code/model_outputs.md`, including:

- Accuracy for different architectures on held-out sets.
- 5‑fold cross‑validation results for various augmentation schemes (e.g., horizontal/vertical flips, different dataset versions).
- Attempts at mitigating overfitting (early stopping, learning rate schedulers, modified fully-connected heads).
- Impact of preprocessing (skull stripping with SynthStrip/BET, orientation and registration, intensity standardization).

Overall, InceptionV3 with appropriate augmentation and hyperparameters emerged as a strong candidate for deployment in the backend.

---

## Clinical Disclaimer & Limitations

- This system is **not a medical device** and has **not been validated for clinical decision-making**.
- Performance metrics in `code/model_outputs.md` are based on specific experimental splits and may not generalize to other cohorts or scanners.
- The model predicts **AD/MCI/CN status from a single MRI acquisition** and does not incorporate longitudinal, cognitive, genetic, or demographic information.
- Predictions should be interpreted as **supporting information only**, alongside expert clinical judgment and additional diagnostic tests.

---

## Future Work

Potential extensions for this backend include:

- Exposing **additional endpoints** (e.g., returning Grad-CAM heatmaps to aid interpretability).
- Supporting **subject-wise cross-validation and data leakage safeguards** more explicitly in the pipeline.
- Integrating with a **frontend dashboard** for clinicians to upload scans, visualize slices, and review predictions over time.
- Logging predictions and metadata to a database for retrospective analysis.

---

## Acknowledgments

This project was developed as part of a **senior design project** focused on supporting early detection and staging of Alzheimer’s disease using MRI data and deep learning. It builds upon:

- The **ADNI** initiative and similar neuroimaging datasets.
- Prior research on MRI-based classification of AD, MCI, and CN.
- The open-source ecosystems of **PyTorch**, **FastAPI**, **FSL**, and related tooling.