# 🔬 Skin Lesion Classification — Streamlit Web Application

A clean, professional, portfolio-grade inference web application for dermoscopic skin lesion classification using a fine-tuned **EfficientNetB0 (Mild Sampling)** deep learning model.

This application is designed **strictly for inference deployment**: it does **not** require the HAM10000 training dataset, metadata CSV files, training notebooks, or model training pipelines.

---

## 📌 Project Overview

Skin cancer is among the most diagnosed malignancies worldwide. Early and accurate dermatological evaluation dramatically improves patient outcomes. This project deploys a fine-tuned convolutional neural network (EfficientNetB0) trained on the academic HAM10000 benchmark dataset to classify dermoscopic images into seven diagnostic categories.

> [!CAUTION]
> **Medical & Educational Safety Disclaimer**  
> This web application is an **academic research demonstration and machine learning engineering portfolio project**. It is **NOT** a certified medical diagnostic tool or medical device. Its predictions must **never** be used for self-diagnosis, clinical decision-making, or treatment planning. Anyone with concerns regarding a skin lesion or changing mole must seek advice from a board-certified dermatologist or licensed physician.

---

## 🏗️ Model Architecture & Benchmark Metrics

The classifier leverages Google's **EfficientNetB0** architecture, fine-tuned with mild class-balanced sampling on dermoscopic skin lesion imagery.

- **Model Backbone:** EfficientNetB0 (Mild Sampling Balanced Fine-Tuning)
- **Input Resolution:** `(224, 224, 3)` RGB
- **Input Pixel Range:** `[0.0, 255.0]` (`float32`) with internal EfficientNet rescaling
- **Output:** 7 Softmax probabilities
- **Primary Model File:** `best_efficientnetb0_mild_sampling.keras` (~35.5 MB)

### Research Validation Performance

Evaluated on the independent validation split of the HAM10000 benchmark:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Validation Accuracy** | **77.68%** | Overall correct class assignment |
| **Macro F1-Score** | **0.5431** | Unweighted mean F1 across all 7 classes |
| **Balanced Accuracy** | **50.78%** | Average recall per class accounting for class imbalance |
| **Sampling Strategy** | **Mild Sampling** | Balanced class oversampling to improve sensitivity on rare/malignant lesions |

*Note: These are research validation metrics, not clinical trials or evidence of clinical diagnostic efficacy.*

---

## 🏷️ Target Classes

The model classifies images into seven specific lesion types (exact output index order):

| Index | Code | Full Diagnostic Name | Pathology Type | Clinical Significance |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `akiec` | Actinic Keratoses & Intraepithelial Carcinoma | Pre-cancerous / In-situ | Sun-induced epidermal lesions; may progress to squamous cell carcinoma |
| **1** | `bcc` | Basal Cell Carcinoma | Malignant Cancer | Most common skin cancer; locally destructive; low metastasis rate |
| **2** | `bkl` | Benign Keratosis-like Lesions | Benign Growth | Seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses |
| **3** | `df` | Dermatofibroma | Benign Lesion | Benign cutaneous fibrous nodules, common on lower extremities |
| **4** | `mel` | Melanoma | Malignant Cancer | Highly aggressive melanocytic malignancy; early detection is critical |
| **5** | `nv` | Melanocytic Nevi | Benign Mole | Common pigment-producing cell proliferations (~67% of dataset) |
| **6** | `vasc` | Vascular Lesions | Benign Growth | Cherry angiomas, angiokeratomas, pyogenic granulomas |

---

## 📁 Repository Structure

```text
skin-lesion-streamlit/
├── app.py                                   # Streamlit web application & inference pipeline
├── requirements.txt                         # Compatible Python dependencies
├── README.md                                # Detailed project documentation & guides
├── .gitignore                               # Git ignore rules and deployment notes
└── best_efficientnetb0_mild_sampling.keras  # Trained model weights (~35.5 MB)
```

---

## 🚀 Local Installation & Quickstart

### Prerequisites

- **Python 3.10, 3.11, or 3.12** (Recommended for standard TensorFlow wheels) or **Python 3.13** (TensorFlow 2.18+)
- Standard CPU (GPU is **not** required for inference)
- `pip` package manager

### Step 1: Clone or Navigate to the Project

```bash
cd skin-lesion-streamlit
```

### Step 2: Create a Virtual Environment

#### Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*(If you have multiple Python versions installed, e.g. Python 3.11: `py -3.11 -m venv venv`)*

#### macOS / Linux (Bash):
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Ensure the Model File is Present

Verify that the trained model file `best_efficientnetb0_mild_sampling.keras` is placed in the project folder alongside `app.py`:

```bash
# Verify model file exists
ls -lh best_efficientnetb0_mild_sampling.keras   # Linux/macOS
Get-Item best_efficientnetb0_mild_sampling.keras # Windows PowerShell
```

### Step 5: Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## ⚙️ Configuring a Custom Model Path

By default, `app.py` searches for `best_efficientnetb0_mild_sampling.keras` in the current working directory. You can configure a custom location using the `MODEL_PATH` environment variable:

#### Windows (PowerShell):
```powershell
$env:MODEL_PATH = "C:\models\best_efficientnetb0_mild_sampling.keras"
streamlit run app.py
```

#### Windows (CMD):
```cmd
set MODEL_PATH=C:\models\best_efficientnetb0_mild_sampling.keras
streamlit run app.py
```

#### macOS / Linux:
```bash
export MODEL_PATH="/path/to/best_efficientnetb0_mild_sampling.keras"
streamlit run app.py
```

---

## 🌐 Deployment Options

### 1. Streamlit Community Cloud
1. Push your code to a GitHub repository.
2. Because the model file is ~35.5 MB, you can push it directly to GitHub (GitHub's standard file limit is 100 MB), or track it via [Git LFS](https://git-lfs.com/):
   ```bash
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes best_efficientnetb0_mild_sampling.keras
   git commit -m "Add model weights with Git LFS"
   ```
3. Connect your repository on [share.streamlit.io](https://share.streamlit.io/) and select `app.py` as the entrypoint.

### 2. Hugging Face Spaces (Streamlit SDK)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) with SDK set to **Streamlit**.
2. Clone the Space repository and add `app.py`, `requirements.txt`, and `best_efficientnetb0_mild_sampling.keras`.
3. Commit and push:
   ```bash
   git add .
   git commit -m "Deploy Skin Lesion Classification app"
   git push
   ```

### 3. Docker Container
You can containerize the application with the following `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t skin-lesion-app .
docker run -p 8501:8501 skin-lesion-app
```

---

## ⚠️ Model Limitations & Research Notes

1. **Class Imbalance & Mild Sampling:**  
   The underlying HAM10000 training distribution consists predominantly of benign melanocytic nevi (`nv` > 67%). The mild sampling model introduces re-weighting/re-sampling to assist the classifier in discerning rarer categories (`mel`, `bcc`, `akiec`).

2. **Dermoscopy vs. Consumer Cameras:**  
   The model was trained exclusively on polarized and non-polarized dermatoscope images under standardized optical magnification and fluid immersion. Photographs taken with standard smartphone cameras under ambient lighting are out-of-distribution and will yield invalid predictions.

3. **Softmax Output vs. Bayesian Probability:**  
   The outputs are normalized softmax activations summing to 1.0. In deep neural networks, softmax scores are frequently uncalibrated and reflect relative feature activations rather than calibrated true clinical odds.

4. **Image Artifacts:**  
   Artifacts such as body hair, skin markings, skin pens, immersion gel bubbles, and dark borders can influence convolutional feature activations.

---

## 📜 License & Acknowledgments

- **Model Backbone:** EfficientNet (Tan & Le, Google Research)
- **Reference Dataset:** HAM10000 ("Human Against Machine with 10000 training images") by Tschandl et al.
- **Project Type:** Machine Learning Engineering & Deep Learning Research Portfolio Project.
