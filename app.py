"""
Skin Lesion Classification - Research Inference Workbench
=========================================================
Clinical-grade research inference workstation for dermoscopic skin lesion classification
powered by an EfficientNetB0 convolutional neural network.

Trained and evaluated on the HAM10000 dataset with mild class-balanced sampling.
Designed as an academic machine learning portfolio artifact and clinical research workbench.
"""

import io
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# ==============================================================================
# Page Configuration & Application Metadata
# ==============================================================================
st.set_page_config(
    page_title="Skin Lesion Classification | Research Inference Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Class Definitions & Medical Knowledge Base (HAM10000)
# ==============================================================================
CLASS_NAMES: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_METADATA: Dict[str, Dict[str, str]] = {
    "akiec": {
        "full_name": "Actinic Keratosis / Intraepithelial Carcinoma",
        "category": "Pre-cancerous",
        "severity": "warning",
        "description": (
            "Sun-induced epidermal lesions with atypical keratinocytes (including Bowen's disease). "
            "Precursor lesions with potential progression to invasive squamous cell carcinoma."
        ),
        "clinical_urgency": "Requires dermatological evaluation & management (cryotherapy, topical 5-FU, or excision)",
        "badge_color": "#b45309",
        "badge_bg": "#fef3c7",
        "badge_border": "#fde68a",
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "category": "Malignant Skin Cancer",
        "severity": "danger",
        "description": (
            "The most common cutaneous malignancy arising from basal layer cells. "
            "Locally destructive with tissue invasion, but possesses very low metastatic potential."
        ),
        "clinical_urgency": "Prompt clinical / surgical assessment indicated (Mohs micrographic surgery or excision)",
        "badge_color": "#be123c",
        "badge_bg": "#fff1f2",
        "badge_border": "#fecdd3",
    },
    "bkl": {
        "full_name": "Benign Keratosis-like Lesions",
        "category": "Benign Epidermal Growth",
        "severity": "benign",
        "description": (
            "Encompasses seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses (LPLK). "
            "Non-cancerous epithelial proliferations frequently mimicking melanocytic tumors."
        ),
        "clinical_urgency": "Benign; routine clinical observation unless symptomatic or inflamed",
        "badge_color": "#047857",
        "badge_bg": "#ecfdf5",
        "badge_border": "#a7f3d0",
    },
    "df": {
        "full_name": "Dermatofibroma",
        "category": "Benign Dermal Lesion",
        "severity": "benign",
        "description": (
            "Benign fibrohistiocytic dermal nodules, often occurring on lower extremities. "
            "Typically exhibits a central white patch with peripheral delicate network (positive 'pinch sign')."
        ),
        "clinical_urgency": "Benign; reassurance provided; intervention rarely required",
        "badge_color": "#047857",
        "badge_bg": "#ecfdf5",
        "badge_border": "#a7f3d0",
    },
    "mel": {
        "full_name": "Melanoma",
        "category": "Malignant Skin Cancer",
        "severity": "danger",
        "description": (
            "High-grade cutaneous malignancy of melanocytic origin. "
            "Significant risk of systemic metastasis if not detected and excised in early intraepidermal phase."
        ),
        "clinical_urgency": "URGENT: Expedited dermatological excision biopsy with histopathologic staging required",
        "badge_color": "#be123c",
        "badge_bg": "#fff1f2",
        "badge_border": "#fecdd3",
    },
    "nv": {
        "full_name": "Melanocytic Nevi",
        "category": "Benign Melanocytic Proliferation",
        "severity": "benign",
        "description": (
            "Common benign neoplasms composed of melanocytes (moles). Represents >67% "
            "of public dermoscopy reference sets (HAM10000/ISIC)."
        ),
        "clinical_urgency": "Benign; baseline documentation and periodic ABCDE self-monitoring recommended",
        "badge_color": "#047857",
        "badge_bg": "#ecfdf5",
        "badge_border": "#a7f3d0",
    },
    "vasc": {
        "full_name": "Vascular Lesions",
        "category": "Benign Vascular Proliferation",
        "severity": "benign",
        "description": (
            "Benign cutaneous vascular proliferations including cherry angiomas, angiokeratomas, "
            "and pyogenic granulomas. Characterized dermoscopically by red-to-purple lacunae."
        ),
        "clinical_urgency": "Benign vascular malformation; therapeutic intervention needed only if bleeding",
        "badge_color": "#047857",
        "badge_bg": "#ecfdf5",
        "badge_border": "#a7f3d0",
    },
}

DEFAULT_MODEL_FILENAME = "best_efficientnetb0_mild_sampling.keras"
FALLBACK_MODEL_FILENAME = "best_efficientnetb0_stage3.keras"

# Published research benchmark metrics on HAM10000 validation split
BENCHMARK_METRICS = {
    "validation_accuracy": 0.7768,
    "macro_f1": 0.5431,
    "balanced_accuracy": 0.5078,
    "architecture": "EfficientNetB0 (Fine-Tuned)",
    "sampling_strategy": "Mild Class-Balanced Sampling",
    "dataset": "HAM10000 (ViDIR Group, MedUni Vienna)",
    "input_resolution": "224 × 224 × 3 RGB",
}


# ==============================================================================
# Clinical Workstation Theme & CSS Styling
# ==============================================================================
def apply_workstation_styles():
    """
    Injects custom CSS to transform Streamlit into a clinical research workstation.
    Employs a sophisticated light palette, precise alignment, restrained borders,
    and clear typographic hierarchy.
    """
    st.markdown(
        """
        <style>
        /* Base typography and layout reset */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #0f172a;
            -webkit-font-smoothing: antialiased;
        }

        /* Adjust main container padding */
        .block-container {
            padding-top: 1.25rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1320px !important;
        }

        /* Top Application Header */
        .workbench-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.1rem 1.4rem;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }

        .header-brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .header-mark {
            width: 44px;
            height: 44px;
            background: #f0fdfa;
            border: 1px solid #ccfbf1;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #0f766e;
            font-size: 1.35rem;
            flex-shrink: 0;
        }

        .header-title {
            font-size: 1.28rem;
            font-weight: 750;
            color: #0f172a;
            margin: 0;
            line-height: 1.2;
            letter-spacing: -0.015em;
        }

        .header-subtitle {
            font-size: 0.82rem;
            color: #64748b;
            margin: 3px 0 0 0;
            font-weight: 500;
        }

        .header-meta-group {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            flex-wrap: wrap;
        }

        .header-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            font-size: 0.74rem;
            font-weight: 600;
            color: #475569;
        }

        .header-pill-status {
            background: #ecfdf5;
            border-color: #a7f3d0;
            color: #047857;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #10b981;
        }

        /* Medical Safety Notice Banner */
        .clinical-disclaimer-card {
            background: #fffbeb;
            border: 1px solid #fef3c7;
            border-left: 4px solid #f59e0b;
            border-radius: 8px;
            padding: 0.85rem 1.15rem;
            margin-bottom: 1.25rem;
            font-size: 0.83rem;
            color: #78350f;
            line-height: 1.45;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
        }

        .disclaimer-head {
            font-weight: 700;
            color: #92400e;
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 3px;
        }

        /* Workflow Step Bar */
        .workflow-bar {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }

        .workflow-step {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.65rem 0.9rem;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 500;
        }

        .workflow-step.active {
            background: #f0fdfa;
            border-color: #0f766e;
            color: #0f766e;
            font-weight: 650;
        }

        .step-badge {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #f1f5f9;
            color: #475569;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.72rem;
            font-weight: 700;
            flex-shrink: 0;
        }

        .workflow-step.active .step-badge {
            background: #0f766e;
            color: #ffffff;
        }

        /* Panel Header Styling */
        .panel-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid #f1f5f9;
            margin-bottom: 1rem;
        }

        .panel-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .panel-badge {
            font-size: 0.72rem;
            padding: 2px 8px;
            border-radius: 4px;
            background: #f1f5f9;
            color: #475569;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* Dropzone enhancement */
        .upload-instruction-card {
            background: #f8fafc;
            border: 1px dashed #cbd5e1;
            border-radius: 8px;
            padding: 1.15rem;
            text-align: center;
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 0.5rem;
            line-height: 1.5;
        }

        .image-spec-bar {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            margin-top: 0.75rem;
            margin-bottom: 0.5rem;
        }

        .spec-item {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 0.45rem 0.6rem;
            text-align: center;
        }

        .spec-key {
            font-size: 0.68rem;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.03em;
        }

        .spec-val {
            font-size: 0.82rem;
            font-weight: 700;
            color: #0f172a;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            margin-top: 1px;
        }

        /* Top Diagnostic Prediction Banner */
        .diagnostic-card {
            border-radius: 8px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 1rem;
            border: 1px solid;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }

        .diagnostic-card.danger {
            background: #fff1f2;
            border-color: #fecdd3;
            color: #881337;
        }

        .diagnostic-card.warning {
            background: #fffbeb;
            border-color: #fde68a;
            color: #78350f;
        }

        .diagnostic-card.benign {
            background: #ecfdf5;
            border-color: #a7f3d0;
            color: #064e3b;
        }

        .diag-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }

        .diag-tag-group {
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .code-tag {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            background: #0f172a;
            color: #ffffff;
            text-transform: uppercase;
        }

        .category-tag {
            font-size: 0.76rem;
            font-weight: 600;
            padding: 3px 9px;
            border-radius: 4px;
            background: #ffffff;
            border: 1px solid currentColor;
            opacity: 0.9;
        }

        .rank-indicator {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            opacity: 0.85;
        }

        .score-headline {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            margin: 0.4rem 0 0.2rem 0;
        }

        .score-number {
            font-size: 2.3rem;
            font-weight: 850;
            line-height: 1;
            letter-spacing: -0.02em;
        }

        .score-label {
            font-size: 0.85rem;
            font-weight: 600;
            opacity: 0.85;
        }

        .diag-title {
            font-size: 1.25rem;
            font-weight: 750;
            margin: 0.35rem 0 0.4rem 0;
            line-height: 1.25;
        }

        .diag-description {
            font-size: 0.84rem;
            line-height: 1.45;
            opacity: 0.92;
            margin-bottom: 0.6rem;
        }

        .diag-urgency-box {
            display: flex;
            align-items: flex-start;
            gap: 0.45rem;
            font-size: 0.78rem;
            padding: 0.45rem 0.65rem;
            background: rgba(255, 255, 255, 0.65);
            border-radius: 6px;
            border: 1px solid rgba(0, 0, 0, 0.05);
            font-weight: 600;
        }

        /* Sidebar Styling */
        .sidebar-section-title {
            font-size: 0.76rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 1.1rem;
            margin-bottom: 0.5rem;
        }

        .metric-card-group {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            margin-bottom: 0.85rem;
        }

        .metric-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.6rem 0.75rem;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
        }

        .metric-name {
            font-size: 0.77rem;
            font-weight: 600;
            color: #475569;
        }

        .metric-number {
            font-size: 1.05rem;
            font-weight: 750;
            color: #0f172a;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }

        .sidebar-kv-table {
            width: 100%;
            font-size: 0.78rem;
            border-collapse: collapse;
            margin-bottom: 0.75rem;
        }

        .sidebar-kv-table td {
            padding: 5px 0;
            border-bottom: 1px solid #f1f5f9;
        }

        .sidebar-kv-table td.key {
            color: #64748b;
            font-weight: 500;
            width: 44%;
        }

        .sidebar-kv-table td.val {
            color: #0f172a;
            font-weight: 600;
            text-align: right;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }

        /* Technical Callout */
        .tech-callout {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 3px solid #64748b;
            border-radius: 6px;
            padding: 0.65rem 0.9rem;
            font-size: 0.78rem;
            color: #475569;
            line-height: 1.45;
            margin-top: 0.75rem;
        }

        /* Footer */
        .workbench-footer {
            margin-top: 2.5rem;
            padding-top: 1.25rem;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            font-size: 0.76rem;
            color: #64748b;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# Model Discovery, Loading & Resource Caching
# ==============================================================================
def get_model_candidates() -> List[str]:
    """
    Returns candidate paths to search for the model, prioritizing mild sampling.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("MODEL_PATH", "").strip(),
        os.path.join(current_dir, DEFAULT_MODEL_FILENAME),
        os.path.join(current_dir, "..", DEFAULT_MODEL_FILENAME),
        DEFAULT_MODEL_FILENAME,
        os.path.join(current_dir, FALLBACK_MODEL_FILENAME),
        os.path.join(current_dir, "..", "efficientnetb0_stage3", FALLBACK_MODEL_FILENAME),
        FALLBACK_MODEL_FILENAME,
    ]
    return [c for c in candidates if c]


def resolve_model_path() -> str:
    """
    Determines the active model filepath from candidates.
    """
    candidates = get_model_candidates()
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, DEFAULT_MODEL_FILENAME)


@st.cache_resource(show_spinner=False)
def load_classification_model(model_filepath: str):
    """
    Loads and caches the trained Keras model without compiling.
    Uses st.cache_resource to avoid redundant instantiations across reruns.
    """
    if not os.path.isfile(model_filepath):
        raise FileNotFoundError(f"Trained model file not found at: '{model_filepath}'")
    model = tf.keras.models.load_model(model_filepath, compile=False)
    return model


# ==============================================================================
# Image Preprocessing & Model Inference Pipeline
# ==============================================================================
def preprocess_image(pil_img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocesses a PIL Image strictly matching the training pipeline:
    1. Resolves EXIF orientation tags automatically.
    2. Converts image to 3-channel RGB.
    3. Resizes image to (224, 224) using bilinear interpolation.
    4. Converts image to float32 NumPy array with range [0.0, 255.0].
    5. Adds batch dimension -> shape (1, 224, 224, 3).

    NOTE: Pixel values are in the range [0.0, 255.0] as Keras EfficientNetB0
    incorporates internal input rescaling layers.
    """
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    resample_attr = getattr(Image, "Resampling", Image)
    bilinear_filter = getattr(resample_attr, "BILINEAR", Image.BILINEAR)
    resized_img = pil_img.resize(target_size, resample=bilinear_filter)

    img_array = np.array(resized_img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def run_inference(model, input_batch: np.ndarray) -> np.ndarray:
    """
    Executes model forward pass and validates probability vector output.
    Returns 1D array of 7 class softmax probabilities.
    """
    raw_output = model.predict(input_batch, verbose=0)

    if raw_output.ndim != 2 or raw_output.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Invalid model output shape {raw_output.shape}. "
            f"Expected shape (1, {len(CLASS_NAMES)})."
        )

    probabilities = raw_output[0]
    return probabilities


def generate_benchmark_pattern(lesion_type: str = "melanocytic") -> Image.Image:
    """
    Generates a deterministic synthetic dermoscopy benchmark pattern
    featuring characteristic peripheral vignetting and pigment network.
    Permits instant testing when an evaluator does not have a local image file.
    """
    np.random.seed(101 if lesion_type == "melanocytic" else 202)
    h, w = 360, 360
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Base background skin tone (erythematous / Caucasian dermoscopic skin)
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :] = [218.0, 182.0, 158.0]

    # Circular dermatoscope aperture vignette
    aperture_radius = w * 0.46
    vignette = np.clip(1.0 - (dist / aperture_radius) ** 4, 0.0, 1.0)

    # Central lesion pigment core
    core_radius = 80.0 if lesion_type == "melanocytic" else 70.0
    lesion_mask = np.clip(1.0 - (dist / core_radius) ** 2, 0.0, 1.0)

    # Pigment coloration
    pigment = np.array([85.0, 48.0, 36.0], dtype=np.float32)

    for c in range(3):
        base[:, :, c] = base[:, :, c] * (1.0 - lesion_mask * 0.72) + pigment[c] * (lesion_mask * 0.72)
        base[:, :, c] *= vignette

    img_arr = np.clip(base, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(img_arr)


# ==============================================================================
# Visualization: Clinical-Grade Horizontal Bar Chart
# ==============================================================================
def create_probability_chart(sorted_results: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders an elegant horizontal bar chart ranking all 7 classes.
    Uses clinical palette styling (teal/medical blue for top, slate for others).
    """
    display_results = list(reversed(sorted_results))

    labels = [
        f"<b>{item['full_name']}</b> <span style='color:#64748b;'>({item['code'].upper()})</span>"
        for item in display_results
    ]
    scores = [item["percentage"] for item in display_results]

    hover_texts = [
        f"<b>{item['full_name']}</b><br>"
        f"Class Token: <code>{item['code']}</code><br>"
        f"Pathology: {item['category']}<br>"
        f"Softmax Output: <b>{item['percentage']:.2f}%</b>"
        for item in display_results
    ]

    # Clinical color mapping: Top ranked class highlighted in clinical teal/blue, others in muted slate
    bar_colors = [
        "#0f766e" if item["rank"] == 1 else "#cbd5e1"
        for item in display_results
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            text=[f" {s:.1f}%" for s in scores],
            textposition="outside",
            textfont=dict(size=11, color="#0f172a", family="ui-monospace, monospace"),
            marker=dict(
                color=bar_colors,
                line=dict(color="#94a3b8", width=0.5),
            ),
            hoverinfo="text",
            hovertext=hover_texts,
            cliponaxis=False,
        )
    )

    max_score = max(scores) if scores else 100.0
    x_limit = min(100.0, max(max_score * 1.18, 20.0))

    fig.update_layout(
        margin=dict(l=10, r=45, t=10, b=25),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Model Softmax Output Score (%)", font=dict(size=11, color="#64748b")),
            range=[0, x_limit],
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            ticksuffix="%",
            tickfont=dict(size=10, color="#64748b"),
        ),
        yaxis=dict(
            showgrid=False,
            autorange=True,
            tickfont=dict(size=11, color="#0f172a"),
        ),
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto", size=11),
    )

    return fig


# ==============================================================================
# Main Application Entry Point
# ==============================================================================
def main():
    apply_workstation_styles()

    resolved_path = resolve_model_path()
    model_basename = os.path.basename(resolved_path)
    model_exists = os.path.isfile(resolved_path)

    # --------------------------------------------------------------------------
    # 1. Compact Clinical Application Header
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="workbench-header">
            <div class="header-brand">
                <div class="header-mark">🔬</div>
                <div class="header-title-wrap">
                    <h1 class="header-title">Skin Lesion Classification</h1>
                    <p class="header-subtitle">Research Inference Workbench · HAM10000 Benchmark Evaluation</p>
                </div>
            </div>
            <div class="header-meta-group">
                <span class="header-pill">Backbone: EfficientNetB0</span>
                <span class="header-pill">Strategy: Mild Sampling</span>
                <span class="header-pill {'header-pill-status' if model_exists else ''}">
                    <span class="status-dot"></span>
                    {'System Ready' if model_exists else 'Model Missing'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 2. Compact Medical Safety Notice
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="clinical-disclaimer-card">
            <div class="disclaimer-head">
                <span>⚠️</span> Medical Research Demonstration — Strictly Not for Clinical Diagnostic Use
            </div>
            <div>
                This system is an academic machine learning prototype trained on the HAM10000 reference archive.
                Predictions represent uncalibrated algorithmic softmax outputs and must <strong>never</strong> be used for medical diagnosis,
                treatment, or clinical decision-making. Standard dermoscopic images are required; consumer camera photos will yield unreliable results.
                If you have concerns about a changing skin spot, consult a board-certified dermatologist.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 3. Sidebar: Research Architecture, Metrics, and Target Classes
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; padding-bottom:8px; border-bottom:1px solid #e2e8f0; margin-bottom:12px;">
                <span style="font-size:1.15rem;">🔬</span>
                <div>
                    <div style="font-weight:700; font-size:0.92rem; color:#0f172a;">Research Specification</div>
                    <div style="font-size:0.73rem; color:#64748b;">HAM10000 Multiclass Benchmark</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">Validation Performance</div>', unsafe_allow_html=True)
        st.caption("Benchmark results on the HAM10000 validation split:")

        st.markdown(
            f"""
            <div class="metric-card-group">
                <div class="metric-row">
                    <span class="metric-name" title="Overall multi-class accuracy across all validation samples">Validation Accuracy</span>
                    <span class="metric-number">{BENCHMARK_METRICS['validation_accuracy'] * 100:.2f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name" title="Unweighted mean of F1 scores across all 7 lesion classes">Macro F1 Score</span>
                    <span class="metric-number">{BENCHMARK_METRICS['macro_f1']:.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name" title="Mean recall across all 7 classes, correcting for extreme class imbalance">Balanced Accuracy</span>
                    <span class="metric-number">{BENCHMARK_METRICS['balanced_accuracy'] * 100:.2f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="font-size:0.73rem; color:#64748b; line-height:1.4; margin-bottom:12px;">
                <em>Note:</em> In skin lesion datasets where benign nevi comprise ~67% of cases,
                <strong>Balanced Accuracy</strong> and <strong>Macro F1</strong> provide true diagnostic sensitivity across minority classes (Melanoma & BCC).
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">Model Specifications</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <table class="sidebar-kv-table">
                <tr><td class="key">Architecture</td><td class="val">EfficientNetB0</td></tr>
                <tr><td class="key">Input Tensor</td><td class="val">224 × 224 × 3</td></tr>
                <tr><td class="key">Color Space</td><td class="val">RGB [0, 255]</td></tr>
                <tr><td class="key">Output Units</td><td class="val">7 Softmax</td></tr>
                <tr><td class="key">Sampling</td><td class="val">Mild Balance</td></tr>
                <tr><td class="key">Backbone</td><td class="val">ImageNet Pretrained</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-title">Target Class Reference (7)</div>', unsafe_allow_html=True)
        with st.expander("Explore 7 Diagnostic Classes", expanded=False):
            for code in CLASS_NAMES:
                meta = CLASS_METADATA[code]
                st.markdown(
                    f"""
                    <div style="padding:6px 0; border-bottom:1px solid #f1f5f9; font-size:0.77rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                            <strong style="color:#0f172a;">{meta['full_name']}</strong>
                            <code style="font-size:0.7rem; font-weight:700;">{code.upper()}</code>
                        </div>
                        <span style="font-size:0.7rem; color:{meta['badge_color']}; font-weight:600;">{meta['category']}</span>
                        <div style="color:#475569; font-size:0.72rem; margin-top:2px;">{meta['description']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="sidebar-section-title">Runtime Environment</div>', unsafe_allow_html=True)
        if model_exists:
            st.markdown(
                f"""
                <div style="font-size:0.74rem; background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:8px;">
                    <div style="color:#047857; font-weight:600; margin-bottom:2px;">● Model Weights Loaded</div>
                    <code style="font-size:0.7rem; word-break:break-all;">{model_basename}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="font-size:0.74rem; background:#fff1f2; border:1px solid #fecdd3; border-radius:6px; padding:8px;">
                    <div style="color:#be123c; font-weight:600; margin-bottom:2px;">✕ Weights Not Found</div>
                    <code style="font-size:0.7rem; word-break:break-all;">{DEFAULT_MODEL_FILENAME}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------------------------
    # 4. Model Loading with Error Guard
    # --------------------------------------------------------------------------
    if not model_exists:
        st.error(
            f"### ⚠️ Model Artifact Unavailable\n\n"
            f"The application could not locate the primary model file: **`{DEFAULT_MODEL_FILENAME}`**.\n\n"
            f"**Expected Location:** `{resolved_path}`\n\n"
            f"Please verify that the `.keras` model file exists or configure the `MODEL_PATH` environment variable."
        )
        st.stop()

    try:
        model = load_classification_model(resolved_path)
    except Exception as exc:
        st.error(
            f"### ⚠️ Model Initialization Failure\n\n"
            f"An error occurred while deserializing the Keras model.\n\n"
            f"**Details:** `{str(exc)}`"
        )
        st.stop()

    # --------------------------------------------------------------------------
    # 5. Session State Management
    # --------------------------------------------------------------------------
    if "current_file_id" not in st.session_state:
        st.session_state.current_file_id = None
    if "active_pil_image" not in st.session_state:
        st.session_state.active_pil_image = None
    if "active_image_name" not in st.session_state:
        st.session_state.active_image_name = None
    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None

    # --------------------------------------------------------------------------
    # 6. Main Workspace Layout & Uploader
    # --------------------------------------------------------------------------
    col_uploader, col_helper = st.columns([1.6, 1], gap="medium")

    with col_uploader:
        uploaded_file = st.file_uploader(
            "Upload Dermoscopic Lesion Image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: Standard JPG, JPEG, PNG dermoscopy images. Input is converted to RGB and resized to 224x224.",
            label_visibility="visible",
        )

    with col_helper:
        st.markdown(
            """
            <div class="upload-instruction-card">
                <strong>Standardized Dermoscopic Input Expected</strong><br>
                For valid research inference, input must be a magnified dermoscopy image (contact or cross-polarized).
                Macro photographs or smartphone camera images are out-of-distribution.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Handle file upload or sample selection
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.current_file_id != file_id:
            st.session_state.current_file_id = file_id
            st.session_state.prediction_results = None
            try:
                pil_temp = Image.open(uploaded_file)
                pil_temp.load()
                st.session_state.active_pil_image = pil_temp
                st.session_state.active_image_name = uploaded_file.name
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("❌ Invalid image format. Please upload an uncorrupted JPG or PNG file.")
                st.session_state.active_pil_image = None
    elif st.session_state.active_pil_image is None:
        # Prompt evaluator with sample test option
        sub_col1, sub_col2 = st.columns([1, 1])
        with sub_col1:
            if st.button("🧪 Load Benchmark Test Sample (Melanocytic)", use_container_width=True):
                sample_img = generate_benchmark_pattern("melanocytic")
                st.session_state.current_file_id = "synthetic_melanocytic_sample"
                st.session_state.active_pil_image = sample_img
                st.session_state.active_image_name = "benchmark_melanocytic_sample.png"
                st.session_state.prediction_results = None
                st.rerun()

    # Determine workflow progress steps
    has_active_image = st.session_state.active_pil_image is not None
    has_active_preds = st.session_state.prediction_results is not None

    step1_class = "active" if has_active_image else ""
    step2_class = "active" if has_active_image else ""
    step3_class = "active" if has_active_preds else ""

    # Workflow stepper banner
    st.markdown(
        f"""
        <div class="workflow-bar">
            <div class="workflow-step {'active' if not has_active_image or step1_class else ''}">
                <div class="step-badge">1</div>
                <div>Dermoscopic Image Input</div>
            </div>
            <div class="workflow-step {step2_class}">
                <div class="step-badge">2</div>
                <div>Preprocessing & Inference</div>
            </div>
            <div class="workflow-step {step3_class}">
                <div class="step-badge">3</div>
                <div>Pathology Score Distribution</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 7. Core Workstation Workspace (Inspection & Inference)
    # --------------------------------------------------------------------------
    if not has_active_image:
        st.markdown(
            """
            <div style="background:#ffffff; border:1px dashed #cbd5e1; border-radius:10px; padding:3.5rem 1.5rem; text-align:center; color:#64748b; margin-top:0.5rem;">
                <div style="font-size:2.4rem; color:#0f766e; margin-bottom:0.75rem;">🔬</div>
                <div style="font-weight:700; font-size:1.05rem; color:#0f172a; margin-bottom:0.35rem;">Awaiting Dermoscopic Image</div>
                <div style="font-size:0.85rem; max-width:540px; margin:0 auto; line-height:1.5;">
                    Please select or drop a dermoscopic image above (<code>.jpg</code>, <code>.png</code>) or click <strong>'Load Benchmark Test Sample'</strong> to evaluate the pipeline.
                    The system will frame the lesion, apply bilinear 224×224 RGB scaling, and evaluate multiclass probabilities across all 7 HAM10000 categories.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        pil_image = st.session_state.active_pil_image
        image_name = st.session_state.active_image_name or "dermoscopic_input.jpg"

        # 2-Column Workstation Layout
        col_inspect, col_results = st.columns([1, 1.25], gap="large")

        # Left Column: Image Inspection & Execution
        with col_inspect:
            st.markdown(
                """
                <div class="panel-header-row">
                    <h3 class="panel-title"><span>🔍</span> Dermoscopic Viewport</h3>
                    <span class="panel-badge">Inspection Panel</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Framed image preview preserving aspect ratio
            st.image(
                pil_image,
                use_container_width=True,
            )

            # Metadata specification bar
            st.markdown(
                f"""
                <div class="image-spec-bar">
                    <div class="spec-item">
                        <div class="spec-key">File Name</div>
                        <div class="spec-val" style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{image_name}">{image_name}</div>
                    </div>
                    <div class="spec-item">
                        <div class="spec-key">Resolution</div>
                        <div class="spec-val">{pil_image.width} × {pil_image.height} px</div>
                    </div>
                    <div class="spec-item">
                        <div class="spec-key">Channels</div>
                        <div class="spec-val">{pil_image.mode} → RGB</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            btn_col1, btn_col2 = st.columns([1.8, 1])
            with btn_col1:
                run_btn = st.button(
                    "▶ Run Model Inference",
                    type="primary",
                    use_container_width=True,
                )
            with btn_col2:
                if st.button("↺ Reset View", use_container_width=True):
                    st.session_state.current_file_id = None
                    st.session_state.active_pil_image = None
                    st.session_state.active_image_name = None
                    st.session_state.prediction_results = None
                    st.rerun()

        # Right Column: Inference Execution & Pathology Analysis
        with col_results:
            st.markdown(
                """
                <div class="panel-header-row">
                    <h3 class="panel-title"><span>📊</span> Clinical Inference & Pathology Analysis</h3>
                    <span class="panel-badge">7-Class Softmax</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if run_btn:
                with st.spinner("Executing forward pass through EfficientNetB0..."):
                    try:
                        processed_batch = preprocess_image(pil_image, target_size=(224, 224))
                        probabilities = run_inference(model, processed_batch)

                        results = []
                        for idx, code in enumerate(CLASS_NAMES):
                            prob = float(probabilities[idx])
                            meta = CLASS_METADATA[code]
                            results.append(
                                {
                                    "code": code,
                                    "full_name": meta["full_name"],
                                    "category": meta["category"],
                                    "severity": meta["severity"],
                                    "score": prob,
                                    "percentage": prob * 100.0,
                                    "clinical_urgency": meta["clinical_urgency"],
                                    "description": meta["description"],
                                }
                            )

                        results.sort(key=lambda x: x["score"], reverse=True)
                        for rank_idx, item in enumerate(results, start=1):
                            item["rank"] = rank_idx

                        st.session_state.prediction_results = results
                    except Exception as err:
                        st.error(
                            f"❌ **Inference Failure**: An error occurred during the forward pass.\n\n"
                            f"Details: `{str(err)}`"
                        )
                        return

            # Display results if available
            if st.session_state.prediction_results is None:
                st.markdown(
                    """
                    <div style="background:#f8fafc; border:1px dashed #cbd5e1; border-radius:8px; padding:3.2rem 1.5rem; text-align:center; color:#64748b;">
                        <div style="font-size:1.8rem; margin-bottom:0.5rem;">⚡</div>
                        <div style="font-weight:700; color:#0f172a; margin-bottom:0.3rem;">Inference Pipeline Ready</div>
                        <div style="font-size:0.82rem; line-height:1.5;">
                            Click <strong>'▶ Run Model Inference'</strong> to compute class probabilities across the 7 diagnostic categories.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                results = st.session_state.prediction_results
                top_class = results[0]

                # Top Diagnostic Prediction Card
                severity_class = top_class["severity"]  # danger, warning, benign
                st.markdown(
                    f"""
                    <div class="diagnostic-card {severity_class}">
                        <div class="diag-header-row">
                            <div class="diag-tag-group">
                                <span class="code-tag">{top_class['code']}</span>
                                <span class="category-tag">{top_class['category']}</span>
                            </div>
                            <span class="rank-indicator">Highest Predicted Probability</span>
                        </div>
                        <div class="score-headline">
                            <span class="score-number">{top_class['percentage']:.2f}%</span>
                            <span class="score-label">Softmax Probability Score</span>
                        </div>
                        <div class="diag-title">{top_class['full_name']}</div>
                        <div class="diag-description">{top_class['description']}</div>
                        <div class="diag-urgency-box">
                            <span>📋</span>
                            <span><strong>Clinical Protocol Reference:</strong> {top_class['clinical_urgency']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Distribution Chart
                st.markdown(
                    """
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <strong style="font-size:0.85rem; color:#0f172a;">Pathology Probability Distribution (All 7 Classes)</strong>
                        <span style="font-size:0.73rem; color:#64748b;">Desc. Order</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                chart = create_probability_chart(results)
                st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

                # Technical interpretation note
                st.markdown(
                    """
                    <div class="tech-callout">
                        <strong>Probabilistic Interpretation:</strong> Softmax scores represent relative feature alignment within the closed 7-class training distribution. 
                        They are <strong>not calibrated Bayesian posterior probabilities</strong> of disease. A non-zero probability for high-risk classes (e.g. Melanoma) warrants clinical attention regardless of top rank.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------------------------
        # 8. Full Ranked Pathology Data Grid (All 7 Classes)
        # ----------------------------------------------------------------------
        if st.session_state.prediction_results is not None:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="panel-header-row" style="margin-top:1rem;">
                    <h3 class="panel-title"><span>📋</span> Complete Diagnostic Class Breakdown</h3>
                    <span class="panel-badge">7-Class Full Evaluation</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Build data grid
            table_data = []
            for item in st.session_state.prediction_results:
                table_data.append(
                    {
                        "Rank": f"#{item['rank']}",
                        "Code": item["code"].upper(),
                        "Diagnostic Entity": item["full_name"],
                        "Pathology Type": item["category"],
                        "Model Score": round(item["percentage"], 2),
                        "Clinical Protocol Context": item["clinical_urgency"],
                    }
                )

            st.dataframe(
                table_data,
                column_config={
                    "Rank": st.column_config.TextColumn("Rank", width="small"),
                    "Code": st.column_config.TextColumn("Code", width="small"),
                    "Diagnostic Entity": st.column_config.TextColumn("Diagnostic Entity", width="medium"),
                    "Pathology Type": st.column_config.TextColumn("Pathology Type", width="medium"),
                    "Model Score": st.column_config.ProgressColumn(
                        "Model Score (%)",
                        help="Relative softmax probability output",
                        format="%.2f%%",
                        min_value=0.0,
                        max_value=100.0,
                        width="medium",
                    ),
                    "Clinical Protocol Context": st.column_config.TextColumn("Clinical Reference Protocol", width="large"),
                },
                hide_index=True,
                use_container_width=True,
            )

    # --------------------------------------------------------------------------
    # 9. Technical Specifications & Methodology Drawer
    # --------------------------------------------------------------------------
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    with st.expander("Technical Architecture, Sampling Rationale & Evaluation Methodology", expanded=False):
        t_col1, t_col2 = st.columns(2, gap="medium")
        with t_col1:
            st.markdown(
                """
                **1. Class Imbalance & Mild Sampling Strategy:**  
                The primary challenge in dermoscopic classification is extreme distribution skew (in HAM10000, Melanocytic Nevi represent ~67% of all cases).
                Standard training yields deceptive overall accuracy while severely failing on rare, lethal malignancies (`mel`, `bcc`).
                The active model employs **mild class-balanced sampling** during fine-tuning, increasing the sampling frequency of underrepresented classes
                to achieve a balanced accuracy of **50.78%** and macro F1 of **0.5431** without destabilizing feature representations.

                **2. Preprocessing & Input Rescaling Pipeline:**  
                Images undergo EXIF orientation correction, RGB conversion, and bilinear resizing to **224 × 224 × 3** dimensions.
                Pixel values are retained in the `[0.0, 255.0]` range because the EfficientNetB0 backbone internally incorporates input normalization and rescaling layers.
                """
            )
        with t_col2:
            st.markdown(
                """
                **3. Closed-Set Softmax vs. Calibrated Clinical Risk:**  
                The network's final layer applies a softmax activation over a closed set of 7 dermatological entities.
                Outputs inherently sum to 1.0 (100%), which means that even out-of-distribution inputs (e.g. hair, pen markings, photographic artifacts)
                will be forced into one of the 7 classes with high artificial confidence. Softmax output should be treated as feature correlation, not Bayesian diagnosis.

                **4. Out-of-Distribution Sensitivity & Clinical Preconditions:**  
                Model validity relies strictly on standardized dermoscopic imaging protocols (dermatoscope with immersion fluid or cross-polarized illumination).
                Unstandardized consumer smartphone photography, poor focal depth, shadows, and non-cutaneous lesions violate distribution bounds.
                """
            )

    # --------------------------------------------------------------------------
    # 10. Research Footer
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="workbench-footer">
            <strong>Skin Lesion Classification Research Platform</strong> · Academic Portfolio Artifact<br>
            Backbone: EfficientNetB0 (Fine-Tuned) · Weights: <code>{model_basename}</code> · Evaluation Split: HAM10000<br>
            <em>Educational and research demonstration only. Not cleared or intended for clinical diagnostics or patient care.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
