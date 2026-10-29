"""
Skin Lesion Classification - Research Inference Workbench
=========================================================
Minimal, accessible research dashboard for dermoscopic skin lesion classification
powered by an EfficientNetB0 convolutional neural network.

Trained and evaluated on the HAM10000 dataset with mild class-balanced sampling.
Designed with a clean, high-contrast light theme for clinical research evaluation.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Skin Lesion Classification | Research Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Class Definitions & Medical Reference Data (HAM10000)
# ==============================================================================
CLASS_NAMES: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_METADATA: Dict[str, Dict[str, str]] = {
    "akiec": {
        "full_name": "Actinic Keratosis / Intraepithelial Carcinoma",
        "category": "Pre-cancerous",
        "severity": "warning",
        "description": (
            "Sun-induced epidermal lesion with cellular atypia. "
            "Pre-malignant precursor that may progress to invasive squamous cell carcinoma."
        ),
        "clinical_urgency": "Requires dermatological evaluation (cryotherapy, topical 5-FU, or excision)",
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "category": "Malignant Skin Cancer",
        "severity": "danger",
        "description": (
            "The most common cutaneous malignancy arising from basal layer cells. "
            "Locally tissue-invasive and destructive; rarely metastasizes."
        ),
        "clinical_urgency": "Prompt clinical/surgical excision indicated (Mohs micrographic surgery or excision)",
    },
    "bkl": {
        "full_name": "Benign Keratosis-like Lesions",
        "category": "Benign Epidermal Growth",
        "severity": "benign",
        "description": (
            "Includes seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses (LPLK). "
            "Common non-cancerous epidermal growths frequently mimicking melanocytic lesions."
        ),
        "clinical_urgency": "Benign; periodic observation unless symptomatic or changing",
    },
    "df": {
        "full_name": "Dermatofibroma",
        "category": "Benign Dermal Lesion",
        "severity": "benign",
        "description": (
            "Benign fibrohistiocytic dermal nodules, typically on extremities. "
            "Often displays positive 'pinch sign' and central pale patch."
        ),
        "clinical_urgency": "Benign; reassurance provided; intervention rarely required",
    },
    "mel": {
        "full_name": "Melanoma",
        "category": "Malignant Skin Cancer",
        "severity": "danger",
        "description": (
            "The most lethal form of skin cancer, originating from melanocytes. "
            "High propensity for early systemic metastasis if not promptly excised."
        ),
        "clinical_urgency": "URGENT: Expedited dermatological excision biopsy with histopathologic staging required",
    },
    "nv": {
        "full_name": "Melanocytic Nevi",
        "category": "Benign Melanocytic Proliferation",
        "severity": "benign",
        "description": (
            "Common benign neoplasms composed of melanocytes (moles). Represents >67% "
            "of public reference datasets (HAM10000)."
        ),
        "clinical_urgency": "Common benign mole; periodic ABCDE self-monitoring recommended",
    },
    "vasc": {
        "full_name": "Vascular Lesions",
        "category": "Benign Vascular Proliferation",
        "severity": "benign",
        "description": (
            "Benign cutaneous blood vessel proliferations including cherry angiomas, angiokeratomas, "
            "and pyogenic granulomas. Characterized by vascular lacunae."
        ),
        "clinical_urgency": "Benign vascular malformation; intervention needed only if bleeding or irritated",
    },
}

DEFAULT_MODEL_FILENAME = "best_efficientnetb0_mild_sampling.keras"
FALLBACK_MODEL_FILENAME = "best_efficientnetb0_stage3.keras"

# Research evaluation results on HAM10000 validation split
BENCHMARK_METRICS = {
    "validation_accuracy": 0.7768,
    "macro_f1": 0.5431,
    "balanced_accuracy": 0.5078,
    "sampling_strategy": "Mild Class-Balanced Sampling",
}


# ==============================================================================
# Minimal Light Theme CSS
# ==============================================================================
def apply_minimal_light_theme():
    """
    Enforces a consistent, high-contrast light theme across Streamlit.
    Removes dark backgrounds, overrides inherited white text,
    and sets explicit accessible colors for all components.
    """
    st.markdown(
        """
        <style>
        /* Force light theme colors on app container */
        [data-testid="stAppViewContainer"], .main, .stApp {
            background-color: #F5F7FA !important;
            color: #17212F !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }

        /* Sidebar light theme */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        [data-testid="stSidebar"] * {
            color: #17212F !important;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown span,
        [data-testid="stSidebar"] .stMarkdown td {
            color: #526174 !important;
        }
        [data-testid="stSidebar"] .stMarkdown strong,
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] .stMarkdown h4 {
            color: #17212F !important;
        }

        /* Clean typography scale */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #17212F !important;
            font-weight: 700 !important;
            letter-spacing: -0.015em !important;
        }
        p, span, label, div {
            color: #17212F;
        }

        /* Streamlit main block container */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px !important;
        }

        /* Minimal Header */
        .workbench-header {
            margin-bottom: 1rem;
            padding-bottom: 0.85rem;
            border-bottom: 1px solid #E2E8F0;
        }
        .header-title {
            font-size: 1.55rem !important;
            font-weight: 750 !important;
            color: #17212F !important;
            margin: 0 !important;
            line-height: 1.25 !important;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .header-subtitle {
            font-size: 0.86rem !important;
            color: #526174 !important;
            margin-top: 3px !important;
            margin-bottom: 0 !important;
        }
        .header-badge {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            color: #0F766E;
            background-color: #F0FDFA;
            border: 1px solid #CCFBF1;
            padding: 2px 8px;
            border-radius: 4px;
            margin-left: 8px;
            vertical-align: middle;
        }

        /* Compact Medical Disclaimer Banner */
        .compact-disclaimer {
            background-color: #FFFBEB !important;
            border: 1px solid #FDE68A !important;
            border-left: 4px solid #D97706 !important;
            border-radius: 6px !important;
            padding: 0.7rem 0.95rem !important;
            margin-bottom: 1.25rem !important;
            font-size: 0.82rem !important;
            color: #92400E !important;
            line-height: 1.45 !important;
        }
        .compact-disclaimer strong {
            color: #78350F !important;
        }

        /* Section Title */
        .section-heading {
            font-size: 1.05rem;
            font-weight: 700;
            color: #17212F;
            margin-top: 0.5rem;
            margin-bottom: 0.35rem;
        }

        /* Uploader styling */
        [data-testid="stFileUploader"] {
            margin-bottom: 0.35rem;
        }
        [data-testid="stFileUploader"] section {
            background-color: #FFFFFF !important;
            border: 1px dashed #CBD5E1 !important;
            border-radius: 8px !important;
        }

        /* Image Metadata Pill */
        .image-meta-bar {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            margin-top: 0.65rem;
            margin-bottom: 0.85rem;
            font-size: 0.78rem;
            color: #526174;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Prediction Result Highlight Box */
        .prediction-box {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .pred-tag-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.4rem;
        }
        .pred-code-pill {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.76rem;
            font-weight: 700;
            padding: 2px 7px;
            background: #17212F;
            color: #FFFFFF !important;
            border-radius: 4px;
            text-transform: uppercase;
        }
        .pred-category-pill {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid;
        }
        .cat-danger {
            background-color: #FFF1F2;
            color: #BE123C !important;
            border-color: #FECDD3;
        }
        .cat-warning {
            background-color: #FFFBEB;
            color: #B45309 !important;
            border-color: #FDE68A;
        }
        .cat-benign {
            background-color: #ECFDF5;
            color: #047857 !important;
            border-color: #A7F3D0;
        }

        .pred-score-large {
            font-size: 2.15rem;
            font-weight: 800;
            color: #0F766E;
            line-height: 1.1;
            margin: 0.35rem 0 0.2rem 0;
            letter-spacing: -0.02em;
        }
        .pred-score-label {
            font-size: 0.82rem;
            color: #64748B;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        .pred-title {
            font-size: 1.2rem;
            font-weight: 750;
            color: #17212F;
            margin-bottom: 0.35rem;
        }
        .pred-desc {
            font-size: 0.83rem;
            color: #526174;
            line-height: 1.45;
            margin-bottom: 0.65rem;
        }
        .pred-protocol-box {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            font-size: 0.78rem;
            color: #334155;
            font-weight: 500;
        }

        /* Compact Metrics Row */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
            margin-top: 1rem;
            margin-bottom: 0.75rem;
        }
        .metric-cell {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 0.75rem 1rem;
        }
        .metric-cell-label {
            font-size: 0.72rem;
            color: #64748B;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.04em;
            margin-bottom: 2px;
        }
        .metric-cell-value {
            font-size: 1.25rem;
            font-weight: 750;
            color: #17212F;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }

        /* Sidebar Key-Value Table */
        .sb-table {
            width: 100%;
            font-size: 0.8rem;
            border-collapse: collapse;
            margin-bottom: 0.75rem;
        }
        .sb-table td {
            padding: 5px 0;
            border-bottom: 1px solid #F1F5F9;
        }
        .sb-key {
            color: #64748B !important;
            font-weight: 500;
        }
        .sb-val {
            color: #17212F !important;
            font-weight: 600;
            text-align: right;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }

        /* Technical Callout */
        .technical-callout {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 3px solid #64748B;
            border-radius: 6px;
            padding: 0.65rem 0.85rem;
            font-size: 0.78rem;
            color: #475569;
            line-height: 1.45;
            margin-top: 0.65rem;
        }

        /* Primary Button */
        .stButton button[kind="primary"] {
            background-color: #0F766E !important;
            border-color: #0F766E !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
        }
        .stButton button[kind="primary"]:hover {
            background-color: #115E59 !important;
            border-color: #115E59 !important;
        }

        /* Footer */
        .workbench-footer {
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid #E2E8F0;
            text-align: center;
            font-size: 0.76rem;
            color: #64748B;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# Model Loading & Caching
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
    Uses st.cache_resource to prevent re-instantiating across reruns.
    """
    if not os.path.isfile(model_filepath):
        raise FileNotFoundError(f"Trained model file not found at: '{model_filepath}'")
    model = tf.keras.models.load_model(model_filepath, compile=False)
    return model


# ==============================================================================
# Image Preprocessing & Inference
# ==============================================================================
def preprocess_image(pil_img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocesses a PIL Image strictly matching the training pipeline:
    1. Resolves EXIF orientation tags automatically.
    2. Converts image to 3-channel RGB.
    3. Resizes image to (224, 224) using bilinear interpolation.
    4. Converts image to float32 NumPy array with range [0.0, 255.0].
    5. Adds batch dimension -> shape (1, 224, 224, 3).
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
    Executes model forward pass and returns 1D array of 7 class softmax probabilities.
    """
    raw_output = model.predict(input_batch, verbose=0)
    if raw_output.ndim != 2 or raw_output.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Invalid model output shape {raw_output.shape}. Expected (1, {len(CLASS_NAMES)}).")
    return raw_output[0]


def generate_benchmark_sample() -> Image.Image:
    """
    Generates a deterministic synthetic dermoscopy reference pattern.
    Permits instant testing when an evaluator does not have a local image file.
    """
    np.random.seed(42)
    h, w = 360, 360
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Base skin tone
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :] = [218.0, 180.0, 155.0]

    # Dermatoscope aperture vignette
    vignette = np.clip(1.0 - (dist / (w * 0.46)) ** 4, 0.0, 1.0)

    # Pigmented lesion core
    lesion_mask = np.clip(1.0 - (dist / 75.0) ** 2, 0.0, 1.0)
    pigment = np.array([88.0, 48.0, 35.0], dtype=np.float32)

    for c in range(3):
        base[:, :, c] = base[:, :, c] * (1.0 - lesion_mask * 0.72) + pigment[c] * (lesion_mask * 0.72)
        base[:, :, c] *= vignette

    img_arr = np.clip(base, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(img_arr)


# ==============================================================================
# Visualization: Clean Horizontal Bar Chart
# ==============================================================================
def create_probability_chart(sorted_results: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders a compact, high-contrast horizontal bar chart ranking all 7 classes.
    """
    display_results = list(reversed(sorted_results))

    labels = [
        f"{item['full_name']} ({item['code'].upper()})"
        for item in display_results
    ]
    scores = [item["percentage"] for item in display_results]

    hover_texts = [
        f"<b>{item['full_name']}</b><br>"
        f"Class: <code>{item['code']}</code> ({item['category']})<br>"
        f"Softmax Probability: <b>{item['percentage']:.2f}%</b>"
        for item in display_results
    ]

    # #0F766E for top prediction, #CBD5E1 for others
    bar_colors = [
        "#0F766E" if item["rank"] == 1 else "#CBD5E1"
        for item in display_results
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            text=[f" {s:.1f}%" for s in scores],
            textposition="outside",
            textfont=dict(size=11, color="#17212F", family="monospace"),
            marker=dict(
                color=bar_colors,
                line=dict(color="#94A3B8", width=0.5),
            ),
            hoverinfo="text",
            hovertext=hover_texts,
            cliponaxis=False,
        )
    )

    max_score = max(scores) if scores else 100.0
    x_limit = min(100.0, max(max_score * 1.18, 20.0))

    fig.update_layout(
        margin=dict(l=10, r=40, t=10, b=25),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Model Softmax Probability (%)", font=dict(size=11, color="#526174")),
            range=[0, x_limit],
            showgrid=True,
            gridcolor="#E2E8F0",
            zeroline=False,
            ticksuffix="%",
            tickfont=dict(size=10, color="#526174"),
        ),
        yaxis=dict(
            showgrid=False,
            autorange=True,
            tickfont=dict(size=11, color="#17212F"),
        ),
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto", size=11),
    )

    return fig


# ==============================================================================
# Main Application
# ==============================================================================
def main():
    apply_minimal_light_theme()

    resolved_path = resolve_model_path()
    model_basename = os.path.basename(resolved_path)
    model_exists = os.path.isfile(resolved_path)

    # --------------------------------------------------------------------------
    # 1. Sidebar: Concise Specifications & Reference
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="font-weight:700; font-size:0.95rem; color:#17212F; padding-bottom:6px; border-bottom:1px solid #E2E8F0; margin-bottom:12px;">
                Model Specification
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <table class="sb-table">
                <tr><td class="sb-key">Backbone</td><td class="sb-val">EfficientNetB0</td></tr>
                <tr><td class="sb-key">Input Size</td><td class="sb-val">224 × 224 × 3</td></tr>
                <tr><td class="sb-key">Color Space</td><td class="sb-val">RGB [0, 255]</td></tr>
                <tr><td class="sb-key">Outputs</td><td class="sb-val">7 Softmax</td></tr>
                <tr><td class="sb-key">Sampling</td><td class="sb-val">Mild Balance</td></tr>
                <tr><td class="sb-key">Weights</td><td class="sb-val">ImageNet Pretrained</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="font-weight:700; font-size:0.85rem; color:#17212F; margin-top:16px; margin-bottom:6px;">
                Target Lesion Classes (7)
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Explore 7 Target Classes", expanded=False):
            for code in CLASS_NAMES:
                meta = CLASS_METADATA[code]
                st.markdown(
                    f"""
                    <div style="padding:5px 0; border-bottom:1px solid #F1F5F9; font-size:0.77rem;">
                        <strong style="color:#17212F;">{meta['full_name']}</strong> 
                        <code style="font-size:0.7rem; font-weight:700; color:#0F766E;">{code.upper()}</code><br>
                        <span style="font-size:0.71rem; color:#526174;">{meta['category']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <div style="font-weight:700; font-size:0.85rem; color:#17212F; margin-top:16px; margin-bottom:6px;">
                System Status
            </div>
            """,
            unsafe_allow_html=True,
        )

        if model_exists:
            st.markdown(
                f"""
                <div style="font-size:0.74rem; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:8px;">
                    <div style="color:#047857; font-weight:600; margin-bottom:2px;">● Weights Ready</div>
                    <code style="font-size:0.68rem; color:#17212F; word-break:break-all;">{model_basename}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="font-size:0.74rem; background:#FFF1F2; border:1px solid #FECDD3; border-radius:6px; padding:8px;">
                    <div style="color:#BE123C; font-weight:600; margin-bottom:2px;">✕ Weights Not Found</div>
                    <code style="font-size:0.68rem; color:#17212F; word-break:break-all;">{DEFAULT_MODEL_FILENAME}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------------------------
    # 2. Main Area: Clean Header (No Large Card)
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="workbench-header">
            <h1 class="header-title">
                <span>🔬</span> Skin Lesion Classification
                <span class="header-badge">Research Workbench</span>
            </h1>
            <p class="header-subtitle">Research Inference Platform · HAM10000 Benchmark Evaluation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 3. Compact Safety Disclaimer Notice
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="compact-disclaimer">
            <strong>⚠️ Educational & Research Demonstration Only:</strong>
            This system is an academic machine learning prototype and is <strong>not</strong> certified for medical diagnosis or clinical treatment.
            Predictions represent uncalibrated algorithmic softmax outputs. Standard dermoscopic images are required; consumer camera photos yield unreliable outputs.
            Consult a board-certified dermatologist for medical evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 4. Model Loading Guard
    # --------------------------------------------------------------------------
    if not model_exists:
        st.error(
            f"**Model Artifact Unavailable:** Could not locate `{DEFAULT_MODEL_FILENAME}` at `{resolved_path}`."
        )
        st.stop()

    try:
        model = load_classification_model(resolved_path)
    except Exception as exc:
        st.error(f"**Model Initialization Error:** `{str(exc)}`")
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
    # 6. Upload Section
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-heading">Analyze a Dermoscopic Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a dermoscopic image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG. Image will be converted to RGB and resized to 224x224.",
        label_visibility="collapsed",
    )

    st.caption("Accepted formats: JPG, JPEG, PNG. Standardized dermoscopic magnification expected.")

    # Synchronize uploaded file to session state
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
                st.error("❌ Invalid image file. Please upload an uncorrupted JPG or PNG image.")
                st.session_state.active_pil_image = None
    elif st.session_state.active_pil_image is None:
        # Provide compact benchmark sample option when no image is loaded
        if st.button("🧪 Load Benchmark Test Sample", use_container_width=False):
            st.session_state.current_file_id = "sample_benchmark_pattern"
            st.session_state.active_pil_image = generate_benchmark_sample()
            st.session_state.active_image_name = "benchmark_dermoscopy_sample.png"
            st.session_state.prediction_results = None
            st.rerun()

    # --------------------------------------------------------------------------
    # 7. Workspace: Preview & Prediction Results
    # --------------------------------------------------------------------------
    if st.session_state.active_pil_image is not None:
        pil_image = st.session_state.active_pil_image
        image_name = st.session_state.active_image_name or "dermoscopic_input.jpg"

        col_img, col_pred = st.columns([1, 1.3], gap="large")

        # Left Column: Image Preview & Control
        with col_img:
            st.markdown('<div class="section-heading" style="font-size:0.95rem;">Image Preview</div>', unsafe_allow_html=True)
            st.image(pil_image, use_container_width=True)

            st.markdown(
                f"""
                <div class="image-meta-bar">
                    <span style="font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:60%;" title="{image_name}">{image_name}</span>
                    <span>{pil_image.width} × {pil_image.height} px · RGB</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            btn_col1, btn_col2 = st.columns([1.5, 1])
            with btn_col1:
                run_btn = st.button("▶ Run Inference", type="primary", use_container_width=True)
            with btn_col2:
                if st.button("↺ Reset", use_container_width=True):
                    st.session_state.current_file_id = None
                    st.session_state.active_pil_image = None
                    st.session_state.active_image_name = None
                    st.session_state.prediction_results = None
                    st.rerun()

        # Right Column: Predictions & Distribution
        with col_pred:
            st.markdown('<div class="section-heading" style="font-size:0.95rem;">Prediction Results</div>', unsafe_allow_html=True)

            if run_btn:
                with st.spinner("Executing forward pass..."):
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
                        st.error(f"Inference error: `{str(err)}`")
                        return

            if st.session_state.prediction_results is None:
                st.markdown(
                    """
                    <div style="background:#FFFFFF; border:1px dashed #CBD5E1; border-radius:8px; padding:2.5rem 1.5rem; text-align:center; color:#526174; margin-top:0.25rem;">
                        <div style="font-weight:600; color:#17212F; margin-bottom:4px;">Awaiting Execution</div>
                        <div style="font-size:0.82rem;">Click <strong>'▶ Run Inference'</strong> to compute class probabilities.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                results = st.session_state.prediction_results
                top_class = results[0]

                # Severity tag class
                sev = top_class["severity"]
                cat_class = "cat-danger" if sev == "danger" else ("cat-warning" if sev == "warning" else "cat-benign")

                # Clean Top Prediction Card
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <div class="pred-tag-row">
                            <span class="pred-code-pill">{top_class['code']}</span>
                            <span class="pred-category-pill {cat_class}">{top_class['category']}</span>
                        </div>
                        <div class="pred-score-large">{top_class['percentage']:.2f}%</div>
                        <div class="pred-score-label">Softmax Probability (Uncalibrated Model Output)</div>
                        <div class="pred-title">{top_class['full_name']}</div>
                        <div class="pred-desc">{top_class['description']}</div>
                        <div class="pred-protocol-box">
                            <strong>Reference Note:</strong> {top_class['clinical_urgency']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Distribution Chart
                st.markdown(
                    """
                    <div style="font-weight:600; font-size:0.84rem; color:#17212F; margin-bottom:2px;">
                        Seven-Class Probability Distribution
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                chart = create_probability_chart(results)
                st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

                # Technical note
                st.markdown(
                    """
                    <div class="technical-callout">
                        <strong>Probabilistic Interpretation:</strong> Softmax scores represent relative feature correlation within the closed 7-class training distribution. They are <strong>not calibrated medical probabilities</strong>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------------------------
        # 8. Full Ranked Table Breakdown
        # ----------------------------------------------------------------------
        if st.session_state.prediction_results is not None:
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-heading" style="font-size:0.95rem;">Full Seven-Class Ranking</div>', unsafe_allow_html=True)

            table_data = []
            for item in st.session_state.prediction_results:
                table_data.append(
                    {
                        "Rank": f"#{item['rank']}",
                        "Code": item["code"].upper(),
                        "Entity": item["full_name"],
                        "Category": item["category"],
                        "Score": round(item["percentage"], 2),
                        "Reference Context": item["clinical_urgency"],
                    }
                )

            st.dataframe(
                table_data,
                column_config={
                    "Rank": st.column_config.TextColumn("Rank", width="small"),
                    "Code": st.column_config.TextColumn("Code", width="small"),
                    "Entity": st.column_config.TextColumn("Pathology Entity", width="medium"),
                    "Category": st.column_config.TextColumn("Category", width="small"),
                    "Score": st.column_config.ProgressColumn(
                        "Model Score (%)",
                        format="%.2f%%",
                        min_value=0.0,
                        max_value=100.0,
                        width="medium",
                    ),
                    "Reference Context": st.column_config.TextColumn("Clinical Protocol", width="large"),
                },
                hide_index=True,
                use_container_width=True,
            )

    # --------------------------------------------------------------------------
    # 9. Compact Research Metrics Row
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-heading" style="margin-top:1.5rem;">Research Evaluation Metrics</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-cell">
                <div class="metric-cell-label">Validation Accuracy</div>
                <div class="metric-cell-value">{BENCHMARK_METRICS['validation_accuracy'] * 100:.2f}%</div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Macro F1 Score</div>
                <div class="metric-cell-value">{BENCHMARK_METRICS['macro_f1']:.4f}</div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Balanced Accuracy</div>
                <div class="metric-cell-value">{BENCHMARK_METRICS['balanced_accuracy'] * 100:.2f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Benchmark results on the HAM10000 validation split. "
        "Because benign melanocytic nevi comprise ~67% of samples, Macro F1 and Balanced Accuracy provide accurate sensitivity evaluation across minority classes."
    )

    # --------------------------------------------------------------------------
    # 10. Technical Architecture Details
    # --------------------------------------------------------------------------
    with st.expander("Technical Architecture & Methodology Details", expanded=False):
        t1, t2 = st.columns(2, gap="medium")
        with t1:
            st.markdown(
                """
                **Class Imbalance & Sampling Strategy:**  
                Standard training on HAM10000 yields high nominal accuracy but low recall on rare, critical classes (`mel`, `bcc`).
                Mild class-balanced sampling increases minority-class sampling frequency during fine-tuning, achieving **50.78%** balanced accuracy and **0.5431** macro F1.

                **Input Preprocessing:**  
                Images undergo EXIF orientation correction, RGB conversion, and bilinear resizing to **224 × 224 × 3**.
                Pixel values remain in `[0.0, 255.0]` (internal EfficientNet rescaling).
                """
            )
        with t2:
            st.markdown(
                """
                **Closed-World Softmax Assumption:**  
                Softmax probabilities sum to 100% across the 7 closed categories. High confidence does not imply calibrated posterior probability.

                **Out-of-Distribution Preconditions:**  
                The network expects standardized magnified dermoscopy. Consumer camera images, poor illumination, or non-skin imagery violate distribution bounds.
                """
            )

    # --------------------------------------------------------------------------
    # 11. Minimal Footer
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="workbench-footer">
            Skin Lesion Classification Research Workbench · EfficientNetB0 ({BENCHMARK_METRICS['sampling_strategy']})<br>
            <em>Strictly for educational and research demonstration. Not cleared for medical diagnosis or clinical use.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
