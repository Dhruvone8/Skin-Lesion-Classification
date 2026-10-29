"""
Skin Lesion Classification - Streamlit Web Application
======================================================
Inference web application for dermoscopic skin lesion classification
powered by an EfficientNetB0 convolutional neural network.

Deployed with 'best_efficientnetb0_mild_sampling.keras', fine-tuned using
mild class-balanced sampling on the HAM10000 dataset for research evaluation.
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# ==============================================================================
# Page Configuration & Metadata
# ==============================================================================
st.set_page_config(
    page_title="Skin Lesion Classification | EfficientNetB0 (Mild Sampling)",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Class Definitions & Medical Knowledge Base
# ==============================================================================
CLASS_NAMES: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_METADATA: Dict[str, Dict[str, str]] = {
    "akiec": {
        "full_name": "Actinic Keratoses and Intraepithelial Carcinoma",
        "category": "Pre-cancerous / In-situ Carcinoma",
        "description": (
            "Sun-induced epidermal lesions with cellular atypia, including Bowen's disease. "
            "Pre-malignant precursors that carry potential progression to invasive squamous cell carcinoma."
        ),
        "clinical_urgency": "Requires dermatological evaluation & management",
        "badge_color": "#d97706",
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "category": "Malignant Skin Cancer",
        "description": (
            "The most common cutaneous malignancy arising from the basal layer of the epidermis. "
            "Locally invasive and tissue-destructive, but exhibits very low metastatic potential."
        ),
        "clinical_urgency": "Requires prompt clinical / surgical evaluation",
        "badge_color": "#dc2626",
    },
    "bkl": {
        "full_name": "Benign Keratosis-like Lesions",
        "category": "Benign Epidermal Growth",
        "description": (
            "Includes seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses. "
            "Very common non-cancerous epidermal growths frequently mimicking melanocytic lesions."
        ),
        "clinical_urgency": "Typically benign; periodic monitoring advised",
        "badge_color": "#059669",
    },
    "df": {
        "full_name": "Dermatofibroma",
        "category": "Benign Dermal Lesion",
        "description": (
            "Benign cutaneous nodules consisting of fibroblasts and histiocytes. "
            "Usually asymptomatic and commonly found on extremities (positive 'pinch sign')."
        ),
        "clinical_urgency": "Benign; intervention rarely required",
        "badge_color": "#059669",
    },
    "mel": {
        "full_name": "Melanoma",
        "category": "Malignant Skin Cancer",
        "description": (
            "The most lethal form of skin cancer, arising from melanocytes. "
            "High propensity for rapid metastasis if not recognized and excised at early intraepidermal stages."
        ),
        "clinical_urgency": "Urgent dermatological excision / biopsy required",
        "badge_color": "#dc2626",
    },
    "nv": {
        "full_name": "Melanocytic Nevi",
        "category": "Benign Melanocytic Proliferation (Mole)",
        "description": (
            "Common benign neoplasms composed of melanocytes (moles). Represents over 67% "
            "of dermoscopic images in standard public reference datasets like HAM10000."
        ),
        "clinical_urgency": "Common benign mole; monitor for ABCDE evolution",
        "badge_color": "#059669",
    },
    "vasc": {
        "full_name": "Vascular Lesions",
        "category": "Benign Vascular Proliferation",
        "description": (
            "Benign proliferations of cutaneous blood vessels including cherry angiomas, "
            "angiokeratomas, and pyogenic granulomas. Characterized by vascular lacunae."
        ),
        "clinical_urgency": "Typically benign vascular growth",
        "badge_color": "#059669",
    },
}

DEFAULT_MODEL_FILENAME = "best_efficientnetb0_mild_sampling.keras"
FALLBACK_MODEL_FILENAME = "best_efficientnetb0_stage3.keras"


# ==============================================================================
# Custom CSS for Professional Medical-Research Aesthetics
# ==============================================================================
def apply_custom_styles():
    st.markdown(
        """
        <style>
        /* Base typography and font smoothing */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Top disclaimer box styling */
        .medical-disclaimer-box {
            background-color: #fffbeb;
            border-left: 5px solid #f59e0b;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 24px;
            color: #78350f;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        .medical-disclaimer-title {
            font-size: 1.05rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
            color: #92400e;
        }
        .medical-disclaimer-text {
            font-size: 0.90rem;
            line-height: 1.5;
            margin: 0;
            color: #78350f;
        }

        /* Top prediction highlight card */
        .top-prediction-card {
            background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
            border: 1px solid #bae6fd;
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
            transition: all 0.2s ease-in-out;
        }
        .prediction-class-title {
            font-size: 1.45rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 6px;
            margin-bottom: 4px;
        }
        .prediction-code-tag {
            display: inline-block;
            background-color: #0284c7;
            color: #ffffff;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-right: 8px;
        }
        .prediction-category-tag {
            display: inline-block;
            background-color: #ffffff;
            color: #334155;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 12px;
            border: 1px solid #cbd5e1;
        }
        .prediction-confidence-value {
            font-size: 2.3rem;
            font-weight: 800;
            color: #0369a1;
            line-height: 1.1;
            margin-top: 10px;
            margin-bottom: 4px;
        }

        /* Metrics summary pills */
        .metric-pill {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px 14px;
            text-align: center;
        }
        .metric-pill-label {
            font-size: 0.73rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        .metric-pill-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 2px;
        }

        /* Technical note callout */
        .technical-callout {
            background-color: #f8fafc;
            border-left: 4px solid #64748b;
            border-radius: 6px;
            padding: 12px 16px;
            font-size: 0.86rem;
            color: #475569;
            line-height: 1.45;
            margin-top: 14px;
        }

        /* Footer styling */
        .app-footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            font-size: 0.82rem;
            color: #64748b;
            line-height: 1.5;
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
        # Explicit env var first
        os.environ.get("MODEL_PATH", "").strip(),
        # Mild sampling in current dir (Primary)
        os.path.join(current_dir, DEFAULT_MODEL_FILENAME),
        # Mild sampling in parent dir / workspace root
        os.path.join(current_dir, "..", DEFAULT_MODEL_FILENAME),
        DEFAULT_MODEL_FILENAME,
        # Fallback stage 3 model if mild sampling not found
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

    # Return expected primary path if none found (for error reporting)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, DEFAULT_MODEL_FILENAME)


@st.cache_resource(show_spinner=False)
def load_classification_model(model_filepath: str):
    """
    Loads and caches the trained Keras model without compiling it.
    Uses st.cache_resource to prevent re-instantiating the model on each run.
    """
    if not os.path.isfile(model_filepath):
        raise FileNotFoundError(
            f"Trained model file not found at: '{model_filepath}'"
        )
    model = tf.keras.models.load_model(model_filepath, compile=False)
    return model


# ==============================================================================
# Image Preprocessing & Inference Logic
# ==============================================================================
def preprocess_image(pil_img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocesses a PIL Image strictly matching the training pipeline:
    1. Handles EXIF orientation tags automatically.
    2. Converts image to 3-channel RGB.
    3. Resizes image to (224, 224) using bilinear interpolation.
    4. Converts image to float32 NumPy array with shape (224, 224, 3).
    5. Adds batch dimension -> shape (1, 224, 224, 3).

    NOTE: Pixel values are in the range [0.0, 255.0].
    Values are NOT divided by 255 here, because Keras EfficientNetB0
    incorporates internal input rescaling layers.
    """
    # 1. Correct image orientation according to EXIF data
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    # 2. Convert to RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # 3. Resize to target dimension
    resample_attr = getattr(Image, "Resampling", Image)
    bilinear_filter = getattr(resample_attr, "BILINEAR", Image.BILINEAR)
    resized_img = pil_img.resize(target_size, resample=bilinear_filter)

    # 4. Convert to float32 NumPy array with range [0.0, 255.0]
    img_array = np.array(resized_img, dtype=np.float32)

    # 5. Add batch dimension -> (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    return img_batch


def run_inference(model, input_batch: np.ndarray) -> np.ndarray:
    """
    Performs forward pass inference and validates output dimensions.
    Returns 1D array of 7 class probabilities.
    """
    raw_output = model.predict(input_batch, verbose=0)

    # Validate output shape
    if raw_output.ndim != 2 or raw_output.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Invalid model output shape {raw_output.shape}. "
            f"Expected shape (1, {len(CLASS_NAMES)})."
        )

    probabilities = raw_output[0]
    return probabilities


# ==============================================================================
# Visualization: Clean Horizontal Bar Chart
# ==============================================================================
def create_probability_chart(sorted_results: List[Dict[str, any]]) -> go.Figure:
    """
    Renders an interactive horizontal bar chart ranking all 7 classes.
    """
    display_results = list(reversed(sorted_results))

    labels = [
        f"<b>{item['full_name']}</b> ({item['code']})"
        for item in display_results
    ]
    scores = [item["percentage"] for item in display_results]
    hover_texts = [
        f"<b>{item['full_name']}</b><br>"
        f"Class Code: <code>{item['code']}</code><br>"
        f"Pathology: {item['category']}<br>"
        f"Model Confidence: <b>{item['percentage']:.2f}%</b>"
        for item in display_results
    ]

    # Distinguish top prediction with medical blue, others in subtle slate
    bar_colors = [
        "#0284c7" if item["rank"] == 1 else "#94a3b8"
        for item in display_results
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            text=[f" {s:.1f}%" for s in scores],
            textposition="outside",
            marker=dict(
                color=bar_colors,
                line=dict(color="#0f172a", width=0.5),
            ),
            hoverinfo="text",
            hovertext=hover_texts,
        )
    )

    max_score = max(scores) if scores else 100
    x_limit = min(100.0, max(max_score * 1.15, 20.0))

    fig.update_layout(
        title=dict(
            text="<b>Model Score Distribution Across All 7 Lesion Classes</b>",
            font=dict(size=14, color="#0f172a"),
            x=0.0,
        ),
        xaxis=dict(
            title="Model Output Probability (%)",
            range=[0, x_limit],
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            ticksuffix="%",
        ),
        yaxis=dict(
            showgrid=False,
            autorange=True,
        ),
        margin=dict(l=10, r=40, t=40, b=30),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto", size=12),
    )

    return fig


# ==============================================================================
# Main Streamlit Application
# ==============================================================================
def main():
    apply_custom_styles()

    # Determine resolved model path
    resolved_path = resolve_model_path()
    model_basename = os.path.basename(resolved_path)
    is_mild_sampling = "mild_sampling" in model_basename

    # --------------------------------------------------------------------------
    # Sidebar: Model Specs, Validation Benchmarks, Class Reference
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🔬 System Architecture")
        model_display_name = (
            "EfficientNetB0 (Mild Sampling)"
            if is_mild_sampling
            else "EfficientNetB0 (Stage 3 Fine-Tuning)"
        )
        st.markdown(
            f"""
            **Model Backbone:** {model_display_name}  
            **Input Dimensions:** `224 × 224 × 3` (RGB)  
            **Output Units:** 7 Softmax Probabilities  
            **Sampling Strategy:** {'Class-balanced mild oversampling' if is_mild_sampling else 'Standard stratified sampling'}
            """
        )

        st.markdown("---")
        st.markdown("### 📊 Research Validation Metrics")
        st.caption("Benchmark evaluation on the HAM10000 validation split:")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(
                """
                <div class="metric-pill">
                    <div class="metric-pill-label">Accuracy</div>
                    <div class="metric-pill-value">77.68%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                """
                <div class="metric-pill">
                    <div class="metric-pill-label">Macro F1</div>
                    <div class="metric-pill-value">0.5431</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        m_col3, m_col4 = st.columns(2)
        with m_col3:
            st.markdown(
                """
                <div class="metric-pill">
                    <div class="metric-pill-label">Balanced Acc</div>
                    <div class="metric-pill-value">50.78%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col4:
            st.markdown(
                """
                <div class="metric-pill">
                    <div class="metric-pill-label">Strategy</div>
                    <div class="metric-pill-value">Mild Sampling</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "Note: Metrics reflect research-phase evaluation on dermoscopic images. "
            "They do not constitute clinical validation."
        )

        st.markdown("---")
        st.markdown("### 🏷️ Target Class Directory")
        with st.expander("Explore 7 Target Classes", expanded=False):
            for code in CLASS_NAMES:
                meta = CLASS_METADATA[code]
                st.markdown(
                    f"**`{code}`** — {meta['full_name']}  \n"
                    f"*Category:* `{meta['category']}`  \n"
                    f"<small>{meta['description']}</small>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")

        st.markdown("### ⚙️ Active Model Status")
        model_exists = os.path.isfile(resolved_path)

        if model_exists:
            st.success(f"Loaded: `{model_basename}`", icon="✅")
            st.caption(f"Path: `{resolved_path}`")
        else:
            st.error(f"Model missing: `{DEFAULT_MODEL_FILENAME}`", icon="❌")
            st.caption(f"Searched: `{resolved_path}`")

    # --------------------------------------------------------------------------
    # Main Header & Title
    # --------------------------------------------------------------------------
    st.title("🔬 Skin Lesion Classification")
    st.markdown(
        "##### Academic Research Inference Platform using Fine-Tuned EfficientNetB0 (Mild Sampling)"
    )

    # --------------------------------------------------------------------------
    # Prominent Medical Safety Disclaimer (Mandatory)
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="medical-disclaimer-box">
            <div class="medical-disclaimer-title">
                ⚠️ Medical & Educational Safety Notice
            </div>
            <div class="medical-disclaimer-text">
                This application is strictly an educational research demonstration and engineering portfolio artifact.
                <br>
                • <strong>Not a Medical Device:</strong> This system is <em>not</em> certified, approved, or clinically validated to diagnose skin conditions or detect skin cancer.<br>
                • <strong>No Treatment Decisions:</strong> Predictions generated by this model must <em>never</em> be used for clinical decision-making, self-diagnosis, or treatment.<br>
                • <strong>Dermoscopic Standard:</strong> The underlying neural network was trained on standardized dermoscopic imaging; consumer smartphone photos, varied lighting, or out-of-distribution inputs will produce unreliable results.<br>
                • <strong>Professional Care:</strong> If you have any concerns regarding a mole, changing spot, or skin lesion, immediately schedule an examination with a board-certified dermatologist or qualified medical physician.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # Model Loading with Graceful Error Handling
    # --------------------------------------------------------------------------
    if not os.path.isfile(resolved_path):
        st.error(
            f"### ⚠️ Model Artifact Not Found\n\n"
            f"The application could not find the required model weights file: **`{DEFAULT_MODEL_FILENAME}`**.\n\n"
            f"**To resolve this issue:**\n"
            f"1. Ensure `best_efficientnetb0_mild_sampling.keras` is placed in `{os.path.dirname(resolved_path)}`, OR\n"
            f"2. Set the `MODEL_PATH` environment variable to the absolute path of your `.keras` model file.\n\n"
            f"*Dataset files, training notebooks, and metadata CSVs are not required.*"
        )
        st.stop()

    try:
        model = load_classification_model(resolved_path)
    except Exception as exc:
        st.error(
            f"### ⚠️ Model Loading Error\n\n"
            f"An error occurred while loading the Keras model file from `{resolved_path}`.\n"
            f"Please verify that TensorFlow {tf.__version__} is compatible with the saved model format.\n\n"
            f"**Error Details:** `{str(exc)}`"
        )
        st.stop()

    # --------------------------------------------------------------------------
    # Image Upload & Interaction Area
    # --------------------------------------------------------------------------
    st.markdown("### 1. Upload Dermoscopic Image")
    st.write(
        "Upload a dermoscopic image in JPG, JPEG, or PNG format to compute model class probabilities."
    )

    uploaded_file = st.file_uploader(
        "Choose a dermoscopic lesion image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG. Image will be converted to RGB and resized to 224x224.",
        label_visibility="collapsed",
    )

    # Smooth state management for file upload
    if "current_file_id" not in st.session_state:
        st.session_state.current_file_id = None
    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None

    if uploaded_file is None:
        # Reset prediction cache when file is removed
        st.session_state.current_file_id = None
        st.session_state.prediction_results = None

        st.info(
            "👆 Please upload a skin lesion image above to begin analysis. "
            "Supported formats: `.jpg`, `.jpeg`, `.png`.",
            icon="ℹ️",
        )
    else:
        # Check if a new file was uploaded compared to session cache
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.current_file_id != file_id:
            st.session_state.current_file_id = file_id
            st.session_state.prediction_results = None

        # Load and validate PIL Image safely
        try:
            pil_image = Image.open(uploaded_file)
            pil_image.load()  # Force decode to verify data integrity
        except (UnidentifiedImageError, OSError, ValueError):
            st.error(
                "❌ **Invalid or Corrupted Image**: The uploaded file could not be decoded as an image. "
                "Please upload a valid, uncorrupted JPG or PNG file."
            )
            return

        col_img, col_pred = st.columns([1, 1.4], gap="large")

        with col_img:
            st.markdown("#### Image Preview")
            st.image(
                pil_image,
                caption=f"Uploaded: {uploaded_file.name} ({pil_image.width} × {pil_image.height} px, {pil_image.mode} mode)",
                use_container_width=True,
            )

            analyze_button = st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True,
            )

        with col_pred:
            st.markdown("#### Model Predictions")

            # Run inference if analyze button is clicked or if predictions already exist for this file
            if analyze_button:
                with st.spinner("Processing image and computing forward pass..."):
                    try:
                        processed_batch = preprocess_image(pil_image, target_size=(224, 224))
                        probabilities = run_inference(model, processed_batch)

                        # Format and rank all 7 classes
                        results = []
                        for idx, code in enumerate(CLASS_NAMES):
                            prob = float(probabilities[idx])
                            meta = CLASS_METADATA[code]
                            results.append(
                                {
                                    "code": code,
                                    "full_name": meta["full_name"],
                                    "category": meta["category"],
                                    "score": prob,
                                    "percentage": prob * 100.0,
                                    "clinical_urgency": meta["clinical_urgency"],
                                    "description": meta["description"],
                                }
                            )

                        # Sort in descending order of score
                        results.sort(key=lambda x: x["score"], reverse=True)
                        for rank_idx, item in enumerate(results, start=1):
                            item["rank"] = rank_idx

                        st.session_state.prediction_results = results
                    except Exception as inf_err:
                        st.error(
                            f"❌ **Inference Error**: An unexpected error occurred during prediction.\n\n"
                            f"Message: `{str(inf_err)}`"
                        )
                        return

            if st.session_state.prediction_results is None:
                st.markdown(
                    """
                    <div style="background-color: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 10px; padding: 44px 20px; text-align: center; color: #64748b;">
                        <div style="font-size: 2.2rem; margin-bottom: 8px;">🔬</div>
                        <strong>Ready for Evaluation</strong><br>
                        Click the <strong>'Analyze Image'</strong> button to run EfficientNetB0 inference.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                results = st.session_state.prediction_results
                top_class = results[0]

                # --------------------------------------------------------------
                # Top Prediction Callout Card
                # --------------------------------------------------------------
                st.markdown(
                    f"""
                    <div class="top-prediction-card">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div>
                                <span class="prediction-code-tag">{top_class['code']}</span>
                                <span class="prediction-category-tag">{top_class['category']}</span>
                            </div>
                            <span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">Rank 1 of 7</span>
                        </div>
                        <div class="prediction-confidence-value">{top_class['percentage']:.2f}%</div>
                        <div class="prediction-class-title">{top_class['full_name']}</div>
                        <div style="font-size: 0.88rem; color: #475569; margin-top: 6px;">
                            {top_class['description']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # --------------------------------------------------------------
                # Probability Distribution Chart
                # --------------------------------------------------------------
                chart = create_probability_chart(results)
                st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

                st.markdown(
                    """
                    <div class="technical-callout">
                        <strong>Interpretation Warning:</strong> Model scores are raw softmax probabilities generated by a neural network. 
                        They represent relative feature correlation with the training distribution, 
                        <strong>not calibrated medical probabilities</strong>. A low score does not rule out malignancy, 
                        and a high score does not confirm disease.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------------------------
        # Full Ranked Table Breakdown
        # ----------------------------------------------------------------------
        if uploaded_file is not None and st.session_state.get("prediction_results") is not None:
            st.markdown("---")
            st.markdown("### 2. Ranked Classification Breakdown")
            st.caption(
                "Complete ranking of all seven model outputs in descending order of softmax output score."
            )

            results = st.session_state.prediction_results

            # Build table rows for clean rendering
            table_rows = []
            for item in results:
                table_rows.append(
                    {
                        "Rank": f"#{item['rank']}",
                        "Class Code": f"`{item['code']}`",
                        "Diagnostic Category": item["full_name"],
                        "Pathology Type": item["category"],
                        "Model Score": f"{item['percentage']:.2f}%",
                        "Urgency Context": item["clinical_urgency"],
                    }
                )

            st.dataframe(
                table_rows,
                column_config={
                    "Rank": st.column_config.TextColumn("Rank", width="small"),
                    "Class Code": st.column_config.TextColumn("Class Code", width="small"),
                    "Diagnostic Category": st.column_config.TextColumn("Diagnostic Category", width="medium"),
                    "Pathology Type": st.column_config.TextColumn("Pathology Type", width="medium"),
                    "Model Score": st.column_config.TextColumn("Model Score (%)", width="small"),
                    "Urgency Context": st.column_config.TextColumn("Clinical Reference Note", width="large"),
                },
                hide_index=True,
                use_container_width=True,
            )

    # --------------------------------------------------------------------------
    # Technical Limitations & Research Context
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 3. Model Scope & Technical Limitations")

    with st.expander("Read Technical Specifications and Model Limitations", expanded=False):
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.markdown(
                """
                **1. Mild Sampling & Class Imbalance:**  
                The HAM10000 dataset is heavily skewed toward benign melanocytic nevi (`nv` comprises ~67% of samples). The `best_efficientnetb0_mild_sampling` model applies mild class-balanced re-sampling to improve recall on rarer and higher-risk classes (`mel`, `bcc`, `akiec`).

                **2. Fixed Preprocessing Pipeline:**  
                Input images are resized to 224 × 224 pixels and kept in the `[0, 255]` float32 range. EfficientNet's internal architecture incorporates a preprocessing rescaling layer. Bilinear resizing can induce aspect-ratio distortion for non-square images.
                """
            )
        with col_l2:
            st.markdown(
                """
                **3. Softmax Scores vs. Calibrated Clinical Risk:**  
                Softmax outputs sum to 100% across the 7 closed categories. However, high softmax confidence does not necessarily imply high Bayesian posterior probability, especially when encountering artifacts (hair, ruler marks, ink, air bubbles).

                **4. Out-of-Distribution Sensitivity:**  
                The model is trained strictly on dermoscopic imaging. Standard smartphone camera photographs, non-skin tissue, poor illumination, or atypical lesions are out-of-distribution (OOD) and will yield completely invalid predictions.
                """
            )

    # --------------------------------------------------------------------------
    # Application Footer
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="app-footer">
            <strong>Skin Lesion Classification Web Application</strong><br>
            Developed as an academic research and machine learning engineering portfolio artifact.<br>
            Model Architecture: EfficientNetB0 (Mild Sampling) • Active File: <code>{model_basename}</code> • Backbone Weights: ImageNet<br>
            <em>Strictly for educational demonstration. Not for medical diagnosis or clinical healthcare decisions.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
