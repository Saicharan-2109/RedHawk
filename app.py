"""Streamlit interface for the Spatial Matrix Vision Lab."""

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import streamlit as st

from spatial_lab.engine import EngineParameters, TransformResult, build_operators, process_image
from spatial_lab.image_io import (
    DEMO_NAMES,
    ImageValidationError,
    PreparedImage,
    build_download_bundle,
    decode_image,
    demo_image,
    encode_png,
)

st.set_page_config(
    page_title="Spatial Matrix · Vision Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"About": "Spatial Matrix Vision Lab · deterministic computer vision with NumPy and OpenCV."},
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1510px; padding-top: 2.4rem; padding-bottom: 3rem;}
    h1,h2,h3 {letter-spacing: -0.035em;}
    h1 {font-size: clamp(2rem, 4vw, 3.2rem) !important; line-height: 1.12 !important;}
    [data-testid="stSidebar"] {border-right: 1px solid #dce3e9;}
    [data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}
    [data-testid="stMetric"] {background: #fff; border: 1px solid #dce3e9;
        padding: .9rem 1.1rem; border-radius: 12px;}
    [data-testid="stMetricValue"] {font-size: 1.55rem; letter-spacing: -.03em;}
    [data-testid="stTabs"] [role="tablist"] {gap: 1.8rem; border-bottom: 1px solid #dce3e9;}
    [data-testid="stTabs"] [role="tab"] {padding-bottom: .8rem;}
    [data-testid="stImage"] img {border-radius: 8px;}
    [data-testid="stVerticalBlockBorderWrapper"] > div {border-radius: 12px;}
    .brand {font-family: ui-monospace, Consolas, monospace; font-size: .8rem;
        font-weight: 650; letter-spacing: .15em; color: #147d68; margin-bottom: 1rem;}
    .brand-dot {display: inline-block; width: 9px; height: 9px; border-radius: 50%;
        background: #147d68; margin-right: 10px;}
    .hero-copy {color: #5b6875; font-size: 1.05rem; max-width: 780px; line-height: 1.65;}
    .tag {display: inline-block; color: #396458; background: #e5f0eb; padding: .3rem .65rem;
        border-radius: 5px; font-family: ui-monospace, Consolas, monospace;
        font-size: .72rem; margin: .25rem .3rem .8rem 0;}
    .step {font-family: ui-monospace, Consolas, monospace; font-size: .72rem;
        letter-spacing: .12em; color: #147d68; margin-bottom: .25rem;}
    .footer {border-top: 1px solid #dce3e9; padding-top: 1rem; margin-top: 2rem;
        color: #677685; font-size: .8rem;}
    @media (max-width: 700px) {.block-container {padding-top: 1.4rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULTS = asdict(EngineParameters())
PRESETS = {
    "Blueprint defaults": DEFAULTS,
    "Gentle response": {**DEFAULTS, "phi_1": 0.40, "phi_2": 0.20, "phi_3": 0.10,
                        "edge_gain": 2.0, "integration_gain": 0.30},
    "Strong outlines": {**DEFAULTS, "phi_1": 0.60, "phi_2": 0.20, "phi_3": 0.10,
                        "edge_gain": 7.0},
}


def reset_parameters() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["preset"] = "Blueprint defaults"
    st.session_state["max_dimension"] = 700


def apply_preset() -> None:
    for key, value in PRESETS[st.session_state["preset"]].items():
        st.session_state[key] = value


@st.cache_data(max_entries=4, ttl=1800, show_spinner=False)
def prepare_upload(data: bytes, name: str, max_dimension: int):
    return decode_image(data, name, max_dimension)


@st.cache_data(max_entries=4, ttl=1800, show_spinner=False)
def prepare_demo(name: str, max_dimension: int):
    return demo_image(name, max_dimension)


def array_digest(array: np.ndarray) -> tuple:
    """Hash every pixel; Streamlit's default hashes large arrays by sampling."""
    return array.shape, str(array.dtype), hashlib.sha256(array.tobytes()).hexdigest()


@st.cache_data(max_entries=4, ttl=1800, show_spinner=False, hash_funcs={np.ndarray: array_digest})
def transform(gray: np.ndarray, parameters: EngineParameters):
    return process_image(gray, parameters)


def prepared_digest(image: PreparedImage) -> tuple:
    return (array_digest(image.rgb), array_digest(image.gray), image.original_width,
            image.original_height, image.source_name, image.resized)


def result_digest(result: TransformResult) -> tuple:
    return (tuple(array_digest(array) for array in (
        result.edges, result.baseline, result.extension, result.operators.transition,
        result.operators.difference, result.operators.integration)),
        result.elapsed_ms, result.operators.spectral_radius, result.operators.condition_number)


@st.cache_data(max_entries=2, ttl=1800, show_spinner=False,
               hash_funcs={PreparedImage: prepared_digest, TransformResult: result_digest})
def export_bundle(prepared: PreparedImage, result: TransformResult, parameters: EngineParameters) -> bytes:
    """Prepare regular download bytes once per complete experiment."""
    return build_download_bundle(prepared, result, parameters)


for parameter_key, parameter_value in DEFAULTS.items():
    st.session_state.setdefault(parameter_key, parameter_value)

with st.sidebar:
    st.markdown("### ◈ Spatial Matrix")
    st.caption("AN INTERACTIVE VISION LAB")
    st.divider()
    st.markdown("**01 / Image source**")
    source_mode = st.radio("Source", ["Sample gallery", "Upload image"], key="source_mode",
                           label_visibility="collapsed")
    max_dimension = st.select_slider(
        "Processing resolution", options=[350, 500, 700, 1000, 1400, 1600],
        value=700, key="max_dimension", help="Longest image side in pixels. Smaller images use less memory.",
    )
    st.divider()
    st.markdown("**02 / Matrix parameters**")
    st.selectbox("Parameter preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.caption("Edits apply immediately. Presets populate the controls below.")
    st.slider("φ₁ · current state", 0.0, 1.0, step=0.01, key="phi_1")
    st.slider("φ₂ · first lag", 0.0, 1.0, step=0.01, key="phi_2")
    st.slider("φ₃ · second lag", 0.0, 1.0, step=0.01, key="phi_3")
    st.slider("θ₁ · noise weight", -1.0, 1.0, step=0.01, key="theta_1")
    st.caption(f"φ sum = {sum(st.session_state[k] for k in ('phi_1', 'phi_2', 'phi_3')):.2f} · keep below 1 for stability")
    st.slider("Extension width (pixels)", 0, 256, step=1, key="extension_px")
    with st.expander("Display gains & noise seed"):
        st.slider("Edge visibility gain", 0.0, 12.0, step=0.25, key="edge_gain")
        st.slider("Integration visibility gain", 0.0, 2.0, step=0.01, key="integration_gain")
        st.slider("Noise seed", 0.0, 0.20, step=0.005, key="noise_floor", format="%.3f")
    st.button("Reset parameters", key="reset_parameters", on_click=reset_parameters, width="stretch")
    st.divider()
    st.caption("Runs on your local Python server. No account, API key, model download, or GPU required.")

st.markdown('<div class="brand"><span class="brand-dot"></span>MATRIX / VISION LAB</div>', unsafe_allow_html=True)
st.title("See what a matrix can do.")
st.markdown(
    '<p class="hero-copy">Explore a photograph through three spatial transformations. '
    'Reveal directional detail, remap intensity, and forecast the edge of the frame '
    'with transparent linear algebra.</p>'
    '<span class="tag">ARIMA-INSPIRED</span><span class="tag">EXPLICIT EQUATIONS</span>'
    '<span class="tag">LOCAL PROCESSING</span>', unsafe_allow_html=True,
)

parameters = EngineParameters(**{key: st.session_state[key] for key in DEFAULTS})
try:
    operators = build_operators(parameters)
except ValueError as error:
    st.error(str(error), icon="⚠️")
    st.info("Lower the φ sliders until their sum is below 1, or select Reset parameters.")
    st.stop()

workspace, inspector, about = st.tabs(["Workspace", "Matrix inspector", "How it works"])

with workspace:
    st.write("")
    input_col, context_col = st.columns([1.55, 1], gap="large")
    prepared = None
    with input_col:
        st.markdown('<div class="step">01 / THE SOURCE</div>', unsafe_allow_html=True)
        if source_mode == "Sample gallery":
            sample_name = st.selectbox("Sample image", DEMO_NAMES, key="demo_name")
            prepared = prepare_demo(sample_name, max_dimension)
        else:
            uploaded = st.file_uploader(
                "Drop your photograph here", type=["png", "jpg", "jpeg"], key="image_upload",
                help="PNG or JPEG · up to 15 MB and 40 million source pixels · at least 3 pixels wide.",
            )
            if uploaded is not None:
                try:
                    prepared = prepare_upload(uploaded.getvalue(), Path(uploaded.name).name, max_dimension)
                except ImageValidationError as error:
                    st.error(str(error))
            else:
                st.info("Upload a PNG or JPEG to begin, or explore the Sample gallery in the sidebar.")
        if prepared is not None:
            st.image(prepared.rgb, caption=prepared.source_name, width="stretch")
            if prepared.resized:
                st.caption(f"Resized from {prepared.original_width:,} × {prepared.original_height:,} to "
                           f"{prepared.gray.shape[1]:,} × {prepared.gray.shape[0]:,} px, preserving aspect ratio.")
    with context_col:
        st.markdown('<div class="step">A SMALL STATE. THREE PERSPECTIVES.</div>', unsafe_allow_html=True)
        st.subheader("Every pixel has a past.")
        st.write("Each row becomes a sequence. A pixel, its two left neighbours, and a constant noise seed form a four-value state.")
        st.markdown("**01 · Differentiate**  \nApply I − F to emphasize changes along the row.")
        st.markdown("**02 · Integrate**  \nApply (I − F)⁻¹ to remap local intensity through the inverse operator.")
        st.markdown("**03 · Extrapolate**  \nAdvance the final state with F to predict new pixels to the right.")
        st.caption("This is a deterministic spatial experiment. Stable forecasts tend to fade; the method does not infer missing objects.")

    if prepared is not None:
        with st.spinner("Applying spatial matrix operators…"):
            result = transform(prepared.gray, parameters)
        height, width = prepared.gray.shape
        st.write("")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Processing canvas", f"{width} × {height}")
        metric_cols[1].metric("Matrix compute", f"{result.elapsed_ms:.1f} ms")
        metric_cols[2].metric("Forecast width", f"+{parameters.extension_px} px")
        metric_cols[3].metric("Spectral radius", f"{operators.spectral_radius:.3f}")
        st.caption("Compute time measures the matrix engine only. Repeated settings reuse an in-memory cached result and its original timing.")
        st.write("")
        st.markdown('<div class="step">02 / TRANSFORMED OUTPUTS</div>', unsafe_allow_html=True)
        st.subheader("One source. Three views.")
        output_specs = [
            ("01", "Directional detail", "I − F", result.edges, "edges.png",
             "Absolute difference response, amplified for visibility. Sensitive to horizontal changes and the noise seed."),
            ("02", "Intensity baseline", "(I − F)⁻¹", result.baseline, "baseline.png",
             "Inverse-operator response with a display gain. It mixes adjacent values; denoising is not guaranteed."),
            ("03", "Spatial forecast", "Fᵏ", result.extension, "extension.png",
             f"Original grayscale canvas plus {parameters.extension_px} predicted columns on the right."),
        ]
        for column, (number, title, equation, pixels, filename, description) in zip(st.columns(3, gap="medium"), output_specs):
            with column:
                with st.container(border=True):
                    st.caption(f"{number} / {equation}")
                    st.markdown(f"#### {title}")
                    st.image(pixels, width="stretch", clamp=True)
                    st.caption(description)
                    st.download_button("Download PNG", encode_png(pixels), filename, "image/png",
                                       key=f"download_{number}", width="stretch", on_click="ignore")
        st.write("")
        bundle_col, note_col = st.columns([1, 2])
        with bundle_col:
            st.download_button(
                "↓ Download experiment bundle", export_bundle(prepared, result, parameters),
                "spatial-matrix-experiment.zip", "application/zip", type="primary", width="stretch",
                key="download_bundle", on_click="ignore",
            )
        with note_col:
            st.caption("Includes the processed color source, grayscale input, all three PNG outputs, NumPy matrices, and a JSON experiment manifest.")
        with st.expander("Compare grayscale input & pixel responses"):
            left, right = st.columns([1, 1.5], gap="large")
            with left:
                st.image(prepared.gray, caption="Grayscale input at processing resolution", width="stretch")
                st.download_button("Download grayscale input", encode_png(prepared.gray), "grayscale.png",
                                   "image/png", on_click="ignore")
            with right:
                row = st.slider("Inspect row", 0, height - 1, height // 2, key="profile_row") if height > 1 else 0
                st.line_chart({"Input": prepared.gray[row], "Detail": result.edges[row],
                               "Baseline": result.baseline[row]}, x_label="Pixel column", y_label="Intensity (0–255)")
                st.caption("The chart shows display-ready 8-bit values, including gain and clipping.")

with inspector:
    st.write("")
    st.subheader("The equations behind the image")
    st.write("These are the exact operators used by the current settings. Every transformation is inspectable.")
    st.latex(r"X_{r,c}=\begin{bmatrix}p_{r,c}&p_{r,c-1}&p_{r,c-2}&\epsilon\end{bmatrix}^{T},\qquad p=\mathrm{gray}/255")
    matrix_columns = st.columns(3)
    for column, name, matrix in zip(matrix_columns, ["Transition · F", "Difference · I − F", "Inverse · (I − F)⁻¹"],
                                    [operators.transition, operators.difference, operators.integration]):
        with column:
            st.markdown(f"**{name}**")
            st.dataframe(np.round(matrix, 4), width="stretch")
    st.latex(r"E_{r,c}=255g_e\left|e_1^T(I-F)X_{r,c}\right|,\qquad B_{r,c}=255g_i e_1^T(I-F)^{-1}X_{r,c}")
    st.latex(r"s_0=[p_{r,W-1},p_{r,W-2},p_{r,W-3},0]^T,\qquad \hat p_{r,W+k-1}=e_1^T F^k s_0\quad(k\geq1)")
    st.caption("Values are clipped to [0, 255] and converted to uint8. Missing left neighbours repeat the first pixel; the source portion of the forecast is preserved exactly.")
    diagnostics = st.columns(2)
    diagnostics[0].metric("ρ(F) · must be < 1", f"{operators.spectral_radius:.6f}")
    diagnostics[1].metric("Condition number of I − F", f"{operators.condition_number:.2f}")
    st.caption("Forecast stability requires ρ(F) < 1. The inverse is rejected if its condition number exceeds 10⁸.")
    with st.expander("Current experiment parameters", expanded=False):
        st.json(asdict(parameters))

with about:
    st.write("")
    st.subheader("Computer vision, through linear algebra.")
    st.write("Spatial Matrix Vision Lab turns the supplied ARIMA-inspired prototype into a local, interactive image-processing workspace. It uses fixed coefficients and four-dimensional states to transform grayscale images, with no training phase.")
    about_left, about_right = st.columns(2, gap="large")
    with about_left:
        st.markdown("#### The processing pipeline")
        st.markdown("1. **Prepare.** Validate PNG/JPEG data, correct EXIF orientation, flatten transparency on white, and resize without upscaling.\n2. **Normalize.** Convert RGB to grayscale with OpenCV and scale pixels to [0, 1].\n3. **Transform.** Construct F, I − F, and its inverse. Apply the operators to each spatial state using vectorized NumPy operations.\n4. **Forecast.** Step the final three-pixel state through F, once per new column.\n5. **Explore.** Compare outputs, inspect matrices, and export the experiment.")
    with about_right:
        st.markdown("#### What the results mean")
        st.write("The difference view responds to changes along rows. The inverse view remaps adjacent intensities. The forecast uses recent edge intensity to predict a continuation that typically decays toward black under stable coefficients.")
        st.write("The blueprint calls this ARIMA(3,1,1). This implementation follows its specified matrix equations; it does not fit a statistical ARIMA model, estimate innovations, or implement a full differenced time-series pipeline. The inverse response is not a general illumination-restoration or denoising algorithm.")
        st.markdown("#### Local by design")
        st.write("Your browser talks to the local Streamlit Python server. Uploads and result caches stay in server memory; this app does not write uploads to disk or send them to external services. Downloads are saved only when you choose them.")
    st.info("Start with a sample, tune one coefficient at a time, then inspect the matrices. For longer forecasts, expect fading rather than semantic scene completion.")

st.markdown('<div class="footer">SPATIAL MATRIX VISION LAB &nbsp; / &nbsp; Python · Streamlit · OpenCV · NumPy &nbsp; / &nbsp; Built for exploration.</div>', unsafe_allow_html=True)
