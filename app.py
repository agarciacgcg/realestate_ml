# app.py
import os
import sys
import platform
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(
    page_title="Odyyn's Raven - JoCo Home Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# -------------------------------
# Brand / header
# -------------------------------
logo_path = Path("odyyn copy.png")
if logo_path.exists():
    st.image(str(logo_path), width=200)

# -------------------------------
# Hide Streamlit chrome
# -------------------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Custom CSS (fixed fonts/selectors)
# -------------------------------
odyyn_primary = "#c10604"
odyyn_secondary = "#04c9c8"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@400;500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Ubuntu', ubuntu, sans-serif;
    color: #333;
}}

h1 {{ color: #800000; }}
h2, h3 {{ color: {odyyn_secondary}; }}

div.stForm > div {{
    background-color: #fdf5e6;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
}}

div.stForm button[kind="primary"] {{
    background-color: {odyyn_primary};
    color: white;
    border-radius: 6px;
    border: none;
}}

div.stForm button[kind="primary"]:hover {{
    background-color: {odyyn_secondary};
    color: black;
}}
.help {{
    font-size: 0.85em;
    color: {odyyn_primary};
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title and intro
# -------------------------------
st.title("Johnson County Home Price Predictor")

with st.expander("Disclaimer"):
    st.write("""
This model is a simplified learning demo. Do not use it as the sole basis for
financial or real-estate decisions. Always consult qualified professionals.
Use at your own risk.
""")

with st.expander("How to use this app"):
    st.write("""
Adjust the property inputs, then click Predict Price to get an estimate. Fields
include bedrooms, bathrooms, lot size, home size, local demographics, city and metro.
The underlying model was trained on historical data and may not reflect current market conditions.
""")

# -------------------------------
# Helpers for robust model loading
# -------------------------------
def _maybe_import_custom_module():
    """
    If you trained with custom transformers, set CUSTOM_CLASSES_MODULE env var
    to a module name (on sys.path) where those classes/functions are defined.
    Example:
      export CUSTOM_CLASSES_MODULE="model_custom"
    """
    modname = os.getenv("CUSTOM_CLASSES_MODULE")
    if not modname:
        return None
    try:
        return importlib.import_module(modname)
    except Exception:
        # Surface a non-blocking warning; model load might still succeed
        st.sidebar.warning(f"Could not import CUSTOM_CLASSES_MODULE '{modname}'.")
        return None

def _resolve_model_path():
    """
    Priority:
      1) st.secrets["model_path"]
      2) env MODEL_PATH
      3) local files: joco_rf_pipeline.skops, then joco_rf_pipeline.joblib
    """
    secrets_path = st.secrets.get("model_path") if hasattr(st, "secrets") else None
    env_path = os.getenv("MODEL_PATH")
    candidates = [secrets_path, env_path, "joco_rf_pipeline.skops", "joco_rf_pipeline.joblib"]
    for p in candidates:
        if p and Path(p).exists():
            return str(p)
    return None

def _env_diagnostics():
    import sklearn
    return dict(
        python=sys.version.replace("\n", " "),
        platform=platform.platform(),
        sklearn=sklearn.__version__,
        pandas=pd.__version__,
        numpy=np.__version__,
        joblib=joblib.__version__,
    )

# -------------------------------
# Load model (joblib with fallback to skops)
# -------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    _maybe_import_custom_module()  # no-op if not set
    model_path = _resolve_model_path()
    if not model_path:
        st.error(
            "No model file found. Place joco_rf_pipeline.skops or joco_rf_pipeline.joblib "
            "in the app directory, or set MODEL_PATH / secrets['model_path']."
        )
        st.stop()

    # Prefer skops for portability if extension is .skops
    path = Path(model_path)
    if path.suffix.lower() == ".skops":
        try:
            from skops import io as skio
        except Exception:
            st.error(
                "Model is in .skops format but 'skops' is not installed. "
                "Add 'skops' to requirements.txt and redeploy."
            )
            st.stop()
        try:
            return skio.load(str(path))
        except Exception as e:
            st.error("Failed to load .skops model.")
            st.exception(e)
            st.stop()

    # Else try joblib
    try:
        return joblib.load(str(path))
    except AttributeError as e:
        st.error(
            "The saved model references a class/function that isn't importable in this environment. "
            "If you trained with custom transformers, set CUSTOM_CLASSES_MODULE to the module where "
            "they are defined, or re-save the model without lambdas / notebook-local classes."
        )
        st.code("\n".join([f"{k}: {v}" for k, v in _env_diagnostics().items()]))
        st.exception(e)
        st.stop()
    except Exception as e:
        st.error("Failed to load joblib model.")
        st.code("\n".join([f"{k}: {v}" for k, v in _env_diagnostics().items()]))
        st.exception(e)
        st.stop()

model = load_model()

# -------------------------------
# Input form
# -------------------------------
st.markdown("## Enter Property Details")
with st.form("prediction_form"):
    bed = st.slider('Bedrooms (max 6)', 1, 6, 3, help="Total bedrooms")
    bath = st.slider('Bathrooms (max 5)', 1, 5, 2, help="Total bathrooms")

    acre_lot = st.number_input(
        'Lot size (acres, max 5)',
        min_value=0.0, max_value=5.0, value=0.2, step=0.01,
        help="1 acre = 43,560 sq ft"
    )
    house_size = st.number_input(
        'House size (sq ft, max 10,000)',
        min_value=300, max_value=10000, value=1500, step=50,
        help="Average home ~1,500 sq ft"
    )

    population = st.number_input(
        'Population (max 2,000,000)', min_value=1000, max_value=2_000_000, value=50_000, step=1000
    )
    median_income = st.number_input(
        'Median income (USD, max 1,000,000)', min_value=20_000, max_value=1_000_000, value=80_000, step=1_000
    )
    pct_bachelor = st.slider(
        'Pct. Bachelor+',
        min_value=0.0, max_value=100.0, value=40.0, step=0.5,
        help="Share of population with a bachelor’s degree or higher (percent)"
    )
    num_schools = st.slider(
        'Number of Public Schools', min_value=0, max_value=50, value=10
    )

    City = st.selectbox('City', options=['Overland Park', 'Olathe', 'Shawnee', 'Leawood', 'Lenexa'])
    Metro = st.selectbox('Metro Area', options=['Kansas City'])

    submit = st.form_submit_button("Predict Price", type="primary")

# -------------------------------
# Build features and predict
# -------------------------------
def build_input_df():
    # Derived features (keep consistent with training)
    log_house_size = float(np.log(max(house_size, 1)))  # guard log(0)
    size_income = float(house_size) * float(median_income)

    features = {
        'bed': bed,
        'bath': bath,
        'acre_lot': float(acre_lot),
        'house_size': int(house_size),
        'log_house_size': log_house_size,
        'population': int(population),
        'median_income': int(median_income),
        'pct_bachelor_plus': float(pct_bachelor),
        'num_public_schools': int(num_schools),
        'City': City,
        'Metro': Metro,
        'size_income': size_income,
    }
    return pd.DataFrame([features])

def expected_input_columns(trained_model, fallback_cols):
    # Try common places where sklearn stores raw input feature names
    # 1) named step 'preprocess' (e.g., ColumnTransformer)
    pre = None
    if hasattr(trained_model, "named_steps"):
        pre = trained_model.named_steps.get("preprocess")
    if pre is not None and hasattr(pre, "feature_names_in_"):
        return list(pre.feature_names_in_)
    # 2) pipeline / estimator feature_names_in_
    if hasattr(trained_model, "feature_names_in_"):
        return list(trained_model.feature_names_in_)
    # 3) give up: use what we built now
    return list(fallback_cols)

if submit:
    input_df = build_input_df()
    exp_cols = expected_input_columns(model, input_df.columns)

    # Ensure all expected columns exist; fill missing numerics with 0, strings with empty
    for col in exp_cols:
        if col not in input_df.columns:
            # naive fill heuristic; customize if your training expects different defaults
            input_df[col] = 0 if col not in ("City", "Metro") else ""
    # Reorder to match training
    input_df = input_df.reindex(columns=exp_cols)

    # Predict
    try:
        pred_log = float(model.predict(input_df)[0])
        pred_price = float(np.exp(pred_log))
        # Basic sanity clamp for display only
        if not np.isfinite(pred_price) or pred_price < 0:
            raise ValueError("Invalid prediction value.")
        st.metric(label="Predicted Price (USD)", value=f"${pred_price:,.2f}")
    except Exception as e:
        st.error("Prediction failed. See details below.")
        st.exception(e)

# -------------------------------
# Additional info panes
# -------------------------------
with st.expander("Why this matters"):
    st.write("""
• Faster, data-driven valuations — produce evidence-based estimates in seconds.  
• Accuracy compounds — models can improve as you add more recent, high-quality data.  
• Smarter risk management — blend property and demographic signals to flag mispricing early.  
• Differentiation — offering instant estimates positions you as tech-forward.
""")

with st.sidebar:
    st.subheader("Environment")
    diag = _env_diagnostics()
    st.code("\n".join([f"{k}: {v}" for k, v in diag.items()]))
    st.caption("If loading fails with AttributeError, set CUSTOM_CLASSES_MODULE to the module where any custom transformers are defined, or re-export the model using named, importable functions/classes (no lambdas).")
