import base64
import math
import gzip
import os
import pickle

try:
    import anthropic
except ImportError:
    anthropic = None
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components  # type: ignore[import-untyped]
from models import KNNRegressor, MLPRegressor, RidgeRegression  # noqa: F401 — required for pickle

st.set_page_config(
    page_title="NutriReceipt",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,700;1,9..144,400&display=swap');
:root {
  --accent:       #2D8C5A;
  --accent-dark:  #1F6B42;
  --accent-light: #E8F5ED;
  --warm:         #F2A541;
  --warm-light:   #FFF5E6;
  --border:       #E8E8E4;
  --text:         #1A1A1A;
  --text-sec:     #5A5A5A;
  --bg:           #FAFAF7;
  --surface:      #FFFFFF;
  --radius:       14px;
}
html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif !important; }
.stApp                      { background: var(--bg) !important; }
[data-testid="stSidebar"]   { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
#MainMenu, footer, header   { visibility: hidden; }
.stDeployButton             { display: none; }

/* Buttons */
.stButton > button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  padding: 10px 24px !important;
  transition: background 0.15s !important;
}
.stButton > button:hover { background: var(--accent-dark) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"]  { border-bottom: 2px solid var(--border) !important; gap: 4px !important; padding: 0 4px !important; }
.stTabs [data-baseweb="tab"]       { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; font-size: 13px !important; color: #1A1A1A !important; border-radius: 8px 8px 0 0 !important; padding: 10px 18px !important; }
.stTabs [aria-selected="true"]     { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: var(--accent-light) !important; }
.stTabs [data-baseweb="tab"]:hover { background: #F0F0EC !important; color: #1A1A1A !important; }
.stTabs [data-baseweb="tab"] p     { color: #1A1A1A !important; }

/* Sliders */
.stSlider [role="slider"] { background: var(--accent) !important; }

/* Cards */
.nr-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px 32px;
  margin-bottom: 20px;
}
.nr-badge {
  display: inline-flex; align-items: center;
  background: var(--accent-light); color: var(--accent);
  padding: 5px 14px; border-radius: 100px;
  font-size: 12px; font-weight: 600; letter-spacing: 0.02em;
  margin-bottom: 16px;
}
.nr-section-label {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.12em;
  color: var(--accent); font-weight: 600; margin-bottom: 8px;
}
.nr-section-title {
  font-family: 'Fraunces', serif; font-size: 22px; font-weight: 600;
  letter-spacing: -0.02em; color: var(--text); margin-bottom: 20px;
}

/* Hide sidebar entirely */
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Reduce Streamlit's default top padding to match side padding */
[data-testid="stMainBlockContainer"] { padding-top: 1rem !important; }

/* Top navbar */
.nr-navbar {
  display: flex; align-items: center; justify-content: space-between;
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 12px 32px; margin: -1rem -1rem 32px -1rem;
  position: sticky; top: 0; z-index: 999;
}
.nr-navbar-logo {
  display: flex; align-items: center; gap: 10px;
}
.nr-navbar-logo span {
  font-family: 'Fraunces', serif; font-weight: 700; font-size: 18px;
  letter-spacing: -0.02em; color: #1A1A1A;
}
.nr-navbar-links { display: flex; align-items: center; gap: 4px; }

/* Nav link buttons (type=secondary) */
button[kind="secondary"] {
  background: transparent !important;
  color: #1B3D2A !important;
  border: 1px solid #C5D4C9 !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 6px 8px !important;
  border-radius: 8px !important;
  transition: background 0.15s !important;
  white-space: nowrap !important;
}
button[kind="secondary"]:hover { background: #E8F5ED !important; }

/* Bordered container (data card) */
[data-testid="stBorderContainer"] {
  border-color: #E8E8E4 !important;
  border-radius: 14px !important;
  background: #FFFFFF !important;
  padding: 28px 32px !important;
}

/* Placeholder shimmer bars */
.nr-placeholder-bar {
  background: linear-gradient(90deg,#F0F0EC 25%,#E8E8E4 50%,#F0F0EC 75%);
  background-size: 200% 100%;
  animation: shimmer 1.6s infinite;
  border-radius: 6px;
}
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
@keyframes ocr-float {
  0%, 100% { transform: translateY(0); }
  50%       { transform: translateY(-10px); }
}
@keyframes ocr-bounce {
  0%, 80%, 100% { transform: translateY(0);    opacity: 0.35; }
  40%           { transform: translateY(-8px); opacity: 1; }
}

/* Page footer */
.nr-footer {
  margin-top: 64px; padding: 20px 0; border-top: 1px solid var(--border);
  font-size: 12px; color: #2D8C5A; line-height: 1.6; text-align: center;
}

/* File uploader drop zone */
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed #C5D4C9 !important;
  border-radius: 14px !important;
  background: #FAFAF7 !important;
  min-height: 220px !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: flex-end !important;
  padding: 4px !important;
  text-align: center !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] p {
  color: #1A1A1A !important;
  font-weight: 500 !important;
}
[data-testid="stFileUploaderFile"] { display: none !important; }

.nr-stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
}
.nr-stat-card {
  background: #FAFAF7;
  border: 1px solid #ECECE8;
  border-radius: 12px;
  padding: 14px 16px;
}
.nr-stat-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #72726C;
  margin-bottom: 6px;
}
.nr-stat-value {
  font-family: 'Fraunces', serif;
  font-size: 24px;
  font-weight: 600;
  color: #1A1A1A;
  line-height: 1.1;
}
.nr-stat-note {
  font-size: 11px;
  color: #66665F;
  margin-top: 6px;
  line-height: 1.4;
}
.nr-callout {
  background: #F6FBF8;
  border: 1px solid #DDEEE4;
  border-radius: 14px;
  padding: 16px 18px;
  color: #295B3E;
  font-size: 13px;
  line-height: 1.55;
}
.nr-callout strong {
  color: #1F6B42;
}
/* Home page CTA button */
.nr-cta-wrap > div > button {
  font-size: 18px !important;
  font-weight: 700 !important;
  padding: 18px 20px !important;
  border-radius: 14px !important;
  letter-spacing: 0.01em !important;
  box-shadow: 0 6px 28px rgba(45,140,90,0.4) !important;
  transition: box-shadow 0.2s, transform 0.15s !important;
}
.nr-cta-wrap > div > button:hover {
  box-shadow: 0 10px 40px rgba(45,140,90,0.55) !important;
  transform: translateY(-2px) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Model bundle ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "basket_size", "total_calories", "produce_share", "processed_share",
    "fiber_density", "protein_density", "sugar_density", "sodium_density",
    "dept_diversity", "beverage_share",
]

SAMPLE_ITEMS = [
    {"name": "Organic Bananas", "qty": 1},
    {"name": "Baby Spinach 5oz", "qty": 1},
    {"name": "Whole Wheat Bread", "qty": 1},
    {"name": "Greek Yogurt Plain", "qty": 2},
    {"name": "Chicken Breast Boneless", "qty": 1},
    {"name": "Broccoli Florets", "qty": 1},
    {"name": "Cheddar Cheese Block", "qty": 1},
    {"name": "Oat Milk Original", "qty": 1},
    {"name": "Frozen Blueberries 12oz", "qty": 1},
    {"name": "Brown Rice 2lb", "qty": 1},
    {"name": "Tortilla Chips", "qty": 1},
    {"name": "Apple Juice 64oz", "qty": 1},
]


class BundleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            custom_classes = {
                "RidgeRegression": RidgeRegression,
                "KNNRegressor": KNNRegressor,
                "MLPRegressor": MLPRegressor,
            }
            if name in custom_classes:
                return custom_classes[name]
        return super().find_class(module, name)


def _try_load_pkl(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return BundleUnpickler(f).load()
    else:
        with open(path, "rb") as f:
            return BundleUnpickler(f).load()


@st.cache_resource
def load_bundle():
    candidates = ["model_bundle.pkl", "model_bundle.pkl.gz"]
    data = None
    last_err = "No model bundle file found. Place model_bundle.pkl next to app.py."

    for path in candidates:
        try:
            data = _try_load_pkl(path)
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            last_err = str(e)

    if data is None:
        return None, last_err

    # If the bundle was accidentally saved as a file-path string, try loading that path
    if isinstance(data, str) and data.endswith(".pkl"):
        import os
        local = os.path.basename(data)
        for attempt in (local, data):
            try:
                data = _try_load_pkl(attempt)
                break
            except Exception:
                continue
        if isinstance(data, str):
            return None, (
                f"model_bundle.pkl.gz contains a path string ({data!r}) instead of "
                "the actual bundle. Place model_bundle.pkl next to app.py."
            )

    if not isinstance(data, dict):
        return None, (
            f"Bundle loaded as {type(data).__name__}, expected dict. "
            "Re-export model_bundle.pkl from the notebook."
        )
    missing = [k for k in ("train_mean", "train_std", "mlp", "ridge", "knn") if k not in data]
    if missing:
        return None, (
            f"Bundle is missing keys: {missing}. "
            f"Keys found: {list(data.keys())}"
        )
    return data, None


bundle, bundle_error = load_bundle()

MODEL_ORDER = ["knn", "ridge", "mlp"]
MIDPOINT_DEFAULT_MODEL = "knn"
MODEL_LABELS = {"knn": "KNN", "ridge": "Ridge", "mlp": "MLP"}
MODEL_EMOJI = {"knn": "📍", "ridge": "📈", "mlp": "🧠"}
DEFAULT_RESULTS_DF = pd.DataFrame(
    {
        "Model": ["Ridge Regression", "KNN (k=11)", "MLP Regressor"],
        "RMSE": [2.6367, 1.7143, 0.1807],
        "MAE": [2.0015, 1.2757, 0.0673],
        "R²": [0.9775, 0.9905, 0.9999],
        "Train Time (s)": [0.0079, 0.0012, 44.3746],
        "Inference Time (s)": [0.0005, 6.6384, 0.0138],
    }
)


def model_key_from_label(label: str) -> str:
    low = label.lower()
    if "knn" in low:
        return "knn"
    if "ridge" in low:
        return "ridge"
    return "mlp"


def get_model_results_df() -> pd.DataFrame:
    if bundle and "results" in bundle:
        try:
            df = pd.DataFrame(bundle["results"])
            if "Model" in df.columns and "RMSE" in df.columns:
                df = df.copy()
                df["model_key"] = df["Model"].map(model_key_from_label)
                return df
        except Exception:
            pass

    df = DEFAULT_RESULTS_DF.copy()
    df["model_key"] = df["Model"].map(model_key_from_label)
    return df


MODEL_RESULTS_DF = get_model_results_df()


def get_model_metric(model_key: str, column: str) -> float | None:
    match = MODEL_RESULTS_DF.loc[MODEL_RESULTS_DF["model_key"] == model_key, column]
    if match.empty:
        return None
    try:
        return float(match.iloc[0])
    except Exception:
        return None


def get_model_label(model_key: str) -> str:
    match = MODEL_RESULTS_DF.loc[MODEL_RESULTS_DF["model_key"] == model_key, "Model"]
    if not match.empty:
        return str(match.iloc[0])
    return MODEL_LABELS[model_key]


def get_model_picker_label(model_key: str) -> str:
    if model_key == "knn":
        best_k = bundle.get("best_k", 11) if bundle else 11
        return f"{MODEL_EMOJI[model_key]}  KNN Default (k={best_k})"
    if model_key == "ridge":
        best_alpha = bundle.get("best_alpha", 1000) if bundle else 1000
        return f"{MODEL_EMOJI[model_key]}  Ridge (α={best_alpha:.0f})"
    hidden_dim = 64
    if bundle and isinstance(bundle.get("best_mlp_cfg"), dict):
        hidden_dim = bundle["best_mlp_cfg"].get("hidden_dim", hidden_dim)
    return f"{MODEL_EMOJI[model_key]}  MLP ({hidden_dim} hidden)"


def get_model_summary_text(model_key: str) -> str:
    summary = []
    if model_key == MIDPOINT_DEFAULT_MODEL:
        summary.append("midpoint-selected deployment default")
    rmse = get_model_metric(model_key, "RMSE")
    r2 = get_model_metric(model_key, "R²")
    if rmse is not None:
        summary.append(f"saved test RMSE {rmse:.2f}")
    if r2 is not None:
        summary.append(f"R² {r2:.4f}")
    return " · ".join(summary)


def predict_phs(features: dict, model_name: str) -> float | None:
    if bundle is None:
        return None
    feature_cols = bundle.get("feature_cols", FEATURE_COLS)
    x = np.array([features[f] for f in feature_cols], dtype=float)
    mean = np.array(bundle["train_mean"], dtype=float)
    std = np.array(bundle["train_std"], dtype=float)
    std = np.where(std == 0, 1.0, std)
    x_norm = (x - mean) / std
    pred = bundle[model_name].predict(x_norm.reshape(1, -1))
    return float(np.clip(np.array(pred).flatten()[0], 0, 100))


def predict_all_models(features: dict) -> dict[str, float]:
    if bundle is None:
        return {}
    return {
        model_key: score
        for model_key in MODEL_ORDER
        if model_key in bundle
        for score in [predict_phs(features, model_key)]
        if score is not None
    }


# ── Visual helpers ────────────────────────────────────────────────────────────
def tier(score: float) -> tuple[str, str, str]:
    if score >= 80: return "Excellent", "#2D8C5A", "#E8F5ED"
    if score >= 65: return "Good",      "#2D8C5A", "#E8F5ED"
    if score >= 50: return "Fair",      "#F2A541", "#FFF5E6"
    return              "Needs Work",   "#D9534F", "#FDECEA"


def score_ring(score: float, size: int = 160) -> str:
    r = 54
    circ   = 2 * math.pi * r
    offset = circ * (1 - score / 100)
    label, color, _ = tier(score)
    return f"""
<div style="position:relative;width:{size}px;height:{size}px;margin:0 auto 8px;">
  <svg width="{size}" height="{size}" viewBox="0 0 140 140" style="transform:rotate(-90deg)">
    <circle cx="70" cy="70" r="{r}" fill="none" stroke="#E8E8E4" stroke-width="10"/>
    <circle cx="70" cy="70" r="{r}" fill="none" stroke="{color}" stroke-width="10"
            stroke-linecap="round"
            stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}"/>
  </svg>
  <div style="position:absolute;inset:0;display:flex;flex-direction:column;
              align-items:center;justify-content:center;">
    <span style="font-family:'Fraunces',serif;font-size:34px;font-weight:700;
                 color:{color};line-height:1;">{score:.0f}</span>
    <span style="font-size:11px;font-weight:600;color:{color};
                 text-transform:uppercase;letter-spacing:0.08em;margin-top:2px;">{label}</span>
  </div>
</div>"""


def factor_row(icon: str, name: str, value: float, max_val: float, direction: str) -> str:
    pct   = min(abs(value) / max_val * 100, 100) if max_val else 0
    color = "#2D8C5A" if direction == "pos" else "#F2A541"
    sign  = "+" if direction == "pos" else "−"
    return f"""
<div style="display:flex;align-items:center;gap:12px;padding:10px 0;
            border-bottom:1px solid #F4F4F0;font-size:13px;">
  <span style="font-size:16px;width:24px;text-align:center;">{icon}</span>
  <span style="width:130px;font-weight:500;color:#1A1A1A;">{name}</span>
  <div style="flex:1;height:8px;background:#F0F0EC;border-radius:4px;overflow:hidden;">
    <div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:4px;"></div>
  </div>
  <span style="width:48px;text-align:right;font-weight:700;color:{color};">{sign}{abs(value):.2f}</span>
</div>"""


FACTOR_META = [
    ("🥦", "Produce Share",   "produce_share",   "pos", 1.0),
    ("🌾", "Fiber Density",   "fiber_density",   "pos", 10.0),
    ("🥩", "Protein Density", "protein_density", "pos", 30.0),
    ("🍬", "Sugar Density",   "sugar_density",   "neg", 30.0),
    ("🧂", "Sodium Density",  "sodium_density",  "neg", 500.0),
    ("📦", "Processed Share", "processed_share", "neg", 1.0),
]


# ── Paste-items heuristic ─────────────────────────────────────────────────────
PRODUCE_KW = {
    "apple","banana","orange","grape","tomato","lettuce","spinach","kale","broccoli",
    "carrot","onion","garlic","pepper","cucumber","zucchini","avocado","berry","berries",
    "fruit","vegetable","veggie","salad","herb","lemon","lime","mango","peach","plum",
    "pear","melon","strawberry","blueberry","raspberry","cherry","apricot","celery",
    "asparagus","beet","cabbage","cauliflower","ginger","mushroom","potato","corn",
    "pineapple","artichoke","eggplant","leek","radish","arugula","basil","cilantro",
    "parsley","mint","dill","rosemary","thyme","oregano","sweet potato",
}
PROCESSED_KW = {
    "chips","cookie","cracker","candy","cake","pastry","cereal","frozen","instant",
    "snack","bar","pizza","sausage","bacon","hot dog","deli","margarine","spread",
    "syrup","jam","jelly","sauce","ketchup","mayo","mayonnaise","ranch","dressing",
    "canned","condensed","pudding","gelatin","gummy","cheeto","dorito","pretzel",
    "popcorn","granola bar","protein bar","energy bar","ramen","noodle","mac",
    "macaroni","salami","pepperoni","luncheon",
}
BEVERAGE_KW = {
    "juice","soda","water","milk","coffee","tea","beer","wine","drink","beverage",
    "smoothie","lemonade","cider","kombucha","sparkling","almond milk","oat milk",
    "coconut water","sports drink","energy drink","cola","gatorade","powerade",
    "monster","redbull","snapple","vitamin water",
}
DEPT_MAP = {
    "meat":   {"chicken","beef","pork","fish","salmon","tuna","shrimp","turkey","lamb","meat"},
    "dairy":  {"yogurt","cheese","butter","cream","egg","dairy"},
    "bakery": {"bread","bagel","muffin","roll","bun","wrap","tortilla"},
    "grains": {"pasta","rice","quinoa","oat","flour","grain","lentil","bean"},
}


def items_to_features(text: str) -> dict:
    lines = [l.strip().lower() for l in text.strip().splitlines() if l.strip()]
    n = max(len(lines), 1)

    produce   = sum(1 for l in lines if any(k in l for k in PRODUCE_KW))
    processed = sum(1 for l in lines if any(k in l for k in PROCESSED_KW))
    beverages = sum(1 for l in lines if any(k in l for k in BEVERAGE_KW))
    other     = max(n - produce - processed - beverages, 0)

    # Rough per-item averages by category
    total_calories = produce * 40  + processed * 200 + beverages * 60  + other * 120
    total_fiber    = produce * 2.0 + processed * 0.5 + beverages * 0.1 + other * 1.0
    total_protein  = produce * 1.0 + processed * 3.0 + beverages * 0.5 + other * 5.0
    total_sugar    = produce * 5.0 + processed * 8.0 + beverages * 10  + other * 4.0
    total_sodium   = produce * 15  + processed * 350 + beverages * 80  + other * 180

    depts = set()
    for l in lines:
        if any(k in l for k in PRODUCE_KW):   depts.add("produce")
        if any(k in l for k in BEVERAGE_KW):  depts.add("beverages")
        if any(k in l for k in PROCESSED_KW): depts.add("snacks")
        for dept, kws in DEPT_MAP.items():
            if any(k in l for k in kws):       depts.add(dept)
    if not depts:
        depts.add("misc")

    return {
        "basket_size":      n,
        "total_calories":   total_calories,
        "produce_share":    produce   / n,
        "processed_share":  processed / n,
        "fiber_density":    total_fiber   / n,
        "protein_density":  total_protein / n,
        "sugar_density":    total_sugar   / n,
        "sodium_density":   total_sodium  / n,
        "dept_diversity":   len(depts) / n,
        "beverage_share":   beverages / n,
    }


def compute_phs_components(features: dict) -> list[dict]:
    components = [
        {
            "label": "Produce Share",
            "score": np.clip(features["produce_share"] / 0.5, 0, 1) * 25,
            "max_score": 25,
            "note": "Higher fruit and vegetable share raises the baseline.",
        },
        {
            "label": "Fiber Density",
            "score": np.clip(features["fiber_density"] / 3.0, 0, 1) * 20,
            "max_score": 20,
            "note": "Higher fiber per item adds points.",
        },
        {
            "label": "Sugar Density",
            "score": (1 - np.clip(features["sugar_density"] / 20.0, 0, 1)) * 20,
            "max_score": 20,
            "note": "Lower sugar density protects the score.",
        },
        {
            "label": "Sodium Density",
            "score": (1 - np.clip(features["sodium_density"] / 500.0, 0, 1)) * 15,
            "max_score": 15,
            "note": "Lower sodium density keeps more points.",
        },
        {
            "label": "Processed Share",
            "score": (1 - np.clip(features["processed_share"] / 0.6, 0, 1)) * 20,
            "max_score": 20,
            "note": "Fewer processed items improve the basket baseline.",
        },
    ]
    return components


def stat_card(label: str, value: str, note: str) -> str:
    return f"""
<div class="nr-stat-card">
  <div class="nr-stat-label">{label}</div>
  <div class="nr-stat-value">{value}</div>
  <div class="nr-stat-note">{note}</div>
</div>"""


def render_feature_snapshot(features: dict, source_label: str) -> None:
    dept_count = max(1, round(features["dept_diversity"] * features["basket_size"]))
    cards_html = "".join(
        [
            stat_card("Basket Size", f"{int(features['basket_size'])}", "estimated grocery items"),
            stat_card("Produce Share", f"{features['produce_share'] * 100:.0f}%", "higher usually helps"),
            stat_card("Processed Share", f"{features['processed_share'] * 100:.0f}%", "lower usually helps"),
            stat_card("Fiber Density", f"{features['fiber_density']:.1f}", "grams per item"),
            stat_card("Protein Density", f"{features['protein_density']:.1f}", "grams per item"),
            stat_card("Sugar Density", f"{features['sugar_density']:.1f}", "grams per item"),
            stat_card("Sodium Density", f"{features['sodium_density']:.0f}", "mg per item"),
            stat_card("Department Mix", f"{dept_count}", "distinct grocery groupings"),
        ]
    )
    st.markdown(
        f"""
<div class="nr-card" style="margin-top:18px;">
  <div class="nr-section-label">Feature Preview</div>
  <div class="nr-section-title" style="margin-bottom:10px;">Estimated basket snapshot</div>
  <p style="font-size:13px;color:#5A5A5A;margin-bottom:18px;">
    Based on {source_label}. These engineered features are what the saved models use for prediction.
  </p>
  <div class="nr-stat-grid">{cards_html}</div>
</div>""",
        unsafe_allow_html=True,
    )


def _model_picker() -> str:
    """Render centered model-selector buttons; returns current model_choice."""
    if "model_choice" not in st.session_state:
        st.session_state["model_choice"] = MIDPOINT_DEFAULT_MODEL
    mc = st.session_state["model_choice"]
    st.markdown(
        '<div class="nr-section-label" style="text-align:center;">Prediction Model</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(
            get_model_picker_label("knn"),
            key="model_knn",
            type="primary" if mc == "knn" else "secondary",
            width='stretch',
        ):
            st.session_state["model_choice"] = "knn"
            st.rerun()
    with c2:
        if st.button(
            get_model_picker_label("ridge"),
            key="model_ridge",
            type="primary" if mc == "ridge" else "secondary",
            width='stretch',
        ):
            st.session_state["model_choice"] = "ridge"
            st.rerun()
    with c3:
        if st.button(
            get_model_picker_label("mlp"),
            key="model_mlp",
            type="primary" if mc == "mlp" else "secondary",
            width='stretch',
        ):
            st.session_state["model_choice"] = "mlp"
            st.rerun()
    st.caption(f"{MODEL_LABELS[mc]} selected: {get_model_summary_text(mc)}.")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    return st.session_state["model_choice"]


# ── Anthropic receipt OCR ─────────────────────────────────────────────────────
def get_anthropic_key() -> str | None:
    if anthropic is None:
        return None
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


@st.cache_data(show_spinner=False)
def extract_items_from_receipt(image_bytes: bytes, media_type: str, api_key: str) -> str:
    if anthropic is None:
        raise RuntimeError("The optional `anthropic` package is not installed.")
    client = anthropic.Anthropic(api_key=api_key)
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=(
            "You are a grocery receipt parser. "
            "Extract only the food and grocery product names from the receipt image. "
            "Output one item per line with no prices, quantities, store name, totals, "
            "taxes, or other non-food information. Output only the plain list, nothing else."
        ),
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": "Please extract all grocery items from this receipt."},
            ],
        }],
    )
    return message.content[0].text


# ── Top navbar ────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

page = st.session_state["page"]

logo_col, nav_col = st.columns([1, 2])
with logo_col:
    st.markdown(
        """
<div class="nr-navbar-logo" style="padding:6px 0;">
  <div style="width:36px;height:36px;background:#2D8C5A;border-radius:10px;
              display:flex;align-items:center;justify-content:center;font-size:18px;">🥗</div>
  <span>NutriReceipt</span>
</div>""",
        unsafe_allow_html=True,
    )
with nav_col:
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🏠  Home", key="nav_home", type="secondary", width='stretch'):
            st.session_state["page"] = "Home"
            st.rerun()
    with b2:
        if st.button("📊  Score", key="nav_score", type="secondary", width='stretch'):
            st.session_state["page"] = "Score"
            st.rerun()
    with b3:
        if st.button("🔬  Insights", key="nav_insights", type="secondary", width='stretch'):
            st.session_state["page"] = "Insights"
            st.rerun()

st.markdown("<hr style='border:none;border-top:1px solid #E8E8E4;margin:0 0 32px 0;'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown(
        """
<div style="margin-bottom:44px;">
  <div class="nr-badge">✦ Receipt-Based Health Score Predictor</div>
  <h1 style="font-family:'Fraunces',serif;font-size:40px;font-weight:700;
             letter-spacing:-0.03em;line-height:1.15;margin-bottom:14px;color:#1A1A1A;">
    Snap your receipt,<br>see how healthy<br>your groceries are —
    <span style="color:#2D8C5A;font-style:italic;">instantly.</span>
  </h1>
  <p style="font-size:17px;color:#5A5A5A;max-width:580px;">
    Upload your grocery receipt, score your basket, and see what's helping
    (or hurting) your nutrition.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    # How it works
    st.markdown('<div class="nr-section-label">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="nr-section-title">Three steps to your health score</div>', unsafe_allow_html=True)

    steps = [
        ("1", "Enter Your Basket",
         "Upload a receipt photo or paste grocery item names to get started."),
        ("2", "We Analyze",
         "Items are mapped to nutrition features — fiber density, produce share, sodium levels."),
        ("3", "Get Your Score",
         "See a 0–100 Purchase Health Score, your nutrition tier, key contributing factors, and tips to improve."),
    ]
    cols = st.columns(3)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
<div class="nr-card">
  <div style="width:32px;height:32px;background:#2D8C5A;color:#fff;border-radius:9px;
              display:flex;align-items:center;justify-content:center;
              font-size:14px;font-weight:700;margin-bottom:14px;">{num}</div>
  <h3 style="font-size:15px;font-weight:600;margin-bottom:8px;color:#1A1A1A;">{title}</h3>
  <p style="font-size:13px;color:#5A5A5A;line-height:1.55;">{desc}</p>
</div>""",
                unsafe_allow_html=True,
            )

    # Score preview
    st.markdown(
        '<div class="nr-section-label" style="margin-top:12px;">Example Output</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nr-section-title">What you\'ll see</div>', unsafe_allow_html=True)

    ring_col, detail_col = st.columns([1, 2])
    with ring_col:
        st.markdown(score_ring(74, 160), unsafe_allow_html=True)
    with detail_col:
        rows_html = "".join(
            factor_row(ico, name, val, mx, d)
            for ico, name, _, d, mx in FACTOR_META[:4]
            for val in [(0.78 if d == "pos" else 0.35)]
        )
        st.markdown(
            f"""
<div style="padding-top:10px;">
  <h3 style="font-family:'Fraunces',serif;font-size:20px;font-weight:600;margin-bottom:6px;">
    Purchase Health Score</h3>
  <p style="font-size:13px;color:#5A5A5A;margin-bottom:16px;">Based on 18 items in your basket</p>
  {rows_html}
</div>""",
            unsafe_allow_html=True,
        )

    # CTA banner
    st.markdown(
        """
<div style="background:linear-gradient(135deg,#2D8C5A 0%,#1F6B42 100%);
            border-radius:14px;padding:44px 48px;margin-top:48px;">
  <h3 style="font-family:'Fraunces',serif;font-size:26px;font-weight:600;
             color:#fff;margin-bottom:10px;">Ready to score your groceries?</h3>
  <p style="font-size:15px;color:rgba(255,255,255,0.85);margin-bottom:16px;">
    Upload a receipt photo, manually enter items, or paste a grocery list.</p>
  <p style="font-size:15px;color:rgba(255,255,255,0.70);margin-bottom:0;font-style:italic;">
    ⚠ This score estimates purchase quality, not actual consumption or medical status.</p>
</div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    _, cta_col, _ = st.columns([1, 2, 1])
    with cta_col:
        st.markdown('<div class="nr-cta-wrap">', unsafe_allow_html=True)
        if st.button("📷  Score My Receipt →", key="home_cta", type="primary", width='stretch'):
            st.session_state["page"] = "Score"
            st.session_state["_scroll_top"] = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Score":
    if st.session_state.pop("_scroll_top", False):
        st_components.html("""<script>
            function scrollTop() {
                var el = window.parent.document.querySelector('[data-testid="stMain"]')
                      || window.parent.document.querySelector('section.main')
                      || window.parent.document.querySelector('.main');
                if (el) el.scrollTop = 0;
            }
            scrollTop();
            setTimeout(scrollTop, 120);
            setTimeout(scrollTop, 350);
        </script>""", height=1)
    st.markdown(
        """
<h1 style="font-family:'Fraunces',serif;font-size:32px;font-weight:700;
           letter-spacing:-0.02em;margin-bottom:8px;color:#1A1A1A;">📊 Score My Receipt</h1>
<p style="font-size:15px;color:#5A5A5A;max-width:560px;margin-bottom:28px;">
  Upload a receipt photo or enter your grocery items — then hit predict to get your health score.</p>
""",
        unsafe_allow_html=True,
    )

    if bundle_error:
        st.error(
            f"Could not load model bundle: {bundle_error}\n\n"
            "Make sure `model_bundle.pkl` or `model_bundle.pkl.gz` is in the same folder as `app.py`, "
            "and that any custom model classes are importable (see README)."
        )

    # ── Data card: header + tabs ─────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            """
<div style="text-align:center;margin-bottom:16px;">
  <h3 style="font-size:20px;font-weight:700;color:#1A1A1A;margin-bottom:6px;">
    Your Grocery Data</h3>
  <p style="font-size:14px;color:#5A5A5A;margin-bottom:0;">
    Choose how you'd like to share your purchase list.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        # Load Sample Receipt — centered below header
        _, sample_btn_col, _ = st.columns([3, 2, 3])
        with sample_btn_col:
            if st.button(
                "📋  Load Sample Receipt",
                key="load_sample_header",
                type="secondary",
                width='stretch',
            ):
                st.session_state["manual_items"] = [dict(item) for item in SAMPLE_ITEMS]
                st.session_state["active_tab"] = 1
                st.rerun()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Custom tab selector ───────────────────────────────────────────────
        if "active_tab" not in st.session_state:
            st.session_state["active_tab"] = 0
        active_tab = st.session_state["active_tab"]

        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            if st.button(
                "📷  Upload Receipt",
                key="tab_btn_0",
                type="primary" if active_tab == 0 else "secondary",
                width='stretch',
            ):
                st.session_state["active_tab"] = 0
                st.rerun()
        with tc2:
            if st.button(
                "🎛️  Manual Input",
                key="tab_btn_1",
                type="primary" if active_tab == 1 else "secondary",
                width='stretch',
            ):
                st.session_state["active_tab"] = 1
                st.rerun()
        with tc3:
            if st.button(
                "📝  Paste Items",
                key="tab_btn_2",
                type="primary" if active_tab == 2 else "secondary",
                width='stretch',
            ):
                st.session_state["active_tab"] = 2
                st.rerun()

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        current_items_text = ""
        preview_source = ""

        # ── Tab 0: Receipt upload ─────────────────────────────────────────────
        if active_tab == 0:
            api_key = get_anthropic_key()
            if anthropic is None:
                st.info(
                    "Receipt OCR is optional. Install the `anthropic` package and add an API key to turn this tab on.",
                    icon="🧾",
                )
            elif not api_key:
                st.info(
                    "Receipt OCR is optional. Set `ANTHROPIC_API_KEY` in your environment to extract grocery items from receipt images.",
                    icon="🔑",
                )

            _, up_col, _ = st.columns([1, 3, 1])
            with up_col:
                uploaded_img = st.file_uploader(
                    "Drag & drop your receipt image here, or click to browse",
                    type=["jpg", "jpeg", "png", "webp"],
                )
                if not uploaded_img:
                    st.markdown(
                        """
<div style="display:flex;gap:8px;justify-content:center;margin-top:4px;">
  <span style="padding:3px 12px;border:1px solid #D0D8D4;border-radius:100px;
               font-size:11px;font-weight:600;color:#5A5A5A;">JPG</span>
  <span style="padding:3px 12px;border:1px solid #D0D8D4;border-radius:100px;
               font-size:11px;font-weight:600;color:#5A5A5A;">PNG</span>
  <span style="padding:3px 12px;border:1px solid #D0D8D4;border-radius:100px;
               font-size:11px;font-weight:600;color:#5A5A5A;">WEBP</span>
</div>
<p style="text-align:center;font-size:13px;color:#5A5A5A;margin-top:12px;">
  ⚠ This score estimates purchase quality, not actual consumption or medical status.</p>


""",
                        unsafe_allow_html=True,
                    )

            if uploaded_img:
                img_bytes = uploaded_img.read()
                mt_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
                media_type = mt_map.get(uploaded_img.name.rsplit(".", 1)[-1].lower(), "image/jpeg")

                prev_col, edit_col = st.columns([1, 1], gap="large")
                with prev_col:
                    st.image(img_bytes, caption="Uploaded receipt", width='stretch')
                with edit_col:
                    if anthropic is None:
                        st.warning("Install `anthropic` and add an API key to extract grocery items from this receipt.")
                    elif not api_key:
                        st.warning("Set `ANTHROPIC_API_KEY` to extract grocery items from this receipt.")
                    else:
                        _ocr_ph = st.empty()
                        _ocr_ph.markdown(
                            """
<div style="text-align:center;padding:36px 0 28px;">
  <div style="font-size:52px;display:inline-block;
              animation:ocr-float 1.5s ease-in-out infinite;margin-bottom:20px;">🧾</div>
  <div style="display:flex;justify-content:center;gap:10px;margin-bottom:18px;">
    <span style="width:10px;height:10px;border-radius:50%;background:#2D8C5A;display:inline-block;
                 animation:ocr-bounce 1.2s ease-in-out infinite;animation-delay:0s;"></span>
    <span style="width:10px;height:10px;border-radius:50%;background:#2D8C5A;display:inline-block;
                 animation:ocr-bounce 1.2s ease-in-out infinite;animation-delay:0.2s;"></span>
    <span style="width:10px;height:10px;border-radius:50%;background:#2D8C5A;display:inline-block;
                 animation:ocr-bounce 1.2s ease-in-out infinite;animation-delay:0.4s;"></span>
  </div>
  <p style="font-size:14px;font-weight:600;color:#2D8C5A;margin:0;">Reading your receipt…</p>
  <p style="font-size:12px;color:#5A5A5A;margin:6px 0 0;">This usually takes a few seconds.</p>
</div>""",
                            unsafe_allow_html=True,
                        )
                        try:
                            extracted = extract_items_from_receipt(img_bytes, media_type, api_key)
                        except Exception as exc:
                            extracted = ""
                            _ocr_ph.empty()
                            st.error(f"OCR failed: {exc}")
                        else:
                            _ocr_ph.empty()
                        if extracted:
                            items_text_ocr = st.text_area(
                                "Extracted items — edit if needed",
                                value=extracted,
                                height=240,
                            )
                            current_items_text = items_text_ocr
                            preview_source = "OCR-extracted receipt items"

        # ── Tab 1: Manual item search ─────────────────────────────────────────
        elif active_tab == 1:
            if "manual_items" not in st.session_state:
                st.session_state["manual_items"] = []

            search_col, add_col = st.columns([5, 1])
            with search_col:
                new_item = st.text_input(
                    "item_search",
                    placeholder="Search or type a grocery item, e.g. Organic Spinach…",
                    label_visibility="collapsed",
                )
            with add_col:
                if st.button("＋ Add", key="add_item_btn", width='stretch'):
                    if new_item.strip():
                        st.session_state["manual_items"].append({"name": new_item.strip(), "qty": 1})
                        st.rerun()

            items = st.session_state["manual_items"]
            if items:
                st.markdown(
                    f"<p style='font-size:12px;color:#5A5A5A;margin:12px 0 4px;'>"
                    f"{len(items)} item{'s' if len(items) != 1 else ''} · keyword matching active</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
<div style="display:grid;grid-template-columns:1fr 80px 40px;gap:8px;
            padding:6px 8px;background:#F4F4F0;border-radius:8px 8px 0 0;
            font-size:11px;font-weight:700;color:#5A5A5A;text-transform:uppercase;
            letter-spacing:0.06em;">
  <span>Item</span><span style="text-align:center;">Qty</span><span></span>
</div>""",
                    unsafe_allow_html=True,
                )
                for i, item in enumerate(items):
                    name_col, qty_col, rm_col = st.columns([6, 1, 0.5])
                    with name_col:
                        bg = "#FFFFFF" if i % 2 == 0 else "#FAFAF7"
                        st.markdown(
                            f"<div style='padding:10px 8px;font-size:14px;color:#1A1A1A;"
                            f"background:{bg};border-bottom:1px solid #F0F0EC;'>{item['name']}</div>",
                            unsafe_allow_html=True,
                        )
                    with qty_col:
                        new_qty = st.number_input(
                            "qty", min_value=1, max_value=50,
                            value=item["qty"], key=f"qty_{i}",
                            label_visibility="collapsed",
                        )
                        st.session_state["manual_items"][i]["qty"] = new_qty
                    with rm_col:
                        if st.button("✕", key=f"rm_{i}", type="secondary"):
                            st.session_state["manual_items"].pop(i)
                            st.rerun()

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                current_items_text = "\n".join(
                    "\n".join([it["name"]] * int(it["qty"]))
                    for it in st.session_state["manual_items"]
                )
                preview_source = "your manual basket entries"
                clr_col, note_col = st.columns([1, 2])
                with clr_col:
                    if st.button("Clear all", key="clear_items", type="secondary"):
                        st.session_state["manual_items"] = []
                        st.rerun()
                with note_col:
                    st.caption("Quantities repeat items when estimating the basket snapshot below.")
            else:
                st.markdown(
                    "<p style='color:#999;font-size:14px;padding:24px 0;text-align:center;'>"
                    "No items yet — type above to add one, or use <strong>Load Sample Receipt</strong> above.</p>",
                    unsafe_allow_html=True,
                )

        # ── Tab 2: Paste items ────────────────────────────────────────────────
        elif active_tab == 2:
            st.markdown(
                "<p style='font-size:13px;color:#5A5A5A;margin-bottom:12px;'>"
                "Paste grocery items one per line. We'll estimate nutrition features "
                "using keyword-based category matching.</p>",
                unsafe_allow_html=True,
            )
            items_text = st.text_area(
                "Items",
                placeholder=(
                    "Organic Bananas\nWhole Wheat Bread\nGreek Yogurt Plain\n"
                    "Baby Spinach 5oz\nChicken Breast Boneless\nCheddar Cheese Block\n"
                    "Oat Milk Original\nFrozen Blueberries 12oz\nTortilla Chips\nApple Juice"
                ),
                height=200,
                label_visibility="collapsed",
            )
            st.caption("💡 Include keywords like 'spinach', 'chips', 'juice' for better matching.")
            current_items_text = items_text
            preview_source = "your pasted item list"

        # ── Feature Preview ───────────────────────────────────────────────────
        has_items = bool(current_items_text.strip())
        if has_items:
            preview_features = items_to_features(current_items_text)
            render_feature_snapshot(preview_features, preview_source)
        else:
            preview_features = None
            ph_cards = "".join(
                '<div class="nr-stat-card">'
                '<div class="nr-placeholder-bar" style="width:55%;height:10px;margin-bottom:8px;"></div>'
                '<div class="nr-placeholder-bar" style="width:45%;height:22px;margin-bottom:6px;"></div>'
                '<div class="nr-placeholder-bar" style="width:65%;height:9px;"></div>'
                '</div>'
                for _ in range(8)
            )
            st.markdown(
                f"""
<div class="nr-card" style="margin-top:18px;">
  <div class="nr-section-label">Feature Preview</div>
  <div class="nr-section-title" style="margin-bottom:10px;">Estimated basket snapshot</div>
  <p style="font-size:13px;color:#5A5A5A;margin-bottom:18px;">
    Add items above to see your basket features here.
  </p>
  <div class="nr-stat-grid">{ph_cards}</div>
</div>""",
                unsafe_allow_html=True,
            )

        # ── Prediction Model picker ───────────────────────────────────────────
        if not has_items:
            st.markdown(
                '<div style="opacity:0.4;pointer-events:none;user-select:none;">',
                unsafe_allow_html=True,
            )
        _model_picker()
        if not has_items:
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            """
<div class="nr-callout" style="margin:8px 0 16px 0;">
  <strong>Project default:</strong> KNN stays selected by default to match the midpoint deployment plan,
  and you can compare all three saved models after prediction.
</div>
""",
            unsafe_allow_html=True,
        )

        # ── Predict button ────────────────────────────────────────────────────
        _, predict_col, _ = st.columns([1, 2, 1])
        with predict_col:
            st.markdown('<div class="nr-cta-wrap">', unsafe_allow_html=True)
            if st.button("🔍  Predict Health Score", key=f"btn_predict_{active_tab}", type="primary", width='stretch'):
                if not has_items:
                    st.toast("Add at least one grocery item before predicting.", icon="⚠️")
                else:
                    st.session_state["features"] = preview_features
                    st.session_state["run_predict"] = True
                    st.session_state["last_items_text"] = current_items_text
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Results ────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="nr-section-label">Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="nr-section-title">Your Purchase Health Score</div>', unsafe_allow_html=True)

    _has_result = st.session_state.get("run_predict") and st.session_state.get("features")

    if not _has_result:
        # Placeholder skeleton shown before any prediction is made
        ph_score_col, ph_factors_col = st.columns([1, 2])
        with ph_score_col:
            st.markdown(
                """
<div class="nr-card" style="text-align:center;min-height:260px;display:flex;
     flex-direction:column;align-items:center;justify-content:center;gap:16px;">
  <svg width="140" height="140" viewBox="0 0 140 140">
    <circle cx="70" cy="70" r="54" fill="none" stroke="#F0F0EC" stroke-width="16"/>
    <text x="70" y="76" text-anchor="middle" font-size="32" font-weight="700"
          font-family="Fraunces,serif" fill="#C8C8C4">—</text>
  </svg>
  <div class="nr-placeholder-bar" style="width:80px;height:22px;border-radius:100px;"></div>
  <div class="nr-placeholder-bar" style="width:160px;height:13px;"></div>
  <div class="nr-placeholder-bar" style="width:130px;height:13px;"></div>
</div>""",
                unsafe_allow_html=True,
            )
        with ph_factors_col:
            st.markdown(
                """
<div class="nr-card" style="min-height:260px;">
  <div class="nr-placeholder-bar" style="width:55%;height:15px;margin-bottom:6px;"></div>
  <div class="nr-placeholder-bar" style="width:35%;height:11px;margin-bottom:20px;"></div>
  """ + "".join(
      f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">'
      f'<div class="nr-placeholder-bar" style="width:22px;height:22px;border-radius:50%;flex-shrink:0;"></div>'
      f'<div style="flex:1;"><div class="nr-placeholder-bar" style="height:11px;width:60%;margin-bottom:5px;"></div>'
      f'<div class="nr-placeholder-bar" style="height:8px;width:{w}%;border-radius:4px;"></div></div></div>'
      for w in [72, 55, 88, 40, 65, 50, 78, 45, 60, 35]
  ) + """
</div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            """
<div style="text-align:center;padding:20px;color:#AAAAAA;font-size:14px;">
  Add your items above and click <strong>Predict Health Score</strong> to see your results.
</div>""",
            unsafe_allow_html=True,
        )
    else:
        f = st.session_state["features"]

        if bundle is None:
            st.warning("Model bundle not loaded — cannot predict.")
        else:
            selected_model = st.session_state.get("model_choice", MIDPOINT_DEFAULT_MODEL)
            score = predict_phs(f, selected_model)
            if score is None:
                st.error("Prediction failed. Check the console for details.")
            else:
                t_label, t_color, t_bg = tier(score)
                component_scores = compute_phs_components(f)
                component_total = sum(component["score"] for component in component_scores)
                component_cards_html = "".join(
                    stat_card(
                        component["label"],
                        f"{component['score']:.1f} / {component['max_score']}",
                        component["note"],
                    )
                    for component in component_scores
                )
                all_scores = predict_all_models(f)
                comparison_rows = []
                for model_key in MODEL_ORDER:
                    if model_key not in all_scores:
                        continue
                    comparison_rows.append(
                        {
                            "Selected": "Yes" if model_key == selected_model else "",
                            "Model": get_model_label(model_key),
                            "Basket Prediction": round(all_scores[model_key], 1),
                            "Saved Test RMSE": round(get_model_metric(model_key, "RMSE") or 0, 2),
                            "Saved Test R²": round(get_model_metric(model_key, "R²") or 0, 4),
                            "Project Role": "Midpoint default" if model_key == MIDPOINT_DEFAULT_MODEL else "Comparison model",
                        }
                    )
                comparison_df = pd.DataFrame(comparison_rows)
                prediction_spread = (
                    max(all_scores.values()) - min(all_scores.values())
                    if len(all_scores) > 1 else 0.0
                )

                score_col, factors_col = st.columns([1, 2])

                # Score ring card
                with score_col:
                    desc_text = {
                        "Excellent": "Exceptional basket — diverse, nutrient-dense, and low in processed items.",
                        "Good":      "Strong basket with good produce and fiber. Small tweaks can push it higher.",
                        "Fair":      "Decent basket. More produce and fewer processed items would help.",
                        "Needs Work": "Room to improve. Try adding more fruits, veggies, and whole grains.",
                    }[t_label]
                    st.markdown(
                        f"""
<div class="nr-card" style="text-align:center;">
  {score_ring(score, 160)}
  <span style="display:inline-block;padding:4px 14px;border-radius:100px;
               font-size:12px;font-weight:700;text-transform:uppercase;
               letter-spacing:0.06em;background:{t_bg};color:{t_color};
               margin-bottom:10px;">{t_label}</span>
  <p style="font-size:12px;color:#2D8C5A;font-weight:600;margin:-2px 0 10px 0;">
    {MODEL_LABELS[selected_model]} prediction</p>
  <p style="font-size:13px;color:#5A5A5A;line-height:1.55;">{desc_text}</p>
</div>""",
                        unsafe_allow_html=True,
                    )

                # Factor breakdown
                with factors_col:
                    rows_html = "".join(
                        factor_row(ico, name, f[feat], mx, direction)
                        for ico, name, feat, direction, mx in FACTOR_META
                    )
                    st.markdown(
                        f"""
<div class="nr-card">
  <h3 style="font-size:15px;font-weight:600;margin-bottom:4px;color:#1A1A1A;">
    What Influenced Your Score</h3>
  <p style="font-size:12px;color:#999;margin-bottom:14px;">
    Key basket features and their direction of impact</p>
  {rows_html}
</div>""",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"""
<div class="nr-card" style="margin-top:4px;">
  <div class="nr-section-label">Interpretability</div>
  <div class="nr-section-title" style="margin-bottom:10px;">Purchase Health Score component scorecard</div>
  <p style="font-size:13px;color:#5A5A5A;margin-bottom:18px;">
    Weak-label baseline: <strong>{component_total:.1f} / 100</strong>. The trained models learn from these same
    engineered basket features, so this view shows where the underlying score pressure is coming from.
  </p>
  <div class="nr-stat-grid">{component_cards_html}</div>
</div>""",
                    unsafe_allow_html=True,
                )

                if st.button("🔬 View Model Insights →", key="score_to_insights", type="primary", width='stretch'):
                    st.session_state["page"] = "Insights"
                    st.rerun()

                # Nutrition summary
                total_fiber   = f["fiber_density"]   * f["basket_size"]
                total_sodium  = f["sodium_density"]  * f["basket_size"]
                total_sugar   = f["sugar_density"]   * f["basket_size"]
                total_protein = f["protein_density"] * f["basket_size"]

                def nut_card(icon, val, unit, label):
                    return (
                        f'<div style="background:#FAFAF7;border-radius:10px;padding:16px;text-align:center;">'
                        f'<div style="font-size:22px;margin-bottom:8px;">{icon}</div>'
                        f'<div style="font-family:\'Fraunces\',serif;font-size:22px;font-weight:700;color:#1A1A1A;">'
                        f'{val} <span style="font-size:12px;color:#999;">{unit}</span></div>'
                        f'<div style="font-size:11px;color:#5A5A5A;font-weight:500;margin-top:4px;">{label}</div>'
                        f'</div>'
                    )

                st.markdown(
                    f"""
<div class="nr-card" style="margin-top:4px;">
  <h3 style="font-size:15px;font-weight:600;margin-bottom:20px;color:#1A1A1A;">
    Basket Nutrition Summary</h3>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:16px;">
    {nut_card("🔥", f"{f['total_calories']:.0f}", "kcal", "Total Calories")}
    {nut_card("🌾", f"{total_fiber:.1f}",  "g",  "Total Fiber")}
    {nut_card("🥩", f"{total_protein:.1f}", "g", "Total Protein")}
    {nut_card("🧂", f"{total_sodium:.0f}", "mg", "Total Sodium")}
    {nut_card("🍬", f"{total_sugar:.1f}",  "g",  "Total Sugar")}
  </div>
</div>""",
                    unsafe_allow_html=True,
                )

                # Improvement suggestions
                tips = []
                if f["produce_share"]   < 0.25:
                    tips.append("Boost produce share by swapping a packaged snack for a fresh fruit or veggie — could raise your score 3–5 points.")
                if f["processed_share"] > 0.30:
                    tips.append("High processed share detected. Choosing whole-food alternatives for 1–2 items would noticeably improve your score.")
                if f["sodium_density"]  > 200:
                    tips.append("Sodium density is elevated. Try low-sodium versions of canned goods or deli meats.")
                if f["fiber_density"]   < 2.0:
                    tips.append("Low fiber density. Adding legumes, whole grains, or leafy greens would help significantly.")
                if f["beverage_share"]  > 0.25:
                    tips.append("High beverage share can dilute nutrient density. Swap a sugary drink for water or a whole fruit.")
                if not tips:
                    tips.append("Great basket! Keep up the diversity of departments and strong produce share.")

                tips_html = "".join(
                    f'<div style="display:flex;align-items:flex-start;gap:10px;font-size:13px;'
                    f'color:#5A5A5A;margin-bottom:8px;line-height:1.5;">'
                    f'<span style="color:#F2A541;font-weight:700;flex-shrink:0;">→</span>{t}</div>'
                    for t in tips
                )
                st.markdown(
                    f"""
<div style="background:#FFF5E6;border:1px solid #F0DFC0;border-radius:14px;
            padding:24px 28px;margin-top:4px;">
  <h3 style="font-size:14px;font-weight:600;margin-bottom:12px;display:flex;
             align-items:center;gap:8px;color:#1A1A1A;">💡 Tips to Improve Next Time</h3>
  {tips_html}
</div>""",
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Insights":
    st.markdown(
        """
<h1 style="font-family:'Fraunces',serif;font-size:32px;font-weight:700;
           letter-spacing:-0.02em;margin-bottom:8px;color:#1A1A1A;">🔬 Model Insights</h1>
<p style="font-size:15px;color:#5A5A5A;max-width:520px;margin-bottom:32px;">
  How the Purchase Health Score is computed and what each model learned.</p>
""",
        unsafe_allow_html=True,
    )

    # Performance table
    st.markdown('<div class="nr-section-label">Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="nr-section-title">Model Comparison</div>', unsafe_allow_html=True)
    insights_df = MODEL_RESULTS_DF.copy()
    insights_df["Project Role"] = insights_df["model_key"].map(
        lambda key: "Midpoint default deployment choice" if key == MIDPOINT_DEFAULT_MODEL else "Comparison model"
    )
    st.dataframe(
        insights_df[
            ["Model", "RMSE", "MAE", "R²", "Train Time (s)", "Inference Time (s)", "Project Role"]
        ].set_index("Model"),
        width='stretch',
    )
    if bundle:
        mlp_cfg = bundle.get("best_mlp_cfg", {})
        st.markdown(
            f"""
<div class="nr-callout" style="margin:10px 0 6px 0;">
  <strong>Saved tuning snapshot:</strong> Ridge α = {bundle.get('best_alpha', 1000):.0f},
  KNN k = {bundle.get('best_k', 11)}, and MLP hidden dim = {mlp_cfg.get('hidden_dim', 64)},
  lr = {mlp_cfg.get('lr', 0.01)}, α = {mlp_cfg.get('alpha', 0.001)}.
</div>
""",
            unsafe_allow_html=True,
        )
    st.bar_chart(insights_df.set_index("Model")[["RMSE", "MAE"]])

    # Ridge coefficients
    if bundle and "ridge" in bundle:
        st.markdown(
            '<div class="nr-section-label" style="margin-top:32px;">Interpretability</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="nr-section-title">Ridge Regression Coefficients</div>',
            unsafe_allow_html=True,
        )
        try:
            model = bundle["ridge"]
            coefs = (
                model.w if hasattr(model, "w") else
                np.array(model.weights).flatten()[:len(FEATURE_COLS)] if hasattr(model, "weights") else
                model.coef_ if hasattr(model, "coef_") else
                None
            )
            if coefs is not None:
                coef_series = pd.Series(
                    np.array(coefs).flatten(),
                    index=FEATURE_COLS,
                    name="Coefficient",
                ).sort_values(key=abs, ascending=False)
                st.bar_chart(coef_series)
                st.caption(
                    "Positive coefficients increase PHS; negative ones decrease it. "
                    "Values shown after Z-score normalization."
                )
            else:
                st.info("Could not locate coefficient attribute on the Ridge model object.")
        except Exception as exc:
            st.info(f"Could not extract coefficients: {exc}")

    # Methodology
    st.markdown(
        '<div class="nr-section-label" style="margin-top:32px;">Methodology</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nr-section-title">How PHS is Computed</div>', unsafe_allow_html=True)
    methodology_cards = "".join(
        [
            stat_card("Produce Share", "25 max", "Full points at roughly 50% of basket items."),
            stat_card("Fiber Density", "20 max", "Rewards baskets that average 3g fiber per item."),
            stat_card("Sugar Density", "20 max", "Loses points as average sugar approaches 20g per item."),
            stat_card("Sodium Density", "15 max", "Loses points as average sodium approaches 500mg per item."),
            stat_card("Processed Share", "20 max", "Loses points as processed share approaches 60% of items."),
        ]
    )
    st.markdown(
        f"""
<div class="nr-card">
  <p style="font-size:14px;color:#5A5A5A;line-height:1.7;margin-bottom:16px;">
    The <strong>Purchase Health Score (PHS)</strong> is a weakly-supervised nutritional metric (0–100)
    inspired by the USDA Healthy Eating Index. It is derived from five capped sub-scores:
  </p>
  <div class="nr-stat-grid" style="margin-bottom:16px;">{methodology_cards}</div>
  <p style="font-size:13px;color:#5A5A5A;line-height:1.65;margin-bottom:0;">
    The deployed models also use basket size, calories, protein density, beverage share, and department diversity.
    All engineered features are Z-score normalized before prediction.
  </p>
</div>""",
        unsafe_allow_html=True,
    )

# ── Footer (all pages) ────────────────────────────────────────────────────────
st.markdown(
    """
<div class="nr-footer">
  <strong>INFO 5368 · PAML</strong> &nbsp;·&nbsp; Built with Streamlit &nbsp;·&nbsp; Cornell Tech
</div>
""",
    unsafe_allow_html=True,
)
