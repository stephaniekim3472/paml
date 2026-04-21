import base64
import math
import gzip
import os
import pickle

import anthropic
import numpy as np
import pandas as pd
import streamlit as st
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
  font-size: 14px !important;
  font-weight: 600 !important;
  padding: 8px 16px !important;
  border-radius: 8px !important;
  transition: background 0.15s !important;
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
}
[data-testid="stFileUploaderDropzoneInstructions"] { padding: 16px 32px 28px !important; }
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


def _try_load_pkl(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


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


def predict_phs(features: dict, model_name: str) -> float | None:
    if bundle is None:
        return None
    x = np.array([features[f] for f in FEATURE_COLS], dtype=float)
    mean = np.array(bundle["train_mean"], dtype=float)
    std  = np.array(bundle["train_std"],  dtype=float)
    x_norm = (x - mean) / std
    pred = bundle[model_name].predict(x_norm.reshape(1, -1))
    return float(np.clip(np.array(pred).flatten()[0], 0, 100))


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


def _model_picker() -> str:
    """Render centered model-selector buttons; returns current model_choice."""
    if "model_choice" not in st.session_state:
        st.session_state["model_choice"] = "mlp"
    mc = st.session_state["model_choice"]
    st.markdown(
        '<div class="nr-section-label" style="text-align:center;">Prediction Model</div>',
        unsafe_allow_html=True,
    )
    _, c1, c2, c3, _ = st.columns([1, 2, 2, 1.5, 1])
    with c1:
        if st.button("🧠  MLP  (R²=0.9999)", key="model_mlp",
                     type="primary" if mc == "mlp" else "secondary", width="stretch"):
            st.session_state["model_choice"] = "mlp"; st.rerun()
    with c2:
        if st.button("📈  Ridge  (R²=0.9775)", key="model_ridge",
                     type="primary" if mc == "ridge" else "secondary", width="stretch"):
            st.session_state["model_choice"] = "ridge"; st.rerun()
    with c3:
        if st.button("📍  KNN  (R²=0.9905)", key="model_knn",
                     type="primary" if mc == "knn" else "secondary", width="stretch"):
            st.session_state["model_choice"] = "knn"; st.rerun()
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    return st.session_state["model_choice"]


# ── Anthropic receipt OCR ─────────────────────────────────────────────────────
def get_anthropic_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


@st.cache_data(show_spinner=False)
def extract_items_from_receipt(image_bytes: bytes, media_type: str, api_key: str) -> str:
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

logo_col, nav_col = st.columns([3, 2])
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
        if st.button("🏠  Home", key="nav_home", type="secondary", width="stretch"):
            st.session_state["page"] = "Home"
            st.rerun()
    with b2:
        if st.button("📊  Score", key="nav_score", type="secondary", width="stretch"):
            st.session_state["page"] = "Score"
            st.rerun()
    with b3:
        if st.button("🔬  Insights", key="nav_insights", type="secondary", width="stretch"):
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
    Upload a receipt photo, manually enter or paste items, or upload a csv file.</p>
  <p style="font-size:15px;color:rgba(255,255,255,0.70);margin-bottom:0;font-style:italic;">
    ⚠ This score estimates purchase quality, not actual consumption or medical status.</p>
</div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    if st.button("📷  Score My Receipt →", key="home_cta"):
        st.session_state["page"] = "Score"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Score":
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
            "Make sure `model_bundle.pkl.gz` is in the same folder as `app.py`, "
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
            if st.button("📋  Load Sample Receipt", key="load_sample_header", type="secondary", width="stretch"):
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
            if st.button("📷  Upload Receipt", key="tab_btn_0",
                         type="primary" if active_tab == 0 else "secondary", width="stretch"):
                st.session_state["active_tab"] = 0
                st.rerun()
        with tc2:
            if st.button("🎛️  Manual Input", key="tab_btn_1",
                         type="primary" if active_tab == 1 else "secondary", width="stretch"):
                st.session_state["active_tab"] = 1
                st.rerun()
        with tc3:
            if st.button("📝  Paste Items", key="tab_btn_2",
                         type="primary" if active_tab == 2 else "secondary", width="stretch"):
                st.session_state["active_tab"] = 2
                st.rerun()

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        features: dict | None = None

        # ── Tab 0: Receipt upload ─────────────────────────────────────────────
        if active_tab == 0:
            api_key = get_anthropic_key()
            if not api_key:
                st.info("Receipt OCR requires an Anthropic API key. Set `ANTHROPIC_API_KEY` in your environment to enable this feature.", icon="🔑")

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

            if uploaded_img and api_key:
                img_bytes = uploaded_img.read()
                mt_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
                media_type = mt_map.get(uploaded_img.name.rsplit(".", 1)[-1].lower(), "image/jpeg")

                prev_col, edit_col = st.columns([1, 1], gap="large")
                with prev_col:
                    st.image(img_bytes, caption="Uploaded receipt", width="stretch")
                with edit_col:
                    with st.spinner("Extracting items from receipt…"):
                        try:
                            extracted = extract_items_from_receipt(img_bytes, media_type, api_key)
                        except Exception as exc:
                            st.error(f"OCR failed: {exc}")
                            extracted = ""
                    if extracted:
                        items_text_ocr = st.text_area(
                            "Extracted items — edit if needed",
                            value=extracted,
                            height=240,
                        )
                        _model_picker()
                        if st.button("🔍  Predict Health Score", key="btn_receipt") and items_text_ocr.strip():
                            features = items_to_features(items_text_ocr)
                            st.session_state["features"]    = features
                            st.session_state["run_predict"] = True
            elif uploaded_img and not api_key:
                st.warning("Set the `ANTHROPIC_API_KEY` environment variable to extract items from this receipt.")

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
                if st.button("＋ Add", key="add_item_btn", width="stretch"):
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
                _model_picker()
                clr_col, pred_col = st.columns([1, 2])
                with clr_col:
                    if st.button("Clear all", key="clear_items", type="secondary"):
                        st.session_state["manual_items"] = []
                        st.rerun()
                with pred_col:
                    if st.button("🔍  Predict Health Score", key="btn_manual", width="stretch"):
                        items_text = "\n".join(
                            "\n".join([it["name"]] * it["qty"])
                            for it in st.session_state["manual_items"]
                        )
                        features = items_to_features(items_text)
                        st.session_state["features"]    = features
                        st.session_state["run_predict"] = True
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

            _model_picker()
            if st.button("🔍  Predict Health Score", key="btn_paste") and items_text.strip():
                features = items_to_features(items_text)
                st.session_state["features"]    = features
                st.session_state["run_predict"] = True

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
            score = predict_phs(f, st.session_state.get("model_choice", "mlp"))
            if score is None:
                st.error("Prediction failed. Check the console for details.")
            else:
                t_label, t_color, t_bg = tier(score)

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
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;">
    {nut_card("🔥", f"{f['total_calories']:.0f}", "kcal", "Total Calories")}
    {nut_card("🌾", f"{total_fiber:.1f}",  "g",  "Total Fiber")}
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
    st.dataframe(
        pd.DataFrame({
            "Model":       ["Ridge Regression", "KNN (k=11)", "MLP Neural Net"],
            "Test RMSE":   [2.64,  1.71,  0.18],
            "Test R²":     [0.9775, 0.9905, 0.9999],
            "Train Time":  ["0.008 s", "~0 s",  "44 s"],
            "Inference":   ["Instant",  "~6.6 s", "Instant"],
        }).set_index("Model"),
        width="stretch",
    )

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
                model.weights if hasattr(model, "weights") else
                model.coef_   if hasattr(model, "coef_")   else
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
    st.markdown(
        """
<div class="nr-card">
  <p style="font-size:14px;color:#5A5A5A;line-height:1.7;margin-bottom:16px;">
    The <strong>Purchase Health Score (PHS)</strong> is a weakly-supervised nutritional metric (0–100)
    inspired by the USDA Healthy Eating Index. It is derived from five sub-scores:
  </p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
    <div style="background:#F8F8F5;border-radius:10px;padding:14px;">
      <div style="font-weight:600;color:#2D8C5A;margin-bottom:4px;">🥦 Produce Share</div>
      <div style="font-size:12px;color:#5A5A5A;">Fraction of basket items from produce departments</div>
    </div>
    <div style="background:#F8F8F5;border-radius:10px;padding:14px;">
      <div style="font-weight:600;color:#2D8C5A;margin-bottom:4px;">🌾 Fiber Density</div>
      <div style="font-size:12px;color:#5A5A5A;">Average grams of fiber per basket item</div>
    </div>
    <div style="background:#F8F8F5;border-radius:10px;padding:14px;">
      <div style="font-weight:600;color:#2D8C5A;margin-bottom:4px;">🥩 Protein Density</div>
      <div style="font-size:12px;color:#5A5A5A;">Average grams of protein per basket item</div>
    </div>
    <div style="background:#F8F8F5;border-radius:10px;padding:14px;">
      <div style="font-weight:600;color:#F2A541;margin-bottom:4px;">🍬 Sugar / Sodium</div>
      <div style="font-size:12px;color:#5A5A5A;">Average sugar and sodium per item (penalized)</div>
    </div>
    <div style="background:#F8F8F5;border-radius:10px;padding:16px;grid-column:span 2;">
      <div style="font-weight:600;color:#F2A541;margin-bottom:4px;">📦 Processed Share</div>
      <div style="font-size:12px;color:#5A5A5A;">
        Fraction of basket items from processed-food aisles.
        All features are Z-score normalized before being fed to the model.
      </div>
    </div>
  </div>
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
