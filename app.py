import math
import gzip
import pickle

import numpy as np
import pandas as pd
import streamlit as st

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
.stTabs [data-baseweb="tab-list"]  { border-bottom: 2px solid var(--border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"]       { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; color: var(--text-sec) !important; }
.stTabs [aria-selected="true"]     { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }

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


@st.cache_resource
def load_bundle():
    try:
        with gzip.open("model_bundle.pkl.gz", "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, str(e)


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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
<div style="display:flex;align-items:center;gap:10px;margin-bottom:32px;padding-top:4px;">
  <div style="width:36px;height:36px;background:#2D8C5A;border-radius:10px;
              display:flex;align-items:center;justify-content:center;font-size:18px;">🥗</div>
  <span style="font-family:'Fraunces',serif;font-weight:700;font-size:16px;
               letter-spacing:-0.02em;color:#1A1A1A;">NutriReceipt</span>
</div>
""",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        ["🏠  Home", "📊  Score My Receipt", "🔬  Model Insights"],
        label_visibility="collapsed",
        key="nav_page",
    )

    st.markdown(
        """
<div style="margin-top:48px;padding:14px;background:#E8F5ED;border-radius:12px;
            font-size:12px;color:#2D8C5A;line-height:1.5;">
  <strong style="display:block;font-size:13px;margin-bottom:4px;">INFO 5368 · PAML</strong>
  Built with Streamlit · Cornell Tech
</div>
""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if "Home" in page:
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
    Paste your grocery items, enter basket features, or import a CSV.
    Our model scores your basket 0–100 and shows what's helping
    (and hurting) your nutrition.
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
         "Paste grocery item names, adjust feature sliders, or upload a CSV with pre-computed values."),
        ("2", "We Analyze",
         "Items are mapped to nutrition features — fiber density, produce share, sodium levels — then normalized for the model."),
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
            border-radius:14px;padding:36px 40px;margin-top:24px;
            display:flex;align-items:center;justify-content:space-between;">
  <div>
    <h3 style="font-family:'Fraunces',serif;font-size:22px;font-weight:600;
               color:#fff;margin-bottom:6px;">Ready to score your groceries?</h3>
    <p style="font-size:14px;color:rgba(255,255,255,0.85);">
      Enter basket features or paste your items — results in seconds.</p>
  </div>
</div>""",
        unsafe_allow_html=True,
    )
    if st.button("Score My Receipt →", key="home_cta"):
        st.session_state["nav_page"] = "📊  Score My Receipt"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif "Score" in page:
    st.markdown(
        """
<h1 style="font-family:'Fraunces',serif;font-size:32px;font-weight:700;
           letter-spacing:-0.02em;margin-bottom:8px;color:#1A1A1A;">📊 Score My Receipt</h1>
<p style="font-size:15px;color:#5A5A5A;max-width:540px;margin-bottom:28px;">
  Enter basket features manually, paste grocery items, or import a CSV
  to get your Purchase Health Score.</p>
""",
        unsafe_allow_html=True,
    )

    if bundle_error:
        st.error(
            f"Could not load model bundle: {bundle_error}\n\n"
            "Make sure `model_bundle.pkl.gz` is in the same folder as `app.py`, "
            "and that any custom model classes are importable (see README)."
        )

    model_choice = st.selectbox(
        "Model",
        ["mlp", "ridge", "knn"],
        format_func=lambda x: {
            "mlp":   "MLP Neural Net  (best accuracy, R²=0.9999)",
            "ridge": "Ridge Regression  (interpretable, R²=0.9775)",
            "knn":   "KNN  (k=11, R²=0.9905 — slower inference)",
        }[x],
        index=0,
    )

    tab_manual, tab_paste, tab_csv = st.tabs(
        ["🎛️  Manual Input", "📝  Paste Items", "📁  Import CSV"]
    )

    features: dict | None = None

    # ── Tab 1: Manual sliders ──────────────────────────────────────────────────
    with tab_manual:
        st.markdown(
            "<p style='font-size:13px;color:#5A5A5A;margin-bottom:16px;'>"
            "Adjust each feature to match your grocery basket.</p>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            basket_size      = st.slider("Basket Size (# items)",             1,    100,   15)
            total_calories   = st.slider("Total Calories (kcal)",              0,  10000, 2000, step=50)
            produce_share    = st.slider("Produce Share (0–1)",              0.0,    1.0, 0.25, step=0.01)
            processed_share  = st.slider("Processed Share (0–1)",           0.0,    1.0, 0.20, step=0.01)
            dept_diversity   = st.slider("Dept Diversity (unique depts/n)", 0.0,    1.0, 0.35, step=0.01)
        with c2:
            fiber_density    = st.slider("Fiber Density (g/item)",           0.0,   20.0,  2.0, step=0.1)
            protein_density  = st.slider("Protein Density (g/item)",         0.0,   50.0,  5.0, step=0.5)
            sugar_density    = st.slider("Sugar Density (g/item)",           0.0,   50.0,  5.0, step=0.5)
            sodium_density   = st.slider("Sodium Density (mg/item)",         0.0, 1000.0, 150.0, step=5.0)
            beverage_share   = st.slider("Beverage Share (0–1)",             0.0,    1.0, 0.10, step=0.01)

        if st.button("🔍 Predict Health Score", key="btn_manual"):
            features = dict(
                basket_size=basket_size, total_calories=total_calories,
                produce_share=produce_share, processed_share=processed_share,
                fiber_density=fiber_density, protein_density=protein_density,
                sugar_density=sugar_density, sodium_density=sodium_density,
                dept_diversity=dept_diversity, beverage_share=beverage_share,
            )
            st.session_state["features"]    = features
            st.session_state["run_predict"] = True

    # ── Tab 2: Paste items ─────────────────────────────────────────────────────
    with tab_paste:
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

        if st.button("🔍 Predict Health Score", key="btn_paste") and items_text.strip():
            features = items_to_features(items_text)
            st.session_state["features"]    = features
            st.session_state["run_predict"] = True
            with st.expander("Estimated features (from item keywords)"):
                st.dataframe(
                    pd.DataFrame([features]).T.rename(columns={0: "estimated value"}),
                    use_container_width=True,
                )

    # ── Tab 3: CSV import ──────────────────────────────────────────────────────
    with tab_csv:
        st.markdown(
            "<p style='font-size:13px;color:#5A5A5A;margin-bottom:12px;'>"
            "Upload a CSV with pre-computed basket features.<br>"
            f"Required columns: <code>{', '.join(FEATURE_COLS)}</code></p>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df_csv = pd.read_csv(uploaded)
            st.dataframe(df_csv.head(), use_container_width=True)
            missing = [c for c in FEATURE_COLS if c not in df_csv.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                row_idx = st.selectbox(
                    "Row to predict",
                    range(len(df_csv)),
                    format_func=lambda i: f"Row {i + 1}",
                )
                if st.button("🔍 Predict Health Score", key="btn_csv"):
                    features = df_csv.iloc[row_idx][FEATURE_COLS].to_dict()
                    st.session_state["features"]    = features
                    st.session_state["run_predict"] = True

    # ── Results ────────────────────────────────────────────────────────────────
    if st.session_state.get("run_predict") and st.session_state.get("features"):
        f = st.session_state["features"]

        if bundle is None:
            st.warning("Model bundle not loaded — cannot predict.")
        else:
            score = predict_phs(f, model_choice)
            if score is None:
                st.error("Prediction failed. Check the console for details.")
            else:
                t_label, t_color, t_bg = tier(score)

                st.markdown("---")
                st.markdown('<div class="nr-section-label">Results</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="nr-section-title">Your Purchase Health Score</div>',
                    unsafe_allow_html=True,
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
elif "Insights" in page:
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
        use_container_width=True,
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
