import streamlit as st
import pickle, json, numpy as np

# Load model files
import os
BASE = os.path.dirname(__file__)

model     = pickle.load(open(os.path.join(BASE, "model.pkl"), "rb"))
encoder   = pickle.load(open(os.path.join(BASE, "encoder.pkl"), "rb"))
locations = json.load(open(os.path.join(BASE, "locations.json")))
# Page config
st.set_page_config(page_title="BLR House Price AI", page_icon="🏙️", layout="centered")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700;900&family=Inter:wght@400;600&display=swap');

/* Background */
.stApp {
    background-color: #020c04;
    background-image:
        radial-gradient(ellipse at 20% 20%, #003300 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, #001a00 0%, transparent 50%);
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Main title */
.main-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 900;
    color: #00ff41;
    text-align: center;
    text-shadow: 0 0 20px #00ff41, 0 0 40px #00cc33;
    letter-spacing: 2px;
    margin-bottom: 4px;
}
.sub-title {
    font-family: 'Share Tech Mono', monospace;
    color: #00aa28;
    text-align: center;
    font-size: 0.95rem;
    letter-spacing: 3px;
    margin-bottom: 30px;
}

/* Section headers */
.section-label {
    font-family: 'Share Tech Mono', monospace;
    color: #00ff41;
    font-size: 0.8rem;
    letter-spacing: 2px;
    margin-bottom: 6px;
    text-transform: uppercase;
}

/* All input boxes, selects */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #061a08 !important;
    border: 1px solid #00aa28 !important;
    border-radius: 8px !important;
    color: #00ff41 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
div[data-baseweb="select"] * { color: #00ff41 !important; }

/* Slider */
div[data-testid="stSlider"] > div > div > div {
    background: #00ff41 !important;
}
div[data-testid="stSlider"] label {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00cc33 !important;
}

/* Number input */
input[type="number"] {
    background: #061a08 !important;
    color: #00ff41 !important;
    border: 1px solid #00aa28 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Button */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #003300, #006600) !important;
    color: #00ff41 !important;
    border: 1px solid #00ff41 !important;
    border-radius: 8px !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 2px !important;
    width: 100% !important;
    padding: 14px !important;
    text-shadow: 0 0 10px #00ff41 !important;
    box-shadow: 0 0 15px #00440022 !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] button:hover {
    box-shadow: 0 0 25px #00ff4155 !important;
    background: linear-gradient(135deg, #004400, #008800) !important;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #020c04, #041a06);
    border: 1px solid #00ff41;
    border-radius: 16px;
    padding: 30px 24px;
    text-align: center;
    box-shadow: 0 0 30px #00ff4133, inset 0 0 30px #00110033;
    margin-top: 20px;
}
.result-label {
    font-family: 'Share Tech Mono', monospace;
    color: #00aa28;
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.result-price {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    color: #00ff41;
    text-shadow: 0 0 20px #00ff41, 0 0 50px #00cc33;
    margin: 10px 0 4px;
}
.result-unit {
    font-family: 'Share Tech Mono', monospace;
    color: #00cc33;
    font-size: 0.9rem;
    letter-spacing: 2px;
}
.result-chips {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    margin-top: 18px;
}
.chip {
    background: #041a06;
    border: 1px solid #00aa2855;
    border-radius: 6px;
    padding: 5px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #00cc33;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 12px;
    margin: 28px 0 20px;
}
.stat-box {
    flex: 1;
    background: #020c04;
    border: 1px solid #00440033;
    border-radius: 12px;
    padding: 16px 8px;
    text-align: center;
}
.stat-num {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    color: #00ff41;
    text-shadow: 0 0 10px #00ff41;
}
.stat-desc {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #006622;
    margin-top: 4px;
    letter-spacing: 1px;
}

/* Divider */
.green-divider {
    border: none;
    border-top: 1px solid #00440055;
    margin: 20px 0;
}

/* Labels for widgets */
label, .stSelectbox label, .stNumberInput label, .stSlider label {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00cc33 !important;
    letter-spacing: 1px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
# ── Credits (fixed top right) ───────────────────────────────────────────────
st.markdown("""
<div style="
    position: fixed;
    top: 56px;
    right: 18px;
    background: #020c04;
    border: 1px solid #00440077;
    border-radius: 10px;
    padding: 12px 16px;
    z-index: 9999;
    text-align: right;
    box-shadow: 0 0 12px #00ff4122;
">
    <div style="font-family:'Share Tech Mono',monospace; color:#00aa28; font-size:0.68rem; letter-spacing:2px; margin-bottom:8px; text-transform:uppercase;">Done By</div>
    <div style="font-family:'Share Tech Mono',monospace; color:#00ff41; font-size:0.75rem; line-height:2; text-shadow: 0 0 8px #00ff4177;">
        R.Harish<br>
        D.A.Kadhiravan<br>
        T.Bhavesh<br>
        S.Dinesh<br>
        P.Jishnu<br>
        S.Sundar
    </div>
</div>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏙️ BLR HOUSE AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">> BENGALURU REAL ESTATE PRICE PREDICTOR</div>', unsafe_allow_html=True)

# Stats bar
st.markdown("""
<div class="stats-row">
  <div class="stat-box"><div class="stat-num">13K+</div><div class="stat-desc">PROPERTIES</div></div>
  <div class="stat-box"><div class="stat-num">30+</div><div class="stat-desc">LOCATIONS</div></div>
  <div class="stat-box"><div class="stat-num">56%</div><div class="stat-desc">R² SCORE</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

# ── Inputs ───────────────────────────────────────────────────────────────────
location = st.selectbox("📍 LOCATION", locations)
sqft     = st.number_input("📐 TOTAL SQUARE FEET", min_value=100, max_value=10000, value=1200, step=50)

col1, col2 = st.columns(2)
with col1:
    bhk  = st.slider("🛏 BHK", 1, 6, 2)
with col2:
    bath = st.slider("🚿 BATHROOMS", 1, 6, 2)

balcony = st.slider("🌿 BALCONIES", 0, 3, 1)

st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("⚡  RUN PREDICTION"):
    loc_enc = encoder.transform([location if location in encoder.classes_ else "Other"])[0]
    price   = max(round(model.predict(np.array([[sqft, bath, balcony, bhk, loc_enc]]))[0], 2), 5)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">> ESTIMATED MARKET PRICE</div>
        <div class="result-price">₹ {price}</div>
        <div class="result-unit">LAKHS ( INDIAN RUPEES )</div>
        <div class="result-chips">
            <div class="chip">📍 {location}</div>
            <div class="chip">📐 {sqft} sq.ft</div>
            <div class="chip">🛏 {bhk} BHK</div>
            <div class="chip">🚿 {bath} Bath</div>
            <div class="chip">🌿 {balcony} Balcony</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
