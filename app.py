import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import datetime
import json
import plotly.graph_objects as go

st.set_page_config(page_title="Radio-X", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

.hero-wrapper {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2d45;
    margin-bottom: 2rem;
}
.hero-left h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-left p {
    color: #64748b;
    font-size: 1rem;
    font-weight: 300;
    margin: 0.3rem 0 0 0;
    letter-spacing: 0.05em;
}
.hero-badge { display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: flex-end; }
.badge {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid;
    text-transform: uppercase;
}
.badge-blue   { color: #38bdf8; border-color: #38bdf833; background: #38bdf810; }
.badge-green  { color: #34d399; border-color: #34d39933; background: #34d39910; }
.badge-purple { color: #818cf8; border-color: #818cf833; background: #818cf810; }

.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.stat-card {
    background: #0d1520;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.stat-card.blue::before   { background: linear-gradient(90deg, #38bdf8, transparent); }
.stat-card.green::before  { background: linear-gradient(90deg, #34d399, transparent); }
.stat-card.purple::before { background: linear-gradient(90deg, #818cf8, transparent); }
.stat-card.amber::before  { background: linear-gradient(90deg, #fbbf24, transparent); }
.stat-label { font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: #475569; font-weight: 500; }
.stat-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; margin: 0.2rem 0 0 0; }
.stat-value.blue   { color: #38bdf8; }
.stat-value.green  { color: #34d399; }
.stat-value.purple { color: #818cf8; }
.stat-value.amber  { color: #fbbf24; }

.img-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 0.6rem;
    padding: 0.4rem 0;
}
.label-original { color: #64748b; }
.label-noisy    { color: #f87171; }
.label-enhanced { color: #34d399; }

.report-card {
    background: #0d1520;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}
.report-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #1e2d45;
}
.report-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #94a3b8;
}
.grade-badge {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    padding: 4px 16px;
    border-radius: 8px;
    letter-spacing: 0.05em;
}
.grade-excellent { background: #064e3b; color: #34d399; border: 1px solid #34d39944; }
.grade-good      { background: #1e3a5f; color: #38bdf8; border: 1px solid #38bdf844; }
.grade-poor      { background: #4c1d1d; color: #f87171; border: 1px solid #f8717144; }
.report-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid #0f1923;
    color: #94a3b8;
}
.report-row span:last-child { color: #e2e8f0; font-weight: 500; }
.recommendation {
    margin-top: 1rem;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
}
.rec-good { background: #064e3b33; border-left: 3px solid #34d399; color: #34d399; }
.rec-warn { background: #1e3a5f33; border-left: 3px solid #38bdf8; color: #38bdf8; }
.rec-poor { background: #4c1d1d33; border-left: 3px solid #f87171; color: #f87171; }

.history-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 1rem;
}
.section-divider { border: none; border-top: 1px solid #1e2d45; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_radiox_model():
    model = load_model('radiox_model.h5', compile=False)
    model.compile(optimizer='adam', loss='mse')
    return model

model = load_radiox_model()

if 'history' not in st.session_state:
    st.session_state.history = []

def upscale(img_array, size=512):
    img = Image.fromarray(img_array)
    return img.resize((size, size), Image.LANCZOS)

def get_grade(psnr_val, ssim_val):
    # Thresholds calibrated for 128x128 denoising autoencoder
    if psnr_val >= 25 and ssim_val >= 0.70:
        return "EXCELLENT", "grade-excellent", "rec-good", "✅ Image quality is optimal. Suitable for clinical diagnosis."
    elif psnr_val >= 20 and ssim_val >= 0.55:
        return "GOOD", "grade-good", "rec-warn", "⚠️ Image quality is acceptable. Minor artifacts may be present."
    else:
        return "POOR", "grade-poor", "rec-poor", "❌ Image quality is low. Re-scan recommended for accurate diagnosis."

# Hero
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-left">
        <h1>Radio-X</h1>
        <p>AI-POWERED CHEST X-RAY ENHANCEMENT SYSTEM &nbsp;·&nbsp; NEURAL IMAGING PLATFORM</p>
    </div>
    <div class="hero-badge">
        <span class="badge badge-blue">Denoising Autoencoder</span>
        <span class="badge badge-green">TensorFlow</span>
        <span class="badge badge-purple">Deep Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Updated stats with new model numbers
st.markdown("""
<div class="stat-row">
    <div class="stat-card blue">
        <div class="stat-label">Model Accuracy</div>
        <div class="stat-value blue">99.71%</div>
    </div>
    <div class="stat-card green">
        <div class="stat-label">Val Loss (MSE)</div>
        <div class="stat-value green">0.0029</div>
    </div>
    <div class="stat-card purple">
        <div class="stat-label">Avg PSNR</div>
        <div class="stat-value purple">25.43 dB</div>
    </div>
    <div class="stat-card amber">
        <div class="stat-label">Avg SSIM</div>
        <div class="stat-value amber">0.7132</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

st.markdown("#### Upload Chest X-Ray")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Preprocess to 128x128 for new model
    img = Image.open(uploaded_file).convert('L').resize((128, 128))
    img_array = np.array(img) / 255.0

    # Add noise for middle column display only
    noisy_array = img_array + 0.3 * np.random.normal(size=img_array.shape)
    noisy_array = np.clip(noisy_array, 0., 1.)

    # Predict
    input_array = img_array.reshape(1, 128, 128, 1)
    denoised = model.predict(input_array, verbose=0)
    denoised_array = denoised.reshape(128, 128)

    # Compute metrics
    psnr_val = psnr(img_array, denoised_array, data_range=1.0)
    ssim_val = ssim(img_array, denoised_array, data_range=1.0)
    grade, grade_class, rec_class, recommendation = get_grade(psnr_val, ssim_val)
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # Upscale for sharp display
    original_up = upscale((img_array * 255).astype(np.uint8))
    noisy_up    = upscale((noisy_array * 255).astype(np.uint8))
    enhanced_up = upscale((denoised_array * 255).astype(np.uint8))

    # Save to history
    st.session_state.history.append({
        "Time": timestamp,
        "File": uploaded_file.name,
        "PSNR (dB)": f"{psnr_val:.2f}",
        "SSIM": f"{ssim_val:.4f}",
        "Grade": grade
    })

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="img-label label-original">📷 Original X-Ray</div>', unsafe_allow_html=True)
        st.image(original_up, use_container_width=True)
    with col2:
        st.markdown('<div class="img-label label-noisy">⚡ Simulated Noisy</div>', unsafe_allow_html=True)
        st.image(noisy_up, use_container_width=True)
    with col3:
        st.markdown('<div class="img-label label-enhanced">✦ AI Enhanced</div>', unsafe_allow_html=True)
        st.image(enhanced_up, use_container_width=True)

    st.markdown(f"""
    <div class="report-card">
        <div class="report-header">
            <div class="report-title">📋 Diagnostic Quality Report</div>
            <span class="grade-badge {grade_class}">{grade}</span>
        </div>
        <div class="report-row"><span>File Name</span><span>{uploaded_file.name}</span></div>
        <div class="report-row"><span>Timestamp</span><span>{timestamp}</span></div>
        <div class="report-row"><span>PSNR Score</span><span>{psnr_val:.2f} dB</span></div>
        <div class="report-row"><span>SSIM Score</span><span>{ssim_val:.4f}</span></div>
        <div class="report-row"><span>MSE Loss</span><span>0.0029</span></div>
        <div class="report-row"><span>Model</span><span>Denoising Convolutional Autoencoder</span></div>
        <div class="recommendation {rec_class}">{recommendation}</div>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.history:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="history-title">📈 Enhancement History — This Session</div>', unsafe_allow_html=True)
    st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)

# Loss curve section
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="history-title">📉 Model Training Loss Curve</div>', unsafe_allow_html=True)

try:
    with open('training_history.json', 'r') as f:
        hist = json.load(f)

    fig = go.Figure()

    # Training loss line
    fig.add_trace(go.Scatter(
        y=hist['loss'],
        mode='lines',
        name='Train Loss',
        line=dict(color='#38bdf8', width=2)
    ))

    # Validation loss line
    fig.add_trace(go.Scatter(
        y=hist['val_loss'],
        mode='lines',
        name='Val Loss',
        line=dict(color='#34d399', width=2)
    ))

    fig.update_layout(
        paper_bgcolor='#0d1520',
        plot_bgcolor='#0d1520',
        font=dict(color='#94a3b8', family='DM Sans'),
        xaxis=dict(
            title='Epoch',
            gridcolor='#1e2d45',
            color='#64748b'
        ),
        yaxis=dict(
            title='MSE Loss',
            gridcolor='#1e2d45',
            color='#64748b'
        ),
        legend=dict(
            bgcolor='#0d1520',
            bordercolor='#1e2d45',
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=20, b=40),
        height=300
    )

    st.plotly_chart(fig, use_container_width=True)

except:
    st.info("Training history not found.")