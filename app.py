"""
app.py
======
Beautiful Streamlit UI for the Image Matching System.
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import time
import cv2
from PIL import Image
import io

# ── Page config — MUST be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="Image Matching System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — dark theme, clean cards, glowing buttons ───────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:   #0a0e1a;
    --bg-card:      #111827;
    --bg-card2:     #1a2235;
    --border:       #1e293b;
    --accent-cyan:  #00d4ff;
    --accent-green: #00ff88;
    --accent-purple:#8b5cf6;
    --text-primary: #e2e8f0;
    --text-muted:   #64748b;
    --text-code:    #7dd3fc;
    --success:      #10b981;
    --warning:      #f59e0b;
    --danger:       #ef4444;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Main container ── */
.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a1040 50%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at 20% 50%, rgba(0,212,255,0.07) 0%, transparent 60%),
                radial-gradient(circle at 80% 20%, rgba(139,92,246,0.07) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #8b5cf6, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1rem;
    color: #64748b;
    font-weight: 400;
    margin: 0;
}
.hero-badges {
    display: flex;
    gap: 0.6rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
}
.badge.purple {
    background: rgba(139,92,246,0.1);
    border-color: rgba(139,92,246,0.3);
    color: #a78bfa;
}
.badge.green {
    background: rgba(0,255,136,0.1);
    border-color: rgba(0,255,136,0.25);
    color: #00ff88;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00d4ff;
    margin: 0 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,212,255,0.3), transparent);
}

/* ── Upload cards ── */
.upload-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    transition: border-color 0.2s;
}
.upload-card:hover {
    border-color: rgba(0,212,255,0.3);
}

/* ── Pipeline stage indicators ── */
.pipeline-stages {
    display: flex;
    gap: 0;
    margin: 1.5rem 0;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}
.stage-item {
    flex: 1;
    padding: 0.8rem 0.5rem;
    text-align: center;
    font-size: 0.72rem;
    font-weight: 500;
    color: #475569;
    border-right: 1px solid var(--border);
    position: relative;
}
.stage-item:last-child { border-right: none; }
.stage-item.active { color: #00d4ff; background: rgba(0,212,255,0.05); }
.stage-item.done   { color: #00ff88; background: rgba(0,255,136,0.04); }
.stage-num {
    display: block;
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    opacity: 0.6;
    margin-bottom: 0.2rem;
}

/* ── Stats grid ── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 0.75rem;
    margin: 1rem 0;
}
.stat-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
}
.stat-label {
    font-size: 0.65rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.stat-value {
    font-size: 1.3rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #e2e8f0;
}
.stat-value.green { color: #00ff88; }
.stat-value.cyan  { color: #00d4ff; }
.stat-value.yellow{ color: #fbbf24; }

/* ── Result image container ── */
.result-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}
.result-label {
    padding: 0.7rem 1rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    border-bottom: 1px solid var(--border);
    background: var(--bg-card2);
}

/* ── Run button ── */
.stButton > button {
    background: linear-gradient(135deg, #0066ff, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 20px rgba(0,102,255,0.3) !important;
    transition: all 0.2s !important;
    width: 100% !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(0,102,255,0.45) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-primary) !important;
}

/* ── Streamlit elements ── */
.stFileUploader {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(0,212,255,0.03) !important;
    border: 1.5px dashed rgba(0,212,255,0.25) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(0,212,255,0.5) !important;
    background: rgba(0,212,255,0.06) !important;
}
.stSelectbox > div, .stSlider > div {
    background: var(--bg-card2) !important;
}
div[data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Success / error messages ── */
.success-banner {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3);
    border-left: 3px solid #10b981;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
    color: #6ee7b7;
    font-size: 0.9rem;
}
.error-banner {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 3px solid #ef4444;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
    color: #fca5a5;
    font-size: 0.9rem;
}
.info-banner {
    background: rgba(0,212,255,0.06);
    border: 1px solid rgba(0,212,255,0.2);
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
    color: #7dd3fc;
    font-size: 0.88rem;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #00d4ff, #8b5cf6) !important;
    border-radius: 4px !important;
}

/* ── Image captions ── */
.img-caption {
    font-size: 0.72rem;
    color: #475569;
    text-align: center;
    margin-top: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card2) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Import pipeline ──────────────────────────────────────────────────────────
from utils import load_image_from_upload, numpy_to_pil
from main  import run_matching_pipeline


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🔍 Image Matching System</h1>
    <p class="hero-sub">Hybrid Deep Learning Pipeline · IIIT Sonepat Research Project</p>
    <div class="hero-badges">
        <span class="badge">Stage 1: Hybrid Extraction</span>
        <span class="badge purple">Stage 2: Attention Matching</span>
        <span class="badge green">Stage 3: Hierarchical Refinement</span>
        <span class="badge">Stage 4: RANSAC Filtering</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Pipeline flow indicator ───────────────────────────────────────────────
st.markdown("""
<div class="pipeline-stages">
  <div class="stage-item">
    <span class="stage-num">01</span>Hybrid<br>Extraction
  </div>
  <div class="stage-item">
    <span class="stage-num">02</span>Attention<br>Matching
  </div>
  <div class="stage-item">
    <span class="stage-num">03</span>Hierarchical<br>Refinement
  </div>
  <div class="stage-item">
    <span class="stage-num">04</span>RANSAC<br>Filter
  </div>
  <div class="stage-item">
    <span class="stage-num">05</span>Visual<br>Output
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Settings
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")
    st.markdown("---")

    st.markdown("**🖼️ Feature Extraction**")
    feature_mode = st.selectbox(
        "Extraction Mode",
        options=['hybrid', 'sift_only', 'orb_only'],
        index=0,
        help="hybrid = SIFT+ORB fusion (best quality)\nsift_only = most accurate\norb_only = fastest"
    )

    n_features = st.slider(
        "Max Keypoints per Image",
        min_value=200, max_value=3000, value=1200, step=100,
        help="More keypoints = more matches but slower"
    )

    st.markdown("---")
    st.markdown("**🤝 Attention Matching**")

    max_layers = st.slider(
        "Max Attention Layers",
        min_value=1, max_value=9, value=7, step=1,
        help="PDF uses 9 layers. Fewer = faster on CPU."
    )

    confidence_thresh = st.slider(
        "Early Exit Confidence",
        min_value=0.5, max_value=0.99, value=0.80, step=0.01,
        help="Higher = more layers used before exiting"
    )

    ratio_thresh = st.slider(
        "Lowe's Ratio Threshold",
        min_value=0.5, max_value=0.9, value=0.75, step=0.01,
        help="Lower = stricter match quality"
    )

    st.markdown("---")
    st.markdown("**🔬 Refinement & Filtering**")

    use_hierarchical = st.checkbox(
        "Hierarchical Refinement",
        value=True,
        help="Multi-scale 0.25x→0.5x→1x (slower but more accurate)"
    )

    ransac_thresh = st.slider(
        "RANSAC Threshold (px)",
        min_value=1.0, max_value=10.0, value=4.0, step=0.5,
        help="Max reprojection error for inlier matches"
    )

    st.markdown("---")
    st.markdown("**🖥️ Performance**")
    max_img_size = st.select_slider(
        "Max Image Size",
        options=[512, 640, 768, 1024],
        value=768,
        help="Larger = better quality but slower on CPU"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#334155; line-height:1.6;">
    <b style="color:#475569">About this project:</b><br>
    Implements the 4-stage pipeline from the paper:<br>
    <i>Image Matching using Hybrid Deep Learning</i><br>
    by Ramswroop Ogra, IIIT Sonepat (2026)
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — Upload Section
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">📁 Upload Images</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("**Image 1**  `— reference`")
    uploaded1 = st.file_uploader(
        "Upload first image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        key='img1',
        label_visibility='collapsed'
    )
    if uploaded1:
        st.image(uploaded1, use_container_width=True)
        st.markdown(f'<p class="img-caption">📎 {uploaded1.name}</p>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("**Image 2**  `— to match`")
    uploaded2 = st.file_uploader(
        "Upload second image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        key='img2',
        label_visibility='collapsed'
    )
    if uploaded2:
        st.image(uploaded2, use_container_width=True)
        st.markdown(f'<p class="img-caption">📎 {uploaded2.name}</p>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Info box when no images uploaded
if not uploaded1 or not uploaded2:
    st.markdown("""
    <div class="info-banner">
    💡 <b>Tip:</b> Upload two photos of the same scene from different angles, 
    lighting conditions, or distances — e.g., two photos of the same building, 
    monument, room, or object. The system will find and visualize matching features.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# RUN BUTTON
# ════════════════════════════════════════════════════════════════════════════
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    run_btn = st.button(
        "🚀  Run Image Matching Pipeline",
        disabled=(uploaded1 is None or uploaded2 is None),
        use_container_width=True
    )

if run_btn and uploaded1 and uploaded2:

    # ── Load images ────────────────────────────────────────────────────────
    with st.spinner("Loading images..."):
        try:
            uploaded1.seek(0)
            uploaded2.seek(0)
            img1_bgr, img1_rgb, img1_gray, size1 = load_image_from_upload(
                uploaded1, max_size=max_img_size)
            img2_bgr, img2_rgb, img2_gray, size2 = load_image_from_upload(
                uploaded2, max_size=max_img_size)
        except Exception as e:
            st.markdown(f'<div class="error-banner">❌ Failed to load images: {e}</div>',
                        unsafe_allow_html=True)
            st.stop()

    # ── Progress UI ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">⚙️ Pipeline Execution</p>',
                unsafe_allow_html=True)

    progress_bar  = st.progress(0)
    status_text   = st.empty()
    stage_cols    = st.columns(4)
    stage_statuses = [st.empty() for _ in range(4)]

    stage_labels = [
        "🔍 Feature Extraction",
        "🤝 Attention Matching",
        "🔬 Hierarchical Refine",
        "✅ RANSAC Filter"
    ]
    for i, (col, lbl) in enumerate(zip(stage_cols, stage_labels)):
        with col:
            stage_statuses[i].markdown(
                f'<div style="text-align:center;font-size:0.72rem;'
                f'color:#334155;padding:0.5rem;background:#111827;'
                f'border-radius:8px;border:1px solid #1e293b;">'
                f'⏳ {lbl}</div>', unsafe_allow_html=True)

    def progress_cb(stage_name, pct):
        progress_bar.progress(int(pct))
        status_text.markdown(
            f'<p style="font-size:0.82rem;color:#64748b;">{stage_name}</p>',
            unsafe_allow_html=True)

        # Update stage indicators
        stage_map = {
            'Stage 1': 0, 'Stage 2': 1, 'Stage 3': 2, 'Stage 4': 3
        }
        for s_key, idx in stage_map.items():
            if s_key in stage_name:
                with stage_cols[idx]:
                    stage_statuses[idx].markdown(
                        f'<div style="text-align:center;font-size:0.72rem;'
                        f'color:#00d4ff;padding:0.5rem;background:rgba(0,212,255,0.07);'
                        f'border-radius:8px;border:1px solid rgba(0,212,255,0.2);">'
                        f'⚡ {stage_labels[idx]}</div>', unsafe_allow_html=True)

    # ── Run pipeline ───────────────────────────────────────────────────────
    t_start = time.time()
    result  = run_matching_pipeline(
        img1_rgb, img1_gray,
        img2_rgb, img2_gray,
        n_features=n_features,
        feature_mode=feature_mode,
        max_attention_layers=max_layers,
        confidence_threshold=confidence_thresh,
        ratio_threshold=ratio_thresh,
        ransac_thresh=ransac_thresh,
        use_hierarchical=use_hierarchical,
        progress_callback=progress_cb
    )
    t_total = time.time() - t_start

    # Mark all stages done
    for i, col in enumerate(stage_cols):
        with col:
            stage_statuses[i].markdown(
                f'<div style="text-align:center;font-size:0.72rem;'
                f'color:#00ff88;padding:0.5rem;background:rgba(0,255,136,0.06);'
                f'border-radius:8px;border:1px solid rgba(0,255,136,0.2);">'
                f'✅ {stage_labels[i]}</div>', unsafe_allow_html=True)

    progress_bar.progress(100)
    status_text.empty()

    # ── Check result ───────────────────────────────────────────────────────
    if not result['success']:
        st.markdown(f"""
        <div class="error-banner">
        ❌ <b>Matching Failed:</b> {result.get('error', 'Unknown error')}<br>
        <small>Try uploading images of the same scene with more visual overlap.</small>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ════════════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    n_inliers = result['ransac_stats']['n_inliers']
    n_total   = result['match_stats']['n_raw']
    ratio     = result['ransac_stats']['inlier_ratio']

    if n_inliers >= 20:
        quality = "Excellent"
        quality_color = "#00ff88"
        quality_icon  = "🟢"
    elif n_inliers >= 10:
        quality = "Good"
        quality_color = "#fbbf24"
        quality_icon  = "🟡"
    elif n_inliers >= 4:
        quality = "Weak"
        quality_color = "#f97316"
        quality_icon  = "🟠"
    else:
        quality = "Poor"
        quality_color = "#ef4444"
        quality_icon  = "🔴"

    st.markdown(f"""
    <div class="success-banner">
    {quality_icon} <b>Matching Complete!</b> — Quality: 
    <span style="color:{quality_color};font-weight:700;">{quality}</span> &nbsp;|&nbsp;
    {n_inliers} inlier matches &nbsp;|&nbsp; {ratio}% inlier ratio &nbsp;|&nbsp;
    {t_total*1000:.0f}ms total
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics Row ────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📊 Pipeline Metrics</p>',
                unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    metrics_data = [
        (m1, "Keypoints (Img1)",
         str(result['feature_info']['n_kpts1']), "cyan"),
        (m2, "Keypoints (Img2)",
         str(result['feature_info']['n_kpts2']), "cyan"),
        (m3, "Raw Matches",
         str(result['match_stats']['n_raw']), "yellow"),
        (m4, "Inlier Matches",
         str(result['ransac_stats']['n_inliers']), "green"),
        (m5, "Inlier Ratio",
         f"{result['ransac_stats']['inlier_ratio']}%", "green"),
        (m6, "Total Time",
         f"{t_total*1000:.0f}ms", "cyan"),
    ]
    for col, label, value, color in metrics_data:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value {color}">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Keypoints visualization ────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🔵 Detected Keypoints (Stage 1)</p>',
                unsafe_allow_html=True)

    kp_col1, kp_col2 = st.columns(2)
    with kp_col1:
        st.markdown('<div class="result-container">'
                    '<div class="result-label">IMAGE 1 KEYPOINTS</div>',
                    unsafe_allow_html=True)
        st.image(result['kpts1_img'], use_container_width=True)
        st.markdown(
            f'<p class="img-caption">'
            f'{result["feature_info"]["n_kpts1"]} keypoints detected</p>'
            '</div>', unsafe_allow_html=True)

    with kp_col2:
        st.markdown('<div class="result-container">'
                    '<div class="result-label">IMAGE 2 KEYPOINTS</div>',
                    unsafe_allow_html=True)
        st.image(result['kpts2_img'], use_container_width=True)
        st.markdown(
            f'<p class="img-caption">'
            f'{result["feature_info"]["n_kpts2"]} keypoints detected</p>'
            '</div>', unsafe_allow_html=True)

    # ── Final match result ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">🟢 Final Matching Result (Stages 2–4)</p>',
                unsafe_allow_html=True)

    st.markdown('<div class="result-container">'
                '<div class="result-label">'
                '🟢 INLIER MATCHES (green) · 🔴 OUTLIERS (red)</div>',
                unsafe_allow_html=True)
    st.image(result['final_result_img'], use_container_width=True)
    st.markdown(
        f'<p class="img-caption">'
        f'{n_inliers} verified inlier matches · {n_total} total raw matches</p>'
        '</div>', unsafe_allow_html=True)

    # ── Raw matches (before RANSAC) ────────────────────────────────────────
    with st.expander("🔎 View Raw Matches (before RANSAC)", expanded=False):
        st.markdown('<div class="result-container">'
                    '<div class="result-label">RAW MATCHES (before geometric filtering)</div>',
                    unsafe_allow_html=True)
        st.image(result['raw_result_img'], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Attention + Refinement Stats ───────────────────────────────────────
    with st.expander("⚙️ Detailed Pipeline Stats", expanded=False):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Stage 2 — Attention Matching**")
            st.markdown(f"""
            <div style="font-size:0.85rem;line-height:2;font-family:'JetBrains Mono',monospace;color:#94a3b8;">
            Exit Layer: <span style="color:#00d4ff">{result['match_stats']['exit_layer']}</span> / {max_layers}<br>
            Confidence: <span style="color:#00ff88">{result['match_stats']['confidence']:.4f}</span><br>
            Raw Matches: <span style="color:#fbbf24">{result['match_stats']['n_raw']}</span><br>
            Stage Time: <span style="color:#a78bfa">{result['match_stats']['time_ms']:.0f}ms</span>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("**Stage 3 — Hierarchical Refinement**")
            ri = result['refinement_info']
            st.markdown(f"""
            <div style="font-size:0.85rem;line-height:2;font-family:'JetBrains Mono',monospace;color:#94a3b8;">
            0.25x (Coarse): <span style="color:#00d4ff">{ri.get('0.25x', 0)} matches</span><br>
            0.50x (Refine): <span style="color:#00d4ff">{ri.get('0.5x', 0)} matches</span><br>
            1.00x (Final):  <span style="color:#00ff88">{ri.get('1.0x', 0)} matches</span><br>
            Total Refined:  <span style="color:#fbbf24">{ri.get('total_refined', 0)}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**All Stats**")
        for k, v in result['pipeline_stats'].items():
            col_k, col_v = st.columns([3, 2])
            with col_k:
                st.markdown(f'<span style="font-size:0.82rem;color:#475569;">{k}</span>',
                            unsafe_allow_html=True)
            with col_v:
                st.markdown(f'<span style="font-size:0.82rem;color:#94a3b8;'
                            f'font-family:\'JetBrains Mono\',monospace;">{v}</span>',
                            unsafe_allow_html=True)

    # ── Download button ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">💾 Export Results</p>', unsafe_allow_html=True)

    dl_col1, dl_col2, dl_col3 = st.columns(3)

    # Convert result images to bytes
    final_pil   = numpy_to_pil(result['final_result_img'])
    final_buf   = io.BytesIO()
    final_pil.save(final_buf, format='JPEG', quality=92)
    final_bytes = final_buf.getvalue()

    # Fix: pad images to same height before hstacking
    kp1 = result['kpts1_img']
    kp2 = result['kpts2_img']
    h_max = max(kp1.shape[0], kp2.shape[0])
    def pad_to_height(img, h):
        if img.shape[0] == h:
            return img
        pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
        return np.vstack([img, pad])
    kp_combined = np.hstack([pad_to_height(kp1, h_max), pad_to_height(kp2, h_max)])
    kp_pil      = numpy_to_pil(kp_combined)
    kp_buf      = io.BytesIO()
    kp_pil.save(kp_buf, format='JPEG', quality=90)
    kp_bytes    = kp_buf.getvalue()

    raw_pil = numpy_to_pil(result['raw_result_img'])
    raw_buf = io.BytesIO()
    raw_pil.save(raw_buf, format='JPEG', quality=90)
    raw_bytes = raw_buf.getvalue()

    with dl_col1:
        st.download_button(
            "⬇️ Final Match Result",
            data=final_bytes,
            file_name="match_result.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    with dl_col2:
        st.download_button(
            "⬇️ Keypoints Image",
            data=kp_bytes,
            file_name="keypoints.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    with dl_col3:
        st.download_button(
            "⬇️ Raw Matches",
            data=raw_bytes,
            file_name="raw_matches.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.72rem;color:#1e293b;padding:1rem 0;">
    Image Matching System · IIIT Sonepat · Hybrid Deep Learning Pipeline ·
    SIFT + ORB + Attention + Hierarchical Refinement + RANSAC
</div>
""", unsafe_allow_html=True)
