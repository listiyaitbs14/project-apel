import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import datetime
import io
import base64
import time
import random
import cv2
import tempfile
import os
from fpdf import FPDF

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Sistem Klasifikasi Kualitas Apel",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# ASET: kelopak bunga apel & daun
# ============================================================
PETAL_SVG = """<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><g fill="currentColor"><ellipse cx="20" cy="9" rx="6.2" ry="9.2" transform="rotate(0 20 20)"/><ellipse cx="20" cy="9" rx="6.2" ry="9.2" transform="rotate(72 20 20)"/><ellipse cx="20" cy="9" rx="6.2" ry="9.2" transform="rotate(144 20 20)"/><ellipse cx="20" cy="9" rx="6.2" ry="9.2" transform="rotate(216 20 20)"/><ellipse cx="20" cy="9" rx="6.2" ry="9.2" transform="rotate(288 20 20)"/></g><circle cx="20" cy="20" r="3.4" fill="#C7A468"/></svg>"""

LEAF_SVG = """<svg viewBox="0 0 24 40" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C3 11 2 24 12 38C22 24 21 11 12 2Z" fill="currentColor"/><path d="M12 7V33" stroke="rgba(255,255,255,0.35)" stroke-width="1"/></svg>"""

def _petal_span(cls):
    return f'<span class="petal {cls}">{PETAL_SVG}</span>'

AMBIENT_PETALS_HTML = '<div class="petal-field" aria-hidden="true">' + "".join(
    _petal_span(f"p{i}") for i in range(1, 10)
) + '</div>'

BLOOM_LOADER_HTML = f"""
<div class="bloom-wrap">
  <div class="bloom-ring"><div class="bloom-flower">{PETAL_SVG}</div></div>
  <p class="bloom-caption">Menganalisis mutu buah…</p>
</div>
"""

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500&family=Quicksand:wght@400;500;600;700&display=swap');

:root {
    --blush-1: #FBF1EC;
    --blush-2: #F6E2DA;
    --cream: #FFFBF8;
    --rose-soft: #E6C3B8;
    --rose-mid: #CE9A8E;
    --rose-deep: #9C6B5E;
    --mauve-ink: #4A3A38;
    --mauve-fade: #8A7470;
    --gold-foil: #C7A468;
    --rust: #BD7666;
}

html, body, .stApp {
    background: linear-gradient(160deg, var(--blush-1) 0%, var(--blush-2) 55%, var(--blush-1) 100%);
    color: var(--mauve-ink);
    font-family: 'Quicksand', sans-serif;
}

.block-container, [data-testid="stAppViewContainer"] .block-container {
    max-width: 620px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    position: relative;
    z-index: 1;
}

/* ---------- Ambient petal field ---------- */
.petal-field { position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: 0; }
.petal { position: absolute; top: -10vh; color: var(--rose-soft); opacity: 0; animation: petal-drift linear infinite; }
.petal.leafy { color: var(--rust); }
@keyframes petal-drift {
    0%   { transform: translateY(-10vh) translateX(0) rotate(0deg); opacity: 0; }
    8%   { opacity: .5; }
    92%  { opacity: .45; }
    100% { transform: translateY(110vh) translateX(36px) rotate(280deg); opacity: 0; }
}
.petal.p1{left:4%;  width:15px; height:15px; animation-duration:23s; animation-delay:0s;}
.petal.p2{left:16%; width:22px; height:22px; animation-duration:29s; animation-delay:4s; color:var(--rose-mid);}
.petal.p3{left:29%; width:13px; height:13px; animation-duration:19s; animation-delay:7s;}
.petal.p4{left:44%; width:18px; height:18px; animation-duration:27s; animation-delay:1s; color:var(--gold-foil);}
.petal.p5{left:58%; width:16px; height:16px; animation-duration:25s; animation-delay:10s;}
.petal.p6{left:71%; width:24px; height:24px; animation-duration:31s; animation-delay:5s; color:var(--rose-mid);}
.petal.p7{left:82%; width:14px; height:14px; animation-duration:21s; animation-delay:8s;}
.petal.p8{left:91%; width:18px; height:18px; animation-duration:28s; animation-delay:2.5s;}
.petal.p9{left:37%; width:11px; height:11px; animation-duration:18s; animation-delay:13s;}

/* ---------- Header ---------- */
.av-header { text-align: center; margin-bottom: 1.2rem; }
.av-eyebrow {
    font-size: .7rem; letter-spacing: 2.5px; text-transform: uppercase;
    color: var(--rose-deep); font-weight: 600;
}
.av-title {
    font-family: 'Playfair Display', serif; font-weight: 600;
    font-size: clamp(2rem, 6vw, 2.65rem); margin: 6px 0 4px 0;
    background: linear-gradient(120deg, var(--rose-deep), var(--gold-foil) 50%, var(--rose-mid));
    background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; animation: title-shimmer 7s ease-in-out infinite;
}
@keyframes title-shimmer { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
.av-subtitle { color: var(--mauve-fade); font-size: .92rem; font-style: italic; font-family: 'Playfair Display', serif; }

/* ---------- Banner kustom ---------- */
.av-banner {
    display: flex; align-items: flex-start; gap: 10px;
    background: var(--b-bg); border: 1px solid var(--b-accent); border-radius: 14px;
    padding: 12px 16px; margin: 14px 0; font-size: .9rem; line-height: 1.5;
}
.av-banner-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--b-accent); margin-top: 6px; flex-shrink: 0; }
.av-banner code { background: rgba(74,58,56,.08); padding: 1px 6px; border-radius: 5px; font-family: monospace; font-size: .85em; }
.av-banner b { color: var(--mauve-ink); }

/* ---------- Uploader & gambar preview ---------- */
[data-testid="stFileUploaderDropzone"], .stFileUploader section {
    background: var(--cream) !important;
    border: 1.5px dashed var(--rose-soft) !important;
    border-radius: 16px !important;
    color: #FFFFFF !important;
}

[data-testid="stFileUploader"] label p {
    color: var(--rose-deep) !important;
    font-weight: 600;
}
            
[data-testid="stFileUploaderDropzone"] small {
    color: rgba(255, 255, 255, 0.8) !important; 
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: var(--rose-deep) !important;
    color: #FFFFFF !important;
    border-color: var(--rose-deep) !important;
}
            
[data-testid="stImage"] img, .stImage img {
    border-radius: 16px; border: 1px solid var(--rose-soft);
    box-shadow: 0 8px 22px rgba(156,107,94,.16);
}

/* ---------- Tombol ---------- */
.stButton button, [data-testid="stButton"] button {
    background: linear-gradient(135deg, var(--rose-mid), var(--rose-deep));
    color: var(--cream); border: none; border-radius: 999px;
    padding: .65rem 1.4rem; font-family: 'Quicksand', sans-serif; font-weight: 600;
    letter-spacing: .3px; box-shadow: 0 6px 18px rgba(156,107,94,.32);
    transition: transform .25s ease, box-shadow .25s ease;
}
.stButton button:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(156,107,94,.42); }
.stButton button:active { transform: translateY(0); }
.stButton button:focus-visible { outline: 2px solid var(--gold-foil); outline-offset: 3px; }

/* ---------- Bloom loader ---------- */
.bloom-wrap { display:flex; flex-direction:column; align-items:center; margin: 50px 0; }
.bloom-ring {
    width: 92px; height: 92px; border-radius: 50%;
    border: 2px solid var(--rose-soft); border-top-color: var(--gold-foil);
    display: flex; align-items:center; justify-content:center; animation: ring-spin 2.6s linear infinite;
}
.bloom-flower { width: 42px; height: 42px; color: var(--rose-mid); animation: bloom-pulse 1.7s ease-in-out infinite; }
@keyframes ring-spin { to { transform: rotate(360deg); } }
@keyframes bloom-pulse { 0%,100% { transform: scale(.85); opacity:.7; } 50% { transform: scale(1.05); opacity:1; } }
.bloom-caption { margin-top: 16px; font-family: 'Playfair Display', serif; font-style: italic; color: var(--rose-deep); font-size: .95rem; }

/* ---------- Kartu Blossom ---------- */
.blossom-card-container { perspective: 1100px; display:flex; justify-content:center; margin: 28px 0; position:relative; z-index:1; }
.blossom-card {
    width: 100%; max-width: 330px; background: var(--cream); border-radius: 22px; padding: 18px;
    border: 1px solid var(--rose-soft);
    box-shadow: 0 4px 10px rgba(156,107,94,.1), 0 18px 38px rgba(156,107,94,.18);
    position: relative; animation: card-reveal .55s cubic-bezier(.16,1,.3,1) both;
    transition: transform .3s ease, box-shadow .3s ease;
}
.blossom-card:hover { transform: translateY(-4px) rotateX(2deg); box-shadow: 0 8px 16px rgba(156,107,94,.14), 0 26px 46px rgba(156,107,94,.24); }
@keyframes card-reveal { 0% { transform: rotateY(85deg) scale(.92); opacity: 0; } 100% { transform: rotateY(0deg) scale(1); opacity: 1; } }
.blossom-card::after { content:''; position:absolute; inset:6px; border:1px solid var(--rose-soft); border-radius:16px; pointer-events:none; opacity:.6; }
.bc-eyebrow { display:block; text-align:center; font-size:.62rem; letter-spacing:2px; text-transform:uppercase; color:var(--rose-deep); font-weight:600; margin-bottom:4px; }
.bc-title-row { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid var(--blush-2); }
.bc-title { font-family:'Playfair Display', serif; font-weight:600; font-size:1.2rem; color:var(--mauve-ink); margin:0; }
.bc-seal {
    color: var(--accent-deep); background: var(--accent-soft); border: 1px solid var(--accent-deep);
    border-radius: 50%; width: 50px; height: 50px;
    display:flex; align-items:center; justify-content:center; flex-direction:column;
}
.bc-seal-num { font-family:'Playfair Display', serif; font-weight:600; font-size:.8rem; }
.bc-seal small { font-size:.46rem; font-weight:500; opacity:.8; }
.bc-photo { width:100%; height:190px; border-radius:14px; overflow:hidden; border:1px solid var(--rose-soft); margin-bottom:12px; background:var(--blush-1); }
.bc-photo img { width:100%; height:100%; object-fit:cover; }
.bc-status-line { display:flex; align-items:center; gap:8px; margin-bottom:12px; font-size:.85rem; }
.bc-dot { width:9px; height:9px; border-radius:50%; background:var(--accent-deep); flex-shrink:0; }
.bc-status-text { font-weight:600; color:var(--mauve-ink); }
.bc-confidence { margin-left:auto; color:var(--mauve-fade); font-size:.78rem; }
.bc-stats { background:var(--blush-1); border-radius:14px; padding:12px 14px; border:1px solid var(--blush-2); }
.bc-stat-row { margin-bottom:10px; }
.bc-stat-row:last-child { margin-bottom:0; }
.bc-stat-label { display:flex; justify-content:space-between; font-size:.72rem; text-transform:uppercase; letter-spacing:.4px; color:var(--mauve-fade); margin-bottom:4px; font-weight:600; }
.bc-stat-val { color:var(--mauve-ink); }
.bc-bar-track { width:100%; height:6px; background:var(--blush-2); border-radius:6px; overflow:hidden; }
.bc-bar-fill { height:100%; border-radius:6px; background:var(--accent-deep); transition: width 1.1s ease-out; }
.bc-bar-fill.gold { background: linear-gradient(90deg, var(--gold-foil), var(--rose-mid)); }
.bc-stat-row.meta { font-size:.74rem; color:var(--mauve-fade); display:flex; justify-content:flex-end; }
.bc-footer { text-align:center; margin-top:12px; font-size:.66rem; letter-spacing:1px; text-transform:uppercase; color:var(--mauve-fade); }

/* ---------- One-shot shower kelopak ---------- */
.petal-shower { position: fixed; inset:0; overflow:hidden; pointer-events:none; z-index: 2; }
.fall { position:absolute; top:-8vh; opacity:0; animation: fall-once 2.6s ease-in forwards; }
@keyframes fall-once {
    0%   { transform: translateY(0) rotate(0deg); opacity: 0; }
    12%  { opacity: .85; }
    100% { transform: translateY(60vh) rotate(220deg); opacity: 0; }
}

[data-testid="stExpander"] {
    background: var(--cream) !important;
    border: 1px solid var(--rose-soft) !important;
    border-radius: 14px !important;
}
[data-testid="stExpander"] summary {
    background: var(--blush-1) !important;
    color: var(--mauve-ink) !important;
    border-radius: 14px !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: var(--mauve-ink) !important;
}
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, var(--rose-mid), var(--rose-deep)) !important;
    color: var(--cream) !important;
    border: none !important;
}

/* ---------- Footer ---------- */
.av-footer-divider { display:flex; align-items:center; gap:10px; margin: 30px 0 8px 0; }
.av-footer-divider span { flex:1; height:1px; background: var(--rose-soft); }
.av-footer-divider i { font-style:normal; color: var(--rose-deep); font-size:.9rem; }
.av-footer-note { text-align:center; font-size:.78rem; color:var(--mauve-fade); font-style:italic; font-family:'Playfair Display', serif; }
            
/* ---------- Modal popup "bukan apel" ---------- */
.av-modal-overlay {
    position: fixed; inset: 0; z-index: 999;
    background: rgba(74,58,56,.55);
    backdrop-filter: blur(2px);
    display: flex; align-items: center; justify-content: center;
    animation: overlay-fade .25s ease both;
}
@keyframes overlay-fade { from { opacity: 0; } to { opacity: 1; } }
.av-modal-card {
    background: var(--cream); border-radius: 20px; padding: 30px 26px 85px 26px;
    max-width: 340px; width: 88%; text-align: center;
    border: 1px solid var(--rose-soft);
    box-shadow: 0 24px 60px rgba(0,0,0,.28);
    animation: shake-once .55s cubic-bezier(.36,.07,.19,.97) both;
}

/* ---------- Library card riwayat ---------- */
.history-card {
    display: flex; align-items: center; gap: 12px;
    background: var(--h-bg); border: 1px solid var(--h-accent);
    border-radius: 14px; padding: 12px 14px; margin: 8px 0 2px 0;
}
.history-card-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--h-accent); flex-shrink: 0;
}
.history-card-info { display: flex; flex-direction: column; }
.history-card-title { font-weight: 600; color: var(--mauve-ink); font-size: .92rem; }
.history-card-sub { font-size: .74rem; color: var(--mauve-fade); }
.history-detail {
    background: var(--cream); border-left: 3px solid var(--h-accent);
    border-radius: 10px; padding: 10px 14px; margin: 0 0 12px 0;
    font-size: .82rem; color: var(--mauve-ink); line-height: 1.5;
}

.st-key-popup_close_btn {
    position: fixed;
    z-index: 1000;
    top: 63%;
    left: 50%;
    transform: translateX(-50%);
    width: 84%;
    max-width: 320px;
    transition: none !important;
}
.st-key-popup_close_btn * {
    transition: none !important;
}
            
@keyframes shake-once {
    10%, 90% { transform: translateX(-1px); }
    20%, 80% { transform: translateX(3px); }
    30%, 50%, 70% { transform: translateX(-6px); }
    40%, 60% { transform: translateX(6px); }
}
.av-modal-icon { font-size: 2.4rem; margin-bottom: 6px; }
.av-modal-title { font-family: 'Playfair Display', serif; font-weight: 600; font-size: 1.2rem; color: var(--rust); margin: 4px 0 8px 0; }
.av-modal-text { color: var(--mauve-ink); font-size: .88rem; line-height: 1.5; margin-bottom: 4px; }

/* ---------- Aksesibilitas ---------- */
@media (prefers-reduced-motion: reduce) {
    .petal, .av-title, .bloom-ring, .bloom-flower, .blossom-card, .fall, .av-modal-card, .av-modal-overlay { animation: none !important; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNGSI BANTU
# ============================================================
def get_image_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def render_banner(text, kind="success"): 
    palette = {
        "success": ("#6F8560", "rgba(143,163,126,.16)"),
        "error":   ("var(--rust)", "rgba(189,118,102,.14)"),
    }
    accent, bg = palette.get(kind, palette["success"])
    st.markdown(
        f'<div class="av-banner" style="--b-accent:{accent}; --b-bg:{bg};">'
        f'<span class="av-banner-dot"></span><span>{text}</span></div>',
        unsafe_allow_html=True
    )

def render_history_dashboard():
    if not st.session_state.history:
        st.caption("Belum ada riwayat scan.")
        return

    accent_map = {
        "Apel Segar": ("#6F8560", "rgba(143,163,126,.16)"),
        "Apel Busuk": ("#8A4F44", "rgba(189,118,102,.14)"),
    }

    for i, item in enumerate(reversed(st.session_state.history)):
        real_idx = len(st.session_state.history) - 1 - i
        accent, bg = accent_map.get(item["class"], ("#9C6B5E", "rgba(156,107,94,.12)"))

        st.markdown(f"""
        <div class="history-card" style="--h-accent:{accent}; --h-bg:{bg};">
            <div class="history-card-dot"></div>
            <div class="history-card-info">
                <span class="history-card-title">Scan ke-{real_idx+1:03d} — {item['class']}</span>
                <span class="history-card-sub">
                    {item['timestamp']} · {item['confidence']*100:.0f}% yakin ·
                    Skor: <b>{item['grade']:.1f}/10</b>
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_petal_shower(mood="fresh", count=12):
    svg = PETAL_SVG if mood == "fresh" else LEAF_SVG
    color_a, color_b = (("var(--rose-mid)", "var(--gold-foil)") if mood == "fresh" else ("var(--rust)", "var(--mauve-fade)"))
    dur_range = (2.0, 3.0) if mood == "fresh" else (3.2, 4.4)
    items = []
    for i in range(count):
        left = random.uniform(2, 96)
        delay = random.uniform(0, 1.3)
        duration = random.uniform(*dur_range)
        size = random.uniform(13, 25)
        top_start = random.uniform(2, 9)
        color = color_a if i % 3 else color_b
        items.append(
            f'<span class="fall" style="left:{left:.1f}%; top:-{top_start:.0f}vh; '
            f'width:{size:.0f}px; height:{size:.0f}px; color:{color}; '
            f'animation-delay:{delay:.2f}s; animation-duration:{duration:.2f}s;">{svg}</span>'
        )
    return '<div class="petal-shower" aria-hidden="true">' + "".join(items) + '</div>'

def generate_pdf_report(image, predicted_class, confidence, grade, timestamp_now):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.convert("RGB").save(tmp.name)
        img_path = tmp.name

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Laporan Analisis Kualitas Apel", ln=True, align="C")
    pdf.ln(4)
    pdf.image(img_path, x=55, w=100)
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Status: {predicted_class}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence*100:.1f}%", ln=True)
    pdf.cell(0, 8, f"Skor: {grade:.1f}/10", ln=True)
    pdf.cell(0, 8, f"Waktu Scan: {timestamp_now}", ln=True)

    os.remove(img_path)
    return bytes(pdf.output(dest="S"))

def extract_frame_from_video(video_bytes, frame_position=0.5):
    """
    Ambil 1 frame dari video buat diproses kayak foto biasa.
    frame_position: 0.0 = awal, 0.5 = tengah, 1.0 = akhir video
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = int(total_frames * frame_position)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    success, frame = cap.read()
    cap.release()
    os.remove(tmp_path)

    if not success:
        raise ValueError("Gagal ambil frame dari video, filenya mungkin corrupt.")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def render_invalid_popup():
    st.markdown("""
    <div class="av-modal-overlay">
        <div class="av-modal-card">
            <div class="av-modal-icon">⚠️</div>
            <h3 class="av-modal-title">Ini Bukan Apel</h3>
            <p class="av-modal-text">
                Gambar yang kamu unggah sepertinya <b>bukan foto apel</b>.
                Coba unggah ulang dengan foto apel yang jelas, ya!
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MODEL CNN (3 KELAS: busuk / segar / bukan apel)
# ============================================================
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('model_apel_cnn.h5')

model = load_my_model()
class_names = ['Apel Busuk', 'Apel Segar', 'Bukan Apel']

if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0
if "show_invalid_popup" not in st.session_state:
    st.session_state.show_invalid_popup = False
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# RENDER HALAMAN
# ============================================================
st.markdown(AMBIENT_PETALS_HTML, unsafe_allow_html=True)

st.markdown("""
<div class="av-header">
    <span class="av-eyebrow">✿ ANALISIS KUALITAS BERBASIS CNN ✿</span>
    <h1 class="av-title">Sistem Klasifikasi Apel</h1>
    <p class="av-subtitle" style="margin-bottom: 2rem;">Sebuah ruang digital untuk menilai kebaikan dan kualitas buah apel</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Unggah Foto atau Video Hasil Kebunmu",
    type=["jpg", "jpeg", "png", "mp4", "mov", "jfif", "avif"]
)

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    if file_ext in ["mp4", "mov"]:
        with st.spinner("Ngambil frame dari video..."):
            image = extract_frame_from_video(uploaded_file.read(), frame_position=0.5)
        st.image(image, caption="Pratinjau (frame dari video)", use_container_width=True)
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Pratinjau", use_container_width=True)

    if st.button("✿ Mulai Analisis", use_container_width=True):

        loading_placeholder = st.empty()
        loading_placeholder.markdown(BLOOM_LOADER_HTML, unsafe_allow_html=True)
        time.sleep(1.6)

        img_resized = image.convert("RGB").resize((180, 180))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_idx = np.argmax(score)
        predicted_class = class_names[predicted_idx]
        confidence = float(np.max(score))
        
        CONFIDENCE_THRESHOLD = 0.40
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_class = 'Bukan Apel'

        loading_placeholder.empty()

        if predicted_class == 'Bukan Apel':
            st.session_state.show_invalid_popup = True
        else:
            st.session_state.show_invalid_popup = False

            timestamp_now = datetime.datetime.now().strftime("%H:%M:%S WIB")
            st.session_state.scan_count += 1
            edition_no = st.session_state.scan_count

            if predicted_class == 'Apel Segar':
                grade = max(6.0, confidence * 10)
                tampil_kesegaran = confidence * 100
                accent_deep = "#6F8560"
                accent_soft = "rgba(143,163,126,.18)"
                status_label = "Apel Segar (Fresh)"
                mood = "fresh"
            else:
                grade = min(5.0, (1.0 - confidence) * 6)
                tampil_kesegaran = (1.0 - confidence) * 100
                accent_deep = "#8A4F44"
                accent_soft = "rgba(189,118,102,.18)"
                status_label = "Apel Busuk (Rotten)"
                mood = "wilted"

            base64_image = get_image_base64(image)

            card_html = f"""
            <div class="blossom-card-container">
                <div class="blossom-card" style="--accent-deep:{accent_deep}; --accent-soft:{accent_soft};">
                    <span class="bc-eyebrow">DIGITAL APPLE INSPECTION</span>
                    <div class="bc-title-row">
                        <h3 class="bc-title">Hasil Analisis</h3>
                        <div class="bc-seal"><span class="bc-seal-num">{grade:.1f}</span><small>/10</small></div>
                    </div>
                    <div class="bc-photo"><img src="{base64_image}" alt="Apel"></div>
                    <div class="bc-status-line">
                        <span class="bc-dot"></span>
                        <span class="bc-status-text">{status_label}</span>
                    </div>
                    <div class="bc-stats">
                        <div class="bc-stat-row">
                            <div class="bc-stat-label"><span>Akurasi Prediksi</span><span class="bc-stat-val">{tampil_kesegaran:.0f}%</span></div>
                            <div class="bc-bar-track"><div class="bc-bar-fill" style="width:{tampil_kesegaran:.0f}%;"></div></div>
                        </div>
                        <div class="bc-stat-row">
                            <div class="bc-stat-label"><span>Skor Keseluruhan</span><span class="bc-stat-val">{grade:.1f}/10</span></div>
                            <div class="bc-bar-track"><div class="bc-bar-fill gold" style="width:{grade*10:.0f}%;"></div></div>
                        </div>
                        <div class="bc-stat-row meta"><span>Dipindai {timestamp_now}</span></div>
                    </div>
                    <div class="bc-footer">SCANNER ID: {edition_no:03d}</div>
                </div>
            </div>
            """

            st.markdown(card_html, unsafe_allow_html=True)

            st.markdown(render_petal_shower(mood=mood, count=14 if mood == "fresh" else 9), unsafe_allow_html=True)

            if predicted_class == 'Apel Segar':
                render_banner(f"Apel berada dalam kondisi segar optimal dengan skor <b>{grade:.1f}/10</b>.", kind="success")
            else:
                render_banner(f"Kualitas buah menurun dengan skor {grade:.1f}/10 — sebaiknya jangan dikonsumsi.", kind="error")

            st.session_state.history.append({
                "class": predicted_class,
                "confidence": confidence,
                "timestamp": timestamp_now,
                "grade": grade
            })

            pdf_bytes = generate_pdf_report(image, predicted_class, confidence, grade, timestamp_now)
            st.download_button(
                "📄 Download Laporan PDF",
                data=pdf_bytes,
                file_name=f"laporan_apel_{edition_no:03d}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# -------- Popup "bukan apel" --------
if st.session_state.show_invalid_popup:
    popup_slot = st.empty()
    with popup_slot.container():
        render_invalid_popup()
        with st.container(key="popup_close_btn"):
            if st.button("Coba Lagi", use_container_width=True, key="close_invalid_popup"):
                popup_slot.empty()
                st.session_state.show_invalid_popup = False
                st.rerun()

with st.expander("📊 Riwayat Scan"):
    render_history_dashboard()

st.markdown("""
<div class="av-footer-divider"><span></span><i>✿</i><span></span></div>
""", unsafe_allow_html=True)