import streamlit as st
import numpy as np
import pandas as pd
import cv2
import io
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── IMPORT MODULES ─────────────────────────────────────────
from config import ALL_ROIS
from corner_detection import find_corner_bubbles, warp_perspective
from scanner import detect_answers, detect_nama, detect_nim, detect_tanggal
from handwriting_ocr import load_or_train, predict_text, postprocess
from eda import grade_from_score, calculate_score
from utils import show_heatmap, show_heatmap_jawaban, show_handwriting

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="LJK Scanner",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --navy:    #111844;
  --mid:     #4B5694;
  --steel:   #7288AE;
  --cream:   #EAE0CF;
  --cream2:  #F5F0E8;
  --white:   #FFFFFF;
  --success: #3D8B6E;
  --warn:    #C47C2B;
  --danger:  #B34040;
  --text:    #EAE0CF;
  --text-dim:#7288AE;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--navy) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--cream) !important;
}
[data-testid="stAppViewContainer"] > .main {
  background: transparent !important;
}
[data-testid="block-container"] {
  padding: 2rem 2.5rem 3rem !important;
}

[data-testid="stSidebar"] {
  background: #0C1235 !important;
  border-right: 1px solid rgba(74,86,148,0.3) !important;
}
[data-testid="stSidebar"] * { color: var(--cream) !important; }
[data-testid="stSidebar"] .stButton > button {
  background: rgba(74,86,148,0.15) !important;
  border: 1px solid rgba(74,86,148,0.4) !important;
  color: var(--cream) !important;
  width: 100%;
  border-radius: 8px;
  font-size: 0.8rem;
  padding: 6px 10px;
  transition: all .2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(74,86,148,0.35) !important;
  border-color: var(--steel) !important;
}

h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: var(--cream) !important; }
.serif-title {
  font-family: 'DM Serif Display', serif;
  font-size: 2.4rem;
  color: var(--cream);
  line-height: 1.15;
  letter-spacing: -0.02em;
}
.serif-title span {
  color: var(--cream);
  font-style: italic;
  background: linear-gradient(135deg, #EAE0CF, #7288AE);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #9BAABF;
  margin-bottom: 0.4rem;
}
.page-subtitle {
  color: var(--steel);
  font-size: 0.95rem;
  font-weight: 300;
  margin-top: 0.3rem;
}

.step-nav {
  display: flex;
  gap: 0;
  margin-bottom: 2rem;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(74,86,148,0.3);
}
.step-item {
  flex: 1;
  padding: 12px 8px;
  text-align: center;
  font-size: 0.78rem;
  font-weight: 500;
  background: rgba(255,255,255,0.02);
  color: var(--text-dim);
  transition: all .2s;
  border-right: 1px solid rgba(74,86,148,0.2);
  cursor: default;
}
.step-item:last-child { border-right: none; }
.step-item.active {
  background: linear-gradient(180deg, rgba(74,86,148,0.3), rgba(17,24,68,0.4));
  color: var(--cream);
  font-weight: 600;
  border-bottom: 2px solid #EAE0CF;
}
.step-item.done {
  background: rgba(74,86,148,0.12);
  color: #7288AE;
}
.step-icon { font-size: 1rem; display: block; margin-bottom: 3px; }

.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(74,86,148,0.25);
  border-top: 2px solid rgba(234,224,207,0.3);
  border-radius: 16px;
  padding: 1.2rem 1.5rem 0.8rem;
  margin-bottom: 1rem;
}
.card-sm {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(74,86,148,0.2);
  border-radius: 12px;
  padding: 1rem 1.2rem;
}

.metrics-row {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  margin: 1.2rem 0;
}
.metric-tile {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(74,86,148,0.2);
  border-top: 2px solid rgba(234,224,207,0.2);
  border-radius: 14px;
  padding: 18px 14px;
  text-align: center;
  transition: transform .2s, border-color .2s;
}
.metric-tile:hover { transform: translateY(-2px); border-color: var(--steel); }
.metric-tile .t-val {
  font-family: 'DM Serif Display', serif;
  font-size: 2rem;
  line-height: 1;
  margin-bottom: 6px;
}
.metric-tile .t-lbl {
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-dim);
}
.c-blue  { color: #7CA4D4; }
.c-green { color: #6DBF9E; }
.c-red   { color: #E07575; }
.c-grey  { color: #7288AE; }
.c-amber { color: #D4A96A; }
.c-cream { color: var(--cream); }

.answer-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(68px, 1fr));
  gap: 6px;
  margin-top: 0.8rem;
}
.ans-cell {
  border-radius: 8px;
  padding: 6px 4px;
  text-align: center;
  font-size: 0.72rem;
  font-family: 'JetBrains Mono', monospace;
  border: 1px solid rgba(74,86,148,0.2);
  background: rgba(255,255,255,0.02);
}
.ans-cell.correct { background: rgba(61,139,110,0.15); border-color: #3D8B6E; color: #6DBF9E; }
.ans-cell.wrong   { background: rgba(179,64,64,0.15);  border-color: #B34040; color: #E07575; }
.ans-cell.empty   { opacity: 0.45; }

.stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox select {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(74,86,148,0.35) !important;
  border-radius: 10px !important;
  color: var(--cream) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--steel) !important;
  box-shadow: 0 0 0 2px rgba(114,136,174,0.15) !important;
}
label, .stSelectbox label, .stTextInput label,
.stNumberInput label, .stTextArea label {
  color: var(--steel) !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
}
.stButton > button {
  background: linear-gradient(135deg, #4B5694, #3d4878) !important;
  color: #EAE0CF !important;
  border: 1px solid rgba(234,224,207,0.15) !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  padding: 10px 24px !important;
  transition: all .2s !important;
  letter-spacing: 0.02em !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #7288AE, #4B5694) !important;
  border-color: rgba(234,224,207,0.35) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 20px rgba(17,24,68,0.5) !important;
}
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.03) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 4px !important;
  border: 1px solid rgba(74,86,148,0.2) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-dim) !important;
  border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(74,86,148,0.35) !important;
  color: var(--cream) !important;
}
[data-testid="stExpander"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid rgba(74,86,148,0.2) !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
  color: var(--cream) !important;
  font-weight: 500 !important;
}
.stFileUploader {
  background: rgba(255,255,255,0.02) !important;
  border: 2px dashed rgba(74,86,148,0.4) !important;
  border-radius: 14px !important;
}
.stDownloadButton > button {
  background: rgba(61,139,110,0.2) !important;
  border: 1px solid rgba(61,139,110,0.5) !important;
  color: #6DBF9E !important;
}
.stDownloadButton > button:hover {
  background: rgba(61,139,110,0.35) !important;
}

hr { border-color: rgba(74,86,148,0.2) !important; }

.heatmap-wrap {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(74,86,148,0.2);
  border-radius: 14px;
  padding: 1.2rem;
}
.heatmap-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--steel);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 0.8rem;
}

.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.badge-blue  { background: rgba(114,136,174,0.2); color: #7CA4D4; border: 1px solid rgba(114,136,174,0.3); }
.badge-green { background: rgba(61,139,110,0.2);  color: #6DBF9E; border: 1px solid rgba(61,139,110,0.3); }
.badge-amber { background: rgba(196,124,43,0.2);  color: #D4A96A; border: 1px solid rgba(196,124,43,0.3); }
.badge-red   { background: rgba(179,64,64,0.2);   color: #E07575; border: 1px solid rgba(179,64,64,0.3); }

.prog-bar-wrap { margin: 6px 0; }
.prog-bar-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.72rem;
  color: var(--text-dim);
  margin-bottom: 3px;
}
.prog-bar-bg {
  background: rgba(74,86,148,0.15);
  border-radius: 4px;
  height: 6px;
  overflow: hidden;
}
.prog-bar-fill { height: 100%; border-radius: 4px; transition: width .6s ease; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(74,86,148,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────
def init_state():
    defaults = {
        'answer_key': {},
        'total_soal': 50,
        'sesi_nama':  '',
        'kode_kelas': '',
        'kode_dosen': '',
        'records':    [],
        'step':       'setup',
        'key_text':   '',
        'bundle':     None,  # SVM model
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

DEFAULT_50 = [
    'B', 'C', 'A', 'D', 'E', 'A', 'B', 'C', 'D', 'A',
    'E', 'B', 'C', 'A', 'D', 'B', 'E', 'A', 'C', 'D',
    'A', 'B', 'E', 'C', 'D', 'B', 'A', 'D', 'C', 'E',
    'A', 'C', 'B', 'D', 'E', 'C', 'A', 'B', 'D', 'E',
    'B', 'D', 'A', 'C', 'E', 'A', 'D', 'B', 'E', 'C',
]

def make_key_text(n):
    lines = []
    for i in range(1, n + 1):
        ans = DEFAULT_50[i - 1] if i <= 50 else 'A'
        lines.append(f"{i}. {ans}")
    return "\n".join(lines)

if not st.session_state.key_text:
    st.session_state.key_text = make_key_text(st.session_state.total_soal)

# ─── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 16px">
      <div style="font-family:'DM Serif Display',serif; font-size:1.9rem; color:#EAE0CF; line-height:1.1; letter-spacing:-0.02em">
        LJK Scanner
      </div>
      <div style="font-size:0.82rem; color:#7288AE; margin-top:6px; letter-spacing:0.1em; text-transform:uppercase; font-weight:500">
        Computer Vision Project
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    steps = [
        ('setup',       '⚙',  'Setup'),
        ('scan',        '📸', 'Scan'),
        ('handwriting', '✍️', 'OCR'),
        ('results',     '📊', 'Hasil'),
    ]
    cur = st.session_state.step
    cur_idx = [s[0] for s in steps].index(cur) if cur in [s[0] for s in steps] else 0

    for i, (s, icon, lbl) in enumerate(steps):
        if i < cur_idx:
            cls = "done"
            dot = "✓"
        elif i == cur_idx:
            cls = "active"
            dot = icon
        else:
            cls = ""
            dot = icon
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;'
            f'border-radius:8px;margin-bottom:4px;'
            f'background:{"rgba(74,86,148,0.2)" if cls == "active" else "rgba(61,139,110,0.1)" if cls == "done" else "transparent"};'
            f'border:1px solid {"rgba(74,86,148,0.4)" if cls == "active" else "rgba(61,139,110,0.25)" if cls == "done" else "transparent"}">'
            f'<span style="font-size:1rem">{dot}</span>'
            f'<span style="font-size:0.92rem;color:{"#EAE0CF" if cls in ["active", "done"] else "#4B5694"};'
            f'font-weight:{"600" if cls == "active" else "400"}">{lbl}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    records = st.session_state.records
    if records:
        scores = [r.get('score', 0) for r in records if r.get('processed')]
        st.markdown('<div class="section-label">Sesi Aktif</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card-sm" style="margin-bottom:8px">
          <div style="font-size:0.78rem;color:#7288AE;margin-bottom:8px">
            {st.session_state.sesi_nama or "—"}<br>
            <span style="font-size:0.68rem;font-family:monospace">{st.session_state.kode_kelas}</span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
            <div style="text-align:center">
              <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#EAE0CF">{len(records)}</div>
              <div style="font-size:0.65rem;color:#4B5694;text-transform:uppercase;letter-spacing:0.1em">Scanned</div>
            </div>
            <div style="text-align:center">
              <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#7CA4D4">{np.mean(scores) if scores else 0.0:.1f}</div>
              <div style="font-size:0.65rem;color:#4B5694;text-transform:uppercase;letter-spacing:0.1em">Avg</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
        for s in scores:
            g = grade_from_score(s)
            if g in grade_counts:
                grade_counts[g] += 1
        grade_colors = {
            'A': '#6DBF9E',
            'B': '#7CA4D4',
            'C': '#D4A96A',
            'D': '#E07575',
            'E': '#9B7E7E',
        }
        for g, cnt in grade_counts.items():
            if cnt == 0:
                continue
            pct = cnt / len(records) * 100
            st.markdown(f"""
            <div class="prog-bar-wrap">
              <div class="prog-bar-label"><span>Grade {g}</span><span>{cnt}</span></div>
              <div class="prog-bar-bg">
                <div class="prog-bar-fill" style="width:{pct}%;background:{grade_colors[g]}"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color:#4B5694;font-size:0.88rem;padding:8px 0;font-style:italic">Belum ada data scan</div>',
            unsafe_allow_html=True
        )

    st.divider()
    if st.button("↺  Reset Sesi"):
        for k in ['answer_key', 'records', 'step', 'sesi_nama', 'kode_kelas', 'kode_dosen', 'key_text', 'bundle']:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

# ─── STEP INDICATOR ────────────────────────────────────────
step_html = '<div class="step-nav">'
for i, (s, icon, lbl) in enumerate(steps):
    if i < cur_idx:
        cls = "done"
        badge_icon = "✓"
    elif i == cur_idx:
        cls = "active"
        badge_icon = icon
    else:
        cls = ""
        badge_icon = icon
    step_html += f'<div class="step-item {cls}"><span class="step-icon">{badge_icon}</span>{lbl}</div>'
step_html += '</div>'
st.markdown(step_html, unsafe_allow_html=True)


# ─── STEP: SETUP ───────────────────────────────────────────
if st.session_state.step == 'setup':
    st.markdown('<div class="section-label">Langkah 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-title">Setup <span>Sesi Ujian</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Konfigurasikan parameter ujian dan input kunci jawaban sebelum memulai scan.</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('''
        <div class="card">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.2rem;
            padding-bottom:12px;border-bottom:1px solid rgba(114,136,174,0.2)">
            <div style="width:3px;height:22px;background:linear-gradient(180deg,#EAE0CF,#7288AE);
              border-radius:2px;flex-shrink:0"></div>
            <div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.05rem;color:#EAE0CF">Informasi Sesi</div>
              <div style="font-size:0.68rem;color:#4B5694;letter-spacing:0.1em;text-transform:uppercase;margin-top:1px">Parameter ujian</div>
            </div>
          </div>
        ''', unsafe_allow_html=True)

        st.session_state.sesi_nama  = st.text_input("Nama Sesi / Mata Kuliah", value=st.session_state.sesi_nama or "Computer Vision UAS")
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.kode_kelas = st.text_input("Kode Kelas", value=st.session_state.kode_kelas or "LK01")
        with c2:
            st.session_state.kode_dosen = st.text_input("Kode Dosen", value=st.session_state.kode_dosen or "DS123")

        new_total = st.number_input("Jumlah Soal", min_value=1, max_value=100, value=st.session_state.total_soal)
        if new_total != st.session_state.total_soal:
            st.session_state.total_soal = new_total
            st.session_state.key_text = make_key_text(new_total)
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('''
        <div class="card">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.2rem;
            padding-bottom:12px;border-bottom:1px solid rgba(114,136,174,0.2)">
            <div style="width:3px;height:22px;background:linear-gradient(180deg,#EAE0CF,#7288AE);
              border-radius:2px;flex-shrink:0"></div>
            <div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.05rem;color:#EAE0CF">Kunci Jawaban</div>
              <div style="font-size:0.68rem;color:#4B5694;letter-spacing:0.1em;text-transform:uppercase;margin-top:1px">Format: 1. A, 2. B, ...</div>
            </div>
          </div>
        ''', unsafe_allow_html=True)

        key_text = st.text_area(
            "Format: `1. A`, `2. B`, ...",
            value=st.session_state.key_text,
            height=280,
            label_visibility="collapsed"
        )
        st.session_state.key_text = key_text

        answer_key = {}
        errors = []
        for line in key_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            if '. ' in line:
                parts = line.split('. ', 1)
            else:
                parts = line.split(',', 1)
            if len(parts) != 2:
                errors.append(f"Format salah: `{line}`")
                continue
            try:
                q   = int(parts[0].strip())
                ans = parts[1].strip().upper()
                if ans not in ['A', 'B', 'C', 'D', 'E']:
                    errors.append(f"Jawaban tidak valid di soal {q}: `{ans}`")
                    continue
                answer_key[q] = ans
            except ValueError:
                errors.append(f"Nomor tidak valid: `{line}`")

        if errors:
            for e in errors[:2]:
                st.error(e)
        else:
            total  = st.session_state.total_soal
            filled = len(answer_key)
            pct    = filled / total * 100 if total > 0 else 0
            color  = "#6DBF9E" if filled == total else "#D4A96A"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
              background:rgba(61,139,110,0.08);border:1px solid rgba(61,139,110,0.25);
              border-radius:10px;margin-top:8px">
              <span style="font-size:1.1rem">{'✅' if filled == total else '⚠️'}</span>
              <div>
                <div style="font-size:0.85rem;color:{color};font-weight:600">
                  {filled} / {total} kunci jawaban valid
                </div>
                <div style="font-size:0.7rem;color:#4B5694">{pct:.0f}% terisi</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.answer_key = answer_key

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("Mulai Scan  →", disabled=len(answer_key) == 0, width='stretch'):
            if not answer_key:
                st.error("Kunci jawaban belum diisi.")
            else:
                st.session_state.step = 'scan'
                st.rerun()

# ─── STEP: SCAN ────────────────────────────────────────────
elif st.session_state.step == 'scan':
    st.markdown('<div class="section-label">Langkah 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-title">Scan <span>Lembar Jawaban</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subtitle">'
        f'{st.session_state.sesi_nama} &nbsp;·&nbsp; '
        f'{st.session_state.kode_kelas} &nbsp;·&nbsp; '
        f'{st.session_state.total_soal} soal &nbsp;·&nbsp; '
        f'{len(st.session_state.records)} lembar ter-scan</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 5])
    with c1:
        if st.button("← Setup", use_container_width=True):
            st.session_state.step = 'setup'
            st.rerun()
    with c2:
        if st.button("Lihat OCR →", disabled=len(st.session_state.records) == 0, use_container_width=True):
            st.session_state.step = 'handwriting'
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:0.6rem">
      <div style="font-family:'DM Serif Display',serif;font-size:1.05rem;color:#EAE0CF;margin-bottom:4px">
        Upload atau Foto Langsung
      </div>
      <div style="font-size:0.82rem;color:#7288AE">Pilih salah satu cara di bawah untuk memasukkan LJK.</div>
    </div>
    """, unsafe_allow_html=True)

    upload_tab, camera_tab = st.tabs(["📁  Upload File", "📷  Ambil Foto (Webcam)"])

    uploaded_files = []

    with upload_tab:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _files = st.file_uploader(
            "Pilih foto LJK (JPG / PNG) — bisa lebih dari satu",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="visible",
        )
        if _files:
            uploaded_files = _files

    with camera_tab:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.8rem;color:#7288AE;margin-bottom:8px">'
            'Arahkan kamera ke LJK, pastikan semua sudut terlihat jelas.</div>',
            unsafe_allow_html=True
        )
        camera_image = st.camera_input("Ambil foto LJK", label_visibility="collapsed")
        if camera_image is not None:
            uploaded_files = [camera_image]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if uploaded_files:
        for uploaded in uploaded_files:
            fname = getattr(uploaded, 'name', None) or 'webcam_capture.jpg'
            if any(r['filename'] == fname for r in st.session_state.records):
                st.warning(f"⚠️ `{fname}` sudah di-scan, dilewati.")
                continue

            with st.expander(f"📄  {fname}", expanded=True):
                img_pil = Image.open(uploaded)
                img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                with st.spinner("Mendeteksi 4 sudut LJK..."):
                    selected = find_corner_bubbles(img_bgr, visualize=False)
                    warped   = warp_perspective(img_bgr, selected)

                col_orig, col_warp = st.columns(2, gap="medium")
                with col_orig:
                    st.markdown('<div class="section-label">Input Asli</div>', unsafe_allow_html=True)
                    st.image(img_pil, use_container_width=True)
                with col_warp:
                    st.markdown('<div class="section-label">Setelah Warp Perspective</div>', unsafe_allow_html=True)
                    if warped is not None and warped.shape[0] > 0:
                        st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.error("❌ Gagal mendeteksi 4 sudut LJK. Coba ulang dengan foto yang lebih jelas.")
                        continue

                record = {
                    'filename':   fname,
                    'img_bgr':    warped,
                    'answer_key': st.session_state.answer_key,
                    'total_soal': st.session_state.total_soal,
                    'processed':  False,
                }
                st.session_state.records.append(record)
                st.success(f"✅ `{fname}` berhasil diproses.")

# ─── STEP: OCR ─────────────────────────────────────────────
elif st.session_state.step == 'handwriting':
    st.markdown('<div class="section-label">Langkah 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-title">OCR <span>Tulisan Tangan & OMR</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subtitle">{st.session_state.sesi_nama} &nbsp;·&nbsp; {len(st.session_state.records)} lembar</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    records = st.session_state.records
    if not records:
        st.warning("Belum ada data scan.")
        if st.button("← Scan"):
            st.session_state.step = 'scan'
            st.rerun()
        st.stop()

    c1, c2, c3 = st.columns([1, 1, 5])
    with c1:
        if st.button("← Scan", width='stretch'):
            st.session_state.step = 'scan'
            st.rerun()
    with c2:
        if st.button("Lihat Hasil →", width='stretch'):
            st.session_state.step = 'results'
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.bundle is None:
        with st.spinner("Loading SVM model..."):
            st.session_state.bundle = load_or_train()

    bundle = st.session_state.bundle
    if bundle is None:
        st.error("❌ Gagal load model SVM. Pastikan `svm_emnist.pkl` ada di folder.")
        st.stop()

    for idx, record in enumerate(records):
        warped = record['img_bgr']
        fname  = record['filename']

        with st.expander(f"📝 {fname}", expanded=(not record.get('processed', False))):
            if not record.get('processed', False):
                with st.spinner(f"Memproses data & OMR untuk {fname}..."):
                    nama_text,    density_nama     = detect_nama(warped)
                    nim_text,     density_nim      = detect_nim(warped)
                    tgl_text,     density_tanggal  = detect_tanggal(warped)

                    x1, y1, x2, y2 = ALL_ROIS['NAMA_MATA_KULIAH']
                    roi_matkul = warped[y1:y2, x1:x2]
                    matkul_text, _, _ = predict_text(roi_matkul, bundle, label='NAMA_MATA_KULIAH')
                    matkul_text = postprocess('NAMA_MATA_KULIAH', matkul_text)
                    
                    x1, y1, x2, y2 = ALL_ROIS['KODE_KELAS']
                    roi_kode = warped[y1:y2, x1:x2]
                    kode_text, _, _ = predict_text(roi_kode, bundle, label='KODE_KELAS')
                    kode_text = postprocess('KODE_KELAS', kode_text)
                    
                    x1, y1, x2, y2 = ALL_ROIS['RUANGAN']
                    roi_ruangan = warped[y1:y2, x1:x2]
                    ruangan_text, _, _ = predict_text(roi_ruangan, bundle, label='RUANGAN')
                    ruangan_text = postprocess('RUANGAN', ruangan_text)
                    
                    x1, y1, x2, y2 = ALL_ROIS['NO_MEJA']
                    roi_nomeja = warped[y1:y2, x1:x2]
                    nomeja_text, _, _ = predict_text(roi_nomeja, bundle, label='NO_MEJA')
                    nomeja_text = postprocess('NO_MEJA', nomeja_text)

                    answers, density_jawaban = detect_answers(warped, st.session_state.total_soal)
                    benar, salah, kosong, score = calculate_score(
                        answers,
                        st.session_state.answer_key
                    )

                    record.update({
                        'nama':             nama_text,
                        'nim':              nim_text,
                        'tanggal':          tgl_text,
                        'kode_kelas':       kode_text,
                        'answers':          answers,
                        'score':            score,
                        'benar':            benar,
                        'salah':            salah,
                        'kosong':           kosong,
                        'density_nama':     density_nama,
                        'density_nim':      density_nim,
                        'density_tanggal':  density_tanggal,
                        'density_jawaban':  density_jawaban,
                        'processed':        True,
                        'matkul':           matkul_text,
                        'ruangan':          ruangan_text,
                        'no_meja':          nomeja_text,
                    })
            else:
                nama_text  = record['nama']
                nim_text   = record['nim']
                tgl_text   = record['tanggal']
                kode_text  = record['kode_kelas']
                matkul_text = record.get('matkul', '')
                ruangan_text = record.get('ruangan', '')
                nomeja_text = record.get('no_meja', '')
                answers    = record['answers']
                benar      = record['benar']
                salah      = record['salah']
                kosong     = record['kosong']
                score      = record['score']
                density_nama     = record.get('density_nama',    np.zeros((26, 20)))
                density_nim      = record.get('density_nim',     np.zeros((10, 10)))
                density_tanggal  = record.get('density_tanggal', np.zeros((10, 6)))
                density_jawaban  = record.get('density_jawaban', np.zeros((st.session_state.total_soal, 5)))
            # Identity card
            st.markdown(f"""
            <div class="card-sm" style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap">
              <div>
                <div class="section-label">Nama (OMR)</div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#EAE0CF">{nama_text or "—"}</div>
              </div>
              <div>
                <div class="section-label">NIM (OMR)</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;color:#7CA4D4">{nim_text or "—"}</div>
              </div>
              <div>
                <div class="section-label">Tanggal (OMR)</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;color:#EAE0CF">{tgl_text or "—"}</div>
              </div>
              <div>
                <div class="section-label">Kode Kelas</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;color:#7CA4D4">{kode_text or "—"}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Handwriting Recognition section
        st.markdown('<div class="section-label">Tulisan Tangan (Handwriting OCR)</div>', unsafe_allow_html=True)
        from utils import show_handwriting
        
        hw_col1, hw_col2 = st.columns(2)
        with hw_col1:
            show_handwriting(
                roi_img=warped[ALL_ROIS['NAMA_MATA_KULIAH'][1]:ALL_ROIS['NAMA_MATA_KULIAH'][3],
                               ALL_ROIS['NAMA_MATA_KULIAH'][0]:ALL_ROIS['NAMA_MATA_KULIAH'][2]],
                ocr_text=matkul_text,
                label_name="Nama Mata Kuliah"
            )
        with hw_col2:
            show_handwriting(
                roi_img=warped[ALL_ROIS['KODE_KELAS'][1]:ALL_ROIS['KODE_KELAS'][3],
                               ALL_ROIS['KODE_KELAS'][0]:ALL_ROIS['KODE_KELAS'][2]],
                ocr_text=kode_text,
                label_name="Kode Kelas"
            )
        
        hw_col3, hw_col4 = st.columns(2)
        with hw_col3:
            show_handwriting(
                roi_img=warped[ALL_ROIS['RUANGAN'][1]:ALL_ROIS['RUANGAN'][3],
                               ALL_ROIS['RUANGAN'][0]:ALL_ROIS['RUANGAN'][2]],
                ocr_text=ruangan_text,
                label_name="Ruangan"
            )
        with hw_col4:
            show_handwriting(
                roi_img=warped[ALL_ROIS['NO_MEJA'][1]:ALL_ROIS['NO_MEJA'][3],
                               ALL_ROIS['NO_MEJA'][0]:ALL_ROIS['NO_MEJA'][2]],
                ocr_text=nomeja_text,
                label_name="No. Meja"
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Score summary
            st.markdown(f"""
            <div class="metrics-row" style="grid-template-columns: repeat(4, 1fr)">
              <div class="metric-tile">
                <div class="t-val c-green">{benar}</div>
                <div class="t-lbl">Benar</div>
              </div>
              <div class="metric-tile">
                <div class="t-val c-red">{salah}</div>
                <div class="t-lbl">Salah</div>
              </div>
              <div class="metric-tile">
                <div class="t-val c-grey">{kosong}</div>
                <div class="t-lbl">Kosong</div>
              </div>
              <div class="metric-tile">
                <div class="t-val c-blue">{score:.1f}</div>
                <div class="t-lbl">Score</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Answer grid
            st.markdown('<div class="section-label" style="margin-top:1.2rem">Detail Jawaban</div>', unsafe_allow_html=True)
            key       = st.session_state.answer_key
            grid_html = '<div class="answer-grid">'
            for q in range(1, st.session_state.total_soal + 1):
                s_ans = answers.get(q)
                k_ans = key.get(q, '?')
                if s_ans is None:
                    cls = "empty"
                    txt = f"{q}. —"
                elif s_ans == k_ans:
                    cls = "correct"
                    txt = f"{q}. {s_ans}"
                else:
                    cls = "wrong"
                    txt = f"{q}. {s_ans}"
                grid_html += f'<div class="ans-cell {cls}">{txt}</div>'
            grid_html += '</div>'
            st.markdown(grid_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Heatmap tabs
            st.markdown('<div class="section-label">🔥 Analisis Heatmap Kepadatan Piksel</div>', unsafe_allow_html=True)
            h_tab1, h_tab2, h_tab3, h_tab4 = st.tabs(["Nama", "NIM", "Tanggal", "Jawaban OMR"])

            with h_tab1:
                ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                x1, y1, x2, y2 = ALL_ROIS['NAMA']
                show_heatmap(
                    density_map=density_nama,
                    title=f"Heatmap Nama OMR ({nama_text})",
                    y_labels=ALPHABET,
                    x_labels=[f"{i + 1}" for i in range(20)],
                    cmap='RdYlGn',
                    roi_img=warped[y1:y2, x1:x2],
                    roi_title="ROI Nama",
                )
            with h_tab2:
                x1, y1, x2, y2 = ALL_ROIS['NIM']
                show_heatmap(
                    density_map=density_nim,
                    title=f"Heatmap NIM OMR ({nim_text})",
                    y_labels=[str(i) for i in range(10)],
                    x_labels=[f"{i + 1}" for i in range(10)],
                    cmap='RdYlGn',
                    roi_img=warped[y1:y2, x1:x2],
                    roi_title="ROI NIM",
                )
            with h_tab3:
                x1, y1, x2, y2 = ALL_ROIS['TANGGAL']
                show_heatmap(
                    density_map=density_tanggal,
                    title=f"Heatmap Tanggal OMR ({tgl_text})",
                    y_labels=[str(i) for i in range(10)],
                    x_labels=[f"{i + 1}" for i in range(6)],
                    cmap='RdYlGn',
                    roi_img=warped[y1:y2, x1:x2],
                    roi_title="ROI Tanggal",
                )
            with h_tab4:
                show_heatmap_jawaban(
                    warped=warped,
                    density_map=density_jawaban,
                    total_soal=st.session_state.total_soal,
                    cmap='jet',
                )

# ─── STEP: RESULTS ─────────────────────────────────────────
elif st.session_state.step == 'results':
    st.markdown('<div class="section-label">Langkah 04</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-title">Ringkasan <span>Hasil Akhir</span></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    records = st.session_state.records

    if not records or not all(r.get('processed', False) for r in records):
        st.warning("Belum semua data diproses OCR.")
        if st.button("← OCR"):
            st.session_state.step = 'handwriting'
            st.rerun()
        st.stop()

    if st.button("← Kembali ke OCR"):
        st.session_state.step = 'handwriting'
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    summary_data = []
    for r in records:
        summary_data.append({
            'File':    r['filename'],
            'Nama':    r['nama'],
            'NIM':     r['nim'],
            'Tanggal': r['tanggal'],
            'Kelas':   r['kode_kelas'],
            'Benar':   r['benar'],
            'Salah':   r['salah'],
            'Kosong':  r['kosong'],
            'Score':   round(r['score'], 2),
            'Grade':   grade_from_score(r['score']),
        })

    df = pd.DataFrame(summary_data)
    st.dataframe(df, width='stretch', hide_index=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Hasil Rekapan (CSV)",
        data=csv,
        file_name=f"Rekap_Nilai_{st.session_state.kode_kelas or 'LJK'}.csv",
        mime='text/csv',
    )
