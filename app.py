import streamlit as st
import numpy as np
import pandas as pd
import io
from PIL import Image

import config
import utils
import preprocessing
import corner_detection
import scanner
import ocr_inference
import training
import handwriting_ocr
import eda

st.set_page_config(
    page_title="LJK Scanner",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --navy:    #111844;
  --mid:     #4B5694;
  --steel:   #7288AE;
  --cream:   #EAE0CF;
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
.serif-title { font-family: 'DM Serif Display', serif; font-size: 2.4rem; color: var(--cream); line-height: 1.15; letter-spacing: -0.02em; }
.serif-title span { color: var(--cream); font-style: italic; background: linear-gradient(135deg, #EAE0CF, #7288AE); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.section-label { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.18em; text-transform: uppercase; color: #9BAABF; margin-bottom: 0.4rem; }
.page-subtitle { color: var(--steel); font-size: 0.95rem; font-weight: 300; margin-top: 0.3rem; }

.step-nav { display: flex; gap: 0; margin-bottom: 2rem; border-radius: 12px; overflow: hidden; border: 1px solid rgba(74,86,148,0.3); }
.step-item { flex: 1; padding: 12px 8px; text-align: center; font-size: 0.78rem; font-weight: 500; background: rgba(255,255,255,0.02); color: var(--text-dim); transition: all .2s; border-right: 1px solid rgba(74,86,148,0.2); cursor: default; }
.step-item:last-child { border-right: none; }
.step-item.active { background: linear-gradient(180deg, rgba(74,86,148,0.3), rgba(17,24,68,0.4)); color: var(--cream); font-weight: 600; border-bottom: 2px solid #EAE0CF; }
.step-item.done { background: rgba(74,86,148,0.12); color: #7288AE; }
.step-icon { font-size: 1rem; display: block; margin-bottom: 3px; }

.card { background: rgba(255,255,255,0.03); border: 1px solid rgba(74,86,148,0.25); border-top: 2px solid rgba(234,224,207,0.3); border-radius: 16px; padding: 1.2rem 1.5rem 0.8rem; margin-bottom: 1rem; }
.card-sm { background: rgba(255,255,255,0.03); border: 1px solid rgba(74,86,148,0.2); border-radius: 12px; padding: 1rem 1.2rem; }

.metrics-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 1.2rem 0; }
.metric-tile { background: rgba(255,255,255,0.04); border: 1px solid rgba(74,86,148,0.2); border-top: 2px solid rgba(234,224,207,0.2); border-radius: 14px; padding: 18px 14px; text-align: center; transition: transform .2s, border-color .2s, border-top-color .2s; }
.metric-tile:hover { transform: translateY(-2px); border-color: var(--steel); border-top-color: rgba(234,224,207,0.55); }
.metric-tile .t-val { font-family: 'DM Serif Display', serif; font-size: 2rem; line-height: 1; margin-bottom: 6px; }
.metric-tile .t-lbl { font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-dim); }
.c-blue   { color: #7CA4D4; }
.c-green  { color: #6DBF9E; }
.c-red    { color: #E07575; }
.c-grey   { color: #7288AE; }
.c-amber  { color: #D4A96A; }
.c-cream  { color: var(--cream); }

.answer-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(68px, 1fr)); gap: 6px; margin-top: 0.8rem; }
.ans-cell { border-radius: 8px; padding: 6px 4px; text-align: center; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; border: 1px solid rgba(74,86,148,0.2); background: rgba(255,255,255,0.02); }
.ans-cell.correct { background: rgba(61,139,110,0.15); border-color: #3D8B6E; color: #6DBF9E; }
.ans-cell.wrong   { background: rgba(179,64,64,0.15); border-color: #B34040; color: #E07575; }
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
label, .stSelectbox label, .stTextInput label, .stNumberInput label, .stTextArea label {
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
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03) !important; border-radius: 10px !important; padding: 4px !important; gap: 4px !important; border: 1px solid rgba(74,86,148,0.2) !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--text-dim) !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.85rem !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: rgba(74,86,148,0.35) !important; color: var(--cream) !important; }
[data-testid="stExpander"] { background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(74,86,148,0.2) !important; border-radius: 12px !important; }
[data-testid="stExpander"] summary { color: var(--cream) !important; font-weight: 500 !important; }
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }
.stFileUploader { background: rgba(255,255,255,0.02) !important; border: 2px dashed rgba(74,86,148,0.4) !important; border-radius: 14px !important; transition: border-color .2s !important; }
.stFileUploader:hover { border-color: var(--steel) !important; }
.stDownloadButton > button { background: rgba(61,139,110,0.2) !important; border: 1px solid rgba(61,139,110,0.5) !important; color: #6DBF9E !important; }
.stDownloadButton > button:hover { background: rgba(61,139,110,0.35) !important; }

hr { border-color: rgba(74,86,148,0.2) !important; }

.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
.badge-green { background: rgba(61,139,110,0.2); color: #6DBF9E; border: 1px solid rgba(61,139,110,0.3); }
.badge-blue  { background: rgba(114,136,174,0.2); color: #7CA4D4; border: 1px solid rgba(114,136,174,0.3); }
.badge-amber { background: rgba(196,124,43,0.2); color: #D4A96A; border: 1px solid rgba(196,124,43,0.3); }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(74,86,148,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

def init_state():
    defaults = {
        'answer_key': {},
        'total_soal': 50,
        'sesi_nama': '',
        'kode_kelas': '',
        'kode_dosen': '',
        'scoring': 'standard',
        'records': [],
        'step': 'setup',
        'key_text': '',
        'ml_bundle': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

DEFAULT_50 = ['B','C','A','D','E','A','B','C','D','A','E','B','C','A','D','B','E','A','C','D',
              'A','B','E','C','D','B','A','D','C','E','A','C','B','D','E','C','A','B','D','E',
              'B','D','A','C','E','A','D','B','E','C']

def make_key_text(n):
    return "\n".join(f"{i}. {DEFAULT_50[i-1] if i<=50 else 'A'}" for i in range(1, n+1))

if not st.session_state.key_text:
    st.session_state.key_text = make_key_text(st.session_state.total_soal)

with st.sidebar:
    st.markdown('<div style="padding: 12px 0 16px"><div style="font-family:\'DM Serif Display\',serif; font-size:1.9rem; color:#EAE0CF">LJK Scanner</div><div style="font-size:0.82rem; color:#7288AE; margin-top:6px; letter-spacing:0.1em; text-transform:uppercase">Computer Vision</div></div>', unsafe_allow_html=True)
    st.divider()

    steps = [('setup', '⚙', 'Setup'), ('scan', '📸', 'Scan'), ('results', '📊', 'Hasil'), ('handwriting', '✍', 'OCR'), ('eda', '📈', 'EDA')]
    cur = st.session_state.step
    cur_idx = [s[0] for s in steps].index(cur) if cur in [s[0] for s in steps] else 0

    for i, (s, icon, lbl) in enumerate(steps):
        cls = "done" if i<cur_idx else "active" if i==cur_idx else ""
        dot = "✓" if i<cur_idx else icon if i==cur_idx else icon
        st.markdown(f'<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:8px;margin-bottom:4px;background:{"rgba(74,86,148,0.2)" if cls=="active" else "rgba(61,139,110,0.1)" if cls=="done" else "transparent"};border:1px solid {"rgba(74,86,148,0.4)" if cls=="active" else "rgba(61,139,110,0.25)" if cls=="done" else "transparent"}"><span>{dot}</span><span style="font-weight:{"600" if cls=="active" else "400"};color:{"#EAE0CF" if cls else "#4B5694"}">{lbl}</span></div>', unsafe_allow_html=True)

    st.divider()
    if st.button("↺  Reset"): 
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

step_html = '<div class="step-nav">'
for i, (s, icon, lbl) in enumerate(steps):
    cls = "done" if i<cur_idx else "active" if i==cur_idx else ""
    step_html += f'<div class="step-item {cls}"><span class="step-icon">{icon if i>=cur_idx else "✓"}</span>{lbl}</div>'
st.markdown(step_html + '</div>', unsafe_allow_html=True)

if st.session_state.step == 'setup':
    st.markdown('<div class="section-label">Langkah 01</div><div class="serif-title">Setup <span>Sesi Ujian</span></div><div class="page-subtitle">Konfigurasikan ujian & kunci jawaban</div><br>', unsafe_allow_html=True)
    col1, col2 = st.columns([1,1], gap="large")
    with col1:
        st.markdown('<div class="card"><div style="margin-bottom:1rem;padding-bottom:12px;border-bottom:1px solid rgba(114,136,174,0.2)"><div style="font-family:\'DM Serif Display\',serif;font-size:1.05rem;color:#EAE0CF">Informasi Sesi</div></div>', unsafe_allow_html=True)
        st.session_state.sesi_nama = st.text_input("Nama Sesi", value=st.session_state.sesi_nama or "Computer Vision UAS")
        c1, c2 = st.columns(2)
        with c1: st.session_state.kode_kelas = st.text_input("Kode Kelas", value=st.session_state.kode_kelas or "LK01")
        with c2: st.session_state.kode_dosen = st.text_input("Kode Dosen", value=st.session_state.kode_dosen or "DS123")
        new_total = st.number_input("Jumlah Soal", min_value=1, max_value=100, value=st.session_state.total_soal)
        if new_total != st.session_state.total_soal:
            st.session_state.total_soal = new_total
            st.session_state.key_text = make_key_text(new_total)
            st.rerun()
        st.session_state.scoring = st.selectbox("Metode Penilaian", ["standard", "penalty"], format_func=lambda x: "Standar (Benar/Total×100)" if x=="standard" else "Penalty (Benar−Salah×0.25)")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><div style="margin-bottom:1rem;padding-bottom:12px;border-bottom:1px solid rgba(114,136,174,0.2)"><div style="font-family:\'DM Serif Display\',serif;font-size:1.05rem;color:#EAE0CF">Kunci Jawaban</div></div>', unsafe_allow_html=True)
        key_text = st.text_area("Format: `1. A`, `2. B`", value=st.session_state.key_text, height=240, label_visibility="collapsed")
        st.session_state.key_text = key_text
        answer_key = {}
        for line in key_text.strip().split('\n'):
            if not line.strip(): continue
            parts = line.split('. ', 1) if '. ' in line else line.split(',', 1)
            if len(parts)==2:
                try:
                    q, ans = int(parts[0].strip()), parts[1].strip().upper()
                    if ans in ['A','B','C','D','E']: answer_key[q] = ans
                except: pass
        filled = len(answer_key)
        st.markdown(f'<div style="padding:10px;background:rgba(61,139,110,0.08);border:1px solid rgba(61,139,110,0.25);border-radius:10px;margin-top:8px;color:{"#6DBF9E" if filled==st.session_state.total_soal else "#D4A96A"}">{filled}/{st.session_state.total_soal} kunci ✅</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Mulai Scan  →", disabled=len(answer_key)==0):
        st.session_state.answer_key = answer_key
        st.session_state.step = 'scan'
        st.rerun()

elif st.session_state.step == 'scan':
    st.markdown(f'<div class="section-label">Langkah 02</div><div class="serif-title">Scan <span>Lembar Jawaban</span></div><div class="page-subtitle">{st.session_state.sesi_nama} · {st.session_state.total_soal} soal · {len(st.session_state.records)} ter-scan</div><br>', unsafe_allow_html=True)
    c1, c2, _ = st.columns([1,1,4])
    with c1:
        if st.button("← Setup"): st.session_state.step='setup'; st.rerun()
    with c2:
        if st.button("Hasil →", disabled=len(st.session_state.records)==0): st.session_state.step='results'; st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    upload_tab, camera_tab = st.tabs(["📁 Upload", "📷 Foto"])
    with upload_tab:
        files = st.file_uploader("Pilih foto LJK", type=["jpg","jpeg","png"], accept_multiple_files=True)
    with camera_tab:
        files = [st.camera_input("Ambil foto")] if st.camera_input("Ambil foto", label_visibility="collapsed") else []
    if files:
        for f in files:
            if f and not any(r['filename']==getattr(f,'name','cam.jpg') for r in st.session_state.records):
                with st.expander(f"📄 {getattr(f,'name','webcam.jpg')}", expanded=True):
                    try:
                        img = Image.open(f)
                        img_np = np.array(img.convert('RGB'))
                        gray = preprocessing.preprocess_image(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))[1]
                        corners = corner_detection.find_corner_bubbles(gray, visualize=False)
                        warped, ok = corner_detection.warp_perspective(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), corners)
                        if not ok: st.error("❌ Gagal deteksi sudut"); continue
                        nama = scanner.detect_nama(warped)
                        nim = scanner.detect_nim(warped)
                        tanggal = scanner.detect_tanggal(warped)
                        answers = scanner.detect_answers(warped, st.session_state.total_soal)
                        correct = sum(1 for q in answers if answers[q]==st.session_state.answer_key.get(q))
                        wrong = sum(1 for q in answers if answers[q] and answers[q]!=st.session_state.answer_key.get(q))
                        unanswered = st.session_state.total_soal - correct - wrong
                        score = round(correct/st.session_state.total_soal*100, 2) if st.session_state.scoring=="standard" else round(max(0, (correct-wrong*0.25)/st.session_state.total_soal*100), 2)
                        grade = 'A' if score>=80 else 'B' if score>=70 else 'C' if score>=60 else 'D'
                        st.markdown(f'<div class="metrics-row"><div class="metric-tile"><div class="t-val c-blue">{score}</div><div class="t-lbl">Skor</div></div><div class="metric-tile"><div class="t-val c-green">{correct}</div><div class="t-lbl">Benar</div></div><div class="metric-tile"><div class="t-val c-red">{wrong}</div><div class="t-lbl">Salah</div></div><div class="metric-tile"><div class="t-val c-grey">{unanswered}</div><div class="t-lbl">Kosong</div></div><div class="metric-tile"><span class="badge badge-{"green" if grade=="A" else "blue" if grade=="B" else "amber"}">{grade}</span></div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="card-sm">Nama: {nama} | NIM: {nim} | Tgl: {tanggal}</div>', unsafe_allow_html=True)
                        st.session_state.records.append({'filename': getattr(f,'name','cam.jpg'), 'nama': nama, 'nim': nim, 'tanggal': tanggal, 'correct': correct, 'wrong': wrong, 'unanswered': unanswered, 'score': score, 'total_soal': st.session_state.total_soal, 'student_answers': answers, 'answer_key': st.session_state.answer_key})
                        st.success("✅ Disimpan")
                    except Exception as e: st.error(f"❌ {str(e)}")

elif st.session_state.step == 'results':
    records = st.session_state.records
    if not records: st.warning("Belum ada data"); st.stop()
    st.markdown(f'<div class="section-label">Langkah 03</div><div class="serif-title">Hasil & <span>Z-Score</span></div><div class="page-subtitle">{st.session_state.sesi_nama}</div><br>', unsafe_allow_html=True)
    c1, c2, c3, _ = st.columns([1,1,1,3])
    with c1:
        if st.button("← Scan"): st.session_state.step='scan'; st.rerun()
    with c2:
        if st.button("OCR →"): st.session_state.step='handwriting'; st.rerun()
    with c3:
        if st.button("EDA →"): st.session_state.step='eda'; st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    scores = [r['score'] for r in records]
    mean = np.mean(scores); std = np.std(scores) if len(scores)>1 else 1
    st.markdown(f'<div class="metrics-row"><div class="metric-tile"><div class="t-val c-cream">{len(records)}</div><div class="t-lbl">Siswa</div></div><div class="metric-tile"><div class="t-val c-blue">{mean:.1f}</div><div class="t-lbl">Mean</div></div><div class="metric-tile"><div class="t-val c-green">{max(scores):.1f}</div><div class="t-lbl">Max</div></div><div class="metric-tile"><div class="t-val c-red">{min(scores):.1f}</div><div class="t-lbl">Min</div></div><div class="metric-tile"><div class="t-val c-amber">{std:.2f}</div><div class="t-lbl">SD</div></div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋 Tabel", "📊 Z-Score", "💾 Export"])
    with tab1:
        df = pd.DataFrame([{'Nama': r['nama'], 'NIM': r['nim'], 'Skor': r['score'], 'Benar': r['correct'], 'Salah': r['wrong']} for r in records])
        st.dataframe(df, use_container_width=True)
    with tab2:
        z_scores = [(s-mean)/std if std>0 else 0 for s in scores]
        df_z = pd.DataFrame([{'Nama': r['nama'], 'NIM': r['nim'], 'Skor': r['score'], 'Z-Score': round(z, 3)} for r, z in zip(records, z_scores)])
        st.dataframe(df_z, use_container_width=True)
    with tab3:
        df_exp = pd.DataFrame([{'Nama': r['nama'], 'NIM': r['nim'], 'Skor': r['score'], 'Z-Score': round((r['score']-mean)/std if std>0 else 0, 3)} for r in records])
        buf = io.BytesIO()
        df_exp.to_excel(buf, sheet_name='Hasil', index=False, engine='openpyxl')
        buf.seek(0)
        st.download_button("⬇ Excel", buf, file_name=f"hasil_{st.session_state.kode_kelas}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif st.session_state.step == 'handwriting':
    st.markdown('<div class="section-label">Langkah 04</div><div class="serif-title">OCR <span>Handwritten</span></div><br>', unsafe_allow_html=True)
    if st.button("← Hasil"): st.session_state.step='results'; st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">SVM Model untuk handwritten OCR (NAMA_MATA_KULIAH, KODE_KELAS, RUANGAN, NO_MEJA)</div>', unsafe_allow_html=True)
    if st.button("Load Model"):
        st.session_state.ml_bundle = training.load_model()
        st.success("✅ Model loaded" if st.session_state.ml_bundle else "❌ Model not found")

elif st.session_state.step == 'eda':
    st.markdown('<div class="section-label">Langkah 05</div><div class="serif-title">EDA & <span>Statistik</span></div><br>', unsafe_allow_html=True)
    if st.button("← Hasil"): st.session_state.step='results'; st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">Analisis: distribusi nilai, ranking, soal sulit</div>', unsafe_allow_html=True)
