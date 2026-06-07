import streamlit as st
import numpy as np
import pandas as pd
import io
from PIL import Image as PILImage

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
  --white:   #FFFFFF;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--navy) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--cream) !important;
}
[data-testid="stAppViewContainer"] > .main { background: transparent !important; }
[data-testid="block-container"] { padding: 2rem 2.5rem 3rem !important; }
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
}
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: var(--cream) !important; }
.serif-title {
  font-family: 'DM Serif Display', serif;
  font-size: 2.4rem;
  color: var(--cream);
  line-height: 1.15;
}
.section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: #9BAABF;
}
.page-subtitle {
  color: var(--steel);
  font-size: 0.95rem;
  font-weight: 300;
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
  color: var(--steel);
  border-right: 1px solid rgba(74,86,148,0.2);
}
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
.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(74,86,148,0.25);
  border-radius: 16px;
  padding: 1.2rem 1.5rem;
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
  border-radius: 14px;
  padding: 18px 14px;
  text-align: center;
}
.metric-tile .t-val {
  font-family: 'DM Serif Display', serif;
  font-size: 2rem;
  margin-bottom: 6px;
}
.metric-tile .t-lbl {
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--steel);
}
.c-blue   { color: #7CA4D4; }
.c-green  { color: #6DBF9E; }
.c-red    { color: #E07575; }
.c-amber  { color: #D4A96A; }
.c-cream  { color: var(--cream); }
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
}
.badge-green { background: rgba(61,139,110,0.2); color: #6DBF9E; border: 1px solid rgba(61,139,110,0.3); }
.badge-blue  { background: rgba(114,136,174,0.2); color: #7CA4D4; border: 1px solid rgba(114,136,174,0.3); }
.badge-amber { background: rgba(196,124,43,0.2); color: #D4A96A; border: 1px solid rgba(196,124,43,0.3); }
.stButton > button {
  background: linear-gradient(135deg, #4B5694, #3d4878) !important;
  color: #EAE0CF !important;
  border: 1px solid rgba(234,224,207,0.15) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  padding: 10px 24px !important;
}
.stTextInput input, .stNumberInput input, .stSelectbox select {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(74,86,148,0.35) !important;
  border-radius: 10px !important;
  color: var(--cream) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────
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
        'ocr_model': None,
        'ocr_accuracy': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

DEFAULT_50 = ['B','C','A','D','E','A','B','C','D','A','E','B','C','A','D','B','E','A','C','D',
              'A','B','E','C','D','B','A','D','C','E','A','C','B','D','E','C','A','B','D','E',
              'B','D','A','C','E','A','D','B','E','C']

def make_key_text(n):
    return "\n".join(f"{i}. {DEFAULT_50[i-1] if i <= 50 else 'A'}" for i in range(1, n+1))

if not st.session_state.key_text:
    st.session_state.key_text = make_key_text(st.session_state.total_soal)

# ─── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 16px">
      <div style="font-family:'DM Serif Display',serif; font-size:1.9rem; color:#EAE0CF">
        LJK Scanner
      </div>
      <div style="font-size:0.82rem; color:#7288AE; margin-top:6px; letter-spacing:0.1em; text-transform:uppercase">
        Computer Vision
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    steps = [
        ('setup',       '⚙', 'Setup'),
        ('scan',        '📸', 'Scan'),
        ('handwriting', '✍', 'Handwriting'),
        ('eda',         '📊', 'EDA'),
    ]
    cur = st.session_state.step
    cur_idx = [s[0] for s in steps].index(cur)

    for i, (s, icon, lbl) in enumerate(steps):
        cls = "done" if i < cur_idx else "active" if i == cur_idx else ""
        dot = "✓" if i < cur_idx else icon
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;'
            f'border-radius:8px;margin-bottom:4px;'
            f'background:{"rgba(74,86,148,0.2)" if cls=="active" else "rgba(61,139,110,0.1)" if cls=="done" else "transparent"}">'
            f'<span style="font-size:1rem">{dot}</span>'
            f'<span style="color:{"#EAE0CF" if cls in ["active","done"] else "#4B5694"};'
            f'font-weight:{"600" if cls=="active" else "400"}">{lbl}</span></div>',
            unsafe_allow_html=True
        )

    st.divider()
    records = st.session_state.records
    if records:
        scores = [r['score'] for r in records]
        st.markdown(f'<div class="section-label">Sesi Aktif</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card-sm">
          <div style="font-size:0.78rem;color:#7288AE;margin-bottom:8px">
            {st.session_state.sesi_nama or "—"}<br>
            <span style="font-size:0.68rem;font-family:monospace">{st.session_state.kode_kelas}</span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
            <div style="text-align:center">
              <div style="font-family:'DM Serif Display',serif;font-size:1.5rem">{len(records)}</div>
              <div style="font-size:0.65rem;color:#4B5694;text-transform:uppercase">Scanned</div>
            </div>
            <div style="text-align:center">
              <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#7CA4D4">{np.mean(scores):.1f}</div>
              <div style="font-size:0.65rem;color:#4B5694;text-transform:uppercase">Avg</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#4B5694;font-size:0.88rem;padding:8px 0;font-style:italic">Belum ada data</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("↺  Reset Sesi"):
        for k in ['answer_key','records','step','sesi_nama','kode_kelas','kode_dosen','key_text','ocr_model','ocr_accuracy']:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

# ─── STEP INDICATOR ─────────────────────────────────────────
step_html = '<div class="step-nav">'
for i, (s, icon, lbl) in enumerate(steps):
    cls = "done" if i < cur_idx else "active" if i == cur_idx else ""
    badge = "✓" if i < cur_idx else icon
    step_html += f'<div class="step-item {cls}"><span style="font-size:1rem">{badge}</span>{lbl}</div>'
step_html += '</div>'
st.markdown(step_html, unsafe_allow_html=True)


if st.session_state.step == 'setup':
    st.markdown('<div class="section-label">Langkah 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-title">Setup <span>Sesi Ujian</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Konfigurasikan parameter ujian dan kunci jawaban.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;
          padding-bottom:12px;border-bottom:1px solid rgba(114,136,174,0.2)">
          <div style="width:3px;height:22px;background:#7288AE;border-radius:2px"></div>
          <div>
            <div style="font-family:'DM Serif Display',serif;font-size:1rem;color:#EAE0CF">Informasi Sesi</div>
            <div style="font-size:0.65rem;color:#4B5694;letter-spacing:0.1em;text-transform:uppercase;margin-top:1px">Parameter</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.sesi_nama = st.text_input("Nama Sesi / Mata Kuliah", value=st.session_state.sesi_nama or "Computer Vision UAS")
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

        st.session_state.scoring = st.selectbox(
            "Metode Penilaian",
            ["standard", "penalty"],
            format_func=lambda x: "Standar — (Benar / Total) × 100" if x == "standard" else "Penalty — Benar − (Salah × 0.25)"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;
          padding-bottom:12px;border-bottom:1px solid rgba(114,136,174,0.2)">
          <div style="width:3px;height:22px;background:#7288AE;border-radius:2px"></div>
          <div>
            <div style="font-family:'DM Serif Display',serif;font-size:1rem;color:#EAE0CF">Kunci Jawaban</div>
            <div style="font-size:0.65rem;color:#4B5694;letter-spacing:0.1em;text-transform:uppercase;margin-top:1px">Format: 1. A</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        key_text = st.text_area("Format: `1. A`, `2. B`, ...", value=st.session_state.key_text, height=280, label_visibility="collapsed")
        st.session_state.key_text = key_text

        answer_key = {}
        errors = []
        for line in key_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split('. ', 1) if '. ' in line else line.split(',', 1)
            if len(parts) != 2:
                errors.append(f"Format salah: `{line}`")
                continue
            try:
                q = int(parts[0].strip())
                ans = parts[1].strip().upper()
                if ans not in ['A','B','C','D','E']:
                    errors.append(f"Jawaban tidak valid di soal {q}: `{ans}`")
                    continue
                answer_key[q] = ans
            except ValueError:
                errors.append(f"Nomor tidak valid: `{line}`")

        if errors:
            for e in errors[:2]:
                st.error(e)
        else:
            total = st.session_state.total_soal
            filled = len(answer_key)
            pct = filled / total * 100 if total > 0 else 0
            color = "#6DBF9E" if filled == total else "#D4A96A"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
              background:rgba(61,139,110,0.08);border:1px solid rgba(61,139,110,0.25);
              border-radius:10px;margin-top:8px">
              <span style="font-size:1.1rem">{'✅' if filled==total else '⚠️'}</span>
              <div>
                <div style="font-size:0.85rem;color:{color};font-weight:600">
                  {filled} / {total} kunci jawaban valid
                </div>
                <div style="font-size:0.7rem;color:#4B5694">{pct:.0f}% terisi</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("Mulai Scan  →", disabled=len(answer_key) == 0):
            st.session_state.answer_key = answer_key
            st.session_state.step = 'scan'
            st.rerun()


elif st.session_state.step == 'scan':
    st.markdown('<div class="section-label">Langkah 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-title">Scan <span>Lembar Jawaban</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{st.session_state.sesi_nama} · {st.session_state.kode_kelas} · {st.session_state.total_soal} soal · {len(st.session_state.records)} lembar ter-scan</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("← Setup"):
            st.session_state.step = 'setup'
            st.rerun()
    with c2:
        if st.button("Handwriting OCR →", disabled=len(st.session_state.records) == 0):
            st.session_state.step = 'handwriting'
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:0.6rem">
      <div style="font-family:'DM Serif Display',serif;font-size:1rem;color:#EAE0CF;margin-bottom:4px">
        Upload Foto LJK
      </div>
      <div style="font-size:0.82rem;color:#7288AE">Pilih file atau ambil foto dengan webcam.</div>
    </div>
    """, unsafe_allow_html=True)

    upload_tab, camera_tab = st.tabs(["📁  Upload File", "📷  Webcam"])

    uploaded_files = []
    
    with upload_tab:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _files = st.file_uploader("Pilih foto LJK (JPG / PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="visible")
        if _files:
            uploaded_files = _files

    with camera_tab:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.8rem;color:#7288AE;margin-bottom:8px">Arahkan kamera ke LJK.</div>', unsafe_allow_html=True)
        camera_image = st.camera_input("Ambil foto", label_visibility="collapsed")
        if camera_image is not None:
            uploaded_files = [camera_image]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if uploaded_files:
        for uploaded in uploaded_files:
            fname = getattr(uploaded, 'name', None) or 'webcam_capture.jpg'
            if any(r['filename'] == fname for r in st.session_state.records):
                st.warning(f"⚠️ `{fname}` sudah di-scan.")
                continue

            with st.expander(f"📄  {fname}", expanded=True):
                try:
                    img_pil = PILImage.open(uploaded)
                    
                    try:
                        import scanner
                        
                        if not hasattr(scanner, 'warp_ljk'):
                            st.error("❌ Function `scanner.warp_ljk()` belum ada.")
                            st.info("📝 Pastikan scanner.py sudah diimplementasikan dengan benar.")
                            continue
                        
                        warped_np, ok = scanner.warp_ljk(img_pil)
                        
                        col_orig, col_warp = st.columns(2, gap="medium")
                        with col_orig:
                            st.markdown('<div class="section-label">Input Asli</div>', unsafe_allow_html=True)
                            st.image(img_pil, use_container_width=True)
                        with col_warp:
                            st.markdown('<div class="section-label">Setelah Warp</div>', unsafe_allow_html=True)
                            if ok:
                                st.image(warped_np, use_container_width=True)
                            else:
                                st.error("❌ Gagal mendeteksi 4 sudut LJK.")
                                continue
                        
                        with st.spinner("Mendeteksi nama, NIM, tanggal, jawaban..."):
                            nama = scanner.detect_nama(warped_np)
                            nim = scanner.detect_nim(warped_np)
                            tanggal = scanner.detect_tanggal(warped_np)
                            answers = scanner.detect_answers(warped_np, st.session_state.total_soal)
                        
                        correct, wrong, unanswered = 0, 0, 0
                        for q, key in st.session_state.answer_key.items():
                            s = answers.get(q)
                            if s is None:
                                unanswered += 1
                            elif s == key:
                                correct += 1
                            else:
                                wrong += 1

                        scoring = st.session_state.scoring
                        total_soal = st.session_state.total_soal
                        if scoring == "standard":
                            score = round(correct / total_soal * 100, 2) if total_soal > 0 else 0
                        else:
                            score = round(max(0, (correct - wrong * 0.25) / total_soal * 100), 2)

                        if score >= 80:
                            grade, grade_cls = 'A', 'badge-green'
                        elif score >= 70:
                            grade, grade_cls = 'B', 'badge-blue'
                        elif score >= 60:
                            grade, grade_cls = 'C', 'badge-amber'
                        else:
                            grade, grade_cls = 'D/E', 'badge-red'

                        st.markdown(f"""
                        <div class="metrics-row">
                          <div class="metric-tile">
                            <div class="t-val c-blue">{score}</div>
                            <div class="t-lbl">Skor</div>
                          </div>
                          <div class="metric-tile">
                            <div class="t-val c-green">{correct}</div>
                            <div class="t-lbl">Benar</div>
                          </div>
                          <div class="metric-tile">
                            <div class="t-val c-red">{wrong}</div>
                            <div class="t-lbl">Salah</div>
                          </div>
                          <div class="metric-tile">
                            <div class="t-val c-amber">{unanswered}</div>
                            <div class="t-lbl">Kosong</div>
                          </div>
                          <div class="metric-tile">
                            <span class="badge {grade_cls}" style="font-size:1.4rem;padding:6px 16px">{grade}</span>
                            <div class="t-lbl" style="margin-top:8px">Grade</div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="card-sm" style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap">
                          <div>
                            <div class="section-label">Nama</div>
                            <div style="font-family:'DM Serif Display',serif;font-size:1.1rem">{nama}</div>
                          </div>
                          <div>
                            <div class="section-label">NIM</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;color:#7CA4D4">{nim}</div>
                          </div>
                          <div>
                            <div class="section-label">Tanggal</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem">{tanggal}</div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        record = {
                            'filename': fname,
                            'nama': nama,
                            'nim': nim,
                            'tanggal': tanggal,
                            'correct': correct,
                            'wrong': wrong,
                            'unanswered': unanswered,
                            'score': score,
                            'total_soal': total_soal,
                            'student_answers': {str(k): v for k, v in answers.items()},
                            'answer_key': st.session_state.answer_key,
                        }
                        st.session_state.records.append(record)
                        st.success(f"✅  `{fname}` berhasil disimpan.")

                    except ImportError:
                        st.error("❌ Module `scanner.py` tidak ditemukan.")
                        st.info("📝 Pastikan `scanner.py` ada di direktori project.")
                    except AttributeError as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("📝 Pastikan semua function ada: warp_ljk, detect_nama, detect_nim, detect_tanggal, detect_answers")
                    except Exception as e:
                        st.error(f"❌ Error processing: {str(e)}")

                except Exception as e:
                    st.error(f"❌ Error opening image: {str(e)}")


elif st.session_state.step == 'handwriting':
    st.markdown('<div class="section-label">Langkah 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-title">Handwriting <span>OCR Training</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Train SVM model untuk pengenalan karakter tulisan tangan.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("← Scan"):
            st.session_state.step = 'scan'
            st.rerun()
    with c2:
        if st.button("EDA →"):
            st.session_state.step = 'eda'
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:1rem">
      <div style="font-family:'DM Serif Display',serif;font-size:1rem;color:#EAE0CF;margin-bottom:4px">
        Training SVM Model
      </div>
      <div style="font-size:0.82rem;color:#7288AE">Latih model SVM untuk OCR menggunakan EMNIST dataset.</div>
    </div>
    """, unsafe_allow_html=True)

    col_train, col_pred = st.columns(2, gap="large")

    with col_train:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;
          padding-bottom:12px;border-bottom:1px solid rgba(114,136,174,0.2)">
          <div style="width:3px;height:22px;background:#7288AE;border-radius:2px"></div>
          <div>
            <div style="font-family:'DM Serif Display',serif;font-size:1rem;color:#EAE0CF">Train Model</div>
            <div style="font-size:0.65rem;color:#4B5694;letter-spacing:0.1em;text-transform:uppercase;margin-top:2px">SVM + HOG</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄  Train OCR Model"):
            with st.spinner("Training SVM... (tunggu beberapa menit)"):
                try:
                    import handwriting_ocr
                    model, accuracy = handwriting_ocr.load_or_train()
                    st.session_state.ocr_model = model
                    st.session_state.ocr_accuracy = accuracy
                    st.success(f"✅ Model trained! Akurasi: **{accuracy:.2%}**")
                except ImportError:
                    st.error("❌ Module `handwriting_ocr.py` tidak ditemukan.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        if st.session_state.ocr_accuracy is not None:
            st.markdown(f"""
            <div class="metric-tile" style="text-align:center;padding:20px;margin-top:12px">
              <div class="t-val c-green" style="font-size:2rem">{st.session_state.ocr_accuracy:.1%}</div>
              <div class="t-lbl">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_pred:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;
          padding-bottom:12px;border-bottom:1px solid rgba(114,136,174,0.2)">
          <div style="width:3px;height:22px;background:#7288AE;border-radius:2px"></div>
          <div>
            <div style="font-family:'DM Serif Display',serif;font-size:1rem;color:#EAE0CF">Test Prediksi</div>
            <div style="font-size:0.65rem;color:#4B5694;letter-spacing:0.1em;text-transform:uppercase;margin-top:2px">Upload digit</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_digit = st.file_uploader("Upload gambar digit (JPG/PNG)", type=["jpg","jpeg","png"], label_visibility="collapsed")

        if uploaded_digit and st.session_state.ocr_model is not None:
            try:
                img = PILImage.open(uploaded_digit).convert('L')
                st.image(img, caption="Input Digit", width=100)

                try:
                    from skimage.feature import hog
                    import cv2
                    
                    img_arr = np.array(img)
                    if img_arr.shape[0] != 28 or img_arr.shape[1] != 28:
                        img_arr = cv2.resize(img_arr, (28, 28))

                    features = hog(img_arr, orientations=9, pixels_per_cell=(4,4), cells_per_block=(2,2), visualize=False)
                    features = features.reshape(1, -1)

                    pred = st.session_state.ocr_model.predict(features)[0]

                    st.markdown(f"""
                    <div style="background:rgba(61,139,110,0.15);border:1px solid rgba(61,139,110,0.35);
                      border-radius:10px;padding:14px;margin-top:12px">
                      <div style="font-size:0.7rem;color:#4B5694;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px">
                        Prediksi
                      </div>
                      <div style="font-family:'DM Serif Display',serif;font-size:2.2rem;color:#6DBF9E">
                        {chr(pred)}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                except ImportError:
                    st.error("❌ Libaries skimage/cv2 belum tersedia.")
                except Exception as e:
                    st.error(f"❌ Error predicting: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error opening image: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 Module `handwriting_ocr.py` harus punya function `load_or_train()` yang return (model, accuracy).")


elif st.session_state.step == 'eda':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    records = st.session_state.records
    if not records:
        st.warning("Belum ada data scan.")
        if st.button("← Scan"):
            st.session_state.step = 'scan'
            st.rerun()
        st.stop()

    st.markdown('<div class="section-label">Langkah 04</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-title">EDA & <span>Analitik Kelas</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{st.session_state.sesi_nama} · {st.session_state.kode_kelas}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, _ = st.columns([1, 4])
    with c1:
        if st.button("← Handwriting"):
            st.session_state.step = 'handwriting'
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    scores = [r['score'] for r in records]
    total_soal = records[0]['total_soal']

    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-tile">
        <div class="t-val c-cream">{len(records)}</div>
        <div class="t-lbl">Mahasiswa</div>
      </div>
      <div class="metric-tile">
        <div class="t-val c-blue">{np.mean(scores):.1f}</div>
        <div class="t-lbl">Rata-rata</div>
      </div>
      <div class="metric-tile">
        <div class="t-val c-green">{max(scores):.1f}</div>
        <div class="t-lbl">Tertinggi</div>
      </div>
      <div class="metric-tile">
        <div class="t-val c-red">{min(scores):.1f}</div>
        <div class="t-lbl">Terendah</div>
      </div>
      <div class="metric-tile">
        <div class="t-val c-amber">{np.std(scores):.2f}</div>
        <div class="t-lbl">Std Dev</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋  Tabel", "📈  Grafik", "💾  Export"])

    COLORS = {'navy':'#111844','mid':'#4B5694','steel':'#7288AE','cream':'#EAE0CF','green':'#6DBF9E','red':'#E07575','blue':'#7CA4D4','amber':'#D4A96A'}

    def style_axes(ax):
        ax.set_facecolor('#0C1235')
        for sp in ax.spines.values():
            sp.set_color(COLORS['mid'])
        ax.tick_params(colors=COLORS['steel'], labelsize=8)
        ax.xaxis.label.set_color(COLORS['steel'])
        ax.yaxis.label.set_color(COLORS['steel'])
        ax.title.set_color(COLORS['cream'])
        ax.grid(axis='y', color=COLORS['mid'], alpha=0.2, linewidth=0.5)

    with tab1:
        df = pd.DataFrame([{
            'Nama': r['nama'],
            'NIM': r['nim'],
            'Tanggal': r['tanggal'],
            'Benar': r['correct'],
            'Salah': r['wrong'],
            'Kosong': r['unanswered'],
            'Skor': r['score'],
            'Grade': 'A' if r['score']>=80 else 'B' if r['score']>=70 else 'C' if r['score']>=60 else 'D' if r['score']>=50 else 'E'
        } for r in records])
        st.dataframe(df, use_container_width=True, height=400)

    with tab2:
        col_ch1, col_ch2 = st.columns(2, gap="large")
        with col_ch1:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#111844')
            style_axes(ax)
            n, bins, patches = ax.hist(scores, bins=range(0, 105, 10), edgecolor='#111844', linewidth=0.8)
            for patch in patches:
                patch.set_facecolor(COLORS['mid'])
                patch.set_alpha(0.85)
            ax.axvline(np.mean(scores), color=COLORS['amber'], lw=1.5, linestyle='--', label=f'Mean = {np.mean(scores):.1f}')
            ax.set_xlabel("Nilai")
            ax.set_ylabel("Jumlah")
            ax.set_title("Distribusi Nilai", fontsize=11, fontweight='bold', pad=10)
            ax.legend(facecolor='#0C1235', labelcolor=COLORS['cream'], fontsize=8, framealpha=0.8)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close(fig)

        with col_ch2:
            grades = {'A (≥80)':0,'B (70-79)':0,'C (60-69)':0,'D (50-59)':0,'E (<50)':0}
            for s in scores:
                if s>=80:
                    grades['A (≥80)']+=1
                elif s>=70:
                    grades['B (70-79)']+=1
                elif s>=60:
                    grades['C (60-69)']+=1
                elif s>=50:
                    grades['D (50-59)']+=1
                else:
                    grades['E (<50)']+=1
            lp = [k for k,v in grades.items() if v>0]
            sp = [v for v in grades.values() if v>0]
            pie_colors = [COLORS['green'],COLORS['blue'],COLORS['amber'],COLORS['red'],'#9B7E7E'][:len(lp)]
            fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='#111844')
            ax2.set_facecolor('#111844')
            wedges, texts, autotexts = ax2.pie(sp, labels=lp, colors=pie_colors, autopct='%1.0f%%', startangle=90,
                                                  textprops={'color':COLORS['cream'],'fontsize':8},
                                                  wedgeprops={'linewidth':2,'edgecolor':'#111844'})
            for at in autotexts:
                at.set_color('#111844')
                at.set_fontweight('bold')
            ax2.set_title("Distribusi Grade", color=COLORS['cream'], fontsize=11, fontweight='bold', pad=10)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig2)
            plt.close(fig2)

    with tab3:
        st.markdown('<div class="section-label">Download Hasil</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        df_rekap = pd.DataFrame([{
            'Nama': r['nama'],
            'NIM': r['nim'],
            'Tanggal': r['tanggal'],
            'Benar': r['correct'],
            'Salah': r['wrong'],
            'Kosong': r['unanswered'],
            'Skor': r['score'],
            'Grade': 'A' if r['score']>=80 else 'B' if r['score']>=70 else 'C' if r['score']>=60 else 'D' if r['score']>=50 else 'E'
        } for r in records])

        df_stats = pd.DataFrame([{'Metrik':m,'Nilai':v} for m,v in {
            'Total Mahasiswa': len(records),
            'Rata-rata': round(np.mean(scores), 2),
            'Tertinggi': max(scores),
            'Terendah': min(scores),
            'Std Deviasi': round(np.std(scores), 2),
        }.items()])

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_rekap.to_excel(writer, sheet_name='Rekap Nilai', index=False)
            df_stats.to_excel(writer, sheet_name='Statistik', index=False)
        buf.seek(0)

        fname = f"hasil_{st.session_state.kode_kelas}.xlsx"
        st.download_button("⬇  Download Excel", data=buf, file_name=fname,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button("⬇  Download CSV",
                           df_rekap.to_csv(index=False).encode(),
                           file_name=f"rekap_{st.session_state.kode_kelas}.csv",
                           mime="text/csv")

        st.markdown('</div>', unsafe_allow_html=True)
