import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import ALL_ROIS, KUNCI_JAWABAN

CHOICES = ['A', 'B', 'C', 'D', 'E']
JENIS_UJIAN_LABELS = [
    'UJIAN TENGAH SEMESTER',
    'UJIAN AKHIR SEMESTER',
    'KUIS / LATIHAN SOAL',
    'LAINNYA',
]


def apply_clahe(gray):
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def get_bubble_range(gray):
    """Detect vertical range of bubbles in ROI"""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rp = np.sum(bw, axis=1) / 255
    w = gray.shape[1]
    b0 = next((i for i, v in enumerate(rp) if v > w * 0.03), 0)
    b1 = next((i for i in range(len(rp) - 1, 0, -1)
               if rp[i] > w * 0.03), gray.shape[0])
    return b0, b1


def scan_grid(gray, num_cols, num_rows, labels, z_thresh=1.2, z_gap=0.5, per_row=False):
    """
    Scan OMR grid dengan proper density_map initialization
    
    Args:
        gray: ROI grayscale image
        num_cols: Jumlah kolom (5 untuk jawaban, 20 untuk nama, dst)
        num_rows: Jumlah baris (10 untuk nim/tgl, 26 untuk nama, dst)
        labels: List label untuk tiap baris/kolom
        z_thresh: Z-score threshold untuk deteksi filled bubble
        z_gap: Gap minimum antara top 2 z-scores
        per_row: If True, scan per row (untuk jawaban); else per column
        
    Returns:
        (results_list, density_map, b0, b1, row_h, col_w)
    """
    # ⭐ Preprocessing
    gray = cv2.filter2D(gray, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ⭐ Detect bubble vertical range
    row_proj = np.sum(bw, axis=1)
    rows = np.where(row_proj > (np.max(row_proj) * 0.1))[0]
    
    if len(rows) == 0:
        # Safety: Jika tidak ada bubbles terdeteksi
        return [None] * num_cols if not per_row else [None] * num_rows, \
               np.zeros((num_rows, num_cols)), 0, gray.shape[0], 0, 0
    
    b0, b1 = rows[0], rows[-1]
    bh = b1 - b0
    row_h = bh / num_rows if num_rows > 0 else 1
    col_w = gray.shape[1] / num_cols if num_cols > 0 else 1
    
    # ⭐ Initialize density_map dengan shape yang BENAR
    density_map = np.zeros((num_rows, num_cols), dtype=np.float32)
    
    # ⭐ Extract raw scores dari cell centers
    raw = np.zeros((num_rows, num_cols), dtype=float)

    for r in range(num_rows):
        for c in range(num_cols):
            x0 = int(c * col_w) + 2
            x1 = int((c + 1) * col_w) - 2
            y0 = b0 + int(r * row_h) + 2
            y1 = b0 + int((r + 1) * row_h) - 2
            
            # Safety bounds check
            x0 = max(0, x0)
            x1 = min(gray.shape[1], x1)
            y0 = max(0, y0)
            y1 = min(gray.shape[0], y1)
            
            if x1 <= x0 or y1 <= y0:
                raw[r, c] = 0
                continue
            
            cell = bw[y0:y1, x0:x1]
            if cell.size == 0:
                raw[r, c] = 0
                continue
            
            # Use center region untuk avoid edge artifacts
            ch, cw = cell.shape
            cx = cell[ch//4:3*ch//4, cw//4:3*cw//4]
            raw[r, c] = np.mean(cx) if cx.size > 0 else 0

    # ⭐ Z-score normalization per column atau per row
    results = []
    
    if not per_row:
        # Per column (untuk NAMA, NIM, TANGGAL: vertical scanning)
        for c in range(num_cols):
            arr = raw[:, c]
            mean, std = arr.mean(), arr.std() + 1e-6
            z = (arr - mean) / std
            
            # ✅ Assign ke density_map dengan proper indexing
            density_map[:, c] = z
            
            best = int(np.argmax(z))
            bz = z[best]
            sz = sorted(z)[-2] if len(z) > 1 else 0
            
            if bz > z_thresh and (bz - sz) > z_gap and best < len(labels):
                results.append(labels[best])
            else:
                results.append(None)
    else:
        # Per row (untuk JAWABAN: horizontal scanning)
        for r in range(num_rows):
            arr = raw[r, :]
            mean, std = arr.mean(), arr.std() + 1e-6
            z = (arr - mean) / std
            
            # ✅ Assign ke density_map dengan proper indexing
            density_map[r, :] = z
            
            best = int(np.argmax(z))
            bz = z[best]
            sz = sorted(z)[-2] if len(z) > 1 else 0
            
            if bz > z_thresh and (bz - sz) > z_gap and best < len(labels):
                results.append(labels[best])
            else:
                results.append(None)

    return results, density_map, b0, b1, row_h, col_w


def detect_nama(warped):
    """
    Deteksi NAMA dari OMR bubble grid (A-Z, 20 columns)
    
    Returns:
        (nama_text: str, density_map: ndarray shape (26, 20))
    """
    x1, y1, x2, y2 = ALL_ROIS['NAMA']
    roi = warped[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    NUM_COLS = 20
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi_gray)
    roi_inv = cv2.bitwise_not(roi_eq)
    
    blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_proj = np.sum(bw, axis=1) / 255
    
    roi_h, roi_w = roi_gray.shape
    bubble_start = next((i for i, v in enumerate(row_proj) if v > roi_w * 0.05), 0)
    bubble_end = next((i for i in range(roi_h - 1, 0, -1)
                       if row_proj[i] > roi_w * 0.05), roi_h)
    
    bubble_h = bubble_end - bubble_start
    row_h = bubble_h / 26 if bubble_h > 0 else roi_h / 26
    col_w = roi_w / NUM_COLS
    
    raw_scores = np.zeros((26, NUM_COLS), dtype=float)
    density_map = np.zeros((26, NUM_COLS), dtype=np.float32)
    
    for col in range(NUM_COLS):
        x0 = int(col * col_w) + 2
        x1 = int((col + 1) * col_w) - 2
        
        x0 = max(0, x0)
        x1 = min(roi_w, x1)
        
        for row in range(26):
            y0 = bubble_start + int(row * row_h) + 2
            y1 = bubble_start + int((row + 1) * row_h) - 2
            
            y0 = max(0, y0)
            y1 = min(roi_h, y1)
            
            if x1 <= x0 or y1 <= y0:
                raw_scores[row, col] = 0.0
                continue
            
            cell = roi_inv[y0:y1, x0:x1]
            if cell.size == 0:
                raw_scores[row, col] = 0.0
                continue
            
            ch, cw = cell.shape
            center = cell[ch // 4:3 * ch // 4, cw // 4:3 * cw // 4]
            raw_scores[row, col] = float(np.mean(center)) if center.size > 0 else 0.0
    
    # ⭐ Per-column Z-score normalization
    row_means = raw_scores.mean(axis=1, keepdims=True)
    debiased = raw_scores - row_means
    
    nama_result = []
    for col in range(NUM_COLS):
        arr = debiased[:, col]
        mean = arr.mean()
        std = arr.std() + 1e-6
        z = (arr - mean) / std
        
        # ✅ Assign density map
        density_map[:, col] = z
        
        best_row = int(np.argmax(z))
        best_z = z[best_row]
        second_z = sorted(z)[-2]
        abs_val = raw_scores[best_row, col]
        
        if best_z > 1.2 and (best_z - second_z) > 0.5 and abs_val > 70:
            nama_result.append(ALPHABET[best_row])
        else:
            nama_result.append('_')
    
    nama_str = ''.join(nama_result).replace('_', ' ').strip()
    
    return nama_str, density_map


def detect_nim(warped):
    """
    Deteksi NIM dari OMR bubble grid (0-9, 10 columns)
    
    Returns:
        (nim_text: str, density_map: ndarray shape (10, 10))
    """
    x1, y1, x2, y2 = ALL_ROIS['NIM']
    roi_nim = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    
    r_nim, d_map, _, _, _, _ = scan_grid(
        roi_nim, num_cols=10, num_rows=10,
        labels=[str(i) for i in range(10)], per_row=False)
        
    nim_final = ''.join(r or '_' for r in r_nim)
    
    return nim_final, d_map


def detect_tanggal(warped):
    """
    Deteksi TANGGAL dari OMR bubble grid (0-9, 6 columns YYMMDD)
    
    Returns:
        (tanggal_text: str, density_map: ndarray shape (10, 6))
    """
    x1, y1, x2, y2 = ALL_ROIS['TANGGAL']
    roi_tgl = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    tgl_cols = 6
    
    r_tgl, d_map, _, _, _, _ = scan_grid(
        roi_tgl, num_cols=tgl_cols, num_rows=10,
        labels=[str(i) for i in range(10)], per_row=False)
        
    tgl_final = ''.join(r or '_' for r in r_tgl)
    
    return tgl_final, d_map


def detect_answers(warped, total_soal=50):
    """
    Deteksi JAWABAN dari OMR answer blocks (A-E, multiple blocks)
    
    Structure: ROI_JAWABAN = [
        ('BLOK_1', x1, y1, x2, y2, soal_awal),
        ('BLOK_2', x1, y1, x2, y2, soal_awal),
        ...
    ]
    
    Returns:
        (answers_dict: {soal: answer}, density_map: ndarray shape (total_soal, 5))
    """
    # ⭐ FIX: Ensure ini sesuai dengan config actual
    ROI_JAWABAN = ALL_ROIS.get('JAWABAN', [])
    
    if not ROI_JAWABAN or not isinstance(ROI_JAWABAN[0], (tuple, list)):
        # Fallback jika ROI_JAWABAN format tidak expected
        return {}, np.zeros((total_soal, 5))
    
    all_answers = {}
    soal_done = 0
    heatmap_list = []
    
    for item in ROI_JAWABAN:
        if len(item) >= 6:
            label, x1, y1, x2, y2, q_start = item[:6]
        else:
            continue
            
        if soal_done >= total_soal:
            break
        
        margin = 10
        roi_bgr = warped[y1-margin:y2+margin, x1-margin:x2+margin]
        
        if roi_bgr.size == 0:
            continue
        
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # ⭐ Soal per blok
        soal_di_blok = min(10, total_soal - soal_done)
        
        r_blok, d_map, _, _, _, _ = scan_grid(
            roi_gray, num_cols=5, num_rows=soal_di_blok, 
            labels=CHOICES, per_row=True
        )
        
        if d_map.shape[0] > soal_di_blok:
            d_map = d_map[:soal_di_blok, :]
        
        heatmap_list.append(d_map)
        
        r_aktif = r_blok[:soal_di_blok]
        for i, ans in enumerate(r_aktif):
            q = q_start + i
            all_answers[q] = ans
        
        soal_done += soal_di_blok
    
    # ⭐ Concatenate heatmaps dengan proper handling
    if heatmap_list:
        try:
            full_heatmap = np.vstack(heatmap_list)
            # Pad jika kurang
            if full_heatmap.shape[0] < total_soal:
                pad_rows = total_soal - full_heatmap.shape[0]
                full_heatmap = np.vstack([
                    full_heatmap,
                    np.zeros((pad_rows, 5), dtype=np.float32)
                ])
            full_heatmap = full_heatmap[:total_soal, :]
        except:
            full_heatmap = np.zeros((total_soal, 5), dtype=np.float32)
    else:
        full_heatmap = np.zeros((total_soal, 5), dtype=np.float32)
    
    return all_answers, full_heatmap


def detect_jenis_ujian(warped):
    """
    Deteksi JENIS UJIAN (4 pilihan bubble)
    
    Returns:
        (jenis_ujian: str or None, density_map: ndarray shape (4, 1))
    """
    try:
        x1, y1, x2, y2 = ALL_ROIS['JENIS_UJIAN']
        roi = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        
        r_jenis, d_map, _, _, _, _ = scan_grid(
            roi, num_cols=1, num_rows=4,
            labels=JENIS_UJIAN_LABELS, per_row=False
        )
        
        jenis_text = r_jenis[0] if r_jenis and r_jenis[0] else None
        
        return jenis_text, d_map
    except:
        return None, np.zeros((4, 1))


def calculate_score(answers, answer_key, total_soal=None):
    """
    Hitung score berdasarkan answers dan answer_key
    
    Args:
        answers: dict {soal: jawaban}
        answer_key: dict {soal: jawaban_benar}
        total_soal: Total soal (untuk kosong count)
        
    Returns:
        (benar, salah, kosong, score)
    """
    if total_soal is None:
        total_soal = max(max(answers.keys()), max(answer_key.keys())) + 1
    
    benar = 0
    salah = 0
    kosong = 0
    
    for soal in range(1, total_soal + 1):
        student_ans = answers.get(soal)
        correct_ans = answer_key.get(soal)
        
        if student_ans is None:
            kosong += 1
        elif student_ans == correct_ans:
            benar += 1
        else:
            salah += 1
    
    # Score: 4 * benar (standard OMR scoring)
    score = benar * 4.0
    
    return benar, salah, kosong, score
