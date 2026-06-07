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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def get_bubble_range(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rp = np.sum(bw, axis=1) / 255
    w = gray.shape[1]
    b0 = next((i for i, v in enumerate(rp) if v > w * 0.03), 0)
    b1 = next((i for i in range(len(rp) - 1, 0, -1)
               if rp[i] > w * 0.03), gray.shape[0])
    return b0, b1


def scan_grid(gray, num_cols, num_rows, labels, z_thresh=1.2, z_gap=0.5, per_row=False):
    eq = apply_clahe(gray)
    inv = cv2.bitwise_not(eq)
    h, w = gray.shape
    b0, b1 = get_bubble_range(gray)
    bh = b1 - b0
    row_h = bh / num_rows
    col_w = w / num_cols

    density_map = np.zeros((num_rows, num_cols), dtype=float)
    raw = np.zeros((num_rows, num_cols), dtype=float)

    for r in range(num_rows):
        for c in range(num_cols):
            y0 = b0 + int(r * row_h) + 2
            y1b = b0 + int((r + 1) * row_h) - 2
            x0 = int(c * col_w) + 2
            x1b = int((c + 1) * col_w) - 2
            cell = inv[y0:y1b, x0:x1b]
            if cell.size == 0:
                continue
            ch, cw = cell.shape
            cx = cell[ch // 4:3 * ch // 4, cw // 4:3 * cw // 4]
            raw[r, c] = float(np.mean(cx)) if cx.size > 0 else 0.0

    results = []
    if not per_row:
        for c in range(num_cols):
            arr = raw[:, c]
            mean, std = arr.mean(), arr.std() + 1e-6
            z = (arr - mean) / std
            density_map[:, c] = z
            best = int(np.argmax(z))
            bz = z[best]
            sz = sorted(z)[-2]
            if bz > z_thresh and (bz - sz) > z_gap and best < len(labels):
                results.append(labels[best])
            else:
                results.append(None)
    else:
        for r in range(num_rows):
            arr = raw[r, :]
            mean, std = arr.mean(), arr.std() + 1e-6
            z = (arr - mean) / std
            density_map[r, :] = z
            best = int(np.argmax(z))
            bz = z[best]
            sz = sorted(z)[-2]
            if bz > z_thresh and (bz - sz) > z_gap and best < len(labels):
                results.append(labels[best])
            else:
                results.append(None)

    return results, density_map, b0, b1, row_h, col_w


def detect_nama(warped):
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
    row_h = bubble_h / 26
    col_w = roi_w / NUM_COLS
    
    raw_scores = np.zeros((26, NUM_COLS), dtype=float)
    density_map = np.zeros((26, NUM_COLS), dtype=float)
    
    for col in range(NUM_COLS):
        x0 = int(col * col_w) + 2
        x1 = int((col + 1) * col_w) - 2
        for row in range(26):
            y0 = bubble_start + int(row * row_h) + 2
            y1 = bubble_start + int((row + 1) * row_h) - 2
            cell = roi_inv[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            ch, cw = cell.shape
            center = cell[ch // 4:3 * ch // 4, cw // 4:3 * cw // 4]
            raw_scores[row, col] = float(np.mean(center)) if center.size > 0 else 0.0
    
    row_means = raw_scores.mean(axis=1, keepdims=True)
    debiased = raw_scores - row_means
    
    nama_result = []
    for col in range(NUM_COLS):
        arr = debiased[:, col]
        mean = arr.mean()
        std = arr.std() + 1e-6
        z = (arr - mean) / std
        
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
    
    # MODIFIKASI: Mengembalikan string nama DAN matriks density_map
    return nama_str, density_map


def detect_nim(warped):
    x1, y1, x2, y2 = ALL_ROIS['NIM']
    roi_nim = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    
    # MODIFIKASI: Tangkap output density_map ke variabel d_map
    r_nim, d_map, _, _, _, _ = scan_grid(
        roi_nim, num_cols=10, num_rows=10,
        labels=[str(i) for i in range(10)], per_row=False)
        
    nim_final = ''.join(r or '_' for r in r_nim)
    
    # MODIFIKASI: Mengembalikan string NIM DAN matriks density_map
    return nim_final, d_map


def detect_tanggal(warped):
    x1, y1, x2, y2 = ALL_ROIS['TANGGAL']
    roi_tgl = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    tgl_cols = 6
    
    # MODIFIKASI: Tangkap output density_map ke variabel d_map
    r_tgl, d_map, _, _, _, _ = scan_grid(
        roi_tgl, num_cols=tgl_cols, num_rows=10,
        labels=[str(i) for i in range(10)], per_row=False)
        
    tgl_final = ''.join(r or '_' for r in r_tgl)
    
    # MODIFIKASI: Mengembalikan string Tanggal DAN matriks density_map
    return tgl_final, d_map


def detect_answers(warped, total_soal=50):
    ROI_JAWABAN = [
        ('1-10', 70, 930, 190, 1150, 1),
        ('11-20', 70, 1160, 190, 1390, 11),
        ('21-30', 250, 930, 380, 1150, 21),
        ('31-40', 259, 1160, 380, 1390, 31),
        ('41-50', 450, 930, 580, 1150, 41),
        ('51-60', 450, 1160, 580, 1390, 51),
        ('61-70', 650, 930, 780, 1150, 61),
        ('71-80', 650, 1160, 780, 1390, 71),
        ('81-90', 830, 930, 950, 1150, 81),
        ('91-100', 830, 1160, 950, 1390, 91),
    ]
    
    all_answers = {}
    soal_done = 0
    heatmap_list = []
    
    for label, x1, y1, x2, y2, q_start in ROI_JAWABAN:
        if soal_done >= total_soal:
            break
        
        soal_di_blok = min(10, total_soal - soal_done)
        roi_bgr = warped[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        r_blok, d_map, _, _, _, _ = scan_grid(
            roi_gray, num_cols=5, num_rows=10,
            labels=CHOICES, per_row=True)
        
        heatmap_list.append(d_map[:soal_di_blok])
        
        r_aktif = r_blok[:soal_di_blok]
        for i, ans in enumerate(r_aktif):
            q = q_start + i
            all_answers[q] = ans
        
        soal_done += soal_di_blok
        
    full_heatmap = np.vstack(heatmap_list) if heatmap_list else np.zeros((10, 5))
    
    return all_answers, full_heatmap
