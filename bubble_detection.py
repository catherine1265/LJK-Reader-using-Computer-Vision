import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from config import SAVE_DIR, CHAR_H, CHAR_W
from corner_detection import find_corner_bubbles, warp_perspective


def preprocess_char(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    bw = cv2.resize(bw, (CHAR_W, CHAR_H))
    return bw


def collect_from_uploads(uploaded_dict):
    total = 0
    CHAR_ROIS = {
        'NAMA_MATA_KULIAH': (50, 180, 900, 240),
        'KODE_KELAS': (50, 270, 300, 330),
        'RUANGAN': (50, 360, 300, 420),
        'NO_MEJA': (50, 450, 300, 510),
    }

    for fname, img_bytes in uploaded_dict.items():
        nparr = np.frombuffer(img_bytes, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        print(f'\n  Processing: {fname}')

        selected = find_corner_bubbles(raw_img, visualize=False)
        page = warp_perspective(raw_img, selected)

        saved_this = 0
        for char, (x1, y1, x2, y2) in CHAR_ROIS.items():
            crop = page[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            bw = preprocess_char(crop)
            idx = len([f for f in os.listdir(SAVE_DIR)
                       if f.startswith(f'{char}__')])
            cv2.imwrite(f'{SAVE_DIR}/{char}__{idx:04d}.png', bw)
            saved_this += 1

        print(f'  {fname}: {saved_this} karakter disimpan')
        total += saved_this

    print(f'\n  Total tersimpan: {total} karakter')
    print(f'  Distribusi:')
    for char in sorted(CHAR_ROIS.keys()):
        n = len([f for f in os.listdir(SAVE_DIR)
                 if f.startswith(f'{char}__')])
        print(f'    {char}: {n} sampel')


def preview_dataset():
    chars = sorted(set(f.split('__')[0]
                       for f in os.listdir(SAVE_DIR)
                       if f.endswith('.png')))
    if not chars:
        print('Dataset kosong.')
        return

    fig, axes = plt.subplots(len(chars), 5,
                             figsize=(12, 2 * len(chars)))
    if len(chars) == 1:
        axes = [axes]

    for i, char in enumerate(chars):
        files_c = sorted([f for f in os.listdir(SAVE_DIR)
                          if f.startswith(f'{char}__')])[:5]
        for j in range(5):
            axes[i][j].axis('off')
            if j < len(files_c):
                img = cv2.imread(
                    os.path.join(SAVE_DIR, files_c[j]),
                    cv2.IMREAD_GRAYSCALE)
                axes[i][j].imshow(img, cmap='gray')
                if j == 0:
                    axes[i][j].set_title(f'"{char}"',
                                         fontsize=11,
                                         fontweight='bold')

    plt.suptitle('Dataset — 5 sampel pertama per karakter',
                 fontweight='bold')
    plt.tight_layout()
    plt.show()
