import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import GRAY_THRESHOLD, MORPH_KERNEL, MORPH_ITER


def find_corner_bubbles(img, visualize=True):
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, dark = cv2.threshold(gray, GRAY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k, iterations=MORPH_ITER)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        solidity = area / float(w * h)
        aspect = w / float(h)
        min_a = img_w * img_h * 0.0003
        max_a = img_w * img_h * 0.04
        if min_a < area < max_a and 0.4 < aspect < 2.5 and solidity >= 0.65:
            candidates.append({
                'area': area, 'x': x, 'y': y, 'w': w, 'h': h,
                'cx': x + w // 2, 'cy': y + h // 2
            })

    corners = {
        'TL': (0, 0),
        'TR': (img_w, 0),
        'BR': (img_w, img_h),
        'BL': (0, img_h),
    }

    selected = {}
    for label, (tx, ty) in corners.items():
        if not candidates:
            break
        best = min(candidates,
                   key=lambda a: (a['cx'] - tx) ** 2 + (a['cy'] - ty) ** 2)
        selected[label] = best

    print(f'\n  Total kandidat: {len(candidates)}')
    print(f'\n  4 Kotak terpilih (terdekat ke tiap sudut):')
    for label, a in selected.items():
        print(f'    {label}: cx={a["cx"]:4d}  cy={a["cy"]:4d}  '
              f'size={a["w"]}x{a["h"]}  area={a["area"]:.0f}')

    if visualize:
        COLORS = {'TL': (255, 50, 50), 'TR': (50, 50, 255),
                  'BR': (50, 200, 50), 'BL': (255, 165, 0)}
        vis = img.copy()

        for label, a in selected.items():
            col = COLORS[label]
            cv2.rectangle(vis, (a['x'], a['y']),
                          (a['x'] + a['w'], a['y'] + a['h']), col, 3)
            cv2.circle(vis, (a['cx'], a['cy']), 10, col, -1)
            cv2.putText(vis, label,
                        (a['cx'] + 12, a['cy'] + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        order = ['TL', 'TR', 'BR', 'BL']
        for i in range(4):
            if order[i] in selected and order[(i + 1) % 4] in selected:
                p1 = (selected[order[i]]['cx'], selected[order[i]]['cy'])
                p2 = (selected[order[(i + 1) % 4]]['cx'], selected[order[(i + 1) % 4]]['cy'])
                cv2.line(vis, p1, p2, (0, 255, 0), 2)

        plt.figure(figsize=(7, 10))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title('4 Kotak Sudut LJK\nMerah=TL  Biru=TR  Hijau=BR  Orange=BL',
                  fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return selected


def warp_perspective(raw_img, selected):
    page = raw_img

    try:
        if len(selected) == 4:
            pts = np.array([
                [selected['TL']['cx'], selected['TL']['cy']],
                [selected['TR']['cx'], selected['TR']['cy']],
                [selected['BR']['cx'], selected['BR']['cy']],
                [selected['BL']['cx'], selected['BL']['cy']]
            ], dtype='float32')

            s = pts.sum(1)
            d = np.diff(pts, axis=1)
            rect = np.zeros((4, 2), dtype='float32')
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            rect[1] = pts[np.argmin(d)]
            rect[3] = pts[np.argmax(d)]

            mW = max(int(np.linalg.norm(rect[2] - rect[3])),
                     int(np.linalg.norm(rect[1] - rect[0])))
            mH = max(int(np.linalg.norm(rect[1] - rect[2])),
                     int(np.linalg.norm(rect[0] - rect[3])))

            dst = np.float32([[0, 0], [mW - 1, 0], [mW - 1, mH - 1], [0, mH - 1]])
            M = cv2.getPerspectiveTransform(rect, dst)
            page = cv2.warpPerspective(raw_img, M, (mW, mH))
            page = cv2.resize(page, (1000, 1414))
        else:
            page = raw_img
    except Exception:
        page = raw_img

    return page
