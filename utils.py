import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import numpy as np


def print_step(step, msg):
    print(f" [{step}] {msg}")


def show(img, title='', cmap=None, size=(10, 8)):
    fig = plt.figure(figsize=size)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=13, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_row(images, titles, size=(18, 5)):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=size)
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_heatmap(
    density_map,
    title="Heatmap OMR",
    y_labels=None,
    x_labels=None,
    cmap='RdYlGn',
    roi_img=None,          # crop BGR dari warped (opsional)
    roi_title="ROI",       # judul panel gambar
):
    """
    Menampilkan crop ROI (jika di-passing) di sebelah kiri
    dan heatmap z-score di sebelah kanan.

    Parameters
    ----------
    density_map : np.ndarray  — matriks z-score (rows x cols)
    title       : str         — judul heatmap
    y_labels    : list[str]   — label sumbu Y
    x_labels    : list[str]   — label sumbu X
    cmap        : str         — colormap matplotlib
    roi_img     : np.ndarray  — gambar crop BGR; jika None hanya tampilkan heatmap
    roi_title   : str         — judul panel gambar ROI
    """
    has_roi = roi_img is not None

    if has_roi:
        fig = plt.figure(figsize=(14, 6))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 2], figure=fig)
        ax_img  = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1])

        if len(roi_img.shape) == 2:
            ax_img.imshow(roi_img, cmap='gray')
        else:
            ax_img.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        ax_img.set_title(roi_title, fontsize=11, fontweight='bold')
        ax_img.axis('off')
    else:
        fig, ax_heat = plt.subplots(figsize=(10, 6))

    im = ax_heat.imshow(
        density_map,
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        vmin=-3,
        vmax=3,
    )
    if y_labels is not None:
        ax_heat.set_yticks(range(len(y_labels)))
        ax_heat.set_yticklabels(y_labels, fontsize=8)
    if x_labels is not None:
        ax_heat.set_xticks(range(len(x_labels)))
        ax_heat.set_xticklabels(x_labels, fontsize=8)
    ax_heat.set_title(title, fontweight='bold', fontsize=12)
    plt.colorbar(im, ax=ax_heat, label='Z-Score Kepadatan Piksel')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_heatmap_jawaban(
    warped,
    density_map,
    total_soal=50,
    cmap='jet',
):
    """
    Khusus untuk section Jawaban OMR.
    Menampilkan per-10-soal: crop ROI bubble sheet di kiri, heatmap di kanan.

    Parameters
    ----------
    warped      : np.ndarray  — gambar warped BGR full LJK
    density_map : np.ndarray  — full heatmap hasil detect_answers (total_soal x 5)
    total_soal  : int         — jumlah soal aktif
    cmap        : str         — colormap
    """
    # Koordinat ROI per blok 10 soal, sama persis dengan scanner.py
    ROI_JAWABAN = [
        ('1-10',   70,  930, 190, 1150,  1),
        ('11-20',  70, 1160, 190, 1390, 11),
        ('21-30', 250,  930, 380, 1150, 21),
        ('31-40', 259, 1160, 380, 1390, 31),
        ('41-50', 450,  930, 580, 1150, 41),
        ('51-60', 450, 1160, 580, 1390, 51),
        ('61-70', 650,  930, 780, 1150, 61),
        ('71-80', 650, 1160, 780, 1390, 71),
        ('81-90', 830,  930, 950, 1150, 81),
        ('91-100',830, 1160, 950, 1390, 91),
    ]

    soal_done = 0
    row_offset = 0

    for label, x1, y1, x2, y2, q_start in ROI_JAWABAN:
        if soal_done >= total_soal:
            break

        soal_di_blok = min(10, total_soal - soal_done)
        q_end        = q_start + soal_di_blok - 1

        # Crop ROI bubble sheet untuk blok ini
        roi_bgr = warped[y1:y2, x1:x2]

        # Slice heatmap untuk blok ini
        blok_heat = density_map[row_offset: row_offset + soal_di_blok]

        # Buat label soal untuk sumbu Y
        y_labels_blok = [f"Soal {q_start + i}" for i in range(soal_di_blok)]

        show_heatmap(
            density_map=blok_heat,
            title=f"Heatmap Jawaban Soal {q_start}–{q_end}",
            y_labels=y_labels_blok,
            x_labels=['A', 'B', 'C', 'D', 'E'],
            cmap=cmap,
            roi_img=roi_bgr,
            roi_title=f"Bubble Sheet Soal {q_start}–{q_end}",
        )

        soal_done  += soal_di_blok
        row_offset += soal_di_blok


def show_handwriting(roi_img, ocr_text, label_name="Handwriting"):
    """
    Menampilkan crop ROI tulisan tangan di kiri, hasil OCR di kanan.

    Parameters
    ----------
    roi_img     : np.ndarray  — gambar crop BGR dari ROI tulisan tangan
    ocr_text    : str         — hasil OCR/ekstraksi text
    label_name  : str         — nama field (misal "Nama Mata Kuliah", "Kode Kelas", dll)
    """
    fig = plt.figure(figsize=(12, 4))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
    ax_img = fig.add_subplot(gs[0])
    ax_txt = fig.add_subplot(gs[1])

    if len(roi_img.shape) == 2:
        ax_img.imshow(roi_img, cmap='gray')
    else:
        ax_img.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"ROI {label_name}", fontsize=11, fontweight='bold')
    ax_img.axis('off')

    ax_txt.axis('off')
    result_text = ocr_text if ocr_text else "—"
    ax_txt.text(
        0.5, 0.5,
        result_text,
        fontsize=18,
        fontweight='bold',
        ha='center',
        va='center',
        transform=ax_txt.transAxes,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    ax_txt.set_title(f"OCR Result: {label_name}", fontsize=11, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
