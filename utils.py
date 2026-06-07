import cv2
import matplotlib.pyplot as plt
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
    
    # Renders the plot inside the web browser instead of a pop-up window
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
    
    # Renders the rows inside the web browser instead of a pop-up window
    st.pyplot(fig)
    plt.close(fig)

def show_heatmap(density_map, title="Heatmap OMR", y_labels=None, x_labels=None, cmap='RdYlGn'):
    """
    Fungsi universal untuk merender matriks density/z-score menjadi heatmap di Streamlit.
    Menghindari freeze aplikasi akibat penggunaan plt.show().
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Render matriks data ke grafik heatmap
    im = ax.imshow(density_map, aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=-3, vmax=3)
    
    # Atur label Sumbu Y jika di-passing (misal list alphabet A-Z atau nomor Soal)
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        
    # Atur label Sumbu X jika di-passing (misal list indeks kolom 1-20 atau opsi A-E)
    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=8)
        
    ax.set_title(title, fontweight='bold', fontsize=12)
    plt.colorbar(im, ax=ax, label='Z-Score Kepadatan Piksel')
    plt.tight_layout()
    
    # Renders heatmap inside Streamlit browser
    st.pyplot(fig)
    plt.close(fig)
