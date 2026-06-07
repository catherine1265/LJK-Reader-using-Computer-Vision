"""
EDA Module - Scoring, grade conversion, dan statistik analysis functions
Extracted dari cv_final__w_eda_.py - Keep all original logic intact
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def grade_from_score(score):
    """
    Convert score (0-100) to grade letter
    Sesuai cv_final__w_eda_.py line 1439-1444
    
    Args:
        score: float (0-100)
    
    Returns:
        str: Grade letter ('A', 'B', 'C', 'D', 'E')
    """
    if score >= 85:
        return 'A'
    elif score >= 75:
        return 'B'
    elif score >= 65:
        return 'C'
    elif score >= 55:
        return 'D'
    else:
        return 'E'


def calculate_score(answers, answer_key):
    """
    Calculate score dari dict jawaban siswa
    Sesuai cv_final__w_eda_.py line 1374-1383
    
    Args:
        answers: dict {soal_number: 'A'/'B'/'C'/'D'/'E' or None}
        answer_key: dict {soal_number: 'A'/'B'/'C'/'D'/'E'}
    
    Returns:
        tuple: (benar, salah, kosong, score_percentage)
            - benar: int, jumlah jawaban benar
            - salah: int, jumlah jawaban salah
            - kosong: int, jumlah yang tidak dijawab
            - score_percentage: float, nilai 0-100 (benar / len(answer_key) * 100)
    """
    if not answer_key:
        return 0, 0, 0, 0
    
    # Hitung benar: jawaban yang sama dengan kunci
    benar = sum(1 for q, ans in answers.items()
                if ans and answer_key.get(q) == ans)
    
    # Hitung salah: jawaban yang berbeda dengan kunci (tapi ada jawaban)
    salah = sum(1 for q, ans in answers.items()
                if ans and answer_key.get(q) != ans)
    
    # Hitung kosong: tidak ada jawaban
    kosong = sum(1 for q, ans in answers.items() if not ans)
    
    # Hitung score: benar / total soal * 100 (sesuai Colab)
    score = round(benar / len(answer_key) * 100, 2)
    
    return benar, salah, kosong, score


def visualize_eda(df_records, answer_key, output_path='eda_statistik.png'):
    """
    Visualisasi EDA dengan 5 charts
    Sesuai cv_final__w_eda_.py line 1476-1551
    
    Args:
        df_records: list of dicts with keys: 'nama', 'nim', 'score', 'benar', 'salah', 'kosong'
        answer_key: dict dengan answer key
        output_path: path untuk save gambar
    """
    if not df_records or not answer_key:
        return
    
    df = pd.DataFrame(df_records)
    
    # Tambah grade column
    df['grade'] = df['score'].apply(grade_from_score)
    
    scores = df['score'].values
    grade_counts = df['grade'].value_counts().reindex(['A','B','C','D','E'], fill_value=0)
    
    # Analisis soal paling banyak salah
    soal_salah = {}
    for rec in df_records:
        for q, ans in rec.get('jawaban', {}).items():
            kunci = answer_key.get(q)
            if kunci:
                if soal_salah.get(q) is None:
                    soal_salah[q] = 0
                if ans != kunci:
                    soal_salah[q] += 1
    
    soal_salah_sorted = sorted(soal_salah.items(),
                                key=lambda x: x[1], reverse=True)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45, wspace=0.35)
    
    colors_grade = {'A':'#2ecc71','B':'#3498db',
                    'C':'#f39c12','D':'#e67e22','E':'#e74c3c'}
    
    # 1. Histogram distribusi nilai
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(scores, bins=max(5, len(scores)//2),
             color='#3498db', edgecolor='white', alpha=0.85)
    ax1.axvline(scores.mean(), color='red', linestyle='--',
                linewidth=1.5, label=f'Rata-rata: {scores.mean():.1f}')
    ax1.set_title('Distribusi Nilai', fontweight='bold')
    ax1.set_xlabel('Nilai'); ax1.set_ylabel('Jumlah Siswa')
    ax1.legend(fontsize=8)
    
    # 2. Bar chart grade
    ax2 = fig.add_subplot(gs[0, 1])
    bar_colors = [colors_grade[g] for g in grade_counts.index]
    ax2.bar(grade_counts.index, grade_counts.values,
            color=bar_colors, edgecolor='white')
    ax2.set_title('Distribusi Grade', fontweight='bold')
    ax2.set_xlabel('Grade'); ax2.set_ylabel('Jumlah Siswa')
    for i, (g, v) in enumerate(grade_counts.items()):
        if v > 0:
            ax2.text(i, v + 0.05, str(v),
                     ha='center', fontweight='bold')
    
    # 3. Ranking bar (nilai per siswa)
    ax3 = fig.add_subplot(gs[0, 2])
    df_sorted = df.sort_values('score', ascending=True)
    bar_c = [colors_grade[g] for g in df_sorted['grade']]
    ax3.barh(range(len(df_sorted)), df_sorted['score'],
             color=bar_c, edgecolor='white')
    ax3.set_yticks(range(len(df_sorted)))
    ax3.set_yticklabels(
        [n[:15] for n in df_sorted['nama']], fontsize=7)
    ax3.set_title('Nilai per Siswa', fontweight='bold')
    ax3.set_xlabel('Nilai')
    ax3.axvline(scores.mean(), color='red',
                linestyle='--', linewidth=1, alpha=0.7)
    
    # 4. Soal paling banyak salah (bar)
    ax4 = fig.add_subplot(gs[1, 0:2])
    top10 = soal_salah_sorted[:10]
    if top10:
        q_labels = [f"Soal {q}" for q, _ in top10]
        q_vals   = [v for _, v in top10]
        ax4.bar(q_labels, q_vals,
                color='#e74c3c', edgecolor='white', alpha=0.85)
        ax4.set_title('10 Soal Paling Banyak Salah', fontweight='bold')
        ax4.set_ylabel('Jumlah Siswa Salah')
        ax4.tick_params(axis='x', rotation=30)
    
    # 5. Benar vs Salah per siswa (stacked bar)
    ax5 = fig.add_subplot(gs[1, 2])
    x = range(len(df))
    ax5.bar(x, df['benar'],  label='Benar',  color='#2ecc71')
    ax5.bar(x, df['salah'],  label='Salah',  color='#e74c3c',
            bottom=df['benar'])
    ax5.bar(x, df['kosong'], label='Kosong', color='#bdc3c7',
            bottom=df['benar']+df['salah'])
    ax5.set_xticks(list(x))
    ax5.set_xticklabels(
        [n[:8] for n in df['nama']], rotation=30, fontsize=7)
    ax5.set_title('Benar / Salah / Kosong', fontweight='bold')
    ax5.set_ylabel('Jumlah Soal')
    ax5.legend(fontsize=7)
    
    plt.suptitle('📊 EDA — STATISTIK NILAI UJIAN',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f'\n  Grafik tersimpan → {output_path}')
