import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from config import (
    MIN_PAPERS_FOR_EDA, GRADE_BOUNDARIES, COLORS_GRADE,
    EDA_PLOT_OUTPUT, EDA_EXCEL_OUTPUT, KUNCI_JAWABAN
)


def grade_from_score(score):
    if score >= GRADE_BOUNDARIES['A']:
        return 'A'
    elif score >= GRADE_BOUNDARIES['B']:
        return 'B'
    elif score >= GRADE_BOUNDARIES['C']:
        return 'C'
    elif score >= GRADE_BOUNDARIES['D']:
        return 'D'
    else:
        return 'E'


def calculate_statistics(df, answer_key=None):
    if answer_key is None:
        answer_key = KUNCI_JAWABAN
    
    if not answer_key or df['score'].isna().all():
        return None
    
    scores = df['score'].dropna()
    
    stats = {
        'jumlah_siswa': len(scores),
        'rata_rata': scores.mean(),
        'nilai_max': scores.max(),
        'nama_max': df.loc[df['score'].idxmax(), 'nama'] if len(scores) > 0 else '-',
        'nilai_min': scores.min(),
        'nama_min': df.loc[df['score'].idxmin(), 'nama'] if len(scores) > 0 else '-',
        'median': scores.median(),
        'std_dev': scores.std(),
    }
    
    return stats


def get_most_wrong_questions(semua_hasil, answer_key=None, top_n=10):
    if answer_key is None:
        answer_key = KUNCI_JAWABAN
    
    soal_salah = {}
    for rec in semua_hasil:
        for q, ans in rec['jawaban'].items():
            kunci = answer_key.get(q)
            if kunci:
                if soal_salah.get(q) is None:
                    soal_salah[q] = 0
                if ans != kunci:
                    soal_salah[q] += 1
    
    soal_salah_sorted = sorted(soal_salah.items(),
                                key=lambda x: x[1], reverse=True)
    return soal_salah_sorted[:top_n]


def print_statistics(df, answer_key=None):
    if answer_key is None:
        answer_key = KUNCI_JAWABAN
    
    if not answer_key or df['score'].isna().all():
        print(f"\n  Info: Kunci jawaban belum diset atau score kosong")
        return
    
    stats = calculate_statistics(df, answer_key)
    
    print('\n' + '='*70)
    print('  STATISTIK NILAI')
    print('='*70)
    print(f"  Jumlah siswa  : {stats['jumlah_siswa']}")
    print(f"  Rata-rata     : {stats['rata_rata']:.2f}")
    print(f"  Nilai tertinggi: {stats['nilai_max']:.2f}  → {stats['nama_max']}")
    print(f"  Nilai terendah : {stats['nilai_min']:.2f}  → {stats['nama_min']}")
    print(f"  Median        : {stats['median']:.2f}")
    print(f"  Std deviasi   : {stats['std_dev']:.2f}")
    
    df['grade'] = df['score'].apply(grade_from_score)
    grade_counts = df['grade'].value_counts().reindex(
        ['A', 'B', 'C', 'D', 'E'], fill_value=0)
    
    print(f"\n  Distribusi Grade:")
    for g, cnt in grade_counts.items():
        bar = '█' * cnt
        print(f"    {g}: {bar} ({cnt})")


def print_ranking(df, answer_key=None):
    if answer_key is None:
        answer_key = KUNCI_JAWABAN
    
    if not answer_key or df['score'].isna().all():
        return
    
    df_rank = df[['nama', 'nim', 'benar', 'salah', 'kosong', 'score']].copy()
    df_rank['ranking'] = df_rank['score'].rank(ascending=False,
                                                method='min').astype(int)
    df_rank = df_rank.sort_values('ranking')
    
    print('\n' + '='*70)
    print('  TABEL NILAI & RANKING')
    print('='*70)
    print(df_rank.to_string(index=False))


def print_most_wrong_questions(semua_hasil, answer_key=None):
    if answer_key is None:
        answer_key = KUNCI_JAWABAN
    
    if not answer_key:
        return
    
    print('\n' + '='*70)
    print('  SOAL PALING BANYAK SALAH')
    print('='*70)
    
    top10 = get_most_wrong_questions(semua_hasil, answer_key, 10)
    print(f"  {'Soal':>5}  {'Salah':>5}  {'Kunci':>5}")
    for q, cnt in top10:
        print(f"  {q:>5}  {cnt:>5}  {answer_key.get(q, '-'):>5}")


def visualize_eda(df, semua_hasil, answer_key=None):
    if answer_key is None:
        answer_key = KUNCI_JAWABAN
    
    if not answer_key or df['score'].isna().all():
        return
    
    scores = df['score'].dropna()
    df['grade'] = df['score'].apply(grade_from_score)
    grade_counts = df['grade'].value_counts().reindex(
        ['A', 'B', 'C', 'D', 'E'], fill_value=0)
    
    soal_salah_sorted = get_most_wrong_questions(semua_hasil, answer_key, 10)
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45, wspace=0.35)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(scores, bins=max(5, len(scores)//2),
             color='#3498db', edgecolor='white', alpha=0.85)
    ax1.axvline(scores.mean(), color='red', linestyle='--',
                linewidth=1.5, label=f'Rata-rata: {scores.mean():.1f}')
    ax1.set_title('Distribusi Nilai', fontweight='bold')
    ax1.set_xlabel('Nilai')
    ax1.set_ylabel('Jumlah Siswa')
    ax1.legend(fontsize=8)
    
    ax2 = fig.add_subplot(gs[0, 1])
    bar_colors = [COLORS_GRADE[g] for g in grade_counts.index]
    ax2.bar(grade_counts.index, grade_counts.values,
            color=bar_colors, edgecolor='white')
    ax2.set_title('Distribusi Grade', fontweight='bold')
    ax2.set_xlabel('Grade')
    ax2.set_ylabel('Jumlah Siswa')
    for i, (g, v) in enumerate(grade_counts.items()):
        if v > 0:
            ax2.text(i, v + 0.05, str(v),
                     ha='center', fontweight='bold')
    
    ax3 = fig.add_subplot(gs[0, 2])
    df_sorted = df.sort_values('score', ascending=True)
    bar_c = [COLORS_GRADE[g] for g in df_sorted['grade']]
    ax3.barh(range(len(df_sorted)), df_sorted['score'],
             color=bar_c, edgecolor='white')
    ax3.set_yticks(range(len(df_sorted)))
    ax3.set_yticklabels(
        [n[:15] for n in df_sorted['nama']], fontsize=7)
    ax3.set_title('Nilai per Siswa', fontweight='bold')
    ax3.set_xlabel('Nilai')
    ax3.axvline(scores.mean(), color='red',
                linestyle='--', linewidth=1, alpha=0.7)
    
    ax4 = fig.add_subplot(gs[1, 0:2])
    if soal_salah_sorted:
        q_labels = [f"Soal {q}" for q, _ in soal_salah_sorted]
        q_vals = [v for _, v in soal_salah_sorted]
        ax4.bar(q_labels, q_vals,
                color='#e74c3c', edgecolor='white', alpha=0.85)
        ax4.set_title('10 Soal Paling Banyak Salah', fontweight='bold')
        ax4.set_ylabel('Jumlah Siswa Salah')
        ax4.tick_params(axis='x', rotation=30)
    
    ax5 = fig.add_subplot(gs[1, 2])
    x = range(len(df))
    ax5.bar(x, df['benar'], label='Benar', color='#2ecc71')
    ax5.bar(x, df['salah'], label='Salah', color='#e74c3c',
            bottom=df['benar'])
    ax5.bar(x, df['kosong'], label='Kosong', color='#bdc3c7',
            bottom=df['benar']+df['salah'])
    ax5.set_xticks(list(x))
    ax5.set_xticklabels(
        [n[:8] for n in df['nama']], rotation=30, fontsize=7)
    ax5.set_title('Benar / Salah / Kosong', fontweight='bold')
    ax5.set_ylabel('Jumlah Soal')
    ax5.legend(fontsize=7)
    
    plt.suptitle('EDA — STATISTIK NILAI UJIAN',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(EDA_PLOT_OUTPUT, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'\n  Grafik tersimpan → {EDA_PLOT_OUTPUT}')


def export_to_excel(df, filename=None):
    if filename is None:
        filename = EDA_EXCEL_OUTPUT
    
    df_export = df[['nama', 'nim', 'benar', 'salah', 'kosong', 'score']].copy()
    df_export['grade'] = df_export['score'].apply(
        lambda x: grade_from_score(x) if pd.notna(x) else '-'
    )
    df_export['ranking'] = df_export['score'].rank(
        ascending=False, method='min').astype('Int64')
    df_export = df_export.sort_values('ranking')
    
    df_export.to_excel(filename, sheet_name='Hasil', index=False)
    print(f'\n  Hasil export → {filename}')


def run_eda(semua_hasil, answer_key=None):
    if answer_key is None:
        answer_key = KUNCI_JAWABAN
    
    if len(semua_hasil) < MIN_PAPERS_FOR_EDA:
        sisa = MIN_PAPERS_FOR_EDA - len(semua_hasil)
        print(f"\n  Info: EDA perlu minimal {MIN_PAPERS_FOR_EDA} lembar. Scan {sisa} lembar lagi.")
        return
    
    df = pd.DataFrame(semua_hasil)
    
    print_ranking(df, answer_key)
    print_statistics(df, answer_key)
    print_most_wrong_questions(semua_hasil, answer_key)
    
    visualize_eda(df, semua_hasil, answer_key)
    export_to_excel(df)
