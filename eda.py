import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def grade(s):
    if s >= 85: return 'A'
    elif s >= 75: return 'B'
    elif s >= 65: return 'C'
    elif s >= 55: return 'D'
    else: return 'E'


def grade_from_score(score):
    """Alias untuk grade function"""
    return grade(score)


def visualize_eda(semua_hasil, kunci_jawaban):
    """
    Generate EDA visualization dari semua_hasil & kunci_jawaban
    
    Args:
        semua_hasil: list of dict {nim, nama, score, benar, salah, kosong, jawaban}
        kunci_jawaban: dict {q: ans}
    """
    if not semua_hasil or not kunci_jawaban:
        return None
    
    df = pd.DataFrame(semua_hasil)
    
    # ── Tabel Benar / Salah / Score + Ranking ───────────────────────
    if kunci_jawaban and df['score'].notna().any():
        df_rank = df[['nama','nim','benar','salah','kosong','score']].copy()
        df_rank['ranking'] = df_rank['score'].rank(ascending=False,
                                                    method='min').astype(int)
        df_rank = df_rank.sort_values('ranking')

        print('\n' + '='*70)
        print('  📊 TABEL NILAI & RANKING')
        print('='*70)
        print(df_rank.to_string(index=False))

    # ── Statistik Deskriptif ─────────────────────────────────────────
    print('\n' + '='*70)
    print('  📈 STATISTIK NILAI')
    print('='*70)

    if kunci_jawaban and df['score'].notna().any():
        scores = df['score'].dropna()
        print(f"  Jumlah siswa  : {len(scores)}")
        print(f"  Rata-rata     : {scores.mean():.2f}")
        print(f"  Nilai tertinggi: {scores.max():.2f}  "
              f"→ {df.loc[df['score'].idxmax(), 'nama']}")
        print(f"  Nilai terendah : {scores.min():.2f}  "
              f"→ {df.loc[df['score'].idxmin(), 'nama']}")
        print(f"  Median        : {scores.median():.2f}")
        print(f"  Std deviasi   : {scores.std():.2f}")

        # ── Distribusi nilai ─────────────────────────────────────────
        df['grade'] = df['score'].apply(grade)
        grade_counts = df['grade'].value_counts().reindex(
            ['A','B','C','D','E'], fill_value=0)

        print(f"\n  Distribusi Grade:")
        for g, cnt in grade_counts.items():
            bar = '█' * cnt
            print(f"    {g}: {bar} ({cnt})")

    # ── Analisis per soal: soal paling banyak salah ──────────────────
    if kunci_jawaban and len(semua_hasil) >= 1:
        print('\n' + '='*70)
        print('  ❌ SOAL PALING BANYAK SALAH')
        print('='*70)

        soal_salah = {}
        for rec in semua_hasil:
            for q, ans in rec['jawaban'].items():
                kunci = kunci_jawaban.get(q)
                if kunci:
                    if soal_salah.get(q) is None:
                        soal_salah[q] = 0
                    if ans != kunci:
                        soal_salah[q] += 1

        soal_salah_sorted = sorted(soal_salah.items(),
                                    key=lambda x: x[1], reverse=True)
        print(f"  {'Soal':>5}  {'Salah':>5}  {'Kunci':>5}")
        for q, cnt in soal_salah_sorted[:10]:
            print(f"  {q:>5}  {cnt:>5}  {kunci_jawaban.get(q,'-'):>5}")

    # ── Visualisasi ──────────────────────────────────────────────────
    if kunci_jawaban and df['score'].notna().any():
        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig,
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
        x   = range(len(df))
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
        plt.savefig('eda_statistik.png', dpi=150, bbox_inches='tight')
        plt.show()
        print('\n  Grafik tersimpan → eda_statistik.png')
        
        return fig
    
    return None
