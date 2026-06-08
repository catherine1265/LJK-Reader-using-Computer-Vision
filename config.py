import os

MODEL_PATH = 'svm_emnist.pkl'
SAVE_DIR = 'dataset/chars'

if os.path.dirname(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

CHAR_H, CHAR_W = 32, 32

ALL_CHARS = sorted(
    list('0123456789') +
    list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
    list('abcdefghijklmnopqrstuvwxyz')
)

ALL_ROIS = {
    'NAMA': (40, 270, 520, 850),
    'NIM': (550, 270, 790, 500),
    'TANGGAL': (820, 270, 970, 500),
    'NAMA_MATA_KULIAH': (540, 545, 940, 625),
    'KODE_KELAS': (540, 658, 740, 725),
    'RUANGAN': (775, 658, 950, 725),
    'NO_MEJA': (50, 450, 300, 510),
}

HANDWRITING_ROIS = [
    (k, *ALL_ROIS[k])
    for k in ('NAMA_MATA_KULIAH', 'KODE_KELAS', 'RUANGAN', 'NO_MEJA')
]

N_CHARS = {
    'RUANGAN': 5,
    'KODE_KELAS': 4,
    'NO_MEJA': 2,
    'NAMA_MATA_KULIAH': 13,
}

BUBBLE_COLS = {
    'A': 550,
    'B': 650,
    'C': 750,
    'D': 850,
}

BUBBLE_ROIS = {}
for q in range(1, 51):
    row = (q - 1) // 10
    col_in_group = (q - 1) % 10
    y = 700 + col_in_group * 60
    
    for opt in ['A', 'B', 'C', 'D']:
        x = BUBBLE_COLS[opt]
        BUBBLE_ROIS[(q, opt)] = (x - 15, y - 15, x + 15, y + 15)

KUNCI_JAWABAN = {}

GRAY_THRESHOLD = 80
MORPH_KERNEL = (3, 3)
MORPH_ITER = 1
DILATE_ITER = 2

MIN_CONTOUR_AREA = 5000
CONTOUR_APPROX_EPSILON = 0.02

BUBBLE_AREA_MIN = 100
BUBBLE_AREA_MAX = 2000
BUBBLE_DARKNESS_THRESHOLD = 0.4

MIN_PAPERS_FOR_EDA = 3

GRADE_BOUNDARIES = {
    'A': 85,
    'B': 75,
    'C': 65,
    'D': 55,
    'E': 0,
}

COLORS_GRADE = {
    'A': '#2ecc71',
    'B': '#3498db',
    'C': '#f39c12',
    'D': '#e67e22',
    'E': '#e74c3c',
}

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

SVM_KERNEL = 'rbf'
SVM_C = 10
SVM_GAMMA = 'scale'
SVM_RANDOM_STATE = 42

AUGMENT_N_SAMPLES = 30
AUGMENT_ROTATION_RANGE = 15
AUGMENT_SCALE_RANGE = (0.85, 1.15)
AUGMENT_NOISE_STD = 8
AUGMENT_SHEAR_RANGE = 4

EDA_PLOT_OUTPUT = 'eda_statistik.png'
EDA_EXCEL_OUTPUT = 'hasil_ujian.xlsx'
