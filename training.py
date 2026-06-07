import os
import pickle
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from config import (
    SAVE_DIR, MODEL_PATH, CHAR_H, CHAR_W,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK,
    SVM_KERNEL, SVM_C, SVM_GAMMA, SVM_RANDOM_STATE,
    AUGMENT_N_SAMPLES, AUGMENT_ROTATION_RANGE, AUGMENT_SCALE_RANGE,
    AUGMENT_NOISE_STD, AUGMENT_SHEAR_RANGE
)


def extract_hog(img):
    return hog(img,
               orientations=HOG_ORIENTATIONS,
               pixels_per_cell=HOG_PIXELS_PER_CELL,
               cells_per_block=HOG_CELLS_PER_BLOCK,
               block_norm='L2-Hys')


def augment_char(img, n=None):
    if n is None:
        n = AUGMENT_N_SAMPLES

    h, w = img.shape
    results = [img]

    for _ in range(n):
        aug = img.copy()

        angle = np.random.uniform(-AUGMENT_ROTATION_RANGE, AUGMENT_ROTATION_RANGE)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        aug = cv2.warpAffine(aug, M, (w, h), borderValue=255)

        sc = np.random.uniform(AUGMENT_SCALE_RANGE[0], AUGMENT_SCALE_RANGE[1])
        aug = cv2.resize(aug, (max(1, int(w * sc)), max(1, int(h * sc))))
        aug = cv2.resize(aug, (w, h))

        noise = np.random.normal(0, AUGMENT_NOISE_STD, aug.shape).astype(np.int16)
        aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if np.random.rand() > 0.5:
            aug = cv2.GaussianBlur(aug, (3, 3), 0)

        dx = np.random.uniform(-AUGMENT_SHEAR_RANGE, AUGMENT_SHEAR_RANGE)
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        pts2 = np.float32([[dx, 0], [w + dx, 0], [0, h]])
        M = cv2.getAffineTransform(pts1, pts2)
        aug = cv2.warpAffine(aug, M, (w, h), borderValue=255)

        results.append(aug)

    return results


def retrain_svm():
    X, y = [], []
    chars_found = []

    for fname in os.listdir(SAVE_DIR):
        if not fname.endswith('.png'):
            continue
        char = fname.split('__')[0]
        img = cv2.imread(os.path.join(SAVE_DIR, fname),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        augmented = augment_char(img)
        for aug in augmented:
            feat = extract_hog(aug)
            X.append(feat)
            y.append(char)
        chars_found.append(char)

    if not X:
        print('  Tidak ada data!')
        return None

    X = np.array(X)
    y = np.array(y)
    print(f'  Training SVM: {len(X)} sampel, '
          f'{len(set(y))} kelas: {sorted(set(y))}')

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA,
                    random_state=SVM_RANDOM_STATE))
    ])
    clf.fit(X, y)

    bundle = {'clf': clf, 'chars': sorted(set(y))}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'  Model tersimpan → {MODEL_PATH}')
    print(f'  Karakter: {sorted(set(y))}')
    return bundle


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f'  Model tidak ditemukan: {MODEL_PATH}')
        return None

    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    print(f'  Model loaded → {MODEL_PATH}')
    return bundle


def delete_model():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print(f'  Model deleted: {MODEL_PATH}')
    else:
        print(f'  Model file not found: {MODEL_PATH}')


def download_model_colab():
    try:
        from google.colab import files
        files.download(MODEL_PATH)
        print(f'  Model downloaded to local machine')
    except ImportError:
        print(f'  Not in Colab environment, skipping download')
