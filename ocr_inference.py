import cv2
import numpy as np
from config import BUBBLE_ROIS, KUNCI_JAWABAN


def detect_bubble_mark(page, q, opt):
    x1, y1, x2, y2 = BUBBLE_ROIS[(q, opt)]
    crop = page[y1:y2, x1:x2]
    
    if crop.size == 0:
        return None
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    filled = np.sum(binary > 0)
    total = binary.size
    fill_ratio = filled / total if total > 0 else 0
    
    return fill_ratio > 0.3


def extract_answers(page, bundle=None):
    all_answers = {}
    
    for q in range(1, 51):
        for opt in ['A', 'B', 'C', 'D']:
            is_marked = detect_bubble_mark(page, q, opt)
            if is_marked:
                all_answers[q] = opt
                break
    
    return all_answers


def calculate_score(all_answers, answer_key=None):
    if answer_key is None:
        answer_key = KUNCI_JAWABAN
    
    benar = sum(1 for q, ans in all_answers.items()
                if ans and answer_key.get(q) == ans)
    salah = sum(1 for q, ans in all_answers.items()
                if ans and answer_key.get(q) != ans)
    kosong = sum(1 for q, ans in all_answers.items() if not ans)
    score = round(benar / len(answer_key) * 100, 2) if answer_key else None
    
    return benar, salah, kosong, score
