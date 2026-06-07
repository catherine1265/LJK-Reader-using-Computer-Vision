import cv2
import numpy as np
from config import (
    GRAY_THRESHOLD, MORPH_KERNEL, MORPH_ITER, DILATE_ITER,
    MIN_CONTOUR_AREA, CONTOUR_APPROX_EPSILON
)
from utils import print_step, show_row


def preprocess_image(img_original):
    img = img_original.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)

    opened = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=MORPH_ITER
    )

    dilated = cv2.dilate(opened, kernel, iterations=DILATE_ITER)

    print_step('4', 'Morphology + Dilasi')

    return img, gray, thresh, opened, dilated


def find_document_contour(dilated, img_original):
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > MIN_CONTOUR_AREA:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(
                cnt,
                CONTOUR_APPROX_EPSILON * peri,
                True
            )

            if len(approx) == 4 and area > max_area:
                largest = approx
                max_area = area

    print_step('5', 'Find Document Contour')
    return largest


def draw_contour(img, largest):
    contour_img = img.copy()

    if largest is not None:
        cv2.drawContours(
            contour_img,
            [largest],
            -1,
            (0, 255, 0),
            4
        )

    return contour_img


def visualize_preprocessing(img, gray, thresh, contour_img):
    show_row(
        [img, gray, thresh, contour_img],
        ['Original', 'Gray', 'Threshold', 'Contour'],
        size=(25, 5)
    )
