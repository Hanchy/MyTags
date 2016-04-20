#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import cv2
import numpy as np
import thinning
from matplotlib import pyplot as plt


def whether_true(matrix):
    # counter = 0.0
    center = matrix[int(matrix.shape[0] / 2.0),
                    int(matrix.shape[1] / 2.0)]
    max_ele = np.max(matrix)
    min_ele = np.min(matrix)
    length = max_ele - min_ele
    #print("[%d, %d]", [center, length])
    if center > 0.3 * length:
        return True
    else:
        return False
    # for i in xrange(matrix.shape[0]):
    #     for j in xrange(matrix.shape[1]):
    #         if center > matrix[i, j]:
    #             counter = counter + 1.0
    #
    # if counter / matrix.size > 0.5:
    #     return True
    # else:
    #     return False


def local_thresholding(img):
    binary = np.zeros(img.shape, np.uint8)
    tau = 4
    for i in xrange(tau, img.shape[0]):
        for j in xrange(tau, img.shape[1]):
            thresh = whether_true(img[i - tau:i + tau + 1, j - tau:j + tau + 1])
            binary[i, j] = thresh
    return binary


def preprocessing(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    horizontal_grad = cv2.Scharr(gray_img, cv2.CV_32F, 1, 0)
    vertical_grad = cv2.Scharr(gray_img, cv2.CV_32F, 0, 1)  # , ksize=3)
    gradient = np.sqrt(np.power(horizontal_grad, 2) +
                       np.power(vertical_grad, 2))
    _, th = cv2.threshold(gradient, 200, 255, cv2.THRESH_BINARY)
    binary = local_thresholding(th)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, 1)
    erosion = cv2.erode(dilated, kernel, iterations=2)
    closed = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations=1)
    thinned = thinning.guo_hall_thinning(closed.astype(np.uint8))

    contours, hierarchy = cv2.findContours(thinned.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = []
    # arc_lengths = []
    for ele in contours:
        if len(ele) > 2:
            retval = cv2.arcLength(ele, False)
            # arc_lengths.append(retval)
            if retval > 160:
                contours2.append(ele)

    trimmed_binary = np.zeros(gray_img.shape, np.uint8)
    cv2.drawContours(trimmed_binary, contours2, -1, (255, 255, 255), 1)
    potential_corners = []
    for i in xrange(1, trimmed_binary.shape[0] - 1):
        for j in xrange(1, trimmed_binary.shape[1] - 1):
            if trimmed_binary[i, j] > 200:
                counter = 0
                if trimmed_binary[i - 1, j - 1] > 200:
                    counter = counter + 1
                if trimmed_binary[i - 1, j + 1] > 200:
                    counter = counter + 1
                if trimmed_binary[i + 1, j - 1] > 200:
                    counter = counter + 1
                if trimmed_binary[i + 1, j + 1] > 200:
                    counter = counter + 1
                if trimmed_binary[i, j + 1] > 200:
                    counter = counter + 1
                if trimmed_binary[i, j - 1] > 200:
                    counter = counter + 1
                if trimmed_binary[i + 1, j] > 200:
                    counter = counter + 1
                if trimmed_binary[i - 1, j] > 200:
                    counter = counter + 1
                if counter > 2:
                    potential_corners.append([j, i])
    p_corners = np.array(potential_corners)
    for ele in p_corners:
        cv2.circle(img, tuple(ele), 2, (0, 0, 255), 2)

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(thinned, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(trimmed_binary, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(gradient, cmap='gray')
    plt.show()


def count_nonezeros(array):
    count = 0
    x = array.reshape(-1)
    for i in xrange(x.size):
        if np.fabs(x[i]) > 1e-2:
            count = count + 1
    return count


def invert_zeros(img):
    for i in xrange(1, img.shape[0] - 1):
        for j in xrange(1, img.shape[1] - 1):
            if np.fabs(img[i, j]) > 1e-3:
                continue
            count = count_nonezeros(img[i - 1:i + 2, j - 1:j + 2])
            if count > 2:
                img[i, j] = 1


def is_saddle(matrix):
    counter = 0
    # print(matrix.shape)
    for i in xrange(matrix.shape[0]):
        for j in xrange(matrix.shape[1]):
            if matrix[i, j] > 127:
                counter = counter + 1
            else:
                counter = counter - 1
    print(counter, 0.3 * matrix.size)
    if np.abs(counter) < 0.3 * matrix.size:
        return True
    else:
        return False


def detect_lines(matrix):
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return

    lines = cv2.HoughLinesP(matrix, 1, np.pi / 180, 100, 20, 100)
    if lines is not None:
        print(lines)
    # for x1, y1, x2, y2 in lines[0]:
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return lines


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray_img)
        horizontal_grad = cv2.Scharr(gray_img, cv2.CV_32F, 1, 0)
        vertical_grad = cv2.Scharr(gray_img, cv2.CV_32F, 0, 1)  # , ksize=3)
        gradient = np.sqrt(np.power(horizontal_grad, 2) +
                           np.power(vertical_grad, 2))
        _, th = cv2.threshold(gradient, 200, 255, cv2.THRESH_BINARY)
        # binary = local_thresholding(th)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(th, kernel, 1)
        erosion = cv2.erode(dilated, kernel, iterations=1)
        closed = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations=1)
        thinned = thinning.guo_hall_thinning(closed.astype(np.uint8))
        #_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(thinned, 1, np.pi / 180, 100, 5)
        if lines is not None:
            for x1, y1, x2, y2 in lines[0]:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        corners = cv2.goodFeaturesToTrack(gray, 10, 0.01, 25)

        corners = np.float32(corners)

        # print("corners", corners.shape)
        for item in corners:
            x, y = item[0]
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            # detect_lines(binary[x - 10:x + 11, y - 10:y + 11])
            m = thinned[x - 10:x + 11, y - 10:y + 11].copy()
            print(x, y, m.shape)
            if m.shape[0] == 0 or m.shape[1] == 0:
                continue
            detect_lines(m)
            # cv2.imshow("b", m)
            # cv2.waitKey(100)
            # if lines is not None:
            # for rho, theta in lines[0]:
            #     a = np.cos(theta)
            #     b = np.sin(theta)
            #     x0 = a * rho
            #     y0 = b * rho
            #     x1 = int(x0 + 1000 * (-b))
            #     y1 = int(y0 + 1000 * (a))
            #     x2 = int(x0 - 1000 * (-b))
            #     y2 = int(y0 - 1000 * (a))
            #     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # if is_saddle(binary[x - 10:x + 11, y - 10:y + 11]):
            #     cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("features", img)
        cv2.imshow("binary", thinned)
        if (cv2.waitKey(0) == 27):
            continue
