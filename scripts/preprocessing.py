#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import cv2
import numpy as np
import thinning
from matplotlib import pyplot as plt


def preprocessing(img):
    gray_img = img.copy()

    horizontal_grad = cv2.Scharr(gray_img, cv2.CV_64F, 1, 0)
    vertical_grad = cv2.Scharr(gray_img, cv2.CV_64F, 0, 1)  # , ksize=3)
    gradient = np.sqrt(np.power(horizontal_grad, 2) +
                       np.power(vertical_grad, 2))

    print(np.max(gradient))
    # binary = cv2.threshold(gradient,

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(horizontal_grad, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(vertical_grad, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(gradient, cmap='gray')
    plt.show()


def whether_true(matrix):
    counter = 0
    center = matrix[matrix.shape[0] / 2 + 1, matrix.shape[1] / 2 + 1]
    for element in matrix:
        if element > center:
            counter = counter + 1

    if counter / matrix.size < 0.6:
        return True
    else:
        return False


def local_thresholding(img):
    binary = np.zeros(img.shape, np.uint8)
    tau = 4
    for i in xrange(tau, img.shape[0]):
        for j in xrange(tau, img.shape[1]):
            thresh = whether_true(img[i - tau:i + tau + 1, j - tau:j + tau + 1])
            binary[i, j] = thresh


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


if __name__ == '__main__':
    img = cv2.imread("/home/fans/Pictures/caltag.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # reval, corners = cv2.findChessboardCorners(gray, (8, 10), None)
    # print(corners.shape)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    # objp = np.zeros((8 * 10, 3), np.float32)
    # objp[:, :2] = np.mgrid[0:8, 0:10].T.reshape(-1, 2)

    # objpoints = []
    # imgpoints = []

    # if reval == True:
    #     # objpoints.append(objp)
    #     corners = cv2.cornerSubPix(gray, corners, (21, 21), (-1, -1), criteria)
    #     # imgpoints.append(corners2)
    #
    #     # img = cv2.drawChessboardCorners(img, (8, 10), corners, reval)
    #     print(corners.shape)
    #     for i in xrange(80):
    #         # print(corners[i, 0])
    #         cv2.circle(img, tuple(corners[i, 0]), 3, (0, 0, 255), -1)
    #     print(img.shape)
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)
    preprocessing(gray)
