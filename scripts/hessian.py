#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import cv2
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from matplotlib import pyplot as plt


def draw_circle(S, s_min):
    corners = []
    for i in range(S.shape[1]):
        for j in range(S.shape[0]):
            if S[j, i] < s_min:
                corners.append((j, i))
    return corners


if __name__ == '__main__':
    p_circle = np.vectorize(draw_circle)
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        rx = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
        ry = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
        rxx = cv2.Scharr(rx, cv2.CV_32F, 1, 0)
        rxy = cv2.Scharr(rx, cv2.CV_32F, 0, 1)
        ryx = cv2.Scharr(ry, cv2.CV_32F, 1, 0)
        ryy = cv2.Scharr(ry, cv2.CV_32F, 0, 1)
        A = rxx + ryy
        B = np.sqrt(np.power((rxx - ryy), 2) + 4 * rxy * rxy)
        lambda1 = A + B
        # # lambda1 = 0.5 * lambda1
        lambda2 = A - B
        # # lambda2 = 0.5 * lambda2
        #
        # # lambda_max = np.max(lambda1)
        # # epsilon = 0.3 * lambda_max
        #
        S = lambda1 * lambda2
        s_min = 0.5 * np.min(S)
        X = S < s_min
        for i in np.arange(S.shape[1]):
            for j in np.arange(S.shape[0]):
                if X[j, i]:
                    cv2.circle(img, (i, j), 3, (0, 0, 255), -1)

        cv2.imshow("img", img)
        cv2.waitKey(1)

    # plt.subplot(2, 2, 1)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(2, 2, 2)
    # plt.imshow(S, cmap='hot')
    # plt.subplot(2, 2, 3)
    # plt.imshow(lambda1, cmap='gray')
    # plt.subplot(2, 2, 4)
    # plt.imshow(lambda2, cmap='gray')
        # plt.show()
