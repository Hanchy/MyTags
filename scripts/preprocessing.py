#!/usr/bin/env python
# -*- code: utf8 -*-
import cv2
import numpy as np
import thinning
from matplotlib import pyplot as plt


def preprocessing(img):
    gray_img = img.copy()
    # cv2.cvtColor(img, gray_img, CV_RGB2GRAY)

    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobelxabs = sobelx.copy()
    cv2.convertScaleAbs(sobelx, sobelxabs)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    sobelyabs = sobely.copy()
    cv2.convertScaleAbs(sobely, sobelyabs)
    grad = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    # grad = cv2.addWeighted(sobelxabs, 0.5, sobelyabs, 0.5, 0)

    gray_img = grad.copy()
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    invert_zeros(grad)
    median_grad = cv2.medianBlur(np.float32(grad), ksize=3)
    thined_grad = thinning.guo_hall_thinning(np.uint8(median_grad))
    plt.subplot(1, 2, 2)
    plt.imshow(thined_grad, cmap='gray')
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


if __name__ == '__main__':
    img = cv2.imread("/home/fans/Pictures/caltag.jpg", 0)
    preprocessing(img)
