import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
directory_name='./datasets/Image'
for filename in os.listdir(directory_name):
    img = cv2.imread(directory_name + "/" + filename)
    blur_img = cv2.medianBlur(img,5)  # cv2.GaussianBlur(img,(5,5),0)#圖片降噪
    ret, blur_th4 = cv2.threshold(blur_img,127,255,cv2.THRESH_TOZERO)
    # img = cv2.imread('./datasets/Private01/', 0)
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV', 'TOZERO_Blur']
    images = [img, th1, th2, th3, th4, th5,blur_th4]

    # for i in range(6):
    #     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 0+1), plt.imshow(images[0], 'gray')
    plt.title(titles[3])
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 3+1), plt.imshow(images[3], 'gray')
    plt.title(titles[3])
    plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 3, 6+1), plt.imshow(images[6], 'gray')
    # plt.title(titles[3])
    # plt.xticks([]), plt.yticks([])
    # plt.imshow(images[1])
    # plt.show()
    print(filename)
    cv2.imwrite("./datasets/Private03/" + filename, images[3])