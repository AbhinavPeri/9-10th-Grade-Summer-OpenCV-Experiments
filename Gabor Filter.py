import cv2
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('macosx')

plt.ion()
plt.axis('off')
cv2.namedWindow("Convolution")
size = (31, 31)
sigma = 10
theta = 0
gamma = 0.01
lamd = 10
img = cv2.resize(cv2.cvtColor(cv2.imread("/Users/abhinavperi/PycharmProjects/OpenCV/OpenCV Pics/polka dots.jpeg"), cv2.COLOR_BGR2GRAY), (640, 480))
cv2.imshow("Img", img)


def update_size(val):
    global size
    size = (val, val)
    update()


def update_sigma(val):
    global sigma
    sigma = val
    update()


def update_theta(val):
    global theta
    theta = val/180 * np.pi
    update()


def update_gamma(val):
    global gamma
    gamma = val/100
    update()


def update_lambda(val):
    global lamd
    lamd = val
    update()


def update():
    g = cv2.getGaborKernel(size, sigma, theta, gamma, lamd)
    imshow(g)
    print('HI')
    #convolved_img = sig.convolve2d(img, g)
    #cv2.imshow("Convolution", convolved_img)


cv2.createTrackbar("Kernel Size", "Convolution", 3, 101, update_size)
cv2.createTrackbar("Sigma", "Convolution", 10, 200, update_sigma)
cv2.createTrackbar("Theta", "Convolution", 0, 360, update_theta)
cv2.createTrackbar("Gamma", "Convolution", 1, 100, update_gamma)
cv2.createTrackbar("Lambda", "Convolution", 10, 200, update_lambda)


def imshow(kernel, **kwargs):
    # utility function to show image
    plt.imshow(kernel, cmap=plt.gray(), **kwargs)
    plt.draw()
    plt.pause(0.1)


