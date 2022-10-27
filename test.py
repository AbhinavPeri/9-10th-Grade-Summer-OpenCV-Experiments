import cv2
import imutils
from sklearn.decomposition import PCA
from scipy import signal as sig
from skimage.filters import gabor_kernel
import math
import numpy as np
import matplotlib.pyplot as plt

grass = cv2.resize(
    cv2.cvtColor(cv2.imread("/Users/abhinavperi/PycharmProjects/OpenCV/OpenCV Pics/grass.jpg"), cv2.COLOR_BGR2GRAY),
    (640, 480))
wall = cv2.resize(cv2.cvtColor(cv2.imread("/Users/abhinavperi/PycharmProjects/OpenCV/OpenCV Pics/brick wall.jpeg"),
                               cv2.COLOR_BGR2GRAY), (640, 480))
cat_face = cv2.resize(cv2.imread("/Users/abhinavperi/PycharmProjects/OpenCV/OpenCV Pics/cat_face_1.jpeg"), (640, 480))
dots = cv2.resize(cv2.imread("/Users/abhinavperi/PycharmProjects/OpenCV/OpenCV Pics/polka dots.jpeg"), (640, 480))
purple = cv2.resize(cv2.imread("/Users/abhinavperi/PycharmProjects/OpenCV/OpenCV Pics/purple.png"), (640, 480))
objects = cv2.resize(
    cv2.cvtColor(cv2.imread("/Users/abhinavperi/PycharmProjects/OpenCV/OpenCV Pics/objects.jpeg"), cv2.COLOR_BGR2GRAY),
    (640, 480))
img = cv2.cvtColor(cat_face, cv2.COLOR_BGR2GRAY)

feature_map = None
'''
for theta in range(8):
    theta = theta / 8 * np.pi
    for sigma in range(1, 2, 4):
        for lambd in range(5, 30, 24):
            for gamma in range(1, 10, 4):
                gamma /= 10
                print("sigma: " + str(sigma) + " theta: " + str(theta) + " lambda: " + str(lambd) + " gamma: " + str(gamma))
                kernel = np.real(cv2.getGaborKernel((11, 11), sigma, theta, lambd, gamma))
                conv = cv2.resize(sig.convolve2d(img, kernel), (640, 480))
                conv = cv2.morphologyEx(conv, cv2.MORPH_ERODE, (5, 5))
                if feature_map is None:
                    feature_map = np.reshape(conv, (1, -1))
                else:
                    feature_map = np.vstack((feature_map, conv.reshape((1, -1))))
pca = PCA(0.3)
pca.fit(feature_map)
print(pca.components_[0].shape)

mask = np.zeros(pca.components_[0].shape)
for comp in pca.components_:
    comp /= np.std(comp)
    comp *= -128/comp.min()
    mask = comp
    plt.imshow(comp.reshape(480, 640), cmap="gray")
    cv2.waitKey(0)
    plt.pause(2)
mask = np.float32(mask.reshape(480, 640))
plt.show()
cv2.imshow("Mask", mask)
cv2.waitKey(0)




for theta in range(8):
    theta = theta / 8 * np.pi
    for sigma in range(10, 11, 4):
        for lambd in range(5, 30, 10):
            for gamma in range(1, 10, 4):
                gamma /= 10
                print("sigma: " + str(sigma) + " theta: " + str(theta) + " lambda: " + str(lambd) + " gamma: " + str(gamma))
                kernel = np.real(cv2.getGaborKernel((21, 21), sigma, theta, lambd, gamma))
                conv = cv2.resize(sig.convolve2d(mask, kernel), (640, 480))
                conv = cv2.morphologyEx(conv, cv2.MORPH_ERODE, (5, 5))
                if feature_map is None:
                    feature_map = np.reshape(conv, (1, -1))
                else:
                    feature_map = np.vstack((feature_map, conv.reshape((1, -1))))
'''

freq = math.sqrt(2)
theta = 0
while freq < 160 and theta < 135:
    theta = 0
    for theta in (0, 180, 45):
        theta = math.radians(theta)
        kernel = np.real(gabor_kernel(freq, theta))
        convolved = sig.convolve2d(img, kernel)
        cv2.imshow("Convolved", convolved)
        cv2.waitKey(100)
        print("******\n")
    freq *= 2

'''
pca = PCA(0.7)
pca.fit(feature_map)
print(pca.components_[0].shape)

mask = np.zeros(pca.components_[0].shape)
for comp in pca.components_:
    comp /= np.std(comp)
    comp *= -128/comp.min()
    mask = comp
    mask = np.float32(mask.reshape(480, 640))
    cv2.imshow("Mask", mask)
    cv2.imshow("Image", cat_face)
    plt.imshow(comp.reshape(480, 640), cmap="gray")
    cv2.waitKey(0)
    plt.pause(2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


'''
feature_map = None
identity = np.zeros((7, 7))
identity[3, 3] = 1

filters = []
# prepare filter bank kernels
for theta in range(4):
    theta = theta / 4. * np.pi
    for frequency in range(1, 8):
        frequency *= 0.05
        kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=1, sigma_y=1))
        filters.append(kernel)
        conv = cv2.resize(sig.convolve2d(objects, kernel), (480, 640))
        if feature_map is None:
            feature_map = conv.reshape((-1, 1))
        else:
            feature_map = np.hstack((feature_map, conv.reshape((-1, 1))))


feature_map = np.hstack((feature_map, cv2.resize(sig.convolve2d(objects, identity), (480, 640)).reshape(-1, 1)))

feature_map = np.float32(feature_map)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 2
ret, label, center = cv2.kmeans(feature_map, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
for lab in range(K):
    output_mean = np.float32((label == lab).reshape((480, 640)) * 255)
    cv2.imshow("Center" + str(lab), output_mean)
cv2.imshow("Wall", objects)
cv2.waitKey(0)
'''

'''
def check_uniform_regions(img, thresh, min_size, bit_mask, mean):
    if img.shape[0] > img.shape[1]:
        slice_obj1 = slice(0, img.shape[0] // 2)
        slice_obj2 = slice(img.shape[0] // 2, img.shape[0])
        sub_img1 = img[slice_obj1]
        sub_img2 = img[slice_obj2]
        top = True
    else:
        slice_obj1 = slice(0, img.shape[1] // 2)
        slice_obj2 = slice(img.shape[1] // 2, img.shape[1])
        sub_img1 = img[:, slice_obj1]
        sub_img2 = img[:, slice_obj2]
        top = False

    for sub_img, num in zip((sub_img1, sub_img2), (1, 2)):
        if num == 1:
            slice_obj = slice_obj1
        else:
            slice_obj = slice_obj2
        std = np.std(sub_img, axis=(0, 1))
        if sub_img.size > min_size:
            if np.all(std < thresh):
                if np.median(np.abs(mean - np.mean(sub_img))) < 25:
                    if top:
                        bit_mask[slice_obj] *= 255
                    else:
                        bit_mask[:, slice_obj] *= 255
            else:
                if top:
                    check_uniform_regions(sub_img, thresh, min_size, bit_mask[slice_obj], mean)
                else:
                    check_uniform_regions(sub_img, thresh, min_size, bit_mask[:, slice_obj], mean)
    return np.uint8(bit_mask)


dots = cv2.medianBlur(dots, 11)
mask = np.ones((480, 640))
mask = check_uniform_regions(dots, np.array([10, 10, 10]), 5, mask, np.array([60, 60, 60]))
cv2.imshow("Mask", mask)
cv2.imshow("Dots", dots)
cv2.waitKey(0)
'''
