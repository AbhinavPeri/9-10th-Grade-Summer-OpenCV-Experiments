import cv2
import numpy as np
import time
from eulerAngles import EulerAngles
import math

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
camera_matrix = np.array([[477.70629631,   0.,         305.66414982],
                        [  0.,         477.83388597, 183.12365019],
                        [  0.,           0.,          1.        ]])

f = 2.123139094711111
pp = (1.3585073325333332, 0.34335684410625)
index_params = dict(algorithm=6, trees=4)
search_params = dict(checks=20)   # or pass empty dictionary




cap = cv2.VideoCapture(0)
fast = cv2.FastFeatureDetector.create(threshold=25, nonmaxSuppression=True)
_, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, (240, 135))
prev_frame = cv2.GaussianBlur(prev_frame, (3, 3), 5)
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
old_points = fast.detect(prev_frame)
old_points = np.array([x.pt for x in old_points], dtype=np.float32)
t_global = np.zeros((3, 1))
R_global = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
identity = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
R_global_euler = EulerAngles((0, 0, 0))
while True:
    startTime = time.time_ns()
    _, frame = cap.read()
    frame = cv2.resize(frame, (240, 135))
    frame = cv2.GaussianBlur(frame, (3, 3), 5)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, frame)
    current_frame = frame.copy()
    if old_points.shape[0] < 2000:
        old_points = fast.detect(prev_frame)
        old_points = np.array([x.pt for x in old_points], dtype=np.float32)
    new_points, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, old_points, None, **lk_params)
    st = st.reshape(st.shape[0])
    old_points = old_points[st == 1]
    new_points = new_points[st == 1]
    #E, _ = cv2.findEssentialMat(new_points, old_points, focal=1, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    E, _ = cv2.findEssentialMat(old_points, new_points, cameraMatrix=camera_matrix, method=cv2.RANSAC, prob=0.9, threshold=1.0)
    print(type(E))
    R1, R2, t = cv2.decomposeEssentialMat(E)
    R1_euler = EulerAngles(R1)
    R2_euler = EulerAngles(R2)
    if abs(abs(R1_euler.pitch) + abs(R1_euler.yaw)) > abs(abs(R2_euler.pitch) + abs(R2_euler.yaw)):
        R = R2
    else:
        R = R1

    t_global = t_global + R.dot(t)
    R_global = R.dot(R_global)
    R_euler = EulerAngles(R)
    R_global_euler = EulerAngles(R_global)
    cv2.imshow("Frame", prev_frame)
    prev_frame = current_frame.copy()
    old_points = new_points.copy()
    key = cv2.waitKey(1)
    if key == 27:
        break
    endTime = str(1/((time.time_ns() - startTime)/1000000000))
    #print(str(t) + " " + str(endTime))
    #print("R1: \n" + str(R1_euler) + "\nR2: \n" + str(R2_euler) + "\nR: \n" + str(R_euler) + "\nGlobal: \n" + str(R_global_euler) + "\nTime: " + endTime + "\n")



