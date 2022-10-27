import cv2
import numpy as np
import math
import time

camera_matrix = np.array([[477.70629631,   0.,         305.66414982],
                        [  0.,         477.83388597, 183.12365019],
                        [  0.,           0.,          1.        ]])
cap = cv2.VideoCapture(0)
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher()
_, prev_frame = cap.read()
old_points, old_des = orb.detectAndCompute(prev_frame, None)
t_global = np.zeros((3, 1))
R_global = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
startTime = time.time_ns()
while True:
    _, frame = cap.read()
    current_frame = frame
    new_points, new_des = orb.detectAndCompute(current_frame, None)
    matches = bf.knnMatch(old_des, new_des, k=2)
    good = []
    list_old = []
    list_new = []

    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])
            list_old.append(old_points[m.queryIdx].pt)
            list_new.append(new_points[m.trainIdx].pt)
    list_old = np.int32(list_old)
    list_new = np.int32(list_new)
    E, _ = cv2.findEssentialMat(list_old, list_new, cameraMatrix=camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1)
    _, R, t, _ = cv2.recoverPose(E, list_old, list_new, camera_matrix)
    R = np.round(R, 3)
    t_global = t_global + R.dot(t)
    R_global = R.dot(R_global)
    matching_result = None
    matching_result = cv2.drawMatchesKnn(prev_frame, old_points, current_frame, new_points, good, matching_result, flags=2)
    cv2.imshow("Matches", matching_result)
    prev_frame = current_frame
    old_points = new_points
    old_des = new_des
    key = cv2.waitKey(1)
    if key == 27:
        break
    print(str(math.degrees(math.atan(R_global[2][1] / R_global[2][2]))) + " " + str((startTime - time.time_ns())/1000000000))
    startTime = time.time_ns()
