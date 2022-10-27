import cv2
import numpy as np
import math
from Perspective_Vision import Camera, VisualOdometry
import time

cap = cv2.VideoCapture("/Users/abhinavperi/PycharmProjects/OpenCV/OpenCV Pics/Nanna Phone Test.mov")
'''
intrinsic = [[976.77820184, 0., 604.02815296],
             [0., 977.05528384, 412.1739181],
             [0., 0., 1.]]

intrinsic = [[477.70629631, 0., 305.66414982],
             [0., 477.83388597, 183.12365019],
             [0., 0., 1.]]
'''
intrinsic = [[1.85789978e+03, 0.00000000e+00, 9.75344296e+02],
             [0.00000000e+00, 1.80208411e+03, 5.72277486e+02],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
I = np.eye(3)
rots = np.array([I, I, I])
camera = Camera(477.70629631, (640, 360), (305.66414982, 83.12365019), np.array([[0], [0], [0]]), rots)
camera.intrinsic_matrix = intrinsic
lk_params = dict(winSize=(15, 15),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)


def detectKeyPoints(image):
    image = cv2.blur(image, (7, 7))
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame, (50, 0, 190), (100, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (21, 21))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not len(contours) == 0:
        selected = contours[0]
        for cnt in contours:
            if cv2.contourArea(cv2.convexHull(cnt)) > cv2.contourArea(cv2.convexHull(selected)):
                selected = cnt
        selected = np.reshape(selected, (-1, 2))
        topLeft = selected[0]
        topRight = selected[0]
        bottomLeft = selected[0]
        bottomRight = selected[0]
        for corner in selected:
            x = corner[0]
            y = corner[1]
            if x + y < topLeft[0] + topLeft[1]:
                topLeft = corner
            if x + y > bottomRight[0] + bottomRight[1]:
                bottomRight = corner
            if x - y < bottomLeft[0] - bottomLeft[1]:
                bottomLeft = corner
            if x - y > topRight[0] - topRight[1]:
                topRight = corner
        return np.array([topLeft, bottomLeft, bottomRight, topRight]), True
    return [], False


recent_angles = []
recent_t = []
colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 255, 255)]
pts1 = np.array([[0, 0], [157, 272], [471, 272], [628, 0]])
vo = VisualOdometry(camera, pts1)
_, prev = cap.read()
prev = cv2.resize(prev, (640, 360))

ref_point, status = detectKeyPoints(prev)
ref_point = np.float32(ref_point)
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
traj = np.zeros((200, 200))

while True:
    time.sleep(0.01)
    _, img = cap.read()
    img = cv2.resize(img, (640, 360))
    pts, status = detectKeyPoints(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if status:
        if ref_point is None:
            ref_point = np.float32(pts)
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, gray, ref_point, None, **lk_params)
        p1 = cv2.cornerSubPix(gray, p1, (5, 5), (-1, -1), criteria)
        points = p1
        error = sum(
            [math.sqrt(((pt_1[0] - pt_0[0]) ** 2 + (pt_1[1] - pt_0[1]) ** 2)) for pt_0, pt_1 in zip(pts, p1)])
        if error > 20:
            print("hi")
            p1 = np.float32(pts)

        for i in range(len(p1)):
            cv2.circle(img, (p1[i][0], p1[i][1]), 2, colors[1])
        ref_point = p1
        new_pts = np.array([[[[p1[i][0]], [p1[i][1]], [1]] for i in range(p1.shape[0])]]).reshape((-1, 3, 1))
        T, _, euler = vo.compute_pose(new_pts)
        if T is not None or euler is not None:
            # print("Translation: " + str(np.round(T.T, 3)) + " Rotation " + str(np.round(euler, 3)))
            recent_t.append(T)
            recent_angles.append(euler)
            if len(recent_t) >= 3:
                T = np.array(recent_t).mean(axis=0)
                euler = np.array(recent_angles).mean(axis=0)
                del recent_t[0]
                del recent_angles[0]
            cv2.putText(img, str(np.round(T, 3)), (300, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
            cv2.putText(img, str(np.round(euler, 3)), (300, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
            cv2.circle(traj, (int(T[0]) + 100, int(T[2])), 1, 255)

    prev = gray.copy()
    # draw = cv2.resize(img, (640, 360))
    cv2.imshow("Frame", img)
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()
