import numpy as np
import cv2
import time

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


cap = cv2.VideoCapture("/Users/abhinavperi/PycharmProjects/OpenCV/OpenCV Pics/Calibration Vid Nana's Phone.mov")
_, image = cap.read()
count = 0
divider = 0

while True:
    _, img = cap.read()
    try:
        divider += 1
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        result = gray
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6) ,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_ACCURACY)

        # If found, add object points, image points (after refining them)
        if ret == True and divider > 10:
            divider = 0
            #cv2.imwrite("Calibration Pics/Photo" + str(count) + ".jpg", img)
            count += 1
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(19,19),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
            result = img

        result = cv2.resize(result, (400, 400))
        cv2.imshow("Result", result)
    except Exception:
        pass
    key = cv2.waitKey(1)
    if key == 27:
        break

print(count)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
h,  w = img.shape[:2]
new_mtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


print(mtx)
print(dist)
print(new_mtx)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print(ret)
print("total error: ", mean_error/len(objpoints))
cv2.waitKey(0)
cv2.destroyAllWindows()