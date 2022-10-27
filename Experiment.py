import cv2
import numpy as np

if __name__ == '__main__':
    Y = -1000
    f_x = 10
    f_y = 10
    c_x = 320
    c_y = 0
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    x = np.arange(640)
    y = np.arange(480)

    grid = np.concatenate([np.array(np.meshgrid(x, y)), np.ones((1, 480, 640))], axis=0)
    normalized = K_inv.dot(grid.reshape((3, -1))).reshape((3, 480, 640))
    reconstructed = normalized * Y/normalized[1]
    invalid = normalized[1] <= 0
    depth = np.sqrt(np.sum(reconstructed**2, axis=0))
    depth[invalid] = 0
    depth = 255 * (depth - np.min(depth))/(np.max(depth) - np.min(depth))
    cv2.imshow("Depth", depth)
    while True:
        if cv2.waitKey(1) == '27':
            break
    cv2.destroyAllWindows()


