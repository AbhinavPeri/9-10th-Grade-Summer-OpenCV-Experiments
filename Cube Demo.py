import numpy as np
from Perspective_Vision import Cube, CameraGUI, Camera

if __name__ == '__main__':
    camera_pos = np.array([[0, 0, -100]]).T
    rotation_x = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
    rotation_y = rotation_x
    rotation_z = rotation_x
    size = (400, 400)
    cx = 200
    cy = 200
    f = 60
    points = Cube.generate_points([0, 0, 0], 70)
    camera = Camera(f, size, (cx, cy), camera_pos, [rotation_x, rotation_y, rotation_z])
    gui = CameraGUI(camera, 'camera_gui')
    CameraGUI.draw((400, 1000), points, gui)