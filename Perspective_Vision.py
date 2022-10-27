import cv2
import numpy as np
import math
import time

class Camera:
    def __init__(self, focal_distance, image_size, center, camera_position, camera_rotation_matrices):
        self.focal = focal_distance
        self.image_size = image_size
        self.center = center
        self.camera_pos = camera_position
        self.camera_rotations = camera_rotation_matrices
        self.camera_rotation_matrix = camera_rotation_matrices[0] @ camera_rotation_matrices[1] @ \
                                      camera_rotation_matrices[2]
        self.intrinsic_matrix = np.array([[self.focal, 0, self.center[0]],
                                          [0, self.focal, self.center[1]],
                                          [0, 0, 1]])
        self.extrinsic_matrix = np.concatenate([self.camera_rotation_matrix, self.camera_pos], axis=1)

    def render(self, global_points):
        global_points = [np.matmul(self.extrinsic_matrix, point) for point in global_points]
        global_points = [point1 / point1[2] for point1 in global_points]
        global_points = [np.matmul(self.intrinsic_matrix, point2) for point2 in global_points]
        return global_points

    def simulate(self, global_points, angle_x, angle_y, angle_z, x, y, z):
        intrinsic_matrix = self.intrinsic_matrix
        t = np.array([[x, y, z]]).T
        x_rotation = np.array([[1, 0, 0],
                               [0, math.cos(math.radians(angle_x)), -math.sin(math.radians(angle_x))],
                               [0, math.sin(math.radians(angle_x)), math.cos(math.radians(angle_x))]])
        y_rotation = np.array([[math.cos(math.radians(angle_y)), 0, math.sin(math.radians(angle_y))],
                               [0, 1, 0],
                               [-math.sin(math.radians(angle_y)), 0, math.cos(math.radians(angle_y))]])
        z_rotation = np.array([[math.cos(math.radians(angle_z)), -math.sin(math.radians(angle_z)), 0],
                               [math.sin(math.radians(angle_z)), math.cos(math.radians(angle_z)), 0],
                               [0, 0, 1]])
        rotation_matrix = np.matmul(np.matmul(x_rotation, y_rotation), z_rotation)
        extrinsic_matrix = np.concatenate([rotation_matrix, t], axis=1)
        pts = [np.matmul(extrinsic_matrix, point) for point in global_points]
        pts = [point1 / point1[2] for point1 in pts]
        pts = [np.matmul(intrinsic_matrix, point2) for point2 in pts]
        image = np.zeros(self.image_size)
        Cube.draw(image, pts)
        return pts, image, extrinsic_matrix


class CameraGUI:
    frames = dict()

    def __init__(self, camera_object, gui_name):
        self.gui_name = gui_name
        self.camera = camera_object
        self.x = 1
        self.y = 1
        self.z = 1
        self.x_angle_switch = 1
        self.y_angle_switch = 1
        self.z_angle_switch = 1
        cv2.namedWindow(gui_name)
        cv2.createTrackbar("Camera X: " + gui_name, gui_name, 0, 100, self.update_camera_x)
        cv2.createTrackbar("Camera Y: " + gui_name, gui_name, 0, 100, self.update_camera_y)
        cv2.createTrackbar("Camera Z: " + gui_name, gui_name, 0, 100, self.update_camera_z)
        cv2.createTrackbar("Camera X_angle: " + gui_name, gui_name, 0, 89, self.update_camera_x_angle)
        cv2.createTrackbar("Camera Y_angle: " + gui_name, gui_name, 0, 89, self.update_camera_y_angle)
        cv2.createTrackbar("Camera Z_angle: " + gui_name, gui_name, 0, 89, self.update_camera_z_angle)
        cv2.createTrackbar("Camera X negative: " + gui_name, gui_name, 0, 1, self.update_camera_x_negative)
        cv2.createTrackbar("Camera Y negative: " + gui_name, gui_name, 0, 1, self.update_camera_y_negative)
        cv2.createTrackbar("Camera Z negative", gui_name, 0, 1, self.update_camera_z_negative)
        cv2.createTrackbar("Camera X_angle negative: " + gui_name, gui_name, 0, 1, self.update_camera_x_angle_negative)
        cv2.createTrackbar("Camera Y_angle negative: " + gui_name, gui_name, 0, 1, self.update_camera_y_angle_negative)
        cv2.createTrackbar("Camera Z_angle negative: " + gui_name, gui_name, 0, 1, self.update_camera_z_angle_negative)

    def update_extrinsic_matrix(self):
        self.camera.extrinsic_matrix = np.concatenate([self.camera.camera_rotation_matrix, self.camera.camera_pos],
                                                      axis=1)

    def update_camera_x(self, val):
        self.camera.camera_pos[0][0] = self.x * val
        self.update_extrinsic_matrix()

    def update_camera_y(self, val):
        self.camera.camera_pos[1][0] = self.y * val
        self.update_extrinsic_matrix()

    def update_camera_z(self, val):
        self.camera.camera_pos[2][0] = self.z * val
        self.update_extrinsic_matrix()

    def update_rotation_matrix(self):
        self.camera.camera_rotation_matrix = np.matmul(
            np.matmul(self.camera.camera_rotations[0], self.camera.camera_rotations[1]),
            self.camera.camera_rotations[2])
        self.update_extrinsic_matrix()

    def update_camera_x_angle(self, val):
        radians = math.radians(self.x_angle_switch * val)
        self.camera.camera_rotations[0] = np.array([[1, 0, 0],
                                                    [0, math.cos(radians), -math.sin(radians)],
                                                    [0, math.sin(radians), math.cos(radians)]])
        self.update_rotation_matrix()

    def update_camera_y_angle(self, val):
        radians = math.radians(self.y_angle_switch * val)
        self.camera.camera_rotations[1] = np.array([[math.cos(radians), 0, math.sin(radians)],
                                                    [0, 1, 0],
                                                    [-math.sin(radians), 0, math.cos(radians)]])
        self.update_rotation_matrix()

    def update_camera_z_angle(self, val):
        radians = math.radians(self.z_angle_switch * val)
        self.camera.camera_rotations[2] = np.array([[math.cos(radians), -math.sin(radians), 0],
                                                    [math.sin(radians), math.cos(radians), 0],
                                                    [0, 0, 1]])
        self.update_rotation_matrix()

    @staticmethod
    def switch(val):
        if val:
            return 1
        return -1

    def update_camera_x_negative(self, val):
        self.x = self.switch(val)

    def update_camera_y_negative(self, val):
        self.y = self.switch(val)

    def update_camera_z_negative(self, val):
        self.z = self.switch(val)

    def update_camera_x_angle_negative(self, val):
        self.x_angle_switch = self.switch(val)

    def update_camera_y_angle_negative(self, val):
        self.y_angle_switch = self.switch(val)

    def update_camera_z_angle_negative(self, val):
        self.z_angle_switch = self.switch(val)

    @staticmethod
    def draw(img_size, input_points, *guis):
        while True:
            for gui_obj in guis:
                image = np.zeros(img_size)
                transformed_points = gui_obj.camera.render(input_points)
                Cube.draw(image, transformed_points)
                CameraGUI.frames[gui_obj.gui_name] = image
                cv2.imshow(gui_obj.gui_name, image)
            k = cv2.waitKey(1)
            if k == 27:
                break

        cv2.destroyAllWindows()

    @staticmethod
    def visual_odometry_with_homography(img_size, input_points, gui):
        original = convertPointToPlanar(input_points)
        vo = VisualOdometry(gui.camera, original)
        start = time.time()
        while True:
            image = np.zeros(img_size)
            transformed_points = gui.camera.render(input_points)
            transformed_points = np.array([[[i[0][0]], [i[1][0]], [1]] for i in transformed_points])
            t, r, _ = vo.compute_pose(transformed_points)
            if time.time() - start > 2:
                print("Homography: " +"\n" + str(vo.H))
                print("Rotation: " + "\n" + str(r))
                print("Translation:" + "\n" + str(t.reshape(1, 3)))
                start = time.time()
            t = np.round(t, 2)
            r = np.round(r, 2)
            cv2.putText(image, "T: " + str(t), (220, 20), cv2.FONT_HERSHEY_PLAIN, 1, 255)
            cv2.putText(image, "R: " + str(r[0]), (220, 50), cv2.FONT_HERSHEY_PLAIN, 1, 255)
            cv2.putText(image, str(r[1]), (240, 70), cv2.FONT_HERSHEY_PLAIN, 1, 255)
            cv2.putText(image, str(r[2]), (240, 90), cv2.FONT_HERSHEY_PLAIN, 1, 255)
            Cube.draw_square(image, transformed_points)
            CameraGUI.frames[gui.gui_name] = image
            cv2.imshow(gui.gui_name, image)
            k = cv2.waitKey(1)
            if k == 27:
                break

        cv2.destroyAllWindows()


class Cube:
    @staticmethod
    def generate_points(starting_point, length):
        point_list = []
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    point_list.append(np.array([[starting_point[0] + length * a], [starting_point[1] + length * b],
                                                [starting_point[2] + length * c], [1]]))
        return point_list

    @staticmethod
    def draw(image, point_list):
        for point in point_list:
            if point[0][0] is not float('nan') and point[1][0] is not float('nan'):
                cv2.circle(image, (int(point[0][0]), int(point[1][0])), 2, 255)

        cv2.line(image, (int(point_list[0][0][0]), int(point_list[0][1][0])),
                 (int(point_list[1][0][0]), int(point_list[1][1][0])), 255)
        cv2.line(image, (int(point_list[0][0][0]), int(point_list[0][1][0])),
                 (int(point_list[2][0][0]), int(point_list[2][1][0])), 255)
        cv2.line(image, (int(point_list[0][0][0]), int(point_list[0][1][0])),
                 (int(point_list[4][0][0]), int(point_list[4][1][0])), 255)
        cv2.line(image, (int(point_list[3][0][0]), int(point_list[3][1][0])),
                 (int(point_list[1][0][0]), int(point_list[1][1][0])), 255)
        cv2.line(image, (int(point_list[3][0][0]), int(point_list[3][1][0])),
                 (int(point_list[2][0][0]), int(point_list[2][1][0])), 255)
        cv2.line(image, (int(point_list[3][0][0]), int(point_list[3][1][0])),
                 (int(point_list[7][0][0]), int(point_list[7][1][0])), 255)
        cv2.line(image, (int(point_list[5][0][0]), int(point_list[5][1][0])),
                 (int(point_list[4][0][0]), int(point_list[4][1][0])), 255)
        cv2.line(image, (int(point_list[5][0][0]), int(point_list[5][1][0])),
                 (int(point_list[1][0][0]), int(point_list[1][1][0])), 255)
        cv2.line(image, (int(point_list[5][0][0]), int(point_list[5][1][0])),
                 (int(point_list[7][0][0]), int(point_list[7][1][0])), 255)
        cv2.line(image, (int(point_list[6][0][0]), int(point_list[6][1][0])),
                 (int(point_list[2][0][0]), int(point_list[2][1][0])), 255)
        cv2.line(image, (int(point_list[6][0][0]), int(point_list[6][1][0])),
                 (int(point_list[7][0][0]), int(point_list[7][1][0])), 255)
        cv2.line(image, (int(point_list[6][0][0]), int(point_list[6][1][0])),
                 (int(point_list[4][0][0]), int(point_list[4][1][0])), 255)

    @staticmethod
    def draw_square(image, point_list):
        for point in point_list:
            if point[0][0] is not float('nan') and point[1][0] is not float('nan'):
                cv2.circle(image, (int(point[0][0]), int(point[1][0])), 2, 255)

        cv2.line(image, (int(point_list[0][0][0]), int(point_list[0][1][0])),
                 (int(point_list[1][0][0]), int(point_list[1][1][0])), 255)
        cv2.line(image, (int(point_list[0][0][0]), int(point_list[0][1][0])),
                 (int(point_list[2][0][0]), int(point_list[2][1][0])), 255)
        cv2.line(image, (int(point_list[1][0][0]), int(point_list[1][1][0])),
                 (int(point_list[3][0][0]), int(point_list[3][1][0])), 255)
        cv2.line(image, (int(point_list[2][0][0]), int(point_list[2][1][0])),
                 (int(point_list[3][0][0]), int(point_list[3][1][0])), 255)


class VisualOdometry:
    def __init__(self, cam, pts1):
        self.K = cam.intrinsic_matrix
        self.camera = cam
        self.p1 = pts1
        self.p2 = None
        self.H = None

    def compute_pose(self, pts2):
        self.p2 = pts2
        self.H_prev = self.H
        self.H, _ = cv2.findHomography(self.p1, self.p2)
        try:
            t, r = self.decomposeHomography()
            return t, r, rotationMatrixToEulerAngles(r)
        except:
            return None, None, None

    def decomposeHomography(self):
        H = self.H.T
        h1 = H[0]
        h2 = H[1]
        h3 = H[2]
        K_inv = np.linalg.inv(self.camera.intrinsic_matrix)
        L = 1 / np.linalg.norm(np.dot(K_inv, h1))
        r1 = L * np.dot(K_inv, h1)
        r2 = L * np.dot(K_inv, h2)
        r3 = np.cross(r1, r2)
        T = L * (K_inv @ h3.reshape(3, 1))
        R = np.array([[r1], [r2], [r3]])
        R = np.reshape(R, (3, 3))
        U, S, V = np.linalg.svd(R, full_matrices=True)
        R = U @ V
        return T, R


def convertPointToPlanar(pt_list):
    return np.array([[[pt[0][0]], [pt[1][0]], [1]] for pt in pt_list])


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])


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
CameraGUI.draw((480, 640), points, gui)


'''
camera_pos = np.array([[0, 0, -40]]).T
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
planar_points = np.array([points[0], points[2], points[4], points[6]])
camera = Camera(f, size, (cx, cy), camera_pos, [rotation_x, rotation_y, rotation_z])
camera.intrinsic_matrix = np.array([[160, 0, 159.5],
                                    [0, 120, 119.5],
                                    [0, 0, 1]])
vo = VisualOdometry(camera, planar_points)
vo.H = np.array([[6.6016, -2.45742, 217.475],
                 [0.606495, 3.06217, 147.038],
                 [0.00224692, -0.00685587, 1]])
t, r = vo.decomposeHomography()
print(t)
print(rotationMatrixToEulerAngles(r))
R = np.array([[0.986389, 0.159555, 0.0397377],
              [-0.14876, 0.968901, -0.197743],
              [-0.0700528, 0.18914, 0.979448]])
'''


'''
R = np.array([[1.0000000, 0.0000000, 0.0000000],
              [0.0000000, 0.9396926, -0.3420202],
              [0.0000000, 0.3420202, 0.9396926]])
u, s, v = np.linalg.svd(E_theoretical)
t = u[:, 2]
Y = np.array([[0, -1, 0],
              [1, 0, 0],
              [0, 0, 1]])
R1 = -1 * u.dot(Y).dot(v)
R2 = -1 * u.dot(Y.T).dot(v)
print(R1)
print(R2)
'''
