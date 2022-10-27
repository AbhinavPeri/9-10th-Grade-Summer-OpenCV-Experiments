import math
import numpy as np


class EulerAngles:
    def __init__(self, rotations):
        if type(rotations) is tuple:
            self.roll = round(rotations[0], 3)
            self.pitch = round(rotations[1], 3)
            self.yaw = round(rotations[2], 3)
        elif type(rotations) is np.ndarray:
            self.pitch = round(math.degrees(-math.asin(rotations[2][0])), 3)
            self.yaw = round(math.degrees(math.atan2(rotations[2][1], rotations[2][2])), 3) if rotations[2][2] != 0 else math.inf
            self.roll = round(math.degrees(math.atan2(rotations[1][0] / self.pitch, rotations[0][0] / self.pitch)),
                              3) if self.pitch != 0 and rotations[0][0] != 0 else math.inf

        self.capAngles()

    def __add__(self, other):
        return EulerAngles((self.roll + other.roll, self.pitch + other.pitch, self.yaw + other.yaw))

    def __iadd__(self, other):
        return EulerAngles((self.roll + other.roll, self.pitch + other.pitch, self.yaw + other.yaw))

    def __mul__(self, other):
        return EulerAngles((self.roll *other, self.pitch * other, self.yaw * other))

    def __imul__(self, other):
        return EulerAngles((self.roll *other, self.pitch * other, self.yaw * other))

    def __repr__(self):
        return "Roll: " + (str(self.roll) if math.isfinite(self.roll) else "NaN") + "\nPitch: " + str(
            self.pitch) + "\nYaw: " + (str(self.yaw) if math.isfinite(self.yaw) else "NaN") + "\n----------------------"

    def capAngles(self):
        if self.roll > 180:
            self.roll -= 360
        if self.roll < -180:
            self.roll += 360

        if self.pitch > 180:
            self.pitch -= 360
        if self.pitch < -180:
            self.pitch += 360

        if self.yaw > 180:
            self.yaw -= 360
        if self.yaw < -180:
            self.yaw += 360

