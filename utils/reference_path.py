from dataclasses import dataclass
import numpy as np

@dataclass
class ReferencePath:
    x   : np.ndarray
    y   : np.ndarray
    yaw : np.ndarray = None

    def get_points(self):
        return np.column_stack((self.x, self.y))

    def nearest_index(self,x_vehicle,y_vehicle):
        dx = self.x - x_vehicle
        dy = self.y - y_vehicle
        dist = np.hypot(dx,dy)
        return np.argmin(dist)

    def target_index(self,x_vehicle, y_vehicle, lookahead_distance):
        nearest     =  self.nearest_index(x_vehicle, y_vehicle)
        for i in range(nearest, len(self.x)):
            dx      =  self.x[i] - x_vehicle
            dy      =  self.y[i] - y_vehicle 
            dist    = np.hypot(dx,dy)
            if dist >= lookahead_distance:
                return i
        return len(self.x) - 1

    def compute_yaw(self):
        dx = np.gradient(self.x)
        dy = np.gradient(self.y)
        self.yaw = np.arctan2(dy,dx)
        return self.yaw

    def compute_curvature(self):

        dx = np.gradient(self.x)
        dy = np.gradient(self.y)

        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        denom = np.power(
            dx**2 + dy**2,
            1.5
        )

        denom = np.maximum(
            denom,
            1e-6
        )

        self.curvature = (
            dx * ddy -
            dy * ddx
        ) / denom

        return self.curvature
