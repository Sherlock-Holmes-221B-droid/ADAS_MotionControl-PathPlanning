import numpy as np
import os
import sys 

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.geometry import *

class StanleyController:
    def __init__(self, wheelbase, k=1.0):
        self.wheelbase = wheelbase
        self.k = k
        self.old_idx = 0  

    def compute_steering(self, vehicle_state, path):  
        fx                = vehicle_state.x + self.wheelbase * np.cos(vehicle_state.yaw)
        fy                = vehicle_state.y + self.wheelbase * np.sin(vehicle_state.yaw)         
        search_range      = slice(self.old_idx, min(self.old_idx + 100, len(path.x)))
        window_x          = path.x[search_range]
        window_y          = path.y[search_range]        
        dx_front          = window_x - fx
        dy_front          = window_y - fy
        local_idx         = np.argmin(np.hypot(dx_front, dy_front))
        target_idx        = self.old_idx + local_idx
        self.old_idx      = target_idx 

        if path.yaw is None:
            path.compute_yaw()
            
        path_yaw         = path.yaw[target_idx]
        vehicle_yaw      = vehicle_state.yaw
        heading_error    = wrap_to_pi(path_yaw - vehicle_yaw)        
        tx               = path.x[target_idx]
        ty               = path.y[target_idx]
        dx               = tx - fx
        dy               = ty - fy        
        front_normal     = np.array([-np.sin(vehicle_state.yaw), np.cos(vehicle_state.yaw)])
        cte              = np.dot(np.array([dx, dy]), front_normal)    # Cross tracking error
        cte_term         = np.arctan2(self.k * cte, vehicle_state.v + 1e-06)
        steering         = (heading_error + cte_term)
        
        metrics = {
            "target_idx": target_idx,
            "target_point": (tx, ty),
            "window_x": window_x,
            "window_y": window_y
        }
        
        return steering, metrics
