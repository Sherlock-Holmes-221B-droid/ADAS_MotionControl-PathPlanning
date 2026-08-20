import numpy as np
import os
import sys 

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from utils.geometry import *

class PurePursuit:
    def __init__(self, wheelbase, lookahead_gain=0.5, min_lookahead=2.0, max_lookahead=12.0):
        self.wheelbase      = wheelbase
        self.lookahead_gain = lookahead_gain  # L_k: scales look-ahead with speed
        self.min_lookahead  = min_lookahead   # L_min: baseline distance at standstill
        self.max_lookahead  = max_lookahead   # Upper safety ceiling limit
        self.old_idx        = 0  

    def compute_steering(self, vehicle_state, path):
        current_lookahead   = self.lookahead_gain * vehicle_state.v + self.min_lookahead
        current_lookahead   = np.clip(current_lookahead, self.min_lookahead, self.max_lookahead)

        search_range        = slice(self.old_idx, min(self.old_idx + 50, len(path.x)))
        window_x            = path.x[search_range]
        window_y            = path.y[search_range]

        dx                  = window_x - vehicle_state.x
        dy                  = window_y - vehicle_state.y
        closest_local_idx   = self.old_idx + np.argmin(np.hypot(dx, dy))
        
        target_idx = closest_local_idx
        while target_idx < len(path.x) - 1:
            dist = np.hypot(path.x[target_idx] - vehicle_state.x, path.y[target_idx] - vehicle_state.y)
            if dist >= current_lookahead:
                break
            target_idx += 1
            
        self.old_idx = closest_local_idx 
        tx             = path.x[target_idx]
        ty             = path.y[target_idx]
        target_heading = np.arctan2(ty - vehicle_state.y, tx - vehicle_state.x)
        alpha          = (target_heading - vehicle_state.yaw)
        alpha          = wrap_to_pi(alpha)
        steering       = np.arctan2(2 * self.wheelbase * np.sin(alpha), current_lookahead)

        metrics = {
            "closest_idx": closest_local_idx,
            "closest_point": (path.x[closest_local_idx], path.y[closest_local_idx]),
            "target_point": (tx, ty),
            "window_x": window_x,
            "window_y": window_y,
            "current_lookahead": current_lookahead 
        }

        return steering, metrics
