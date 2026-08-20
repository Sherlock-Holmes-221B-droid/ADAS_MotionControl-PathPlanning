import numpy as np
from scipy.linalg import solve_discrete_are
import os
import sys 

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.geometry import *


class LQRController:

    def __init__(self, wheelbase, dt=0.1):

        self.wheelbase          = wheelbase
        self.dt                 = dt
        self.old_idx            = 0
        self.X                  = np.zeros((4, 1))

    def get_target_index(self, vehicle_state, path):

        if path.yaw is None:
            path.compute_yaw()
        if not hasattr(path, "curvature"):
            path.compute_curvature()
            
        search_start = self.old_idx
        search_end = min(self.old_idx + 50,len(path.x))
        fx                      = vehicle_state.x
        fy                      = vehicle_state.y
        dx                      = path.x[search_start:search_end] - fx
        dy                      = path.y[search_start:search_end] - fy
        local_idx               = np.argmin(np.hypot(dx,dy))
        target_idx              = search_start + local_idx
        self.old_idx            = max(self.old_idx,target_idx)
        tx                      = path.x[target_idx]
        ty                      = path.y[target_idx]
        path_yaw                = path.yaw[target_idx]
        front_normal            = np.array([-np.sin(path_yaw),np.cos(path_yaw)])
        error_vec               = np.array([fx - tx,fy - ty])
        cte                     = np.dot(error_vec, front_normal)
        heading_error           = wrap_to_pi(vehicle_state.yaw - path_yaw)
        yaw_rate                = (vehicle_state.v/ self.wheelbase* np.tan(vehicle_state.delta))
        e_dot                   = vehicle_state.v * np.sin(heading_error)
        curvature               = path.curvature[target_idx]
        self.X                  = np.array([cte,e_dot,heading_error,yaw_rate]).reshape(4, 1)

        return {
            "cte": cte,
            "e_dot": e_dot,
            "heading_error": heading_error,
            "heading_dot": yaw_rate,
            "curvature": curvature,
            "target_idx": target_idx
        }
    
    def build_state_space(self,velocity,current_delta,curvature=0.0):
        v = max(1.0, velocity)

        A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, v,   0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]
        ])

        B = np.array([
            [0.0],
            [0.0],
            [0.0],
            [v / self.wheelbase]
        ])

        Q = np.diag([
            35.0,    # cte
            1.0,     # e_dot
            20.0,    # heading error
            1.0      # yaw rate
        ])

        R                         = np.array([[10.0]])
        Ad                        = np.eye(4) + A * self.dt
        Bd                        = B * self.dt
        P                         = solve_discrete_are(Ad,Bd,Q,R)
        K                         = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
        delta_fb                  = -float((K @ self.X).item())
        delta_ff                  = np.arctan(self.wheelbase * curvature)
        delta_cmd                 = (delta_ff + delta_fb)
        max_steer                 = np.deg2rad(35)
        delta_cmd                 = np.clip(delta_cmd, -max_steer, max_steer)
        tau                       = 0.5
        steer_rate                = (delta_cmd - current_delta) / tau
        steer_rate                = np.clip(steer_rate,-1.0, 1.0)
        return steer_rate, K