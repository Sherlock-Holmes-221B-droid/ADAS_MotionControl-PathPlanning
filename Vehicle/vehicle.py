import numpy as np
import os
import sys 
from .Vehicle_config import VehicleConfig
from .vehicle_state import VehicleState
from .control_input import ControlInput

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from utils.geometry import *

class Vehicle :

    def __init__(self, config : VehicleConfig, initial_state : VehicleState):
        self.cfg = config
        self.state = initial_state
        self.history = { "x":[] , "y":[], "yaw":[], "v":[],"delta" : []}
        self.record_history()

    def get_state_vector(self):
        return np.array([self.state.x , self.state.y, self.state.yaw, self.state.v, self.state.delta])

    @property
    
    def pose(self):
        return (self.state.x, self.state.y, self.state.yaw)

    def record_history(self):
        self.history["x"].append(self.state.x)
        self.history["y"].append(self.state.y)
        self.history["yaw"].append(self.state.yaw)
        self.history["v"].append(self.state.v)
        self.history["delta"].append(self.state.delta)

    def step(self, control, dt):
        accel           =  np.clip(control.accel, self.cfg.max_decel, self.cfg.max_accel)
        steer_rate      =  np.clip(control.steer_rate,-self.cfg.max_steering_rate,self.cfg.max_steering_rate)  
        
        x               =  self.state.x
        y               =  self.state.y
        yaw             =  self.state.yaw
        v               =  self.state.v
        delta           =  self.state.delta
        
        # Bicycle Model

        x_dot           =  v * np.cos(yaw)
        y_dot           =  v * np.sin(yaw)
        yaw_dot         =  (v/self.cfg.wheelbase) * np.tan(delta)

        # Euler Integration

        x               +=  x_dot * dt
        y               +=  y_dot * dt
        yaw             +=  yaw_dot * dt
        v               +=  accel * dt
        delta           +=  steer_rate * dt

        yaw = wrap_to_pi(yaw)
        v = np.clip(v,0,self.cfg.max_speed)
        delta           =  np.clip(delta, -self.cfg.max_steer,self.cfg.max_steer )        

        self.state.x     =  x  
        self.state.y     =  y  
        self.state.yaw   =  yaw
        self.state.v     =  v  
        self.state.delta =  delta

        self.record_history()








