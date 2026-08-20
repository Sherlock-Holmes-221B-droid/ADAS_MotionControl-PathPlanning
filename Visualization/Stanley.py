import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
from VehiclePlotter import VehiclePlotter

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Vehicle.Vehicle_config import VehicleConfig
from Vehicle.vehicle_state import VehicleState
from Vehicle.control_input import ControlInput
from Vehicle.vehicle_state import *
from Vehicle.vehicle import *
from utils.geometry import *
from utils.reference_path import *
from utils.path_generation import *
from Controllers.pid_speed_controller import *
from Controllers.Stanley_controller import StanleyController

cfg = VehicleConfig()
state = VehicleState()
state.x, state.y, state.yaw, state.v, state.delta = 0.0, 0.0, 0.0, 0.0, 0.0  

vehicle_run = Vehicle(cfg, state)
plotter = VehiclePlotter(cfg)

history_snapshots = []
state_snapshots = []
target_points = []
search_windows_history = []
closest_points_history = []

# path = generate_spline_path()
path = generate_sine_path()

st = StanleyController(wheelbase=cfg.wheelbase, k=1.0) 
pid = PIDSpeedController(kp=1.0, ki=0.1, kd=1)

for frame_count in range(1000):
    accel = pid.compute_control(target_speed=5.55, current_speed=vehicle_run.state.v, dt=0.1)
    desired_steer, tracking_metrics = st.compute_steering(vehicle_run.state, path)
    current_idx = tracking_metrics["target_idx"]
    if current_idx >= len(path.x) - 15:
        break

    steer_error = (desired_steer - vehicle_run.state.delta)
    steer_rate = 4.5 * steer_error
    control = ControlInput(accel=accel, steer_rate=steer_rate)    
    vehicle_run.step(control, dt=0.1)
    target_points.append(tracking_metrics["target_point"])
    closest_points_history.append(tracking_metrics["target_point"])
    search_windows_history.append((tracking_metrics["window_x"], tracking_metrics["window_y"]))    
    history_snapshots.append(copy.deepcopy(vehicle_run.history))
    state_snapshots.append(copy.deepcopy(vehicle_run.state))

animation_data = {
    "plotter": plotter,
    "path": path,
    "search_windows": search_windows_history,
    "closest_points": closest_points_history,
    "target_points": target_points,
    "states": state_snapshots,
    "history": history_snapshots,
    "title": "Stanley Controller"
}

if __name__ == "__main__":

    fig, ax = plt.subplots(figsize=(10, 10))
    ani = FuncAnimation(
        fig,
        lambda frame: plotter.animate(
            frame,
            ax,
            path,
            search_windows_history,
            closest_points_history,
            target_points,
            state_snapshots,
            history_snapshots,
            title_name="Stanley Controller"
        ),
        frames=len(state_snapshots),
        interval=30
    )
    plt.show()
