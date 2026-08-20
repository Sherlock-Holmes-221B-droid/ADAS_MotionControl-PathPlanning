import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from VehiclePlotter import VehiclePlotter

from Vehicle.Vehicle_config import VehicleConfig
from Vehicle.vehicle_state import VehicleState
from Vehicle.vehicle import Vehicle
from Vehicle.control_input import ControlInput
from Controllers.pid_speed_controller import PIDSpeedController
from Controllers.LQR_controller import LQRController
from Controllers.MPC_controller import MPCController
from utils.path_generation import *


cfg = VehicleConfig()

state = VehicleState(
    x=0.0,
    y=0.0,
    yaw=0.0,
    v=0.0,
    delta=0.0
)
vehicle_run                     = Vehicle(cfg, state)
plotter                         = VehiclePlotter(cfg)
Lqr                             = LQRController(wheelbase=cfg.wheelbase,dt=0.1)
mpc                             = MPCController(wheelbase = cfg.wheelbase,horizon = 30, dt = 0.1)
pid                             = PIDSpeedController(kp=1.0,ki=0.1,kd=1.0)
# path                            = generate_spline_path()
path                            = generate_sine_path()
path.compute_yaw()
path.compute_curvature()

history_snapshots = []
state_snapshots = []
target_points = []
closest_points_history = []
search_windows_history = []

for frame_count in range(1000):

    current_velocity           = max(0.1,vehicle_run.state.v)
    accel                      = pid.compute_control(target_speed=5.55,current_speed=current_velocity,dt=0.1)
    metrics                    = Lqr.get_target_index(vehicle_state=vehicle_run.state,path=path)
    x0                         = Lqr.X
    idx                        = metrics["target_idx"]

    if idx >= len(path.x) - 1:
        print("Path completed")
        break

    steer_rate = mpc.steer(wheelbase = cfg.wheelbase,curvature = metrics["curvature"], x0 = Lqr.X, velocity = vehicle_run.state.v, delta = vehicle_run.state.delta )
    control = ControlInput(accel=accel,steer_rate=steer_rate)
    vehicle_run.step(control,dt=0.1)
    px = path.x[idx]
    py = path.y[idx]
    target_points.append((px, py))
    closest_points_history.append((px, py))
    window_size = 50
    s_start = max(0, idx - window_size)
    s_end = min(len(path.x), idx + window_size)
    search_windows_history.append((path.x[s_start:s_end],path.y[s_start:s_end]))
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
    "title": "MPC Controller"
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
            title_name="MPC Controller"
        ),
        frames=len(state_snapshots),
        interval=30
    )
    plt.show()    