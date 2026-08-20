from dataclasses import dataclass

@dataclass

class ControlInput :

    accel       : float = 0.0
    steer_rate  : float = 0.0
    