import numpy as np
from dataclasses import dataclass

@dataclass
class VehicleConfig:

    # Vehicle configurations

    wheelbase           : float = 2.8
    length              : float = 4.5
    width               : float = 1.8
    rear_overhang       : float = 0.8
    front_overhang      : float = 0.9

    # Wheel Geometry

    wheel_length        : float = 0.6
    wheel_width         : float = 0.25


    # Limits for our vehicle

    max_steer           : float = np.deg2rad(35.0)
    max_accel           : float = 3
    max_decel           : float = -6
    max_speed           : float = 50
    max_steering_rate   : float = np.deg2rad(120)

    @property
    def vehicle_length(self):
        return(self.front_overhang + self.wheelbase + self.rear_overhang)
    

    

