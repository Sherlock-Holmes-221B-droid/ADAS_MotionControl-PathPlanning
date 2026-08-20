import numpy as np
from .reference_path import *
from scipy.interpolate import CubicSpline

def generate_straight_path(
        length = 100,
        resolution = 0.5
):
    x = np.arange(0,length,resolution)
    y = np.zeros_like(x)

    return ReferencePath(x=x,y=y)

def generate_sine_path(length =100,amplitude = 5, frequency = 0.1):
    x = np.linspace(0,length,1000)
    y = amplitude*np.sin(frequency * x)
    return ReferencePath(x=x, y=y)

def generate_circle_path(radius = 20, n_points = 1000):
    theta = np.linspace(0,2*np.pi, n_points)
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    return ReferencePath(x=x,y=y)

def generate_figure_eight_path(size=50, n_points=1000):
    t = np.linspace(0, 2*np.pi, n_points)
    # Lemniscate of Bernoulli equations
    x = (size * np.cos(t)) / (1 + np.sin(t)**2)
    y = (size * np.sin(t) * np.cos(t)) / (1 + np.sin(t)**2)
    return ReferencePath(x=x, y=y)

def generate_racetrack_path(length=60, width=30, n_points=1000):
    r = width / 2.0
    perimeter = 2 * length + 2 * np.pi * r
    s = np.linspace(0, perimeter, n_points)
    
    x, y = np.zeros_like(s), np.zeros_like(s)
    for i, si in enumerate(s):
        if si < length:
            x[i], y[i] = si, -r
        elif si < length + np.pi * r:
            theta = -np.pi/2 + (si - length) / r
            x[i], y[i] = length + r * np.cos(theta), r * np.sin(theta)
        elif si < 2 * length + np.pi * r:
            x[i], y[i] = length - (si - length - np.pi * r), r
        else:
            theta = np.pi/2 + (si - 2 * length - np.pi * r) / r
            x[i], y[i] = r * np.cos(theta), r * np.sin(theta)
            
    return ReferencePath(x=x, y=y)


x_coords = [0.0,  20.0,  35.0,  50.0,  80.0, 100.0 , 60]
y_coords = [0.0,   5.0, -15.0,  10.0,  25.0,  15.0 , 0]
def generate_spline_path(waypoints_x = x_coords, waypoints_y = y_coords, n_points=1000):
    # Calculate cumulative distance between waypoints
    dx = np.diff(waypoints_x)
    dy = np.diff(waypoints_y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0], np.cumsum(ds)))
    
    # Fit cubic splines over distance
    cs_x = CubicSpline(s, waypoints_x)
    cs_y = CubicSpline(s, waypoints_y)
    
    # Interpolate fine points
    s_fine = np.linspace(0, s[-1], n_points)
    return ReferencePath(x=cs_x(s_fine), y=cs_y(s_fine))

# x_coords = [0.0,  20.0,  35.0,  50.0,  80.0, 100.0]
# y_coords = [0.0,   5.0, -15.0,  10.0,  25.0,  15.0]

# complex_track = generate_waypoint_spline_path(x_coords, y_coords, n_points=1200)
