


# Helper_types.py
import numpy as np
from dataclasses import dataclass


@dataclass
class Robot_Disturbance_Params:
    dt: float = 0.1
    v_max: float = 1.0
    w_max: float = 1.5
    nu: float = 3.0
    sigma_xy: float = 0.02
    sigma_th: float = 0.01
    seed: int = 1


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi
