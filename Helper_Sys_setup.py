from dataclasses import dataclass
import numpy as np
from scipy.stats import t as student_t
from Helper_nominal import unicycle_step_nominal
# ----------------------------
# 1) System definition
# ----------------------------
@dataclass
class Robot_Disturbance_Params:
    dt: float = 0.1
    v_max: float = 1.0          # m/s
    w_max: float = 1.5          # rad/s
    # heavy-tail noise (Student-t)
    nu: float = 3.0             # degrees of freedom (smaller => heavier tails), large nu = 30 is almost Gaussian
    sigma_xy: float = 0.02      # position disturbance scale (m) per step
    sigma_th: float = 0.01      # heading disturbance scale (rad) per step, heading disturbance 0.01 rad is around 0.57 degree
    seed: int = 1

@dataclass
class Scenario:
    goal: np.ndarray               # [xg, yg]
    obstacle_center: np.ndarray    # [xo, yo]
    obstacle_radius: float = 0.6
    d_safe: float = 0.8            # safety distance threshold (must keep d >= d_safe)


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def sample_student_t_noise(rng, nu, scale, size):
    """
    Student-t noise with df=nu, scaled to 'scale'.
    """
    return student_t.rvs(df=nu, size=size, random_state=rng) * scale

def robot_move_one_step(x, u, p: Robot_Disturbance_Params, rng):
    """
    x = [x, y, theta]
    u = [v, w]
    with additive heavy-tailed disturbance on x,y,theta.
    """
    dt = p.dt
    v = np.clip(u[0], -p.v_max, p.v_max)
    w = np.clip(u[1], -p.w_max, p.w_max)

    x_pos, y_pos, th = x
    x_next = np.array([
        x_pos + dt * v * np.cos(th),
        y_pos + dt * v * np.sin(th),
        wrap_angle(th + dt * w),
    ])

    # additive heavy-tailed disturbance
    w_xy = sample_student_t_noise(rng, p.nu, p.sigma_xy, size=2)
    w_th = sample_student_t_noise(rng, p.nu, p.sigma_th, size=1)
    x_next[0:2] += w_xy
    x_next[2] = wrap_angle(x_next[2] + w_th.item())
    return x_next



def unicycle_step_stochastic(x, u, p: Robot_Disturbance_Params, rng):
    """With Student-t noise."""
    x_next = unicycle_step_nominal(x, u, p)
    w_xy = sample_student_t_noise(rng, p.nu, p.sigma_xy, size=2)
    w_th = sample_student_t_noise(rng, p.nu, p.sigma_th, size=1)
    x_next[0:2] += w_xy
    x_next[2] = wrap_angle(x_next[2] + w_th.item())
    return x_next