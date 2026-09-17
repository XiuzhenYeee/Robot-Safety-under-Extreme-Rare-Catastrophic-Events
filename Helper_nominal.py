import numpy as np
from Helper_types import Robot_Disturbance_Params, wrap_angle



def unicycle_step_nominal(x, u, p: Robot_Disturbance_Params):
    """No noise: nominal step."""
    dt = p.dt
    v = np.clip(u[0], -p.v_max, p.v_max)
    w = np.clip(u[1], -p.w_max, p.w_max)
    x_pos, y_pos, th = x
    x_next = np.array([
        x_pos + dt * v * np.cos(th),
        y_pos + dt * v * np.sin(th),
        wrap_angle(th + dt * w),
    ])
    return x_next


def rollout_nominal(x0, U, p: Robot_Disturbance_Params):
    X = np.zeros((U.shape[0] + 1, 3))
    X[0] = x0
    for k in range(U.shape[0]):
        X[k+1] = unicycle_step_nominal(X[k], U[k], p)
    return X