from Helper_Sys_setup import Scenario, Robot_Disturbance_Params, robot_move_one_step
from Helper_safety_metric import distance_to_obstacle
import numpy as np

def rollout(x0, T, controller_fn, sc: Scenario, p: Robot_Disturbance_Params, rng):
    """
    Simulate for T steps.
    Returns:
      X: (T+1, 3) states
      U: (T, 2) inputs
      clearance: (T+1,) obstacle clearance (distance to boundary)
    """
    X = np.zeros((T + 1, len(x0)))
    U = np.zeros((T, 2))
    clearance = np.zeros(T + 1)

    X[0] = x0
    clearance[0] = distance_to_obstacle(X[0], sc)

    for k in range(T):
        u = controller_fn(X[k], sc, p)
        U[k] = u
        X[k + 1] = robot_move_one_step(X[k], u, p, rng)
        clearance[k + 1] = distance_to_obstacle(X[k + 1], sc)

    return X, U, clearance