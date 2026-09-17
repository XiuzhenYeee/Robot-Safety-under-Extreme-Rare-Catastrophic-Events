from Helper_Sys_setup import Robot_Disturbance_Params, Scenario
import numpy as np


def linearize_unicycle(xbar, ubar, p: Robot_Disturbance_Params):
    """
    Linearize x_{k+1} = f(x_k,u_k) around (xbar, ubar)
    x = [x,y,theta], u=[v,omega]
    """
    dt = p.dt
    th = xbar[2]
    v = ubar[0]

    A = np.array([
        [1.0, 0.0, -dt * v * np.sin(th)],
        [0.0, 1.0,  dt * v * np.cos(th)],
        [0.0, 0.0,  1.0]
    ])

    B = np.array([
        [dt * np.cos(th), 0.0],
        [dt * np.sin(th), 0.0],
        [0.0,             dt]
    ])
    return A, B


def linearized_obstacle_halfspace(Xbar, sc: Scenario):
    """
    Linearize constraint: clearance >= d_safe
    clearance = ||p - o|| - r_o
    Want: ||p-o|| >= r_o + d_safe

    Linearize g(p)=||p-o|| around pbar:
      g(p) ≈ g(pbar) + n^T (p - pbar),  where n=(pbar-o)/||pbar-o||

    Constraint g(p) >= r_o + d_safe becomes:
      n^T p >= (r_o + d_safe) + n^T pbar - g(pbar)

    Returns arrays (n_k, b_k) so that:
      n_k^T p_k >= b_k
    """
    o = sc.obstacle_center
    R = sc.obstacle_radius + sc.d_safe
    N = Xbar.shape[0] - 1

    n_list = []
    b_list = []
    for k in range(N+1):
        pbar = Xbar[k, :2]
        diff = pbar - o
        norm = np.linalg.norm(diff) + 1e-12
        n = diff / norm
        g = norm  # ||pbar-o||
        # n^T p >= R + n^T pbar - g
        b = R + n @ pbar - g
        n_list.append(n)
        b_list.append(b)
    return np.array(n_list), np.array(b_list)