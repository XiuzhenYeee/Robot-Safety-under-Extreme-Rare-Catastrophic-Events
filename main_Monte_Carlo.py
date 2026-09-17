"""
Robotics
mobile robot with heavy-tailed disturbances.
- Discrete-time dynamics with additive noise
- controller: go-to-goal + obstacle avoidance
- Monte Carlo rollouts under Student-t disturbances
"""
import numpy as np
import matplotlib.pyplot as plt
from Helper_Sys_setup import Scenario, Robot_Disturbance_Params, sample_student_t_noise
from Helper_Monte_Carlo import monte_carlo_safety, pot_gpd_quantile, gaussian_quantile
from Helper_baseline_controller import go_to_goal


if __name__ == "__main__":
    # Scenario: start left, goal right, obstacle in the middle
    sc = Scenario(
        goal=np.array([5.0, 5.0]),
        obstacle_center=np.array([3.0, 3.0]),
        obstacle_radius=0.6,
        d_safe=0.8
    )

    p = Robot_Disturbance_Params(
        dt=0.1, v_max=1.0, w_max=1.5,
        nu=3.0, sigma_xy=0.03, sigma_th=0.015,
        seed=1
    )


    ## ========================= Monte Carlo safety =======================
    x0 = np.array([0.0, 0.0, 0.2])
    T = 120  # 12 seconds
    M = 2000
    min_clr, vio = monte_carlo_safety(x0, T, M, go_to_goal, sc, p)
    p_hat = vio.mean()
    print(f"Empirical violation probability P(min clearance < d_safe): {p_hat:.4f} (M={M})")
    # EVT data: define a scalar "violation severity"
    # X = d_safe - min_clearance  (bigger = worse). We model tail of X.
    Xsev = sc.d_safe - min_clr
    # Fit EVT quantile at epsilon
    eps = 1e-3
    try:
        q_hat, info = pot_gpd_quantile(Xsev, eps=eps, u_quantile=0.90)
        print(f"EVT POT/GPD estimated q(1-{eps}) for severity X: {q_hat:.4f}")
        print("EVT fit info:", info)
    except ValueError as e:
        print("EVT fit skipped:", e)
        q_hat, info = None, None
    # --- Plot 3: distribution of min clearance (MC) ---
    plt.figure()
    plt.hist(min_clr, bins=60)
    plt.axvline(sc.d_safe, linestyle="--")
    plt.title("Monte Carlo: distribution of minimum clearance")
    plt.xlabel("min clearance over rollout (m)"); plt.ylabel("count")

    plt.show()




# ----------------------------
# 6) Monte Carlo simulate error dynamics and compute q_k (generic quantile fn)
# ----------------------------
def estimate_qk_generic(A_list, B_list, K_list, n_list, eps, p: Robot_Disturbance_Params,
                        quantile_fn, M_evt=2000, u_quantile=0.90, **quantile_kwargs):
    """
    Same as estimate_qk_evt, but the quantile estimator is swappable via
    `quantile_fn` (e.g., pot_gpd_quantile or gaussian_quantile), so the
    identical error-rollout machinery can produce either an EVT-based or a
    naive/Gaussian-based tightening for direct comparison.

    We enforce half-space: n_k^T p_k >= b_k (nominal tightened)
    Error e affects p via p = pbar + e_pos. Violation due to error:
      n^T(pbar + e_pos) < b  =>  n^T e_pos < (b - n^T pbar)
    The "bad" direction for tightening is the negative slack from error:
      X_k := max(0, -(n^T e_pos))
    """
    rng = np.random.default_rng(p.seed)
    N = len(A_list)

    # collect samples of X_k for each k (0..N), vectorized over all M_evt
    # paths at once (all paths share the same rollout machinery as the
    # original per-sample loop, just drawn and propagated as (M_evt,3)
    # arrays instead of one path at a time; this is ~80x faster, which
    # matters once M_evt is pushed well above a few thousand to control
    # the finite-sample bias of the online GPD quantile fit below).
    X_samples = [np.zeros(M_evt) for _ in range(N+1)]

    e = np.zeros((M_evt, 3))  # start with zero deviation, all paths
    X_samples[0][:] = np.maximum(0.0, -(e[:, :2] @ n_list[0]))
    for k in range(N):
        # Student-t disturbance w_k in state coordinates, all paths at once
        w_xy = rng.standard_t(p.nu, size=(M_evt, 2)) * p.sigma_xy
        w_th = rng.standard_t(p.nu, size=(M_evt, 1)) * p.sigma_th
        w = np.hstack([w_xy, w_th])

        A = A_list[k]
        B = B_list[k]
        K = K_list[k]
        AK = A - B @ K
        e = e @ AK.T + w

        X_samples[k+1][:] = np.maximum(0.0, -(e[:, :2] @ n_list[k+1]))
    # (debug print of per-k error-tail stats silenced -- was firing on every
    # single MPC solve, i.e. once per real-time step per trial, and drowned
    # out the actual results. Uncomment locally if you need to inspect the
    # error-rollout tail at a specific horizon step.)
    # for k in range(1, min(6, N+1)):
    #     xk = X_samples[k]
    #     print(f"k={k}: frac>0={np.mean(xk>0):.3f}, max={xk.max():.4g}, mean={xk.mean():.4g}")

    # fit quantile for each step using the supplied quantile_fn
    q_list = [0.0]
    fit_info = [dict(method="fixed", reason="e0=0 so X0=0")]

    for k in range(1, N+1):
        qk, info = quantile_fn(X_samples[k], eps=eps, u_quantile=u_quantile, **quantile_kwargs)
        q_list.append(qk)
        fit_info.append(info)

    return np.array(q_list), fit_info


def estimate_qk_evt(A_list, B_list, K_list, n_list, eps, p: Robot_Disturbance_Params,
                    M_evt=2000, u_quantile=0.90):
    """
    Backward-compatible wrapper: EVT/POT-GPD quantile estimation, matching
    the original signature/behavior. Kept so existing callers don't break.
    """
    return estimate_qk_generic(
        A_list, B_list, K_list, n_list, eps, p,
        quantile_fn=pot_gpd_quantile,
        M_evt=M_evt, u_quantile=u_quantile,
        min_exceed=20, fallback_empirical=True,
    )


def estimate_qk_gaussian(A_list, B_list, K_list, n_list, eps, p: Robot_Disturbance_Params,
                         M_evt=2000, u_quantile=0.90):
    """
    Naive/Gaussian counterpart of estimate_qk_evt, for baseline comparison.
    """
    return estimate_qk_generic(
        A_list, B_list, K_list, n_list, eps, p,
        quantile_fn=gaussian_quantile,
        M_evt=M_evt, u_quantile=u_quantile,
    )