"""
Extremal-index (theta) estimation for the closed-loop projected-error
process, and the theta-corrected epsilon target, matching manuscript
Section 4 ("Clustering of Rare Events under Closed-Loop Dynamics"):

    theta_star_eps = -ln(1-eps) / (theta * N)      (Eq. eq_theta_corrected_epsilon)

used to replace the per-step quantile target eps with eps_star inside the
SAME pot_gpd_quantile-based tightening already used for the (non-theta-
corrected) "evt" controller -- the manuscript's theta-corrected constraint
(Eq. eq_theta_corrected_constraint) only changes the target level passed
to the quantile estimator, nothing else about the estimator itself.

theta is the extremal index of Z_k = c^T x_k, the closed-loop-projected
state process (here: the projected closed-loop ERROR e_k along the
obstacle-constraint normal n, matching the convention already used in
main_Monte_Carlo.estimate_qk_generic's X_samples = max(0, -(e @ n))).

This codebase's system is a linearize-and-relinearize (SCP) LTV design, not
the fixed-(A,B,K) LTI system the manuscript's closed-form Proposition 1
assumes -- exactly the same gap that already exists for the (non-theta)
per-step q_k tightening, which is why estimate_qk_generic already uses an
empirical Monte-Carlo/GPD fit instead of a closed-form formula. Consistent
with that precedent, theta here is estimated EMPIRICALLY (Section 4.3's
Ferro-Segers intervals estimator, not the closed-form Proposition 1
formula) from ONE long synthetic path of the closed-loop error dynamics,
frozen at a single representative linearization point (AK, n) -- reasonable
since AK's eigenvalues were found to vary only mildly across the horizon
for this scenario (~0.8-0.92 throughout, see quick_infeasibility_isolate2.py
investigation), so a single frozen linearization is a fair proxy for "the
system" in the sense the manuscript's fixed-LTI theory means it.
"""
import numpy as np
from Helper_Sys_setup import Scenario, Robot_Disturbance_Params
from Helper_baseline_controller import go_to_goal
from Helper_nominal import unicycle_step_nominal, rollout_nominal
from Helper_linearization import linearize_unicycle, linearized_obstacle_halfspace
from Helper_dlqr import dlqr


def ferro_segers_theta(Y, quantile):
    """
    Empirical extremal index of the sequence Y at the given threshold
    quantile, via the intervals estimator of Ferro and Segers (2003).
    Identical formula to Code_clustering_unicycle/validate_theta_unicycle.py's
    ferro_segers_theta and manuscript Eq. eq_ferro_segers -- duplicated here
    (rather than imported across the two separate project folders) since
    it's a small, self-contained, dependency-free function.

    Returns (theta_hat, N) where N is the exceedance count.
    """
    Y = np.asarray(Y)
    u = np.quantile(Y, quantile)
    exceed_idx = np.flatnonzero(Y > u)
    N = exceed_idx.size
    if N < 2:
        return np.nan, N
    T = np.diff(exceed_idx).astype(float)
    if T.max() <= 2:
        num = 2.0 * (T.sum()) ** 2
        den = (N - 1) * np.sum(T ** 2)
    else:
        Tm1 = T - 1.0
        num = 2.0 * (Tm1.sum()) ** 2
        den = (N - 1) * np.sum(Tm1 * (T - 2.0))
    theta_hat = num / den if den > 0 else np.nan
    return min(1.0, theta_hat), N


def estimate_theta_for_scenario(x0, sc: Scenario, p: Robot_Disturbance_Params,
                                 N=12, k_rep=0, n_steps=500_000,
                                 u_quantile=0.95, seed=1):
    """
    Estimate theta for this scenario by:
      1. Building the same go_to_goal reference trajectory used to
         initialize the real SCP loop (one_evt_tube_smpc_step step (a)),
      2. Linearizing at horizon step k_rep to get a representative (A,B,K,n),
      3. Simulating ONE long synthetic path of the closed-loop error
         e_{t+1} = (A - B K) e_t + w_t under the same heavy-tailed
         disturbance model used elsewhere in this codebase,
      4. Running the Ferro-Segers estimator on the n-projected path.

    This is a one-time (per-scenario) cost, not something to call inside
    the per-MPC-step tightening loop -- n_steps=500_000 takes a few seconds
    and only needs to be done once before a whole run_comparison sweep.
    """
    Ubar = np.zeros((N, 2))
    x_tmp = x0.copy()
    for k in range(N):
        Ubar[k] = go_to_goal(x_tmp, sc, p)
        x_tmp = unicycle_step_nominal(x_tmp, Ubar[k], p)
    Xbar = rollout_nominal(x0, Ubar, p)

    A, B = linearize_unicycle(Xbar[k_rep], Ubar[k_rep], p)
    Qe = np.diag([5.0, 5.0, 1.0])
    Re = np.diag([1.0, 1.0])
    K = dlqr(A, B, Qe, Re)
    AK = A - B @ K

    n_list, b_list = linearized_obstacle_halfspace(Xbar, sc)
    n = n_list[k_rep]

    rng = np.random.default_rng(seed)
    e = np.zeros(3)
    Z = np.zeros(n_steps)
    for t in range(n_steps):
        w_xy = rng.standard_t(p.nu, size=2) * p.sigma_xy
        w_th = rng.standard_t(p.nu, size=1) * p.sigma_th
        w = np.concatenate([w_xy, w_th])
        e = AK @ e + w
        Z[t] = -(e[:2] @ n)

    theta_hat, n_exceed = ferro_segers_theta(Z, u_quantile)
    return theta_hat, dict(n_exceed=n_exceed, u_quantile=u_quantile,
                            AK_eig=np.abs(np.linalg.eigvals(AK)), n=n, k_rep=k_rep)


def theta_corrected_eps(eps, theta, N):
    """
    eps_star = -ln(1-eps) / (theta * N)   (manuscript Eq. eq_theta_corrected_epsilon)
    N here is the MPC PREDICTION HORIZON (not the closed-loop rollout
    length) -- see manuscript Section 4, line 364: "the per-step target eps
    ... across the horizon i=1,...,N".
    """
    return -np.log(1.0 - eps) / (theta * N)
