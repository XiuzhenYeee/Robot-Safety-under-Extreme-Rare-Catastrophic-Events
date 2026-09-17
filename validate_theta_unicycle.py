"""
Validate the closed-form extremal index theta_Y (Proposition 1, Section 4) on
a closed-loop gain actually derived from the unicycle obstacle-avoidance
scenario used in the Simulation section, rather than on a generic synthetic
A_K as in Code_theta_Y/estimate_theta_Y.py.

This ties Section 4's clustering/extremal-index contribution to the same
applied example that already validates Section 3's contribution, instead of
leaving Section 4 validated only on an abstract toy system.

Model
-----
    e_{k+1} = A_K e_k + w_k,   w_k = zeta_k * b,   Y_k = c^T e_k

A_K = A - B K is the DLQR gain of the *linearized* unicycle tube-error
dynamics, frozen at one representative operating point: the state where the
robot is passing the obstacle at exactly the tightened safety distance,
heading tangentially around it. This matches Assumption 4 / Proposition 1
exactly (a single fixed A_K), unlike the real closed-loop MPC simulation
(Section 5), where A_K is re-linearized at every step. This is the "Option A"
simplification discussed with the user: freeze A_K at a representative point
rather than trying to validate the formula on a genuinely time-varying gain.

c is the obstacle-normal direction (the same direction n used to tighten the
chance constraint in Section 3), extended with a zero heading component.
b = c / ||c|| : the disturbance is modeled as acting purely along the
constraint-relevant direction (the honest, in-scope special case of
Assumption 4), with tail index alpha = nu (the Student-t degrees of freedom
used throughout the simulation section) and symmetric tail-balance p=q=0.5
(Student-t noise is symmetric).

No scipy/cvxpy dependency: the discrete algebraic Riccati equation is solved
here by direct fixed-point iteration (converges quickly for this small,
well-behaved system), so this script runs anywhere numpy + matplotlib do.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) Representative operating point (from the Simulation-section scenario)
# ----------------------------------------------------------------------
# Scenario (matches main_compare_controllers.py): goal=(6,0), obstacle at
# (3,0), obstacle_radius=0.6, d_safe=0.8 -> tightened stand-off R=1.4.
obstacle_center = np.array([3.0, 0.0])
R_safe = 0.6 + 0.8  # obstacle_radius + d_safe

dt = 0.1
v_bar = 1.0     # forward speed at the binding point (v_max)
th_bar = 0.2    # heading while passing the obstacle (rad)
# Representative position: passing the obstacle tangentially at exactly the
# tightened stand-off distance R_safe, directly "above" the obstacle center.
pbar = obstacle_center + np.array([0.0, R_safe])
xbar = np.array([pbar[0], pbar[1], th_bar])
ubar = np.array([v_bar, 0.3])  # some turning rate to go around

# Linearized unicycle dynamics x_{k+1} = A x_k + B u_k (Helper_linearization.py)
th = xbar[2]
v = ubar[0]
A = np.array([
    [1.0, 0.0, -dt * v * np.sin(th)],
    [0.0, 1.0,  dt * v * np.cos(th)],
    [0.0, 0.0,  1.0],
])
B = np.array([
    [dt * np.cos(th), 0.0],
    [dt * np.sin(th), 0.0],
    [0.0,             dt],
])

# Obstacle-normal direction at pbar (Helper_linearization.linearized_obstacle_halfspace)
diff = pbar - obstacle_center
n = diff / np.linalg.norm(diff)
print(f"Operating point: pbar={pbar}, theta={th_bar}, obstacle normal n={n}")

# ----------------------------------------------------------------------
# 2) DLQR gain via fixed-point Riccati iteration (no scipy)
# ----------------------------------------------------------------------
def dlqr_no_scipy(A, B, Q, R, n_iter=2000, tol=1e-14):
    P = Q.copy()
    for _ in range(n_iter):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        P_new = Q + A.T @ P @ (A - B @ K)
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P

Qe = np.diag([5.0, 5.0, 1.0])
Re = np.diag([1.0, 1.0])
K, P = dlqr_no_scipy(A, B, Qe, Re)
Ak = A - B @ K
eigs = np.linalg.eigvals(Ak)
print(f"A_K eigenvalues (magnitude): {np.abs(eigs)}")
assert np.all(np.abs(eigs) < 1.0), "A_K not stable, pick a different operating point"

# ----------------------------------------------------------------------
# 3) Disturbance direction and tail parameters
# ----------------------------------------------------------------------
c = np.array([n[0], n[1], 0.0])          # constraint direction, extended to 3D
b = c / np.linalg.norm(c)                 # disturbance assumed along c (Assumption 4)
sigma_xy = 0.03                           # matches Robot_Disturbance_Params in the sim section
nu = 3.0                                  # Student-t degrees of freedom = tail index alpha
alpha = nu
p_tail, q_tail = 0.5, 0.5                 # symmetric Student-t noise

# ----------------------------------------------------------------------
# 4) Closed-form theta_Y (Proposition 1)
# ----------------------------------------------------------------------
def closed_form_theta(Ak, b, c, alpha, p, q, n_lags=300):
    g = np.array([c @ np.linalg.matrix_power(Ak, j) @ b for j in range(n_lags)])
    g_plus = np.maximum(g, 0.0)
    g_minus = np.maximum(-g, 0.0)
    num = (g_plus.max() ** alpha) * p + (g_minus.max() ** alpha) * q
    den = p * np.sum(g_plus ** alpha) + q * np.sum(g_minus ** alpha)
    return num / den, g_plus.max(), g_minus.max()

theta_Y, sup_gplus, sup_gminus = closed_form_theta(Ak, b, c, alpha, p_tail, q_tail)

# ----------------------------------------------------------------------
# 5) Simulate the closed-loop projection Y_k = c^T e_k
# ----------------------------------------------------------------------
def simulate_Y(n_steps, seed, scale=sigma_xy):
    rng = np.random.default_rng(seed)
    zeta = rng.standard_t(alpha, size=n_steps) * scale
    e = np.zeros(3)
    Y = np.empty(n_steps)
    for k in range(n_steps):
        Y[k] = c @ e
        e = Ak @ e + zeta[k] * b
    return Y

# ----------------------------------------------------------------------
# 6) Ferro-Segers (2003) intervals estimator (pure arithmetic, no scipy)
# ----------------------------------------------------------------------
def ferro_segers_theta(Y, quantile):
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

if __name__ == "__main__":
    import time, os
    t0 = time.time()

    print(f"\nClosed-form theta_Y (Proposition 1, unicycle-derived A_K) = {theta_Y:.4f}")
    print(f"  sup g_plus  = {sup_gplus:.4f}")
    print(f"  sup g_minus = {sup_gminus:.4f}")

    # ------------------------------------------------------------------
    # Ferro-Segers (2003) intervals-estimator cross-check: an independent,
    # purely data-driven estimate of theta_Y, computed directly from
    # simulated trajectories with no reference to the closed-form formula
    # above. If it converges toward the closed-form value as the threshold
    # gets more extreme (more exceedances -> lower quantile -> more biased
    # by non-extreme behavior; fewer exceedances -> higher quantile ->
    # closer to the asymptotic regime the theory assumes, but noisier),
    # that is genuine evidence the closed-form formula is not just
    # internally consistent but actually describes the simulated process.
    #
    # A single realization is noisy at the small-N end (few hundred
    # exceedances), so each target exceedance count is averaged over
    # n_seeds independent trajectories and reported with its std across
    # seeds, rather than quoting one seed's point estimate.
    # ------------------------------------------------------------------
    n_fs = 2_000_000
    target_N_exceed = [160_000, 40_000, 4_000, 400]
    n_seeds_fs = 10

    print(f"\nFerro-Segers data-driven cross-check "
          f"({n_seeds_fs} seeds x {n_fs:,} steps each):")
    print(f"{'N_exceed target':>16}  {'theta_hat mean':>14}  {'std across seeds':>17}")
    fs_results = []
    for N_target in target_N_exceed:
        q = 1 - N_target / n_fs
        thetas = []
        for s in range(n_seeds_fs):
            Y_fs = simulate_Y(n_fs, seed=1000 + s)
            theta_hat, N_actual = ferro_segers_theta(Y_fs, q)
            thetas.append(theta_hat)
        thetas = np.array(thetas)
        fs_results.append((N_target, thetas.mean(), thetas.std(ddof=1)))
        print(f"{N_target:>16,}  {thetas.mean():>14.4f}  {thetas.std(ddof=1):>17.4f}")
    print(f"  (reference: closed-form theta_Y = {theta_Y:.4f})")

    # ---------------- Figure (two panels) ----------------
    fig, (ax_fs, ax) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    fs_N = [r[0] for r in fs_results]
    fs_mean = [r[1] for r in fs_results]
    fs_std = [r[2] for r in fs_results]
    ax_fs.errorbar(fs_N, fs_mean, yerr=fs_std, fmt="o-", color="tab:blue",
                    capsize=4, label=r"Ferro-Segers $\hat\theta$ (mean $\pm$ std, "
                                      f"{n_seeds_fs} seeds)")
    ax_fs.axhline(theta_Y, color="crimson", ls="--", lw=1.2,
                  label=r"closed-form $\theta_Y$")
    ax_fs.set_xscale("log")
    ax_fs.invert_xaxis()
    ax_fs.set_xlabel("number of exceedances (more extreme $\\rightarrow$)")
    ax_fs.set_ylabel(r"$\hat\theta$")
    ax_fs.set_title("(a) data-driven vs closed-form $\\theta_Y$", fontsize=9.5)
    ax_fs.legend(fontsize=7.5)

    # An illustrative trajectory with clustering marked. Using a compact
    # 500-step window (matching the style of the Section 4.1 illustration)
    # rather than a longer one: with theta_Y this small the persistence is
    # strong enough that a short window already shows the "one big jump,
    # then geometric decay" pattern directly, and a shorter window keeps the
    # plot visually zoomed-in rather than sparse.
    # (seed/tau0 chosen, by a small search over seeds, to show a clean
    # example: two near-isolated single-step exceedances plus one large,
    # visibly-decaying burst.)
    n_illus = 500
    Y_illus = simulate_Y(n_illus, seed=9)
    tau0 = 10.0
    u0 = np.quantile(Y_illus, 1 - tau0 / n_illus)
    exceed = Y_illus > u0
    n_clusters = int(np.sum(np.diff(np.r_[0, exceed.astype(int), 0]) == 1))
    ax.plot(Y_illus, lw=0.7, color="steelblue")
    ax.axhline(u0, color="crimson", ls="--", lw=1.2)
    ax.scatter(np.where(exceed)[0], Y_illus[exceed], color="crimson", s=18, zorder=5)
    ax.set_title(f"(b) unicycle-derived $A_K$: {exceed.sum()} exceedances, {n_clusters} cluster(s)",
                 fontsize=9.5)
    ax.set_xlabel("time step $k$")
    ax.set_ylabel("$Y_k$")

    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(os.path.join(out_dir, "theta_unicycle_validation.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "theta_unicycle_validation.pdf"))
    print(f"\nSaved theta_unicycle_validation.png/.pdf to {out_dir}")
    print(f"Total time: {time.time()-t0:.1f}s")
