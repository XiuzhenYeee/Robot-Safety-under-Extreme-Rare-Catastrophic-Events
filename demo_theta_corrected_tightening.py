"""
Simple demo: theta_Y-corrected constraint tightening vs naive per-step
tightening, on the unicycle scenario's own closed-loop gain.

Reuses the frozen operating point / A_K / b / c already derived and
validated in Code_clustering_unicycle/validate_theta_unicycle.py
(theta_Y = 0.1527 for this system). No scipy/cvxpy needed.

Story
-----
Section 3's tightened constraint,

    c^T xbar_i <= z_max - q_hat_i(1 - eps),

chooses q_hat_i so that the MARGINAL, per-step exceedance probability is
eps. It says nothing about the probability of the trajectory ever
experiencing a violation, or about how long a violation lasts once it
starts. Section 4 shows exceedances cluster with extremal index theta_Y,
so both of those trajectory-level quantities are governed by theta_Y, not
recoverable from eps alone:

  (1) P(>= 1 violation episode over T steps) ~= 1 - exp(-theta_Y * T * eps),
      which for T*eps not tiny is noticeably larger than the naive eps.

  (2) mean number of consecutive violating steps once a violation starts
      ~= 1/theta_Y.

This script quantifies both effects at the paper's own scenario parameters
(T=30, eps=1e-3, from main_compare_controllers.py) and shows the corrected
per-step target eps_star that actually delivers a target trajectory-level
risk of eps, mimicking Section 3's QP with the constraint

    c^T xbar_i <= z_max - q_hat_i(1 - eps_star),   eps_star = -ln(1-eps)/(theta_Y*T).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time

# ----------------------------------------------------------------------
# 1) Frozen unicycle operating point (identical to validate_theta_unicycle.py)
# ----------------------------------------------------------------------
obstacle_center = np.array([3.0, 0.0])
R_safe = 0.6 + 0.8

dt = 0.1
v_bar = 1.0
th_bar = 0.2
pbar = obstacle_center + np.array([0.0, R_safe])
xbar = np.array([pbar[0], pbar[1], th_bar])
ubar = np.array([v_bar, 0.3])

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

diff = pbar - obstacle_center
n = diff / np.linalg.norm(diff)

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
assert np.all(np.abs(np.linalg.eigvals(Ak)) < 1.0)

c = np.array([n[0], n[1], 0.0])
b = c / np.linalg.norm(c)
sigma_xy = 0.03
nu = 3.0
alpha = nu
p_tail, q_tail = 0.5, 0.5

def closed_form_theta(Ak, b, c, alpha, p, q, n_lags=300):
    g = np.array([c @ np.linalg.matrix_power(Ak, j) @ b for j in range(n_lags)])
    g_plus = np.maximum(g, 0.0)
    g_minus = np.maximum(-g, 0.0)
    num = (g_plus.max() ** alpha) * p + (g_minus.max() ** alpha) * q
    den = p * np.sum(g_plus ** alpha) + q * np.sum(g_minus ** alpha)
    return num / den

theta_Y = closed_form_theta(Ak, b, c, alpha, p_tail, q_tail)

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
# 2) Naive vs theta-corrected per-step target
#    (T, eps match the actual Section 5.1 scenario: run_comparison(..., T=30, eps=1e-3, ...))
# ----------------------------------------------------------------------
T = 30
eps = 1e-3
eps_star = -np.log(1 - eps) / (theta_Y * T)

print(f"theta_Y (unicycle-derived, Section 5.2)     = {theta_Y:.4f}")
print(f"naive per-step eps                          = {eps:.2e}")
print(f"theta-corrected per-step eps*                = {eps_star:.2e}   (eps/eps* = {eps/eps_star:.2f}x tighter)")

# ----------------------------------------------------------------------
# 3) Empirical quantiles (tightening margins) for both designs
# ----------------------------------------------------------------------
t0 = time.time()
n_ref = 6_000_000
Y_ref = simulate_Y(n_ref, seed=1)
m_naive = np.quantile(Y_ref, 1 - eps)
m_star = np.quantile(Y_ref, 1 - eps_star)
print(f"\nreference simulation: {n_ref:,} steps ({time.time()-t0:.1f}s)")
print(f"naive tightening margin   q(1-eps)   = {m_naive:.4f}")
print(f"corrected tightening margin q(1-eps*) = {m_star:.4f}  (extra safety margin: {m_star - m_naive:.4f})")

# mean consecutive-violation run length under the naive margin (Section 4 prediction: 1/theta_Y)
exceed = Y_ref > m_naive
edges = np.diff(np.r_[0, exceed.astype(int), 0])
starts = np.flatnonzero(edges == 1)
ends = np.flatnonzero(edges == -1)
run_lengths = ends - starts
mean_run_length = run_lengths.mean()
print(f"\nobserved mean consecutive-violation run length (naive margin) = {mean_run_length:.2f}")
print(f"predicted 1/theta_Y                                            = {1/theta_Y:.2f}")
print(f"(n = {run_lengths.size} violation episodes observed)")

# ----------------------------------------------------------------------
# 4) Monte Carlo over independent T-step trajectories: does each design
#    actually deliver the intended trajectory-level risk?
#
#    Run this as n_blocks independent blocks (each with its own seed
#    stream) rather than one single pass of n_reps trajectories. This
#    gives an empirical standard error on p_naive/p_star, not just a
#    single point estimate: at a target rate of ~1e-3 and 200k reps,
#    the expected violation count is only ~200, so a single run's
#    relative sampling noise is ~7% -- large enough that "0.95x target"
#    from one seed is not, by itself, evidence the correction works.
# ----------------------------------------------------------------------
n_blocks = 20
n_reps_per_block = 50_000
n_reps = n_blocks * n_reps_per_block  # 1,000,000 total, vs 200,000 before
rng_master = np.random.default_rng(123)

p_naive_blocks = np.empty(n_blocks)
p_star_blocks = np.empty(n_blocks)
for blk in range(n_blocks):
    viol_naive = 0
    viol_star = 0
    for rep in range(n_reps_per_block):
        Y = simulate_Y(T, seed=int(rng_master.integers(1, 2**31 - 1)))
        if (Y > m_naive).any():
            viol_naive += 1
        if (Y > m_star).any():
            viol_star += 1
    p_naive_blocks[blk] = viol_naive / n_reps_per_block
    p_star_blocks[blk] = viol_star / n_reps_per_block

p_naive = p_naive_blocks.mean()
p_star = p_star_blocks.mean()
# Empirical (block-to-block) standard error of the mean.
se_naive_blk = p_naive_blocks.std(ddof=1) / np.sqrt(n_blocks)
se_star_blk = p_star_blocks.std(ddof=1) / np.sqrt(n_blocks)
# Analytic binomial standard error, as a sanity check that the blocks
# aren't showing extra dispersion beyond plain Bernoulli sampling noise.
se_naive_binom = np.sqrt(p_naive * (1 - p_naive) / n_reps)
se_star_binom = np.sqrt(p_star * (1 - p_star) / n_reps)

print(f"\n=== Trajectory-level (episode) violation probability over T={T} steps, target = {eps:.2e} ===")
print(f"    ({n_blocks} blocks x {n_reps_per_block:,} reps = {n_reps:,} total trajectories)")
print(f"  naive design      (tightened to per-step eps)       -> observed = {p_naive:.2e} +/- {se_naive_blk:.2e} (block SE)"
      f"  [binomial SE {se_naive_binom:.2e}]  ({p_naive/eps:.2f}x target)")
print(f"  theta-corrected   (tightened to per-step eps*)      -> observed = {p_star:.2e} +/- {se_star_blk:.2e} (block SE)"
      f"  [binomial SE {se_star_binom:.2e}]  ({p_star/eps:.2f}x target)")
naive_lo, naive_hi = p_naive - 1.96 * se_naive_blk, p_naive + 1.96 * se_naive_blk
star_lo, star_hi = p_star - 1.96 * se_star_blk, p_star + 1.96 * se_star_blk
print(f"  95% CI naive:            [{naive_lo:.2e}, {naive_hi:.2e}]")
print(f"  95% CI theta-corrected:  [{star_lo:.2e}, {star_hi:.2e}]")
print(f"\nTotal time: {time.time()-t0:.1f}s")

# ----------------------------------------------------------------------
# 5) Figure for advisor meeting
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(["naive\n(per-step $\\epsilon$)", "$\\theta_Y$-corrected\n(per-step $\\epsilon_\\star$)"],
            [p_naive, p_star], yerr=[1.96 * se_naive_blk, 1.96 * se_star_blk],
            capsize=4, color=["tab:red", "tab:blue"])
axes[0].axhline(eps, color="k", ls="--", lw=1.2, label=r"target $\epsilon$")
axes[0].set_ylabel("observed $P(\\geq 1$ violation over $T$ steps$)$")
axes[0].set_title("(a) trajectory-level risk (error bars: 95% CI, "
                   f"{n_blocks}$\\times${n_reps_per_block//1000}k reps)", fontsize=8.5)
axes[0].legend(fontsize=8)

axes[1].bar(["observed", "$1/\\theta_Y$"], [mean_run_length, 1 / theta_Y],
            color=["tab:blue", "tab:red"])
axes[1].set_ylabel("mean consecutive violating steps")
axes[1].set_title("(b) violation episode length")

fig.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "theta_corrected_tightening_demo.png"), dpi=200)
print(f"\nSaved figure to {out_dir}/theta_corrected_tightening_demo.png")
