import numpy as np
from Helper_Sys_setup import Scenario, Robot_Disturbance_Params
from Helper_rollout import rollout
from scipy.stats import genpareto, norm

# ----------------------------
# 4) Monte Carlo evaluation of rare-event safety
# ----------------------------
def monte_carlo_safety(x0, T, M, controller_fn, sc: Scenario, p: Robot_Disturbance_Params):
    """
    Runs M rollouts and returns:
      min_clearances: (M,) min clearance over each rollout
      violation_flags: (M,) whether clearance ever < d_safe
    """
    rng = np.random.default_rng(p.seed)
    min_clearances = np.zeros(M)
    violation_flags = np.zeros(M, dtype=bool)

    for m in range(M):
        # new RNG stream per rollout (still reproducible)
        rng_m = np.random.default_rng(rng.integers(0, 2**32 - 1))
        X, U, clr = rollout(x0, T, controller_fn, sc, p, rng_m)
        min_clearances[m] = np.min(clr)
        violation_flags[m] = np.any(clr < sc.d_safe)

    return min_clearances, violation_flags


def pot_gpd_quantile(samples, eps=1e-3, u_quantile=0.90, min_exceed=30,
                      fallback_empirical=False):
    """
    Estimate (1-eps)-quantile of X using POT/GPD.
    samples: 1D array of X values (higher = worse / more risky)
    eps: target tail probability (e.g., 1e-3)
    u_quantile: choose threshold u as this empirical quantile (e.g., 0.90)
    min_exceed: minimum number of exceedances required to trust the GPD fit
    fallback_empirical: if True, fall back to an empirical-quantile estimate
        (instead of raising) when there are too few samples/exceedances

    Returns:
      q_hat: estimated quantile at level (1-eps)
      info: dict with fitted params and diagnostics
    """
    x = np.asarray(samples).copy()
    x = x[np.isfinite(x)]

    if x.size < 50:
        if fallback_empirical:
            q_hat = np.quantile(x, 1 - eps) if x.size > 0 else 0.0
            info = dict(method="empirical_fallback", reason="too few samples", M=x.size)
            return q_hat, info
        raise ValueError("Need more samples for EVT. Increase M or data length.")

    u = np.quantile(x, u_quantile)
    exceed = x[x > u] - u
    Nu = exceed.size
    M = x.size
    pu_hat = Nu / M

    if Nu < min_exceed:
        if fallback_empirical:
            q_hat = np.quantile(x, 1 - eps)
            info = dict(method="empirical_fallback", reason=f"too few exceedances (Nu={Nu})",
                        Nu=Nu, M=M, u=u, u_quantile=u_quantile)
            return q_hat, info
        raise ValueError(f"Too few exceedances (Nu={Nu}). Lower u_quantile or collect more samples.")

    # Fit GPD to exceedances (loc fixed at 0)
    # genpareto parameterization: shape=c (=xi), loc, scale
    xi_hat, loc_hat, beta_hat = genpareto.fit(exceed, floc=0.0)

    # Quantile formula:
    # q = u + (beta/xi) * ((pu/eps)^xi - 1)    if xi != 0
    # q = u + beta * log(pu/eps)              if xi ~ 0
    if abs(xi_hat) < 1e-6:
        q_hat = u + beta_hat * np.log(pu_hat / eps)
    else:
        q_hat = u + (beta_hat / xi_hat) * ((pu_hat / eps) ** xi_hat - 1.0)

    info = dict(
        method="gpd",
        u=u, u_quantile=u_quantile, Nu=Nu, M=M, pu_hat=pu_hat,
        xi_hat=xi_hat, beta_hat=beta_hat
    )
    return q_hat, info


def gaussian_quantile(samples, eps=1e-3, u_quantile=None, min_exceed=None,
                       fallback_empirical=None):
    """
    Naive light-tailed tightening: assumes X is approximately Gaussian and
    estimates the (1-eps)-quantile from the sample mean/std only.
    q_hat = mu + sigma * z_{1-eps}

    Accepts the same keyword signature as pot_gpd_quantile (u_quantile,
    min_exceed, fallback_empirical) so the two are interchangeable as a
    `quantile_fn` argument; the extra kwargs are unused here.

    Returns:
      q_hat: estimated quantile at level (1-eps)
      info: dict with fitted params
    """
    x = np.asarray(samples).copy()
    x = x[np.isfinite(x)]

    if x.size < 2:
        info = dict(method="gaussian", reason="too few samples", M=x.size)
        return 0.0, info

    mu = x.mean()
    sigma = x.std(ddof=1)
    z = norm.ppf(1 - eps)
    q_hat = mu + sigma * z

    info = dict(method="gaussian", mu=mu, sigma=sigma, z=z, M=x.size)
    return q_hat, info


def oracle_empirical_quantile(samples, eps=1e-3, u_quantile=None, min_exceed=None,
                               fallback_empirical=None):
    """
    Plain large-sample empirical (1-eps)-quantile, no GPD tail fit at all.

    This is NOT the paper's proposed EVT/GPD estimator (Section 4's
    hat q_e via POT/GPD, used by pot_gpd_quantile above); it's the same
    oracle-style estimator used in
    Code_clustering_unicycle/demo_theta_corrected_tightening.py
    (m_naive = np.quantile(Y_ref, 1-eps) on a 6,000,000-sample reference),
    included here only so that main_compare_controllers.py's "evt" curve
    can be checked against that idealized theta-only validation on equal
    footing. Only meaningful with a large M_evt (thousands of samples in
    the tail), since eps=1e-3 needs enough mass past the (1-eps) point for
    the raw empirical quantile itself to be stable.

    Accepts the same keyword signature as pot_gpd_quantile / gaussian_quantile
    so it's interchangeable as a `quantile_fn` argument; the extra kwargs are
    unused here.

    Returns:
      q_hat: the empirical (1-eps)-quantile
      info: dict with diagnostics
    """
    x = np.asarray(samples)
    x = x[np.isfinite(x)]
    q_hat = np.quantile(x, 1 - eps) if x.size > 0 else 0.0
    info = dict(method="oracle_empirical", M=x.size)
    return q_hat, info