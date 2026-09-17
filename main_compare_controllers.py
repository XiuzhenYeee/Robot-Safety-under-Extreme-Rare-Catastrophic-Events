"""
Compare baseline (no tightening) vs Gaussian-tightened vs EVT/POT-GPD-tightened
vs theta-corrected EVT/POT-GPD-tightened tube SMPC, under heavy-tailed
(Student-t) disturbances. The theta-corrected controller (manuscript
Section 4) uses the same POT/GPD tightening as "evt", but targets a
stricter eps_star = -ln(1-eps)/(theta*N) instead of the raw eps, to correct
for the fact that a naive per-step eps target under-protects a horizon of
temporally-clustered (extremal-index theta<1) closed-loop error exceedances
-- see Helper_extremal_index.py.

Produces:
  - printed empirical violation probabilities for all four controllers
  - overlaid histogram of min-clearance-over-rollout for all four
  - boxplot of safety margin (clearance - d_safe) per controller
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from Helper_Sys_setup import Scenario, Robot_Disturbance_Params
from Helper_baseline_controller import baseline_go_to_goal
from Helper_rollout import rollout
from Helper_Monte_Carlo import pot_gpd_quantile, gaussian_quantile, oracle_empirical_quantile
from Helper_simulate_evt_tube_smpc import simulate_evt_tube_smpc
from Helper_extremal_index import estimate_theta_for_scenario, theta_corrected_eps


def run_comparison(x0, sc, p, M=300, T=120, N=12, eps=1e-3, M_evt=1500,
                    M_evt_evt=None, seed=0, checkpoint_path=None, checkpoint_every=10):
    """
    Runs M independent trials of three controllers under matched disturbance
    seeds (per-trial), and returns the min clearance achieved in each trial.

    checkpoint_path: if given, results-so-far (plus theta_hat/eps_star and
    everything needed to resume interpreting them) are saved to this .npz
    path every `checkpoint_every` trials, and again at the end. Intended
    for long unattended (e.g. overnight) runs: if the process is
    interrupted, killed, or the machine sleeps partway through, whatever
    completed so far is still on disk and usable -- current progress is
    tracked via `m_completed` in the saved file (results past that index
    are still the zero-initialized placeholder, not real data).

    Each trial's body is wrapped in a try/except: if a single trial raises
    an unexpected error (not the already-handled per-MPC-step exceptions
    inside simulate_evt_tube_smpc, but something at a higher level), that
    trial's results are left as NaN and the run continues rather than
    losing all remaining trials to one bad draw. Summary statistics
    elsewhere should use NaN-aware functions (nanmean etc.) accordingly.

    The "evt" controller uses pot_gpd_quantile -- the manuscript's actual
    proposed POT/GPD tail estimator -- NOT oracle_empirical_quantile.
    (An earlier version of this script used oracle_empirical_quantile as a
    temporary stand-in, to cross-check against the idealized,
    oracle-calibrated validation in
    Code_clustering_unicycle/demo_theta_corrected_tightening.py; that gave
    an artificially perfect 0% violation rate, which is not a real
    performance number for the paper's method -- oracle_empirical_quantile
    has no tail model, it just reads off the largest few samples directly,
    which only works because it was handed an enormous sample size. The
    real method fits a GPD tail from a modest number of Monte Carlo
    samples, which is necessarily noisier -- expect a non-zero but still
    meaningfully-better-than-gaussian violation rate here.)
    M_evt_evt is unused now (kept only for backward-compatible call
    signatures) since GPD tail fitting doesn't need the huge sample count
    oracle_empirical_quantile required.

    Returns: (results, theta_hat, eps_star) -- results is a dict of
    {controller_name: (M,) array of min-clearance-over-rollout}; theta_hat
    and eps_star are the one-time extremal-index estimate and the resulting
    theta-corrected epsilon target used for the "evt_theta" controller.
    """
    if M_evt_evt is None:
        M_evt_evt = M_evt
    rng = np.random.default_rng(seed)
    results = {"baseline": np.zeros(M), "gaussian": np.zeros(M), "evt": np.zeros(M),
               "evt_theta": np.zeros(M)}
    n_fallback_total = {"gaussian": 0, "evt": 0, "evt_theta": 0}
    n_soft_total = {"gaussian": 0, "evt": 0, "evt_theta": 0}

    # --- theta-corrected epsilon (manuscript Section 4) ---
    # One-time estimate of the extremal index theta for this scenario's
    # closed-loop projected-error process (see Helper_extremal_index.py),
    # then eps_star = -ln(1-eps)/(theta*N) replaces the raw eps ONLY for
    # the quantile target passed into pot_gpd_quantile -- same estimator,
    # same code path as the "evt" controller above, just a stricter target
    # level to correct for the fact that a naive per-step eps target
    # under-protects a T-step-dependent (clustered) closed-loop trajectory.
    theta_hat, theta_info = estimate_theta_for_scenario(x0, sc, p, N=N)
    eps_star = theta_corrected_eps(eps, theta_hat, N)
    print(f"theta-corrected: theta_hat={theta_hat:.4f} (n_exceed={theta_info['n_exceed']}, "
          f"AK_eig={np.round(theta_info['AK_eig'], 3)}), "
          f"eps={eps:.1e} -> eps_star={eps_star:.2e} (N={N})")

    # NaN-initialized so a skipped (errored) trial is visibly distinct from
    # a genuine result, and so nanmean/nanstd/etc. downstream ignore it
    # rather than silently treating a missing trial as clearance=0.
    for name in results:
        results[name][:] = np.nan

    import time
    t_start = time.time()

    for m in range(M):
        trial_seed = int(rng.integers(0, 2**32 - 1))

        try:
            # --- baseline: no tightening at all ---
            rng_baseline = np.random.default_rng(trial_seed)
            _, _, clr = rollout(x0, T, baseline_go_to_goal, sc, p, rng_baseline)
            results["baseline"][m] = clr.min()

            # --- Gaussian-tightened tube SMPC ---
            _, _, clr, _, _, n_fb, n_sf = simulate_evt_tube_smpc(
                x0, sc, p, T_steps=T, N=N, eps=eps, M_evt=M_evt,
                rng_seed=trial_seed, store_debug=False,
                quantile_fn=gaussian_quantile,
            )
            results["gaussian"][m] = clr.min()
            n_fallback_total["gaussian"] += n_fb
            n_soft_total["gaussian"] += n_sf

            # --- EVT (POT/GPD)-tightened tube SMPC -- the paper's actual method ---
            _, _, clr, _, _, n_fb, n_sf = simulate_evt_tube_smpc(
                x0, sc, p, T_steps=T, N=N, eps=eps, M_evt=M_evt,
                rng_seed=trial_seed, store_debug=False,
                quantile_fn=pot_gpd_quantile,
            )
            results["evt"][m] = clr.min()
            n_fallback_total["evt"] += n_fb
            n_soft_total["evt"] += n_sf

            # --- theta-corrected EVT (POT/GPD)-tightened tube SMPC ---
            # Same pot_gpd_quantile estimator as "evt" above, but targeting
            # eps_star instead of the raw eps (see note above the trial loop).
            _, _, clr, _, _, n_fb, n_sf = simulate_evt_tube_smpc(
                x0, sc, p, T_steps=T, N=N, eps=eps_star, M_evt=M_evt,
                rng_seed=trial_seed, store_debug=False,
                quantile_fn=pot_gpd_quantile,
            )
            results["evt_theta"][m] = clr.min()
            n_fallback_total["evt_theta"] += n_fb
            n_soft_total["evt_theta"] += n_sf
        except Exception as e:
            # Unexpected failure at the trial level (not the already-handled
            # per-MPC-step exceptions inside simulate_evt_tube_smpc). Leave
            # this trial's entries as NaN and keep going -- losing one trial
            # out of many is far better than losing an entire overnight run.
            print(f"  [WARNING] trial {m+1}/{M} raised {type(e).__name__}: {e} -- "
                  f"skipping this trial, continuing.")

        if (m + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = elapsed / (m + 1)
            eta_min = rate * (M - (m + 1)) / 60
            print(f"  trial {m+1}/{M} done ({elapsed/60:.1f} min elapsed, "
                  f"~{eta_min:.1f} min remaining)")

        if checkpoint_path is not None and (m + 1) % checkpoint_every == 0:
            # Includes scenario/target info too (not just the raw arrays) so
            # this file is self-sufficient for a basic violation-probability/
            # margin check even if the run is interrupted before the final
            # comparison_data.npz (which also needs the representative
            # trajectories, computed only after this loop finishes) is written.
            np.savez(
                checkpoint_path,
                m_completed=m + 1, M=M,
                **{f"min_clr_{k}": v for k, v in results.items()},
                theta_hat=theta_hat, eps_star=eps_star,
                eps=eps, d_safe=sc.d_safe,
            )

    # Report how often the MPC solve genuinely failed (fallback) vs how
    # often it succeeded but had to use the softened constraint's slack
    # (soft), as a fraction of all (trial x step) MPC solves. A high soft
    # rate means the tightened problem is genuinely infeasible from states
    # the closed loop visits (not just synthetic probe states) at eps=1e-3
    # with these heavy-tailed disturbances -- a real scenario-difficulty
    # finding, not a bug -- see quick_infeasibility_isolate2.py. Fallback
    # should now be rare since ordinary tightened-constraint infeasibility
    # no longer raises (it's absorbed by the slack instead).
    total_steps = M * T
    for name in ["gaussian", "evt", "evt_theta"]:
        frac_fb = n_fallback_total[name] / total_steps
        frac_sf = n_soft_total[name] / total_steps
        print(f"  {name:10s}: fallback rate = {frac_fb:.1%} "
              f"({n_fallback_total[name]}/{total_steps}), "
              f"soft-constraint rate = {frac_sf:.1%} "
              f"({n_soft_total[name]}/{total_steps})")

    return results, theta_hat, eps_star


def summarize_and_plot(results, sc, eps, save_dir="figures", save_name="comparison_hist"):
    print("\n=== Empirical violation probabilities (target eps = {:.1e}) ===".format(eps))
    for name, min_clr in results.items():
        p_hat = (min_clr < sc.d_safe).mean()
        print(f"  {name:10s}: P(min clearance < d_safe) = {p_hat:.4f}  "
              f"(n={len(min_clr)})")

    plt.figure()
    colors = {"baseline": "tab:red", "gaussian": "tab:orange", "evt": "tab:blue",
              "evt_theta": "tab:green"}
    for name, min_clr in results.items():
        plt.hist(min_clr, bins=30, alpha=0.5, label=name, color=colors.get(name))
    plt.axvline(sc.d_safe, linestyle="--", color="k", label="d_safe")
    plt.legend()
    plt.xlabel("min clearance over rollout (m)")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{save_name}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, f"{save_name}.pdf"), bbox_inches="tight")
    print(f"Saved figure to {save_dir}/{save_name}.png and .pdf")

    plt.show()


def plot_margin_analysis(results, sc, eps, save_dir="figures",
                          save_name="margin_analysis"):
    os.makedirs(save_dir, exist_ok=True)

    names = ["baseline", "gaussian", "evt", "evt_theta"]
    colors = {"baseline": "tab:red", "gaussian": "tab:orange", "evt": "tab:blue",
              "evt_theta": "tab:green"}
    margins = {name: results[name] - sc.d_safe for name in names}

    print(f"\n=== Safety margin (min clearance - d_safe), eps={eps:.1e} ===")
    for name in names:
        m = margins[name]
        print(f"  {name:10s}: mean={m.mean():.3f}  min={m.min():.3f}  "
              f"std={m.std():.3f}  (n={len(m)})")

    # ---------------- Boxplot of margin distributions ----------------
    # (Previously this function also produced a bar chart of mean±std and
    # worst-case margin, but that information is a strict subset of what the
    # boxplot already shows (mean marker, whiskers/outliers as worst-case),
    # so the bar chart was dropped as redundant.)
    fig, ax = plt.subplots(figsize=(6, 5))
    box_data = [margins[name] for name in names]
    bp = ax.boxplot(box_data, labels=names, patch_artist=True, showmeans=True)
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)
    ax.axhline(0.0, linestyle="--", color="k")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_name}_boxplot.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, f"{save_name}_boxplot.pdf"),
                bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    # obstacle_center is offset off the start-goal centerline (y=1, not 0).
    # A perfectly head-on obstacle (dead center between x0 and goal) creates
    # a degenerate case for the linearized tangent-line safety constraint:
    # directly on the centerline, that constraint's normal is nearly
    # vertical, so "satisfy it" locally means "back away in x" rather than
    # "curve sideways" -- the MPC solve genuinely converges (SCP succeeds,
    # not a solver bug) but its optimal decision near the boundary is to
    # reverse, producing a stuck reverse/re-approach limit cycle instead of
    # progress toward the goal (confirmed via quick_check_stuck_state.py:
    # applied v~-0.985 right at the boundary, scp_iters=5, a converged
    # optimum, not a failure). Offsetting the obstacle avoids handing the
    # optimizer that exactly-symmetric, degenerate geometry -- matches the
    # off-axis obstacle already used in quick_scp_check.py.
    # NOTE: goal=[6,1] puts the straight line from x0=[1,0] to the goal
    # almost exactly grazing the obstacle (closest approach ~3.12,0.42,
    # clearance ~ -0.01m -- just barely clipping the physical disk), which
    # makes for a more visually interesting forced detour than goal=[6,0]'s
    # comfortable 0.4m clearance. Trade-off: because the obstacle sits
    # almost exactly ON the direct path rather than clearly to one side,
    # this is closer to a coin-flip between curving above vs below it --
    # watch for inconsistent left/right choices across trials/replans if
    # this reintroduces any of the jitter the off-axis fix (previous commit)
    # was meant to avoid.
    sc = Scenario(
        goal=np.array([6.0, 1.0]),
        obstacle_center=np.array([3.0, 1.0]),
        obstacle_radius=0.6,
        d_safe=0.8,
    )
    p = Robot_Disturbance_Params(
        dt=0.1, v_max=1.0, w_max=1.5,
        nu=3.0, sigma_xy=0.03, sigma_th=0.01,
        seed=1,
    )
    # Start close enough to the obstacle that the tightened constraint is
    # actually forced to bind within the MPC horizon (confirmed via the
    # single-state diagnostic: x0=[1,0,0] produces a real detour and a
    # genuine Gaussian-vs-EVT divergence in Ubar/Xbar).
    x0 = np.array([1.0, 0.0, 0.0])

    eps = 1e-3
    # M_evt=1500 for both gaussian and evt now: the "evt" controller uses
    # pot_gpd_quantile (the paper's actual POT/GPD tail estimator, see
    # run_comparison docstring), which -- unlike the oracle_empirical_quantile
    # substitution used previously -- fits a parametric tail model, so it
    # doesn't need an enormous sample count to produce a stable quantile
    # estimate. M_evt_evt is no longer used (kept only so old call sites
    # with that kwarg don't break).
    #
    # NOTE: an earlier attempt at bumping M_evt while using pot_gpd_quantile
    # made results worse instead of better -- a real, previously-unresolved
    # sensitivity in the GPD threshold/fit (u_quantile=0.75 default in
    # one_evt_tube_smpc_step) that's worth revisiting if these numbers look
    # off, rather than reflexively increasing M_evt to "fix" it.
    #
    # OVERNIGHT / publication-quality settings. M=300 trials is enough to
    # start meaningfully resolving evt vs evt_theta's violation-probability
    # gap (recall: at M=15 both often floor at 0, or show only 1-2
    # violations -- too few to trust the ratio between them; M=300 gives
    # much better resolution without needing many thousands of trials).
    # If you want the earlier fast/noisy settings back for quick iteration,
    # use M, T, M_evt = 15, 20, 800 instead.
    M, T, M_evt = 300, 30, 1500
    print(f"OVERNIGHT settings: M={M}, T={T}, M_evt={M_evt} "
          f"(this will take a long time -- see checkpointing below)")

    N = 12
    os.makedirs("data", exist_ok=True)
    checkpoint_path = os.path.join("data", "checkpoint.npz")
    results, theta_hat, eps_star = run_comparison(
        x0, sc, p, M=M, T=T, N=N, eps=eps, M_evt=M_evt, seed=0,
        checkpoint_path=checkpoint_path, checkpoint_every=10,
    )

    # NaN-aware: a trial that errored (see run_comparison's try/except) is
    # left as NaN and excluded here rather than corrupting the statistics.
    print("\n=== Empirical violation probabilities (target eps = {:.1e}) ===".format(eps))
    for name, min_clr in results.items():
        n_valid = np.sum(~np.isnan(min_clr))
        p_hat = np.nanmean(min_clr < sc.d_safe)
        print(f"  {name:10s}: P(min clearance < d_safe) = {p_hat:.4f}  "
              f"(n={n_valid}/{len(min_clr)})")

    print(f"\n=== Safety margin (min clearance - d_safe), eps={eps:.1e} ===")
    for name, min_clr in results.items():
        m = min_clr - sc.d_safe
        n_valid = np.sum(~np.isnan(m))
        print(f"  {name:10s}: mean={np.nanmean(m):.3f}  min={np.nanmin(m):.3f}  "
              f"std={np.nanstd(m):.3f}  (n={n_valid}/{len(m)})")

    # --- one representative single-trajectory rollout per controller, using
    # the SAME disturbance seed for baseline/gaussian/evt (common random
    # numbers) so a trajectory plot isolates controller behavior rather than
    # noise-realization differences. This is separate from the M-trial
    # statistical loop above (which discards X for speed) -- only 3 extra
    # solves, cheap relative to the full comparison. Not needed for
    # evt_theta since Figure 1 only shows baseline/gaussian/evt.
    #
    # Uses a LONGER horizon (T_traj) than the M-trial statistical loop's T:
    # T=20 steps = 2s = at most ~2m of travel at v_max=1.0, nowhere near
    # enough to cover the ~5-6m from x0=[1,0] to goal=[6,1] -- fine for the
    # statistics (the safety-critical obstacle passage happens early, well
    # within T=20), but a trajectory PLOT stopping partway looks like the
    # robot got stuck. T_traj=110 (11s) gives enough margin even for
    # gaussian/evt's wider detour (they dip further from the direct line
    # than baseline, so cover more total distance and need more time to
    # finish -- 70 steps was enough for baseline's tighter arc but left
    # gaussian/evt short of the goal). T_traj is only used for these 3
    # extra solves, so this doesn't slow down the (M-trial-multiplied)
    # statistics above.
    T_traj = 200
    print(f"\nGenerating one representative trajectory per controller "
          f"(T_traj={T_traj} steps, common random numbers)...")
    rep_seed = 12345
    rng_rep = np.random.default_rng(rep_seed)
    X_baseline, _, _ = rollout(x0, T_traj, baseline_go_to_goal, sc, p, rng_rep)
    X_gaussian, _, _, _, _, _, _ = simulate_evt_tube_smpc(
        x0, sc, p, T_steps=T_traj, N=N, eps=eps, M_evt=M_evt,
        rng_seed=rep_seed, store_debug=False, quantile_fn=gaussian_quantile,
    )
    X_evt, _, _, _, _, _, _ = simulate_evt_tube_smpc(
        x0, sc, p, T_steps=T_traj, N=N, eps=eps, M_evt=M_evt,
        rng_seed=rep_seed, store_debug=False, quantile_fn=pot_gpd_quantile,
    )

    # --- save everything needed for plotting, so plot_figures.py can be
    # re-run (and its styling iterated on) WITHOUT re-running any of the
    # slow MPC/Monte-Carlo simulation above.
    os.makedirs("data", exist_ok=True)
    data_path = os.path.join("data", "comparison_data.npz")
    np.savez(
        data_path,
        min_clr_baseline=results["baseline"], min_clr_gaussian=results["gaussian"],
        min_clr_evt=results["evt"], min_clr_evt_theta=results["evt_theta"],
        X_baseline=X_baseline, X_gaussian=X_gaussian, X_evt=X_evt,
        x0=x0, goal=sc.goal, obstacle_center=sc.obstacle_center,
        obstacle_radius=sc.obstacle_radius, d_safe=sc.d_safe,
        eps=eps, theta_hat=theta_hat, eps_star=eps_star,
        M=M, T=T, N=N, M_evt=M_evt,
    )
    print(f"Saved plotting data to {data_path} -- run plot_figures.py to (re)generate "
          f"figures without re-running the simulation.")