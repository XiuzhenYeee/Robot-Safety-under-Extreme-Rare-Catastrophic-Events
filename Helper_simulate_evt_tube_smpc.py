import numpy as np
from Helper_Sys_setup import Scenario, Robot_Disturbance_Params, unicycle_step_stochastic
from Helper_safety_metric import distance_to_obstacle
from Helper_baseline_controller import baseline_go_to_goal, go_to_goal
from Helper_Sys_setup import wrap_angle, sample_student_t_noise
from Helper_nominal import unicycle_step_nominal, rollout_nominal
from Helper_linearization import linearize_unicycle, linearized_obstacle_halfspace
from Helper_dlqr import dlqr
from Helper_Monte_Carlo import pot_gpd_quantile
from main_Monte_Carlo import estimate_qk_generic
from Helper_solve_tightened_smpc import solve_tightened_ltv_qp


def one_evt_tube_smpc_step(x0, sc: Scenario, p: Robot_Disturbance_Params,
                           N=15, eps=1e-3, M_evt=2000,
                           quantile_fn=pot_gpd_quantile, quantile_kwargs=None,
                           max_scp_iters=5, scp_tol=1e-3, step_damping=0.6):
    """
    Returns:
      U0 to apply

    quantile_fn: swappable quantile estimator (e.g., pot_gpd_quantile or
      gaussian_quantile) used to tighten each horizon-step constraint.
    quantile_kwargs: extra kwargs forwarded to quantile_fn (e.g., min_exceed,
      fallback_empirical for pot_gpd_quantile). Defaults to the standard
      EVT settings if quantile_fn is pot_gpd_quantile and none are given.

    max_scp_iters: number of sequential-convex-programming passes. Each
      pass after the first re-linearizes the dynamics AND the obstacle
      half-space around the PREVIOUS pass's solved trajectory (not the
      fixed initial guess) and re-solves the tightened QP. Set to 1 to
      recover the old single-shot linearize-and-solve behavior.
    scp_tol: stop iterating early once the (true, nonlinearly-rolled-out)
      trajectory stops changing by more than this much (max abs state
      change across all horizon steps).
    step_damping: fraction of each iteration's new control sequence to
      actually adopt: Ubar <- Ubar + step_damping*(Ubar_new - Ubar). 1.0
      (full step every time) is standard SCP but diverged in testing on
      this scenario -- NaN/Inf trajectories that then poison the
      downstream Monte Carlo tightening. Defaulting to 0.6 as a more
      conservative starting point; a finite-value guard below (see (g))
      also protects against any remaining divergence regardless of this
      setting, by discarding a bad iteration instead of propagating it.
    """
    if quantile_kwargs is None:
        quantile_kwargs = dict(min_exceed=20, fallback_empirical=True) \
            if quantile_fn is pot_gpd_quantile else {}

    # (a) initialize nominal from the OBSTACLE-AWARE go_to_goal controller,
    # not the blind baseline_go_to_goal. baseline_go_to_goal has no
    # obstacle term at all, so when start/obstacle/goal are close to
    # collinear (as in this scenario), the initial reference trajectory
    # can run straight through the obstacle disk itself -- and since the
    # obstacle half-space is a first-order Taylor expansion of the circle
    # taken AT this reference, linearizing around a point already inside
    # the forbidden disk produces a meaningless constraint there. Starting
    # from a reference that already curves away from the obstacle avoids
    # handing the very first SCP iteration a degenerate expansion point.
    Ubar = np.zeros((N, 2))
    x_tmp = x0.copy()
    for k in range(N):
        Ubar[k] = go_to_goal(x_tmp, sc, p)
        x_tmp = unicycle_step_nominal(x_tmp, Ubar[k], p)
    Xbar = rollout_nominal(x0, Ubar, p)

    q_list, fit_info, n_list, b_list = None, None, None, None
    n_scp_done = 0
    last_slack = None

    for scp_it in range(max_scp_iters):
        # (b) linearize dynamics around the CURRENT nominal (Xbar, Ubar).
        # After iteration 0 this is the previous iteration's own solution,
        # re-rolled out through the TRUE nonlinear dynamics below -- that
        # is what makes this SCP rather than a single linearize-and-solve.
        A_list, B_list = [], []
        for k in range(N):
            A, B = linearize_unicycle(Xbar[k], Ubar[k], p)
            A_list.append(A)
            B_list.append(B)

        # (c) choose stabilizing K_k (LQR on each (A_k,B_k))
        Qe = np.diag([5.0, 5.0, 1.0])
        Re = np.diag([1.0, 1.0])
        K_list = [dlqr(A_list[k], B_list[k], Qe, Re) for k in range(N)]

        # (d) linearize obstacle constraint along the CURRENT Xbar -> (n_k, b_k)
        n_list, b_list = linearized_obstacle_halfspace(Xbar, sc)

        # (e) estimate q_k from Monte Carlo error rollouts, using quantile_fn
        q_list, fit_info = estimate_qk_generic(
            A_list, B_list, K_list, n_list, eps=eps, p=p,
            quantile_fn=quantile_fn, M_evt=M_evt, u_quantile=0.75,
            **quantile_kwargs,
        )

        # (f) solve tightened QP for new nominal plan. The obstacle
        # constraint is SOFTENED (heavily-penalized slack instead of a hard
        # constraint) inside solve_tightened_ltv_qp -- see that function's
        # docstring. When the tightened problem is feasible this recovers
        # the identical hard-constraint solution (slack ~0); when it is not
        # (a real, expected occurrence at eps=1e-3 with heavy tails -- see
        # quick_infeasibility_isolate2.py), it returns the least-violating
        # trajectory instead of raising, so we track how much slack was
        # actually used rather than discovering it only via a raised
        # exception and a full fallback to the zero-margin heuristic.
        goal_state = np.array([sc.goal[0], sc.goal[1], 0.0])  # target heading = 0; adjust if needed
        Xbar_qp, Ubar_qp, s_qp = solve_tightened_ltv_qp(
            x0=x0, A_list=A_list, B_list=B_list,
            n_list=n_list, b_list=b_list, q_list=q_list,
            p=p, goal=goal_state,
        )

        # (g) damp the update and re-roll through the TRUE nonlinear
        # dynamics (not the QP's own linear x.value) so the next
        # iteration's linearization is taken around a trajectory that
        # actually satisfies the real unicycle dynamics, and so `delta`
        # below measures real trajectory movement, not linear-model drift.
        #
        # Guard: if this iteration's QP solution is non-finite (NaN/Inf --
        # observed in testing, likely from a near-singular/degenerate
        # linearization on a step where the obstacle constraint is very
        # tight) or blows up to an implausible magnitude for this scenario,
        # DISCARD it and stop iterating, keeping the last good (Xbar, Ubar)
        # instead of letting garbage values flow into estimate_qk_generic
        # (which silently produces NaN q_k's with no error, since
        # `-(e[:, :2] @ n_list[k])` on a NaN/Inf n_list just makes more
        # NaNs rather than raising).
        if not (np.all(np.isfinite(Xbar_qp)) and np.all(np.isfinite(Ubar_qp))
                and np.all(np.isfinite(s_qp))):
            break

        Ubar_new = Ubar + step_damping * (Ubar_qp - Ubar)
        Xbar_new = rollout_nominal(x0, Ubar_new, p)

        if not (np.all(np.isfinite(Xbar_new)) and np.all(np.isfinite(Ubar_new))):
            break

        delta = np.max(np.abs(Xbar_new - Xbar))
        Xbar, Ubar = Xbar_new, Ubar_new
        n_scp_done = scp_it + 1
        last_slack = s_qp
        if delta < scp_tol:
            break

    max_slack = float(np.max(last_slack)) if last_slack is not None else 0.0
    return Ubar[0], dict(Xbar=Xbar, Ubar=Ubar, q_list=q_list, fit_info=fit_info,
                         n_list=n_list, b_list=b_list, scp_iters=n_scp_done,
                         max_slack=max_slack)

def simulate_evt_tube_smpc(
    x0,
    sc: Scenario,
    p: Robot_Disturbance_Params,
    T_steps=100,          # number of closed-loop steps
    N=12,                 # MPC horizon
    eps=1e-3,
    M_evt=1500,
    rng_seed=None,
    store_debug=True,
    debug_stride=1,       # store dbg every k steps (1 = every step)
    quantile_fn=pot_gpd_quantile,
    quantile_kwargs=None,
):
    """
    Closed-loop (receding-horizon) simulation:
      - at each step, solve tube SMPC (tightened via quantile_fn) -> get u0
      - apply u0 to stochastic unicycle -> update x
      - repeat for T_steps steps

    quantile_fn: swappable quantile estimator, e.g. pot_gpd_quantile (EVT)
      or gaussian_quantile (naive light-tailed baseline), forwarded to
      one_evt_tube_smpc_step at every MPC solve.

    Returns:
      X: (T_steps+1,3) states
      U: (T_steps,2) controls actually applied
      clr: (T_steps+1,) clearance-to-obstacle-boundary at each state
      violated: (T_steps+1,) bool whether clearance < d_safe (hard safety radius)
      dbg_list: list of debug dicts (optional)
    """
    if rng_seed is None:
        rng_seed = p.seed
    rng = np.random.default_rng(rng_seed)

    X = np.zeros((T_steps + 1, 3))
    U = np.zeros((T_steps, 2))
    clr = np.zeros(T_steps + 1)
    violated = np.zeros(T_steps + 1, dtype=bool)

    X[0] = x0
    clr[0] = distance_to_obstacle(X[0][:2], sc)
    violated[0] = (clr[0] < sc.d_safe)

    dbg_list = [] if store_debug else None
    n_fallback = 0  # count of steps where the MPC solve raised (now rare,
    # since the obstacle constraint is softened inside solve_tightened_ltv_qp
    # -- this should mostly only fire for genuine numerical/solver failures,
    # not ordinary tightened-constraint infeasibility).
    n_soft = 0      # count of steps where the MPC solve SUCCEEDED but had to
    # use nonzero slack on the (softened) tightened obstacle constraint --
    # i.e. the intended eps-margin was knowingly compromised by some amount
    # for that step, even though no exception was raised. Tracked separately
    # from n_fallback so "solver genuinely failed" and "solver succeeded but
    # couldn't fully honor the safety margin" aren't conflated. Both counters
    # are unconditional (cheap) regardless of store_debug, so callers that
    # discard dbg_list for speed (e.g. main_compare_controllers.py's M-trial
    # sweep) can still see how often either is happening -- see
    # quick_infeasibility_isolate2.py for why this can be a real, expected
    # occurrence at eps=1e-3 with heavy-tailed disturbances, not a bug.
    SOFT_SLACK_TOL = 1e-4

    for t in range(T_steps):
        # ---- 1) solve MPC at current state ----
        try:
            u0, dbg = one_evt_tube_smpc_step(
                x0=X[t].copy(), sc=sc, p=p, N=N, eps=eps, M_evt=M_evt,
                quantile_fn=quantile_fn, quantile_kwargs=quantile_kwargs,
            )
            if dbg.get("max_slack", 0.0) > SOFT_SLACK_TOL:
                n_soft += 1
        except Exception as e:
            n_fallback += 1
            # If MPC fails outright (genuine solver/numerical failure --
            # ordinary tightened-constraint infeasibility should no longer
            # raise here, see solve_tightened_ltv_qp's softened constraint),
            # fall back to the OBSTACLE-AWARE go_to_goal, not the blind
            # baseline_go_to_goal. Falling back to the blind controller at
            # exactly the moment the MPC solve fails was silently defeating
            # the whole point of the tightening on those steps.
            u0 = go_to_goal(X[t], sc, p)
            dbg = {"status": "fallback", "error": str(e)}

        U[t] = u0

        # optionally store debug
        if store_debug and (t % debug_stride == 0):
            # keep it light: store q_list + fit methods + first control
            slim_dbg = {
                "t": t,
                "u0": u0.copy(),
                "status": dbg.get("status", "ok"),
            }
            if "q_list" in dbg:
                slim_dbg["q_list"] = np.array(dbg["q_list"]).copy()
            if "fit_info" in dbg:
                slim_dbg["fit_methods"] = [d.get("method", None) for d in dbg["fit_info"]]
            if "error" in dbg:
                slim_dbg["error"] = dbg["error"]
            dbg_list.append(slim_dbg)

        # ---- 2) apply u0 to the *stochastic* plant ----
        X[t + 1] = unicycle_step_stochastic(X[t], U[t], p, rng)

        # ---- 3) log safety metrics ----
        clr[t + 1] = distance_to_obstacle(X[t + 1][:2], sc)
        violated[t + 1] = (clr[t + 1] < sc.d_safe)

    return X, U, clr, violated, dbg_list, n_fallback, n_soft