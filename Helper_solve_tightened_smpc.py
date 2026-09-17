import numpy as np
from Helper_Sys_setup import Robot_Disturbance_Params
import cvxpy as cp

# ----------------------------
# 7) Solve tightened linear MPC (QP) on LTV model
# ----------------------------
def solve_tightened_ltv_qp(x0, A_list, B_list, n_list, b_list, q_list, p: Robot_Disturbance_Params,
                           goal=None, Q=None, R=None, slack_penalty=2000.0):
    """
    Decision variables: x_k (k=0..N), u_k (k=0..N-1), s_k (k=0..N, slack >= 0)
    Dynamics: x_{k+1} = A_k x_k + B_k u_k
    SOFTENED tightened obstacle constraint (half-space):
        n_k^T p_k >= b_k + q_k - s_k,  s_k >= 0

    Why softened: at eps=1e-3 with heavy-tailed disturbances, q_k can be a
    large fraction of the scenario's actual geometric clearance budget
    (confirmed via quick_infeasibility_isolate2.py -- q_k~0.3-0.55m against
    ~0.4-0.9m of available clearance), so the exact tightened constraint is
    sometimes genuinely infeasible from states the closed loop visits (not
    a bug -- a real consequence of asking for a 1-in-1000 margin in a tight
    space). Previously this raised RuntimeError and the caller fell back
    entirely to the untightened go_to_goal heuristic for that step (no
    margin enforcement at all). Adding a heavily-penalized slack instead
    means: when the tightened problem IS feasible, `slack_penalty` is large
    enough (relative to the O(1-10) scale of the tracking cost Q,R here)
    that the solver drives s_k to (numerically) zero and recovers the exact
    same solution as before -- this is the standard "exact penalty" result
    for a linear slack penalty. When it is NOT feasible, the solver instead
    returns the trajectory that violates the tightened margin by the LEAST
    amount necessary (rather than failing outright), which is a strictly
    better fallback than reverting to the zero-margin heuristic. The
    returned slack values let the caller detect and log exactly when/how
    much the intended eps-margin was compromised, instead of that
    information being silently lost inside a generic exception.

    goal: target state (3,) [x, y, theta] to track. If None, defaults to
        the origin (backward-compatible with earlier behavior, but note
        that regulating to the origin does NOT drive the robot toward the
        scenario's actual goal -- pass sc.goal explicitly in practice).

    Returns: Xbar_new, Ubar_new, slack  (slack: (N+1,) array, ~0 wherever
        the tightened constraint was actually satisfiable)
    """
    N = len(A_list)  # horizon length
    nx = 3
    nu = 2

    if Q is None:
        Q = np.diag([1.0, 1.0, 0.1])
    if R is None:
        R = np.diag([0.5, 0.2])

    if goal is None:
        goal = np.zeros(nx)
    goal = np.asarray(goal).reshape(nx)

    x = cp.Variable((nx, N+1))
    u = cp.Variable((nu, N))
    s = cp.Variable(N+1, nonneg=True)

    constraints = []
    constraints += [x[:, 0] == x0]

    # dynamics + bounds + tightened constraints
    for k in range(N):
        constraints += [x[:, k+1] == A_list[k] @ x[:, k] + B_list[k] @ u[:, k]]
        constraints += [cp.abs(u[0, k]) <= p.v_max]
        constraints += [cp.abs(u[1, k]) <= p.w_max]

    # obstacle half-space constraints at k=0..N, softened by s_k >= 0:
    for k in range(N+1):
        # n_k^T [x;y] >= b_k + q_k - s_k
        constraints += [n_list[k] @ x[0:2, k] >= b_list[k] + q_list[k] - s[k]]

    # cost: track the goal, not the origin; heavily penalize any slack use
    cost = slack_penalty * cp.sum(s)
    for k in range(N):
        cost += cp.quad_form(x[:, k] - goal, Q) + cp.quad_form(u[:, k], R)
    cost += cp.quad_form(x[:, N] - goal, Q)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"QP failed: {prob.status}")

    return x.value.T, u.value.T, s.value  # Xbar_new, Ubar_new, slack