from Helper_Sys_setup import Robot_Disturbance_Params, Scenario, robot_move_one_step, wrap_angle
from Helper_safety_metric import distance_to_obstacle
import numpy as np



def go_to_goal(x, sc: Scenario, p: Robot_Disturbance_Params,
                             k_v=0.8, k_w=2.0, k_rep=3.0, rep_range=4.0,
                             v_slow_frac=0.7):
    """
      - go-to-goal
      - obstacle repulsion adds heading bias when within rep_range
      - slows down while repulsion is active, shrinking the turning
        radius (v/w_max) exactly when a tight avoidance turn is needed

    Returns u = [v, w]

    Defaults were retuned (from k_rep=0.8, rep_range=1, no speed
    throttling) after finding that the old defaults left this reference
    trajectory clipping the obstacle disk entirely (min clearance as low
    as -0.6, i.e. literally inside the obstacle) from this scenario's
    actual start state, regardless of horizon length -- see
    quick_rep_range_sweep.py. Root cause: rep_range=1 means avoidance only
    activated once distance-to-center < 1.6, already deep inside the
    tightened R_safe=1.4 buffer, too late for the w_max-limited turn rate
    to react; and even once repulsion was strengthened, the robot's
    minimum turning radius (v_max/w_max) at full speed was still too wide
    to clear d_safe -- confirmed by min clearance plateauing at exactly
    0.794 (just under the 0.8 needed) across many (k_rep, rep_range)
    combinations once repulsion was otherwise strong enough. Slowing down
    near the obstacle (v_slow_frac) breaks that plateau and clears d_safe
    with real margin (~1.0-1.14 vs the 0.8 needed) rather than a knife's
    edge. Set v_slow_frac=0 to recover the pre-throttling behavior.
    """
    pos = x[:2]
    theta = x[2]

    # goal direction
    to_goal = sc.goal - pos
    goal_dist = np.linalg.norm(to_goal) + 1e-12
    goal_dir = to_goal / goal_dist
    theta_goal = np.arctan2(goal_dir[1], goal_dir[0])

    # obstacle direction (push away from obstacle center)
    to_obs = pos - sc.obstacle_center
    obs_dist_center = np.linalg.norm(to_obs) + 1e-12
    obs_dir = to_obs / obs_dist_center
    theta_rep = np.arctan2(obs_dir[1], obs_dir[0])

    # repulsion weight (only active near obstacle)
    clearance = distance_to_obstacle(x, sc)  # distance to obstacle boundary
    # map clearance to repulsion weight in [0,1]
    if clearance < rep_range:
        w_rep = np.clip((rep_range - clearance) / rep_range, 0.0, 1.0)
    else:
        w_rep = 0.0

    # obtain desired heading
    # do not blend angle directly to avoid angle discontinuity
    v_goal = np.array([np.cos(theta_goal), np.sin(theta_goal)])
    v_rep = np.array([np.cos(theta_rep), np.sin(theta_rep)])
    v_des = (1.0 * v_goal) + (k_rep * w_rep) * v_rep # k_rep determines how much the robot prioritizes safety vs efficiency
    th_des = np.arctan2(v_des[1], v_des[0])

    # control laws
    heading_err = wrap_angle(th_des - theta)
    v_cmd = np.clip(k_v * goal_dist, 0.0, p.v_max) * (1.0 - v_slow_frac * w_rep)
    w_cmd = np.clip(k_w * heading_err, -p.w_max, p.w_max)
    return np.array([v_cmd, w_cmd])

def baseline_go_to_goal(x, sc: Scenario, p: Robot_Disturbance_Params, k_v=0.8, k_w=2.0):
    pos = x[:2]
    th = x[2]
    to_goal = sc.goal - pos
    dist = np.linalg.norm(to_goal) + 1e-12
    th_goal = np.arctan2(to_goal[1], to_goal[0])
    heading_err = wrap_angle(th_goal - th)
    v_cmd = np.clip(k_v * dist, 0.0, p.v_max)
    w_cmd = np.clip(k_w * heading_err, -p.w_max, p.w_max)
    return np.array([v_cmd, w_cmd])