import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from scipy.stats import t as student_t
from scipy.stats import genpareto
from scipy.linalg import solve_discrete_are
from Helper_Sys_setup import wrap_angle, sample_student_t_noise, Robot_Disturbance_Params
from Helper_nominal import unicycle_step_nominal 
from Helper_linearization import linearize_unicycle
from Helper_simulate_evt_tube_smpc import simulate_evt_tube_smpc
# ----------------------------
# 1) Setting 
# ----------------------------
@dataclass
class Robot_Disturbance_Params:
    dt: float = 0.1
    v_max: float = 1.0
    w_max: float = 1.5
    nu: float = 3.0
    sigma_xy: float = 0.02
    sigma_th: float = 0.01
    seed: int = 1

# ----------------------------
@dataclass
class Scenario:
    goal: np.ndarray
    obstacle_center: np.ndarray
    obstacle_radius: float = 0.6
    d_safe: float = 0.8

 

 
# ----------------------------
# ----------------------------
if __name__ == "__main__":
    p = Robot_Disturbance_Params(dt=0.1, v_max=1.0, w_max=1.5, nu=3.0,
                       sigma_xy=0.03, sigma_th=0.01, seed=1)
    sc = Scenario(goal=np.array([6.0, 0.0]),
                  obstacle_center=np.array([3.0, 0.0]),
                  obstacle_radius=0.6, d_safe=0.8)

    x0 = np.array([0.0, -1.2, 0.2])

    X, U, clr, violated, dbg_list, n_fallback, n_soft = simulate_evt_tube_smpc(
        x0=x0, sc=sc, p=p,
        T_steps=80,   # 8 seconds
        N=12,
        eps=1e-3,
        M_evt=1500,
        store_debug=True,
        debug_stride=1
    )

    print("Done.")
    print("Any violation?", bool(np.any(violated)))
    print("Minimum clearance-to-boundary:", float(np.min(clr)))
    print("Min (clearance - d_safe):", float(np.min(clr - sc.d_safe)))
    print(f"MPC solve fallback rate: {n_fallback}/80 steps ({n_fallback/80:.1%})")
    print(f"Soft-constraint (slack used) rate: {n_soft}/80 steps ({n_soft/80:.1%})")
    print("First 5 controls:\n", U[:5])

    # Sanity check: is pot_gpd_quantile actually doing a real GPD tail fit,
    # or silently falling back to a plain empirical quantile (which happens
    # if a horizon step has too few exceedances above the u_quantile=0.75
    # threshold, per Helper_Monte_Carlo.pot_gpd_quantile's fallback_empirical
    # path)? If it's mostly/always falling back, then despite calling it
    # "evt/GPD" the result is really just an empirical-quantile estimate in
    # disguise -- worth knowing given main_compare_controllers.py's evt
    # result came out nearly identical to the oracle_empirical_quantile run.
    if dbg_list:
        from collections import Counter
        method_counts = Counter()
        for d in dbg_list:
            for m in d.get("fit_methods", []):
                if m is not None:
                    method_counts[m] += 1
        print(f"\nQuantile fit method tally across all steps/horizon-k "
              f"(gpd = genuine tail fit, empirical_fallback = too few "
              f"exceedances, fell back to plain empirical quantile):")
        for method, count in method_counts.items():
            print(f"  {method:20s}: {count}")


