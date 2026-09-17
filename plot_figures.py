import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("data", "comparison_data.npz")
FIG_DIR = "figures"

COLORS = {
    "baseline": "tab:red",
    "gaussian": "tab:orange",
    "evt": "tab:blue",
    "evt_theta": "tab:green",
}
LABELS = {
    "baseline": "Go to goal",
    "gaussian": "Gaussian",
    "evt": "Naive EVT",
    "evt_theta": r"$\theta$-corrected EVT",
}


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found -- run main_compare_controllers.py first "
            f"(it saves this file after running the comparison)."
        )
    d = np.load(DATA_PATH)

    obstacle_center = d["obstacle_center"]
    obstacle_radius = float(d["obstacle_radius"])
    d_safe = float(d["d_safe"])
    goal = d["goal"]
    x0 = d["x0"]
    eps = float(d["eps"]) 
    min_clr_raw = {
        "baseline": d["min_clr_baseline"],
        "gaussian": d["min_clr_gaussian"],
        "evt": d["min_clr_evt"],
        "evt_theta": d["min_clr_evt_theta"],
    }
    n_total = len(min_clr_raw["baseline"])
    min_clr = {name: arr[~np.isnan(arr)] for name, arr in min_clr_raw.items()}
    for name, arr in min_clr.items():
        if len(arr) < n_total:
            print(f"  note: {name} has {len(arr)}/{n_total} valid (non-NaN) trials "
                  f"-- plotting/statistics use only the valid ones.")

    os.makedirs(FIG_DIR, exist_ok=True)

    # ================= Figure 1: scenario + naive-tightening comparison =====
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(11, 5))

    # --- (a) trajectories + obstacle + safety circle ---
    theta_circ = np.linspace(0, 2 * np.pi, 200)
    obs_x = obstacle_center[0] + obstacle_radius * np.cos(theta_circ)
    obs_y = obstacle_center[1] + obstacle_radius * np.sin(theta_circ)
    safe_r = obstacle_radius + d_safe
    safe_x = obstacle_center[0] + safe_r * np.cos(theta_circ)
    safe_y = obstacle_center[1] + safe_r * np.sin(theta_circ)

    ax1a.fill(obs_x, obs_y, color="dimgray", alpha=0.8, label="obstacle")
    ax1a.plot(safe_x, safe_y, "k--", linewidth=1.2, label=f"safe boundary")

    for name in ["baseline", "gaussian", "evt"]:
        X = d[f"X_{name}"]
        ax1a.plot(X[:, 0], X[:, 1], color=COLORS[name], linewidth=1.8,
                  label=LABELS[name])

    ax1a.plot(x0[0], x0[1], "k^", markersize=10, label="start")
    ax1a.plot(goal[0], goal[1], "k*", markersize=14, label="goal")
    ax1a.set_xlabel("x (m)")
    ax1a.set_ylabel("y (m)")  
    ax1a.legend(loc="upper left", fontsize=8, frameon=True)
    ax1a.set_xlim(0, 7)
    ax1a.set_ylim(-2, 4)
    ax1a.grid(alpha=0.3) 

    # --- (b) histogram of min-clearance-over-rollout, 3 controllers ---
    for name in ["baseline", "gaussian", "evt"]:
        ax1b.hist(min_clr[name], bins=20, alpha=0.5, color=COLORS[name],
                  label=LABELS[name])
    ax1b.axvline(d_safe, linestyle="--", color="k", label="safe boundary")
    ax1b.set_xlabel("min distance over rollout (m)")
    ax1b.set_ylabel("count") 
    ax1b.legend(loc="best", fontsize=8)
    ax1b.grid(alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(os.path.join(FIG_DIR, "fig1_scenario_comparison.png"), dpi=300, bbox_inches="tight")
    fig1.savefig(os.path.join(FIG_DIR, "fig1_scenario_comparison.pdf"), bbox_inches="tight")
    print(f"Saved {FIG_DIR}/fig1_scenario_comparison.png / .pdf")

    # ================= Figure 2: theta-correction comparison ================
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 5))

    # --- (a) violation probability bar chart: evt vs evt_theta only ---
    names_theta = ["evt", "evt_theta"]
    p_hat = [(min_clr[n] < d_safe).mean() for n in names_theta]
    bars = ax2a.bar([LABELS[n] for n in names_theta], p_hat,
                     color=[COLORS[n] for n in names_theta], alpha=0.8)
    ax2a.axhline(eps, linestyle="--", color="k", label=f"target eps={eps:.1e}")
    for bar, val in zip(bars, p_hat):
        ax2a.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax2a.set_ylabel("P(min clearance < safe boundary)")  
    ax2a.grid(alpha=0.3, axis="y")

    # --- (b) margin boxplot, all 4 controllers ---
    names_all = ["baseline", "gaussian", "evt", "evt_theta"]
    margins = [min_clr[n] - d_safe for n in names_all]
    bp = ax2b.boxplot(margins, tick_labels=[LABELS[n] for n in names_all],
                       patch_artist=True, showmeans=True,
                       whis=[2.5, 97.5])
    for patch, name in zip(bp["boxes"], names_all):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.6)
    ax2b.axhline(0.0, linestyle="--", color="k")
    ax2b.set_ylabel("Safety Margin Distribution") 
    ax2b.tick_params(axis="x", labelrotation=15)
    ax2b.grid(alpha=0.3, axis="y")

    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG_DIR, "fig2_theta_correction.png"), dpi=300, bbox_inches="tight")
    fig2.savefig(os.path.join(FIG_DIR, "fig2_theta_correction.pdf"), bbox_inches="tight")
    print(f"Saved {FIG_DIR}/fig2_theta_correction.png / .pdf")

    plt.show()


if __name__ == "__main__":
    main()
