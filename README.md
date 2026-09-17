# extreme-rare-events-for-robot-safety

Does a theoretical formula for how rare events cluster in time actually hold
up on a real (simulated) control system, or only on toy math? This repo
answers that on a robot navigating past an obstacle under heavy-tailed
disturbances: it derives a closed-loop safety-violation clustering rate in
closed form, checks it against a purely data-driven estimator, and shows
what happens to the robot's actual safety margin if you ignore clustering
versus correct for it.

## Why does this matter for a robot?

A safety-critical controller (e.g. an obstacle-avoidance MPC) is usually
tuned so that the *per-step* probability of a safety violation is below some
small target $\epsilon$. But a single disturbance doesn't just affect one
time step — its effect persists through the robot's closed-loop dynamics for
several subsequent steps. That means violations don't happen independently:
one near-miss makes the *next* step's near-miss more likely too, so
violations arrive in short clusters rather than scattering uniformly in
time.

This is exactly the extremal index θ ∈ (0,1] idea (see
[extreme-value-clustering-in-time-series](https://github.com/) for a
from-scratch explanation of what θ means in general). Here we go one step
further: instead of illustrating θ on a synthetic process, we derive it
**from a real closed-loop robot system's own dynamics**, and check that the
formula actually predicts what the robot does.

## The setup

A unicycle robot (state = position + heading) is controlled to pass a
circular obstacle at a fixed tightened stand-off distance, heading
tangentially around it — the exact operating point where a safety-margin
violation is most likely to occur. Linearizing the robot's dynamics and its
feedback control law around that point gives a closed-loop error dynamics
matrix $A_K$ (via a standard LQR gain), so the tracking error $e_k$ driven by
disturbances $w_k$ evolves as

```
e_{k+1} = A_K e_k + w_k,      w_k = zeta_k * b,      Y_k = c^T e_k
```

where $\zeta_k$ are i.i.d. heavy-tailed (Student-*t*) disturbances acting
along a fixed direction $b$, and $Y_k = c^T e_k$ is the safety-relevant
projection of the error onto the obstacle-normal direction $c$ — i.e., how
close the robot's actual trajectory drifts toward violating the safety
boundary at step $k$.

Because $A_K$ comes directly from this robot's own linearized dynamics
(not a made-up matrix), the resulting closed-form extremal index
$\theta_Y$ is a genuine prediction about *this specific robot's* clustering
behavior — not just a demonstration that the math is self-consistent.

## What's validated, and how

`validate_theta_unicycle.py` computes $\theta_Y$ two completely independent
ways and compares them:

1. **Closed form** — directly from $A_K$, $b$, $c$, and the disturbance's
   tail parameters, with no simulation involved.
2. **Data-driven** — the **Ferro–Segers (2003) intervals estimator**, which
   estimates θ purely from the *gaps between exceedance times* in simulated
   trajectories of $Y_k$, with no knowledge of $A_K$ or the model at all.
   Averaged over several long, independent trajectories per threshold level
   (the estimator is noisy at any single threshold, so multiple independent
   runs are needed to trust the comparison).

Agreement between the two — across thresholds ranging from relatively
common exceedances (1 in ~12) down to genuinely rare ones (1 in 5,000) — is
the validation that the closed-form formula isn't just internally
consistent, but actually describes this robot's real clustering behavior.

`demo_theta_corrected_tightening.py` then asks the practical follow-up
question: *so what?* Using this robot's own validated $\theta_Y$, it
compares a naive safety margin (sized only for the per-step target
$\epsilon$) against a θ-corrected one (sized for the stricter target
$\epsilon_\star = -\ln(1-\epsilon)/(\theta_Y T)$ over a horizon of length
$T$), and quantifies the difference directly: how much more often does the
robot actually experience a violation *episode* over a full run with the
naive margin, versus the corrected one? (Short answer, at this robot's own
parameters: the naive margin experiences violation episodes several times
more often than intended; the θ-corrected margin lands almost exactly on
target.)

## Install

```bash
pip install -r requirements.txt
```

No `scipy`/`cvxpy` dependency — everything, including the discrete
algebraic Riccati equation for the LQR gain, is solved with `numpy` +
`matplotlib` alone, so this runs anywhere Python + those two packages do.

## Contents

- `validate_theta_unicycle.py` → `theta_unicycle_validation.{png,pdf}`.
  Derives $A_K$ from the robot's linearized dynamics at the critical
  operating point, computes $\theta_Y$ in closed form, then cross-checks it
  against the Ferro–Segers estimator as described above.
- `demo_theta_corrected_tightening.py` → `theta_corrected_tightening_demo.png`.
  Quantifies the θ-correction's real effect on this robot: naive vs.
  corrected episode-level violation probability and expected consecutive-
  violation run length.

## Run

```bash
python validate_theta_unicycle.py
python demo_theta_corrected_tightening.py
```

`validate_theta_unicycle.py` runs several independent long simulated
trajectories for the Ferro–Segers cross-check and takes roughly 1–2 minutes.

## Related

- [smpc-evt-tube](https://github.com/) — the tube-SMPC controller and
  obstacle-avoidance scenario this validation is built on (naive per-step
  tightening only, no clustering correction).
- [extreme-value-clustering-in-time-series](https://github.com/) — a
  from-scratch, paper-independent explanation of the extremal index concept
  on a simple synthetic model, before applying it here to a real system.

## Background

This code accompanies *"Stochastic MPC under Heavy-Tailed Disturbances: An
Extreme Value Theory Approach"* (X. Ye and W. Tang), which develops the
closed-form extremal index result and the θ-corrected constraint tightening
proved here, and integrates them into the full tube-SMPC controller in
[smpc-evt-tube](https://github.com/).
