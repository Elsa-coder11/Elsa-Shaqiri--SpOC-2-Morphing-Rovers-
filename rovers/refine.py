"""Stage 3: refine the forms under the real switching rules (Lloyd with the simulator in the loop).

The clustering in stage 2 assumes the rover always uses the best form on every
patch. The real rover can't:
  * the form is chosen on the patch it is leaving, one step before it is used, and
  * after a morph it is locked in that form for 9 steps (cooldown).
So the terrain a form actually drives on is not the terrain it was fitted to.

The fix keeps Lloyd's structure but lets the official simulator do the
assignment step:

  assign  run the 30 scenarios and log, for every step t, the patch T_t, the
          form m_t in use and its speed nu_t
  update  for every form i:  F_i* = argmin_F  sum_{t: m_t = i}  nu_t / V(F, T_t)

The weight nu_t is the distance step t covered (in units of v_max), so
nu_t / V(F, T_t) is the time the new form F needs for that same distance. The
update minimises a first-order model of the total mission time, with the
switching decisions held fixed. Those decisions do change once the forms change,
so a full step can overshoot. We therefore take a damped step with a line search:

  candidates  F + eta (F* - F)  for eta in {1, 1/2, 1/4}, scored together by the
              official simulator (one batch on the worker pool)
  accept      the best candidate if it beats the current fitness; otherwise stop

Every accepted step lowers the official fitness, so the sequence is monotone and
the result is never worse than the stage-2 starting point.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .controller import assemble
from .masks import _project, velocity
from .official import Evaluator

STEP_SIZES = (1.0, 0.5, 0.25)


def _update_form(form: np.ndarray, patches: torch.Tensor, weights: torch.Tensor,
                 steps: int, lr: float) -> np.ndarray:
    f = torch.as_tensor(form.reshape(-1), dtype=torch.float32).clone().requires_grad_(True)
    optimiser = torch.optim.Adam([f], lr=lr)

    def cost(v: torch.Tensor) -> torch.Tensor:
        return (weights / velocity(v[None, :], patches)[:, 0]).sum()

    with torch.no_grad():
        start_cost = float(cost(f))
    for _ in range(steps):
        optimiser.zero_grad()
        loss = cost(f)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            _project(f)
    with torch.no_grad():
        if float(cost(f)) >= start_cost:
            return form
    return f.detach().numpy().reshape(form.shape)


def update_step(masks: np.ndarray, logs, steps: int = 200, lr: float = 0.05) -> np.ndarray:
    """Lloyd update F* from one simulation log (assignment = forms actually used)."""
    patches = torch.as_tensor(np.concatenate([r[1] for r in logs]))
    modes = np.concatenate([r[2] for r in logs])
    weights = torch.as_tensor(np.concatenate([r[3] for r in logs]), dtype=torch.float32)
    target = masks.copy()
    for i in range(len(masks)):
        chosen = torch.as_tensor(modes == i)
        if int(chosen.sum()) >= 5:
            target[i] = _update_form(masks[i], patches[chosen], weights[chosen], steps, lr)
    return target


@dataclass
class RefineResult:
    masks: np.ndarray
    fitness: float
    history: list[float] = field(default_factory=list)       # accepted official fitness per iteration
    timestamps: list[float] = field(default_factory=list)    # time.perf_counter() per iteration


def refine(masks: np.ndarray, evaluator: Evaluator, switch_threshold: float | None = None,
           iterations: int = 10, steps: int = 200, lr: float = 0.05, verbose: bool = True) -> RefineResult:
    fitness, logs = evaluator.record(assemble(masks, switch_threshold))
    result = RefineResult(masks.copy(), fitness, [fitness], [time.perf_counter()])
    if verbose:
        print(f"  refine start: official fitness {fitness:.4f}", flush=True)
    for iteration in range(1, iterations + 1):
        target = update_step(result.masks, logs, steps, lr)
        candidates = [result.masks + eta * (target - result.masks) for eta in STEP_SIZES]
        scored = evaluator.record_many([assemble(c, switch_threshold) for c in candidates])
        best = int(np.argmin([s[0] for s in scored]))
        new_fitness, new_logs = scored[best]
        if verbose:
            tried = ", ".join(f"eta={eta}: {s[0]:.4f}" for eta, s in zip(STEP_SIZES, scored))
            print(f"  refine iteration {iteration}: {tried}", flush=True)
        if new_fitness >= result.fitness - 1e-6:
            break
        result.masks, result.fitness, logs = candidates[best], new_fitness, new_logs
        result.history.append(new_fitness)
        result.timestamps.append(time.perf_counter())
    return result
