"""Stage 2b: design the 4 rover forms by clustering terrain patches.

With "morph to the best form" switching, the rover's speed on a patch T is
max_i V(F_i, T), where V is the official velocity function. Its total
driving time is therefore proportional to

    J(F_1..F_k) = mean_j  1 / max_i V(F_i, T_j)          (over terrain patches T_j)

Minimising J is a k-clustering problem: each form is the "centre" of the
terrain it handles best. We solve it with Lloyd's algorithm, i.e. k-means with
the official velocity in place of Euclidean distance:

  * seeding  k-means++: each new centre is drawn with probability proportional
             to the squared extra time (1/nu - 1) of the patch under the
             centres chosen so far
  * assign   each patch goes to the form that is fastest on it        O(n k d)
  * update   each form minimises its cluster's mean 1/V by Adam
             gradient steps, started from the cluster mean        O(steps n_i d)
             The update is kept only if it lowers that cluster's cost.

The assign step cannot increase J, and the update step only accepts lower
cluster costs, so J never increases. J >= 1, so the algorithm converges, as
ordinary k-means does. A few random restarts guard against bad local optima.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

EPS_C = 0.03 ** 2          # same constant as the official velocity_function
CENTRE = 60                # index of the centre element in a flattened 11x11 mask
LOWER, UPPER, CENTRE_MIN = -100.0, 100.0, 1e-16


def velocity(forms: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    """Batched copy of the official ``velocity_function``: [n, d] x [k, d] -> [n, k]."""
    f_norm = forms.norm(dim=1)
    t_norm = patches.norm(dim=1)
    fn, tn = f_norm[None, :], t_norm[:, None]
    luminance = (2 * fn * tn + EPS_C) / (fn ** 2 + tn ** 2 + EPS_C)
    cosine = (patches / tn) @ (forms / f_norm[:, None]).T
    unit_distance = torch.sqrt(torch.clamp(2 - 2 * cosine, min=1e-12))  # ||f/|f| - t/|t|||
    metric = (2 - unit_distance) * 0.5 * luminance.sqrt()
    x = (1 - metric) * 2
    return 1.0 / (1 + x ** 3)


def travel_cost(forms: torch.Tensor, patches: torch.Tensor) -> float:
    """J: mean of 1/nu with the best form on every patch (1.0 = top speed everywhere)."""
    with torch.no_grad():
        return float((1.0 / velocity(forms, patches).max(dim=1).values).mean())


def _project(form: torch.Tensor) -> None:
    form.clamp_(LOWER, UPPER)
    form[CENTRE].clamp_(CENTRE_MIN, UPPER)


def _kmeans_pp(patches: torch.Tensor, k: int, rng: np.random.Generator) -> torch.Tensor:
    n = len(patches)
    centres = [patches[rng.integers(n)]]
    for _ in range(1, k):
        nu = velocity(torch.stack(centres), patches).max(dim=1).values
        weight = ((1.0 / nu - 1.0) ** 2).double().numpy()
        centres.append(patches[rng.choice(n, p=weight / weight.sum())])
    return torch.stack(centres).clone()


def _refine(form: torch.Tensor, cluster: torch.Tensor, steps: int, lr: float) -> tuple[torch.Tensor, float]:
    """Gradient descent on the cluster's mean 1/V(form, T), with box projection."""
    form = form.clone().requires_grad_(True)
    optimiser = torch.optim.Adam([form], lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        loss = (1.0 / velocity(form[None, :], cluster)[:, 0]).mean()
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            _project(form)
    form = form.detach()
    with torch.no_grad():
        cost = float((1.0 / velocity(form[None, :], cluster)[:, 0]).mean())
    return form, cost


@dataclass
class MaskResult:
    masks: np.ndarray          # [k, 11, 11]
    cost: float                # J on the training patches
    iterations: int
    history: list[float]
    usage: np.ndarray          # share of patches served by each form


def _assign(forms: torch.Tensor, patches: torch.Tensor, window: torch.Tensor | None) -> torch.Tensor:
    """Fastest form per patch, or per window of consecutive patches (one form per window)."""
    with torch.no_grad():
        nu = velocity(forms, patches)
        if window is None:
            return nu.argmax(dim=1)
        time = torch.zeros(int(window.max()) + 1, forms.shape[0]).index_add_(0, window, 1.0 / nu)
        return time.argmin(dim=1)[window]


def windowed_cost(forms: torch.Tensor, patches: torch.Tensor, window: torch.Tensor | None) -> float:
    with torch.no_grad():
        chosen = _assign(forms, patches, window)
        nu = velocity(forms, patches).gather(1, chosen[:, None])[:, 0]
        return float((1.0 / nu).mean())


def lloyd(patches: torch.Tensor, k: int, rng: np.random.Generator, max_iter: int = 30,
          steps: int = 150, lr: float = 0.05, tol: float = 1e-4,
          window: torch.Tensor | None = None) -> MaskResult:
    forms = _kmeans_pp(patches, k, rng)
    history = [windowed_cost(forms, patches, window)]
    for iteration in range(1, max_iter + 1):
        assignment = _assign(forms, patches, window)
        for i in range(k):
            cluster = patches[assignment == i]
            if len(cluster) == 0:  # empty cluster: re-seed on the slowest patch
                worst = velocity(forms, patches).max(dim=1).values.argmin()
                forms[i] = patches[worst]
                continue
            with torch.no_grad():
                current = float((1.0 / velocity(forms[i][None, :], cluster)[:, 0]).mean())
            start = cluster.mean(dim=0)
            _project(start)
            candidate, cost = _refine(start, cluster, steps, lr)
            if cost < current:
                forms[i] = candidate
        history.append(windowed_cost(forms, patches, window))
        if history[-2] - history[-1] < tol:
            break
    best = _assign(forms, patches, window)
    usage = np.bincount(best.numpy(), minlength=k) / len(patches)
    order = np.argsort(-usage)  # mask 0 = most used form (the rover starts in mask 0)
    masks = forms.numpy()[order].reshape(k, 11, 11)
    return MaskResult(masks, history[-1], iteration, history, usage[order])


def make_windows(owner: np.ndarray, length: int) -> np.ndarray:
    """Window id per patch: consecutive runs of ``length`` patches within each segment."""
    ids = np.empty(len(owner), dtype=np.int64)
    next_id = 0
    for segment in np.unique(owner):
        where = np.flatnonzero(owner == segment)
        ids[where] = next_id + np.arange(len(where)) // length
        next_id = int(ids[where].max()) + 1
    return ids


def fit_masks(patches: np.ndarray, k: int = 4, restarts: int = 4, seed: int = 0,
              owner: np.ndarray | None = None, window_length: int = 1, **kwargs) -> MaskResult:
    """Best of ``restarts`` Lloyd runs (different k-means++ seeds).

    With ``window_length > 1`` (and ``owner``, the segment of each patch), every run of
    that many consecutive patches must share one form. This models the rover's
    morphing cooldown, during which it cannot change form.
    """
    torch.manual_seed(seed)
    data = torch.as_tensor(patches, dtype=torch.float32)
    window = None
    if window_length > 1:
        assert owner is not None, "window clustering needs the segment id of every patch"
        window = torch.as_tensor(make_windows(owner, window_length))
    rng = np.random.default_rng(seed)
    best: MaskResult | None = None
    for _ in range(restarts):
        result = lloyd(data, k, rng, window=window, **kwargs)
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    return best
