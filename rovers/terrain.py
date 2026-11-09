"""Stage 2a: collect the terrain the rover will drive over.

The stage-1 controller drives in a (near) straight line from the landing site
to the sample, facing the sample. So the terrain under the rover is fixed by
geometry alone: it does not depend on the masks. That is what separates the
mask design from the controller design.

We sample the 11x11 terrain patch (``mode_view``, rotated to the heading) every
``spacing`` pixels along each of the 30 segments, using the official
``extract_local_view``. Equal spacing matters: the time to cross a stretch of
length ds is ds / (v_max * nu), so the mean of 1/nu over equally spaced patches
is proportional to the total driving time.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from math import atan2

import numpy as np

from .official import _init_worker, load, scenarios


def segment_patches(map_id: int, scenario_id: int, spacing: float = 6.0) -> np.ndarray:
    """All 11x11 patches (flattened to 121) along one start->sample segment."""
    import torch

    mod = load()
    start, goal = mod.SCENARIO_POSITIONS[map_id][scenario_id].double().view(2, 2)
    delta = goal - start
    length = float(delta.norm())
    heading = atan2(float(delta[1]), float(delta[0]))
    steps = int(length // spacing) + 1
    patches = np.empty((steps, mod.MASK_SIZE * mod.MASK_SIZE), dtype=np.float32)
    for i in range(steps):
        position = (start + delta * min(1.0, i * spacing / length)).float()
        _, mode_view = mod.udp.env.extract_local_view(position, heading, map_id)
        patches[i] = mode_view.flatten().numpy()
    return patches


def _segment_task(args: tuple[int, int, float]) -> np.ndarray:
    return segment_patches(*args)


def sample_all(spacing: float = 6.0, evaluator=None) -> tuple[np.ndarray, np.ndarray]:
    """Patches for all 30 scenarios, plus the scenario index (0..29) of each patch.

    Pass an ``Evaluator`` to reuse its worker pool (saves the start-up time)."""
    tasks = [(m, s, spacing) for (m, s) in scenarios()]
    if evaluator is not None:
        per_segment = evaluator.map(_segment_task, tasks)
    else:
        with ProcessPoolExecutor(initializer=_init_worker) as pool:
            per_segment = list(pool.map(_segment_task, tasks))
    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(per_segment)])
    return np.concatenate(per_segment), owner
