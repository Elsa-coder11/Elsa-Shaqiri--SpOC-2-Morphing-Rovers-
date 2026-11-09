"""Access to ESA's official evaluation code, plus a parallel evaluator.

The official file ``external/morphing_udp.py`` is used unmodified. This module
  * imports it from the repository root (it reads ``./data/...`` at import time), and
  * runs the 30 independent scenarios in parallel worker processes. Every scenario
    goes through the official ``run_single_scenario``, so the score is the same as
    ``udp.fitness(x)`` (up to float rounding in the final sum).
"""
from __future__ import annotations

import importlib
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "spoc2" / "morphing"
UDP_FILE = ROOT / "external" / "morphing_udp.py"


@lru_cache(maxsize=None)
def load():
    """Import and return the official ``morphing_udp`` module (loads the maps once)."""
    if not UDP_FILE.exists():
        raise SystemExit("Official challenge files are missing. Run: python scripts/fetch_challenge.py")
    os.environ.setdefault("MPLBACKEND", "Agg")  # the UDP imports pyplot; no GUI needed
    sys.path.insert(0, str(UDP_FILE.parent))
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # imageio v2-API deprecation notice
            module = importlib.import_module("morphing_udp")
    finally:
        os.chdir(cwd)
    module.PATH = str(DATA_DIR)  # so udp.example() works from any working directory
    _numpy2_compatibility(module)
    return module


def _numpy2_compatibility(module) -> None:
    """Restore the NumPy 1.x arithmetic the official code was written for.

    The official rover does ``angle += MAX_DA * angular_change.numpy()[0]``. Under
    NumPy 1.x (2023), ``python_float * np.float32`` was promoted to float64. Under
    NumPy 2 (NEP 50) it stays float32, and current torchvision then rejects the
    float32 angle in ``rotate`` with "Argument angle should be int or float". Making
    the steering output float64 (an exact conversion) reproduces the original float64
    result and changes nothing else.
    """
    if getattr(module.Controller, "_numpy2_patched", False):
        return
    original_forward = module.Controller.forward

    def forward(self, landscape, state, past_inp):
        mode_command, angle_command, latent = original_forward(self, landscape, state, past_inp)
        return mode_command, angle_command.double(), latent

    module.Controller.forward = forward
    module.Controller._numpy2_patched = True


def scenarios() -> list[tuple[int, int]]:
    mod = load()
    return [(m, s) for m in range(mod.MAPS_PER_EVALUATION) for s in range(mod.SCENARIOS_PER_MAP)]


def official_fitness(x: Sequence[float]) -> float:
    """The exact official call, single process: ``udp.fitness(x)[0]``."""
    return float(load().udp.fitness(np.asarray(x, dtype=float))[0])


def run_scenario(x: Sequence[float], map_id: int, scenario_id: int) -> float:
    """Score of one scenario: (1 + d/d0) * T/T_min, exactly as in ``udp.fitness``."""
    mod = load()
    rover = mod.Rover(np.asarray(x, dtype=float))
    distance, time = mod.udp.run_single_scenario(rover, map_id, scenario_id)
    return float((1 + distance) * time)


def record_scenario(x: Sequence[float], map_id: int, scenario_id: int):
    """Run one scenario through the official loop and log, for every time step,
    the terrain patch under the rover, the form in use and the velocity factor.

    Returns (score, patches [steps, 121], modes [steps], nu [steps]).
    """
    mod = load()
    log: list[tuple[np.ndarray, int, float]] = []

    class RecordingRover(mod.Rover):
        def update_rover_state(self, rover_view, mode_view, distance_vector, original_distance):
            used_mode = int(self.current_mode)  # the form that sets this step's speed
            super().update_rover_state(rover_view, mode_view, distance_vector, original_distance)
            log.append((mode_view.flatten().numpy().copy(), used_mode, float(self.mode_efficiency)))

    rover = RecordingRover(np.asarray(x, dtype=float))
    distance, time = mod.udp.run_single_scenario(rover, map_id, scenario_id)
    patches = np.stack([entry[0] for entry in log]).astype(np.float32)
    modes = np.array([entry[1] for entry in log])
    nu = np.array([entry[2] for entry in log])
    return float((1 + distance) * time), patches, modes, nu


def _record_task(task: tuple[np.ndarray, int, int]):
    return record_scenario(*task)


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)  # tiny tensors: one thread per process is fastest
    load()


def _run_task(task: tuple[np.ndarray, int, int]) -> float:
    return run_scenario(*task)


class Evaluator:
    """Evaluates chromosomes by spreading the 30 scenarios over worker processes."""

    def __init__(self, workers: int | None = None):
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self._pool = ProcessPoolExecutor(self.workers, initializer=_init_worker)
        self.evaluations = 0

    def scenario_scores(self, x: Sequence[float]) -> np.ndarray:
        return self.many_scenario_scores([x])[0]

    def fitness(self, x: Sequence[float]) -> float:
        return float(self.scenario_scores(x).mean())

    def many(self, xs: Iterable[Sequence[float]]) -> list[float]:
        return [float(s.mean()) for s in self.many_scenario_scores(xs)]

    def many_scenario_scores(self, xs: Iterable[Sequence[float]]) -> list[np.ndarray]:
        """Score several chromosomes at once; all their scenarios share the worker pool."""
        xs = [np.asarray(x, dtype=float) for x in xs]
        cases = scenarios()
        tasks = [(x, m, s) for x in xs for (m, s) in cases]
        chunk = max(1, len(tasks) // (4 * self.workers))
        flat = list(self._pool.map(_run_task, tasks, chunksize=chunk))
        self.evaluations += len(xs)
        n = len(cases)
        return [np.array(flat[i * n:(i + 1) * n]) for i in range(len(xs))]

    def map(self, fn, tasks):
        """Run any picklable top-level function on the worker pool."""
        return list(self._pool.map(fn, tasks))

    def record(self, x: Sequence[float]):
        """Fitness plus the per-step log of every scenario (see ``record_scenario``)."""
        return self.record_many([x])[0]

    def record_many(self, xs: Iterable[Sequence[float]]):
        """``record`` for several chromosomes at once (their scenarios share the pool)."""
        xs = [np.asarray(x, dtype=float) for x in xs]
        cases = scenarios()
        flat = list(self._pool.map(_record_task, [(x, m, s) for x in xs for (m, s) in cases]))
        self.evaluations += len(xs)
        n = len(cases)
        out = []
        for i in range(len(xs)):
            results = flat[i * n:(i + 1) * n]
            out.append((float(np.mean([r[0] for r in results])), results))
        return out

    def close(self) -> None:
        self._pool.shutdown()

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
