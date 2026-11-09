"""Solve SpOC-2 "Morphing Rovers" with the decomposition pipeline.

    Stage 1  controller   write a pure-pursuit steering law into the network    O(1), no search
    Stage 2  forms        Lloyd / k-means of terrain windows under the official
                          velocity function                                    ~20 s
    Stage 3  refinement   Lloyd with the official simulator doing the assignment
                          (elitist: keeps the best officially scored result)    ~10 simulations
    Verify                the unmodified official call  udp.fitness(x)

Usage:
    python solve.py                       # full pipeline, writes results/
    python solve.py --workers 4 --quick   # fewer restarts / iterations
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rovers.controller import assemble
from rovers.masks import fit_masks
from rovers.official import Evaluator, ROOT, official_fitness, scenarios
from rovers.refine import refine
from rovers.terrain import sample_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: CPUs - 1)")
    parser.add_argument("--spacing", type=float, default=6.0, help="terrain sampling step in pixels")
    parser.add_argument("--window", type=int, default=10, help="patches per window (models the cooldown)")
    parser.add_argument("--restarts", type=int, default=4, help="k-means++ restarts")
    parser.add_argument("--iterations", type=int, default=10, help="max refinement iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="2 restarts, 4 refinement iterations")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.quick:
        args.restarts, args.iterations = 2, 4

    args.out.mkdir(parents=True, exist_ok=True)
    timeline: list[dict] = []
    t0 = time.perf_counter()

    def log(stage: str, fitness: float | None = None, **extra) -> None:
        entry = {"stage": stage, "seconds": round(time.perf_counter() - t0, 1), **extra}
        if fitness is not None:
            entry["fitness"] = round(fitness, 4)
        timeline.append(entry)
        print(f"[{entry['seconds']:7.1f}s] {stage}" + (f": fitness {fitness:.4f}" if fitness is not None else ""),
              flush=True)

    with Evaluator(args.workers) as evaluator:
        evaluator.fitness(assemble(np.ones((4, 11, 11))))  # start the workers (each loads the maps)
        log(f"worker pool ready ({evaluator.workers} processes)")

        # Stage 1 + 2: terrain along the pure-pursuit paths, then clustering.
        patches, owner = sample_all(args.spacing, evaluator)
        log(f"sampled {len(patches)} terrain patches along the 30 paths")
        clustered = fit_masks(patches, k=4, restarts=args.restarts, seed=args.seed,
                              owner=owner, window_length=args.window)
        log(f"Lloyd clustering done (model cost J = {clustered.cost:.4f}, form usage "
            f"{np.round(clustered.usage, 3).tolist()})")

        # Stage 3: simulator-in-the-loop refinement.
        refined = refine(clustered.masks, evaluator, iterations=args.iterations)
        log("refinement done", refined.fitness, history=[round(h, 4) for h in refined.history])
        x = assemble(refined.masks)
        evaluations = evaluator.evaluations - 1  # minus the pool warm-up call
        per_scenario = evaluator.scenario_scores(x)

    # Verification with the exact, unmodified official call (single process).
    fitness = official_fitness(x)
    log("verified with udp.fitness(x)", fitness)

    np.save(args.out / "solution.npy", x)
    summary = {
        "fitness": fitness,
        "total_seconds": round(time.perf_counter() - t0, 1),
        "parameters": {k: v for k, v in vars(args).items() if k != "out"},
        "official_evaluations": evaluations,
        "per_scenario": {f"map{m + 1}-scenario{s + 1}": round(float(v), 4)
                         for (m, s), v in zip(scenarios(), per_scenario)},
        "timeline": timeline,
        # (seconds since start, official fitness) after each refinement simulation
        "refine_trace": [(round(t - t0, 2), round(f, 4)) for t, f in zip(refined.timestamps, refined.history)],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFinal fitness {fitness:.4f}  (ESA example rover: 7.98; 1.0 = theoretical optimum)")
    print(f"Saved {args.out / 'solution.npy'} and {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
