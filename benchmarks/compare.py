"""Compare the decomposition pipeline with generic black-box optimisers.

Every method is scored with the same official simulator on the same worker pool,
and each black-box method gets the same budget of official evaluations.

    random-search  uniformly random chromosomes within the bounds        19 133 variables
    es-full        (1+1)-ES with the 1/5th success rule, started from
                   ESA's example rover (integer genes kept fixed)        19 126 variables
    es-masks       (1+1)-ES on the masks only, with our analytic
                   controller fixed, started from flat masks                484 variables
    ours           solve.py (decomposition + Lloyd + simulator-in-the-loop Lloyd)

Usage:  python benchmarks/compare.py --evals 100       (run solve.py first)
Writes results/benchmark.json, results/benchmark.md and results/convergence.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rovers.chromosome import N_HYPER, N_MASK_PARAMS  # noqa: E402
from rovers.controller import assemble  # noqa: E402
from rovers.masks import CENTRE  # noqa: E402
from rovers.official import ROOT, Evaluator, load  # noqa: E402

RESULTS = ROOT / "results"


def random_search(ev: Evaluator, budget: int, rng: np.random.Generator):
    lower, upper = (np.asarray(b, dtype=float) for b in load().udp.get_bounds())
    trace, best, start = [], np.inf, time.perf_counter()
    while len(trace) < budget:
        batch = min(ev.workers, budget - len(trace))
        xs = rng.uniform(lower, upper, size=(batch, len(lower)))
        xs[:, -N_HYPER:] = rng.integers(lower[-N_HYPER:], upper[-N_HYPER:] + 1, size=(batch, N_HYPER))
        for f in ev.many(xs):
            best = min(best, f)
            trace.append((time.perf_counter() - start, best))
    return trace


def one_plus_one_es(ev: Evaluator, x0: np.ndarray, mutable: np.ndarray, sigma: float, budget: int,
                    rng: np.random.Generator, project=None):
    """(1+1)-ES, Rechenberg's 1/5th success rule. ``mutable`` = indices that are searched."""
    lower, upper = (np.asarray(b, dtype=float) for b in load().udp.get_bounds())
    start = time.perf_counter()
    parent, f_parent = x0.copy(), ev.fitness(x0)
    trace = [(time.perf_counter() - start, f_parent)]
    while len(trace) < budget:
        child = parent.copy()
        child[mutable] += sigma * rng.standard_normal(len(mutable))
        np.clip(child, lower, upper, out=child)
        if project is not None:
            project(child)
        f_child = ev.fitness(child)
        if f_child <= f_parent:
            parent, f_parent = child, f_child
            sigma *= 1.5
        else:
            sigma *= 1.5 ** -0.25
        trace.append((time.perf_counter() - start, f_parent))
    return trace


def ours_trace() -> tuple[list[tuple[float, float]], dict]:
    summary = json.loads((RESULTS / "summary.json").read_text())
    warm_up = summary["timeline"][0]["seconds"]  # pool start-up is excluded for every method
    trace = [(t - warm_up, f) for t, f in summary["refine_trace"]]
    best, out = np.inf, []
    for t, f in trace:
        best = min(best, f)
        out.append((t, best))
    return out, summary


def plot(traces: dict[str, list[tuple[float, float]]], example: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
    colours = {"ours": "#2a78d6", "es-masks": "#eb6834", "es-full": "#1baf7a", "random-search": "#eda100"}
    labels = {"ours": "Ours (decomposition + Lloyd)", "es-masks": "(1+1)-ES, masks only (484 vars)",
              "es-full": "(1+1)-ES, full vector (19 126 vars)", "random-search": "Random search (19 133 vars)"}

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=9)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.axhline(example, color=muted, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0, example, " ESA example rover", color=ink2, fontsize=9, va="bottom")
    ax.axhline(1.0, color=muted, linewidth=1.2, linestyle=(0, (1, 2)))
    ax.text(0, 1.0, " theoretical optimum (straight line at top speed)", color=ink2, fontsize=9, va="bottom")

    for name in ("random-search", "es-full", "es-masks", "ours"):
        if name not in traces:
            continue
        t, f = zip(*traces[name])
        ax.step(t, f, where="post", color=colours[name], linewidth=2, label=labels[name])
        ax.plot([t[-1]], [f[-1]], "o", color=colours[name], markersize=8, markeredgecolor=surface,
                markeredgewidth=2)
        ax.annotate(f"{f[-1]:.2f}", (t[-1], f[-1]), xytext=(6, 0), textcoords="offset points",
                    color=ink, fontsize=9, va="center")

    ax.set_xlabel("wall-clock time (s, 11 worker processes, pool start-up excluded)", color=ink2, fontsize=9)
    ax.set_ylabel("best fitness so far (lower is better)", color=ink2, fontsize=9)
    ax.set_ylim(0.8, max(example, max(f for tr in traces.values() for _, f in tr[:1])) * 1.08)
    ax.set_title("Morphing Rovers: best official fitness vs. computing time", color=ink, fontsize=12,
                 loc="left", pad=12)
    # The band between the ESA reference line and the full-vector ES curve is empty: legend goes there.
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.70))
    for text in legend.get_texts():
        text.set_color(ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evals", type=int, default=100, help="official evaluations per black-box method")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--plot-only", action="store_true", help="redraw the chart from results/benchmark.json")
    args = parser.parse_args()
    if args.plot_only:
        saved = json.loads((RESULTS / "benchmark.json").read_text())
        plot(saved["traces"], saved["example"], RESULTS / "convergence.png")
        print(f"Saved {RESULTS / 'convergence.png'}")
        return
    rng = np.random.default_rng(args.seed)
    mod = load()

    traces: dict[str, list[tuple[float, float]]] = {}
    with Evaluator(args.workers) as ev:
        ev.fitness(assemble(np.ones((4, 11, 11))))  # start-up, not timed
        example = ev.fitness(mod.udp.example())
        print(f"ESA example: {example:.4f}", flush=True)

        traces["random-search"] = random_search(ev, args.evals, rng)
        print(f"random search: {traces['random-search'][-1][1]:.4f}", flush=True)

        x0 = np.asarray(mod.udp.example(), dtype=float)
        traces["es-full"] = one_plus_one_es(ev, x0, np.arange(len(x0) - N_HYPER), 0.5, args.evals, rng)
        print(f"(1+1)-ES full: {traces['es-full'][-1][1]:.4f}", flush=True)

        def keep_centres_positive(x: np.ndarray) -> None:
            centres = np.arange(4) * 121 + CENTRE
            x[centres] = np.clip(x[centres], 1e-16, 100)

        flat = assemble(np.ones((4, 11, 11)))
        traces["es-masks"] = one_plus_one_es(ev, flat, np.arange(N_MASK_PARAMS), 0.5, args.evals, rng,
                                             keep_centres_positive)
        print(f"(1+1)-ES masks: {traces['es-masks'][-1][1]:.4f}", flush=True)

    traces["ours"], summary = ours_trace()
    rows = [
        ("Random search", "19 133", args.evals, traces["random-search"]),
        ("(1+1)-ES, full chromosome", "19 126", args.evals, traces["es-full"]),
        ("(1+1)-ES, masks only (our controller)", "484", args.evals, traces["es-masks"]),
        ("**Ours: decomposition + Lloyd + simulator-in-the-loop Lloyd**", "structured",
         summary["official_evaluations"], traces["ours"]),
    ]
    lines = ["| Method | Variables searched | Official evaluations | Wall-clock (s) | Best fitness |",
             "|---|---|---|---|---|",
             f"| ESA example rover (reference) | - | 1 | - | {example:.4f} |"]
    for name, variables, evals, trace in rows:
        lines.append(f"| {name} | {variables} | {evals} | {trace[-1][0]:.0f} | {trace[-1][1]:.4f} |")
    table = "\n".join(lines)
    print("\n" + table)
    (RESULTS / "benchmark.md").write_text(table + "\n")
    (RESULTS / "benchmark.json").write_text(json.dumps({"example": example, "traces": traces}, indent=1))
    plot(traces, example, RESULTS / "convergence.png")
    print(f"Saved {RESULTS / 'benchmark.md'} and {RESULTS / 'convergence.png'}")


if __name__ == "__main__":
    main()
