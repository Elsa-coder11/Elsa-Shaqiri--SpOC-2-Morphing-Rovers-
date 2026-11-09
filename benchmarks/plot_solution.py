"""Save ESA's own plots (rover forms + 30 trajectories) for a solution.

Usage:  python benchmarks/plot_solution.py [results/solution.npy]
Writes results/forms.png and results/trajectories.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rovers.official import ROOT, load  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "solution.npy"
    mod = load()  # sets the non-interactive matplotlib backend
    import matplotlib.pyplot as plt

    mod.udp.plot(np.load(path))  # official plotting: forms figure, then trajectories figure
    figures = [plt.figure(n) for n in plt.get_fignums()]
    names = ["forms.png", "trajectories.png"]
    for figure, name in zip(figures, names):
        figure.savefig(ROOT / "results" / name, dpi=110, bbox_inches="tight")
        print(f"Saved results/{name}")


if __name__ == "__main__":
    main()
