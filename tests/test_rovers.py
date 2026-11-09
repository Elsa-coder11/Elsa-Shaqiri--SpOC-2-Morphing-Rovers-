"""Fast checks that our building blocks agree with the official evaluation code.

Run:  python -m unittest discover -s tests -v
(needs the official files: python scripts/fetch_challenge.py)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rovers.chromosome import N_MASK_PARAMS, Chromosome  # noqa: E402
from rovers.controller import assemble, build_controller  # noqa: E402
from rovers.masks import fit_masks, make_windows, velocity  # noqa: E402
from rovers.official import ROOT, load, run_scenario  # noqa: E402


class TestChromosome(unittest.TestCase):
    def test_size_matches_challenge(self):
        mod = load()
        self.assertEqual(Chromosome().size, N_MASK_PARAMS + mod.NUM_NN_PARAMS + 7)
        self.assertEqual(Chromosome().size, 19133)

    def test_views_write_into_flat_vector(self):
        c = Chromosome()
        c["output.bias"][0] = 3.0
        start = N_MASK_PARAMS + 144  # output bias is the last of the 146 biases
        self.assertEqual(c.x[start], 3.0)

    def test_assembled_solution_is_within_bounds(self):
        lower, upper = (np.asarray(b) for b in load().udp.get_bounds())
        x = assemble(np.random.default_rng(0).normal(0, 300, (4, 11, 11)))
        self.assertTrue(np.all(x >= lower) and np.all(x <= upper))
        self.assertTrue(np.all(x[-7:] == np.round(x[-7:])))


class TestController(unittest.TestCase):
    def test_steering_is_exact_pure_pursuit(self):
        mod = load()
        x = build_controller().x
        network = mod.Controller(x[N_MASK_PARAMS:])
        landscape = torch.rand(89, 89)
        latent = torch.zeros(40)
        for bearing in np.linspace(-np.pi, np.pi, 13):
            # torch.Tensor (float32), exactly as the official rover builds its state vector
            state = torch.Tensor([0.5, 0.0, bearing / np.pi, 0.7, 0.1, 1.0, 0.0, 0.0, 0.0])
            mode, steer, _ = network(landscape, state, latent)
            expected = np.clip(4 * bearing / np.pi, -1, 1)  # turn = pi/4 * steer = bearing if |bearing| <= pi/4
            self.assertAlmostEqual(float(steer[0]), expected, places=5)
            self.assertGreater(float(mode[0]), 0)  # always asks to morph

    def test_hysteresis_switching(self):
        mod = load()
        network = mod.Controller(build_controller(switch_threshold=0.8).x[N_MASK_PARAMS:])
        for efficiency, should_switch in ((0.5, True), (0.95, False)):
            state = torch.Tensor([efficiency, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0])
            mode, _, _ = network(torch.zeros(89, 89), state, torch.zeros(40))
            self.assertEqual(bool(mode[0] > 0), should_switch)


class TestMasks(unittest.TestCase):
    def test_velocity_matches_official(self):
        mod = load()
        rng = np.random.default_rng(1)
        forms = torch.tensor(rng.normal(1, 4, (4, 121)), dtype=torch.float32)
        patches = torch.tensor(rng.normal(1, 4, (6, 121)), dtype=torch.float32)
        ours = velocity(forms, patches)
        for j in range(6):
            for i in range(4):
                official = float(mod.velocity_function(forms[i].view(11, 11), patches[j].view(11, 11)))
                self.assertAlmostEqual(float(ours[j, i]), official, places=5)

    def test_lloyd_recovers_planted_clusters(self):
        rng = np.random.default_rng(0)
        centres = rng.normal(1, 5, (4, 121))
        centres[:, 60] = 1.0
        patches = np.concatenate([c + rng.normal(0, 0.05, (50, 121)) for c in centres]).astype(np.float32)
        result = fit_masks(patches, k=4, restarts=3, max_iter=10, steps=50)
        self.assertLess(result.cost, 1.02)  # ~top speed on every patch

    def test_windows_stay_inside_segments(self):
        owner = np.array([0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(make_windows(owner, 2), [0, 0, 1, 2, 2, 3, 3])


class TestSolution(unittest.TestCase):
    def test_rover_reaches_a_sample(self):
        solution = ROOT / "results" / "solution.npy"
        if not solution.exists():
            self.skipTest("run solve.py first")
        score = run_scenario(np.load(solution), 5, 3)  # map 6, scenario 4
        self.assertLess(score, 2.0)  # reached the sample (a miss scores far higher on this route)


if __name__ == "__main__":
    unittest.main()
