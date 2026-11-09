"""Stage 1: write the steering controller directly into the network weights.

No training is needed. The rover's state vector already contains the angle to the
target, ``s[2] = angle_diff / pi``, and the network can pass it to the steering
output almost unchanged:

    s[2] --inp--> h1[0] --lin3--> h2[0] --lin4--> h3[0] --output--> steering

Every hidden layer uses Hardtanh, which is the identity on [-1, 1], and
|s[2]| <= 1, so the signal reaches the output exactly. With output gain 4:

    steering = clamp(4 * angle_diff / pi, -1, 1),   turn = pi/4 * steering

So whenever the target is within 45 degrees, the rover turns to face it exactly
in one step (pure pursuit). Otherwise it turns at the maximum rate. The
convolutional path and recurrent weights are left at zero.

Morphing: if the network asks to morph (output[0] > 0) and the cooldown is over,
the simulator itself picks the form with the highest velocity on the current
terrain. Two policies are available:
  * always morph       output[0] = +1 (constant bias)
  * hysteresis tau     output[0] = K * (tau - efficiency), i.e. morph only when the
                       current form runs below a fraction tau of top speed. This
                       saves the 9-step cooldown for when it pays off.
The efficiency (state ``s[0]``, in [0, 1]) is routed through hidden unit 1 in the
same way as the angle.
"""
from __future__ import annotations

import numpy as np

from .chromosome import HARDTANH, MAXPOOL, Chromosome

STEER_GAIN = 4.0  # = 1 / (pi/4) in units of pi: cancels the heading error in one step
SWITCH_GAIN = 50.0


def build_controller(chromosome: Chromosome | None = None,
                     switch_threshold: float | None = None,
                     steer_gain: float = STEER_GAIN) -> Chromosome:
    """Write the analytic controller into ``chromosome`` (masks are left untouched)."""
    c = chromosome if chromosome is not None else Chromosome()
    for name in c.names:
        c[name][...] = 0.0
    c.hyper[:] = [MAXPOOL, MAXPOOL, HARDTANH, HARDTANH, HARDTANH, HARDTANH, HARDTANH]

    # Steering channel (hidden unit 0): angle to target -> angular velocity.
    c["inp.weight"][0, 2] = 1.0
    c["lin3.weight"][0, 0] = 1.0
    c["lin4.weight"][0, 0] = 1.0
    c["output.weight"][1, 0] = steer_gain

    # Morphing channel.
    if switch_threshold is None:
        c["output.bias"][0] = 1.0
    else:
        c["inp.weight"][1, 0] = 1.0          # h1[1] = current efficiency
        c["lin3.weight"][1, 1] = 1.0
        c["lin4.weight"][1, 1] = 1.0
        c["output.weight"][0, 1] = -SWITCH_GAIN
        c["output.bias"][0] = SWITCH_GAIN * switch_threshold
    return c


def assemble(masks: np.ndarray, switch_threshold: float | None = None) -> np.ndarray:
    """Complete decision vector: four masks + analytic controller, clipped to the bounds."""
    c = Chromosome()
    c.masks[...] = masks
    build_controller(c, switch_threshold)
    return c.clip_to_bounds().x
