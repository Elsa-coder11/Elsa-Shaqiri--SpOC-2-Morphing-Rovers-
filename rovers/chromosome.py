"""Named, writable views into the flat 19 133-element decision vector.

Layout (from the challenge description, derived here from the official network so it
can never drift out of sync):

    [ 4 masks x 11 x 11 | 146 biases | 18 496 weights | 7 integer hyper-parameters ]

``Chromosome()["lin3.weight"]`` returns a NumPy *view* with the PyTorch shape, so
writing into it writes straight into the flat vector.
"""
from __future__ import annotations

from functools import lru_cache
from math import prod

import numpy as np

from .official import load

N_MODES = 4
MASK_SIZE = 11
N_MASK_PARAMS = N_MODES * MASK_SIZE * MASK_SIZE  # 484
N_HYPER = 7

# Activation ids used by the official controller.
SIGMOID, HARDSIGMOID, TANH, HARDTANH, SOFTSIGN, SOFTPLUS, RELU = range(7)
MAXPOOL, AVGPOOL = 0, 1


@lru_cache(maxsize=None)
def layout() -> tuple[dict[str, tuple[int, tuple[int, ...]]], int]:
    """Map parameter name -> (start index, shape), and the offset of the hyper-parameters."""
    mod = load()
    probe = mod.Controller(np.zeros(mod.NUM_NN_PARAMS + N_HYPER))
    index: dict[str, tuple[int, tuple[int, ...]]] = {}
    bias_at = N_MASK_PARAMS
    weight_at = N_MASK_PARAMS + mod.NUM_BIASES
    for name, param in probe.named_parameters():
        shape = tuple(param.shape)
        if param.dim() == 1:
            index[name] = (bias_at, shape)
            bias_at += param.numel()
        else:
            index[name] = (weight_at, shape)
            weight_at += param.numel()
    assert bias_at == N_MASK_PARAMS + mod.NUM_BIASES
    assert weight_at == N_MASK_PARAMS + mod.NUM_NN_PARAMS
    return index, weight_at


class Chromosome:
    def __init__(self, x: np.ndarray | None = None):
        index, hyper_at = layout()
        self._index = index
        self._hyper_at = hyper_at
        self.size = hyper_at + N_HYPER
        if x is None:
            self.x = np.zeros(self.size)
        else:
            self.x = np.array(x, dtype=float)
            assert self.x.shape == (self.size,), f"expected {self.size} values, got {self.x.shape}"

    def __getitem__(self, name: str) -> np.ndarray:
        start, shape = self._index[name]
        return self.x[start:start + prod(shape)].reshape(shape)

    @property
    def names(self) -> list[str]:
        return list(self._index)

    @property
    def masks(self) -> np.ndarray:
        return self.x[:N_MASK_PARAMS].reshape(N_MODES, MASK_SIZE, MASK_SIZE)

    @property
    def hyper(self) -> np.ndarray:
        return self.x[self._hyper_at:]

    def clip_to_bounds(self) -> "Chromosome":
        lower, upper = (np.asarray(b, dtype=float) for b in load().udp.get_bounds())
        np.clip(self.x, lower, upper, out=self.x)
        self.x[-N_HYPER:] = np.round(self.x[-N_HYPER:])
        return self
