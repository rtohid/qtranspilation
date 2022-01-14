# Copyright (c) 2021 R. Tohid (@rtohid)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import time

from typing import Tuple

from qtranspiler.architectures import Grid


class Profile:

    def __init__(self, func, args, kwargs, return_value, timing):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.return_value = return_value
        self.time = timing

    def __repr__(self):
        name = f"function: {self.func.__name__}\n"
        args = f"args: {self.args}\n"
        kwargs = f"kwargs: {self.kwargs}\n"
        timing = f"time: {self.time}"
        return name + args + kwargs + timing


def local_randomizer(map: np.ndarray, depth: int) -> np.ndarray:
    for i in range(0, len(map), depth):
        h, w = np.shape(map)
        map[i:i + depth] = np.random.permutation(
            map.flatten()[h * i:h * (i + depth)]).reshape(depth, w)


def random_map(grid: Grid) -> np.ndarray:
    """Randomly generates a new mapping for a given grid."""
    size = grid.size()
    new_map = np.arange(size)
    np.random.shuffle(new_map)
    return new_map


def profile_func(func, *args, **kwargs):
    start = time.perf_counter()
    return_value = func(*args, **kwargs)
    end = time.perf_counter()
    return Profile(func, args, kwargs, return_value, end - start)
