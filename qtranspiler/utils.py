# Copyright (c) 2021 R. Tohid (@rtohid)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import time

from typing import Tuple

from qtranspiler.architectures import Grid


class Profile:

    def __init__(self, func):
        self.func = func
        self.args = list()
        self.kwargs = list()
        self.return_values = list()
        self.times = list()

    def add_run(self, args, kwargs):
        start = time.perf_counter()
        return_value = self.func(*args, **kwargs)
        end = time.perf_counter()
        self.args.append(args)
        self.kwargs.append(kwargs)
        self.return_values.append(return_value)
        self.times.append(end - start)

    def average_value(self, key=lambda x: x):
        values = [key(v) for v in self.return_values]
        return sum(values) / self.num_runs()

    def average_time(self):
        return sum(self.times) / self.num_runs()

    def num_runs(self):
        return len(self.times)

    def __repr__(self):
        name = f"function: {self.func.__name__}\n"
        args = f"args: {self.args}\n"
        kwargs = f"kwargs: {self.kwargs}\n"
        timing = f"time: {self.times}"
        return name + args + kwargs + timing


def blocked_random_map(grid: Grid,
                       depth: int = None,
                       width: int = None) -> np.ndarray:
    mapping = grid.labels()
    h, w = grid.shape
    if not depth:
        depth = h
    if not width:
        width = w

    # Break the grid down into (almost) equal blocks.
    rows = np.array_split(mapping.reshape(grid.shape), depth, axis=0)
    cols = [np.array_split(row, width, axis=1) for row in rows]

    # Locally randomize the grid by randomizing labels of each block
    for col in cols:
        for c in col:
            mapping[c.flatten()] = np.random.permutation(c.flatten())

    return mapping.reshape(grid.shape)


def local_random_permutations(grid: Grid,
                              cluster_size: int,
                              delta_row: int = 0,
                              delta_col: int = 0):

    def random_cluster_bounds():
        while True:
            index = np.random.randint(grid.size)
            if index in not_yet_picked:
                row = index // grid.width
                column = index - (row * grid.width)
                lower_row = np.random.randint(max(0, row - delta_row), row + 1)
                upper_row = np.random.randint(
                    row,
                    min(row + delta_row, grid.height) + 1)
                lower_col = np.random.randint(max(0, column - delta_col),
                                              column + 1)
                upper_col = np.random.randint(
                    column,
                    min(column + delta_col, grid.width) + 1)
                if lower_col == upper_col or lower_row == upper_row:
                    continue
                not_yet_picked.remove(index)
                return (lower_row, upper_row, lower_col, upper_col)

    not_yet_picked = set(range(grid.size))
    num_clusters = grid.size // cluster_size
    new_map = np.arange(grid.size).reshape(grid.shape)

    for i in range(num_clusters):
        lower_row, upper_row, lower_col, upper_col = random_cluster_bounds()
        height = upper_row - lower_row
        width = upper_col - lower_col
        new_map[lower_row:upper_row,
                lower_col:upper_col] = np.random.permutation(
                    new_map[lower_row:upper_row,
                            lower_col:upper_col].flatten()).reshape(
                                height, width)
    return new_map


def random_map(grid: Grid) -> np.ndarray:
    """Randomly generates a new mapping for a given grid."""
    size = grid.size
    new_map = np.arange(size)
    np.random.shuffle(new_map)
    return new_map.reshape(grid.shape)


def profile_func(func, num_runs, *args, **kwargs):
    runs = Profile(func)
    for _ in range(num_runs):
        runs.add_run(args, kwargs)

    return runs
