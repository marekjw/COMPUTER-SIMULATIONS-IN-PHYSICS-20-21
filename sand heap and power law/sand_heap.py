import numpy as np
import queue
import matplotlib.pyplot as plt

from fastprogress import progress_bar
from numpy.lib.function_base import gradient


def random_pair(n):
    return np.random.randint(0, n), np.random.randint(0, n)


class SandHeap:
    neighbours = [
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0)
    ]

    def __init__(self, N, n_critical, supercritical=False) -> None:
        self.N = N
        self.grains = np.zeros((N, N), dtype=int)
        self.n_critical = n_critical
        self.was_cascade = np.zeros((N, N), dtype=bool)

        if supercritical:
            self.grains.fill(2*(n_critical+1) - 1)

    def get_array(self):
        return self.grains

    def single_cascade(self, x, y, arr):
        for dx, dy in SandHeap.neighbours:
            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < self.N and 0 <= new_y < self.N:
                arr[new_x, new_y] += 1

    def do_one_iteration(self, xs, ys):
        res = 0
        grains_copy = np.copy(self.grains)

        grains_copy[xs, ys] -= len(SandHeap.neighbours)

        for x, y in zip(xs, ys):
            if not self.was_cascade[x, y]:
                self.was_cascade[x, y] = True
                res += 1

            self.single_cascade(x, y, grains_copy)

        self.grains = grains_copy
        return res

    def cascade(self, xs, ys) -> int:
        res = 0
        self.was_cascade[:] = False

        while len(xs) != 0:
            res += self.do_one_iteration(xs, ys)
            xs, ys = np.where(self.grains > self.n_critical)

        return res

    def drop_random_grain(self) -> int:
        return self.drop_grain(random_pair(self.N))

    def just_cascade(self):
        xs, ys = np.where(self.grains > self.n_critical)
        self.do_one_iteration(xs, ys)

    def drop_grain(self, index):
        x, y = index
        self.grains[x, y] += 1
        if(self.grains[x, y] > self.n_critical):
            return self.cascade([x], [y])
        return 0

    def get_n_grains(self) -> int:
        return self.grains.sum()

    def is_stable(self) -> bool:
        pass

    def reset(self, supercritical=False):
        if not supercritical:
            self.grains.fill(0)
        else:
            self.grains.fill((self.n_critical+1) * 2 - 1)


def simple_simulation(heap: SandHeap, steps: int):
    n_grains = []
    for step in progress_bar(range(steps)):
        heap.drop_random_grain()
        n_grains.append(heap.get_n_grains())

    return n_grains


def run_simulation(N, n_critical, steps, step_treshold=None):
    heap = SandHeap(N, n_critical)

    if step_treshold == None:
        return simple_simulation(heap, steps)

    step_counter = 0
    greatest_amount = 0

    n_grains_record = []

    while step_counter < step_treshold:
        step_counter += 1

        heap.drop_random_grain()

        n_grains = heap.get_n_grains()

        n_grains_record.append(n_grains)

        if n_grains > greatest_amount:
            greatest_amount = n_grains
            step_counter = 0

    print('Stan stacjonarny osiągnięty')

    fig = plt.figure(figsize=(10, 10))
    plt.plot(n_grains_record)

    avalanches = np.zeros(N * N + 1, dtype=int)

    for i in progress_bar(range(steps)):
        avalanche_size = heap.drop_random_grain()
        if avalanche_size > 0:
            avalanches[avalanche_size] += 1

    return avalanches


"""
res = run_simulation(31, 3, 140000, 500)

fig = plt.figure(figsize=(10, 10))

plt.plot(res)
plt.xscale('log')
plt.yscale('log')
plt.show()
"""
