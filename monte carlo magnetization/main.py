import numpy as np
from fastprogress import progress_bar


class Spins:

    neighbours = [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1)
    ]

    def __init__(self, L, temp, J=1) -> None:
        self.spin_array = np.ones((L, L))
        self.L = L
        self.beta = 1/temp
        self.J = J

    def get_treshold(self, delta):
        return 1/(1 + np.exp(-self.beta * delta))

    def random_idx(self):
        return (np.random.randint(0, self.L),
                np.random.randint(0, self.L))

    def get_energy_difference(self, idx):
        x, y = idx
        return 2 * self.J * \
            sum(self.spin_array[(x + dx) % self.L, (y+dy) % self.L]
                for (dx, dy) in Spins.neighbours)

    def monte_carlo_step(self):
        idx = self.random_idx()
        delta = self.get_energy_difference(idx)

        if np.random.random() < self.get_treshold(delta):
            self.spin_array[idx] = 1
        else:
            self.spin_array[idx] = -1

    def get_magnetization(self):
        return np.sum(self.spin_array)


def run_simulation(L, termalize_steps, record_steps, temperature):
    spins = Spins(L, temperature)

    for i in range(termalize_steps):
        spins.monte_carlo_step()

    res = 0
    for i in range(record_steps):
        spins.monte_carlo_step()
        res += abs(spins.get_magnetization())/record_steps

    return res / L ** 2
