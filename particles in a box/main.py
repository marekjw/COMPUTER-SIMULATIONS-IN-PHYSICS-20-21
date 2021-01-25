from matplotlib.pyplot import sca
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import copy
from numpy import array
from matplotlib.patches import Circle
from fastprogress import progress_bar

DIMENSION = 2


class Particles:
    def __init__(self, r, v, force, boxsize,  mass=1, dt=1e-4):
        self.r = r
        self.v = v
        self.mass = mass
        self.dt = dt
        self.force = force
        self.n = self.r.shape[0]
        self.boxsize = boxsize
        self.volume = self.boxsize**2

    def get_mean_kinetic_energy(self):
        return np.sum(self.v**2)/self.n/2*self.mass

    def get_pressure(self):
        return self.get_mean_kinetic_energy() * self.n / self.volume - 1 / 2 / \
            self.volume * self.force.pressure_sum

    def get_positions(self):
        return self.r, self.v

    def get_position(self, i):
        return self.r[i]

    def next_position(self):
        # calculate the next position using verlet's methods
        next_v = self.v + self.force(self.r)/self.mass*self.dt
        next_r = self.r + next_v * self.dt

        self.r, self.v = next_r, next_v

        self.r %= self.boxsize

    def reverse(self):
        next_v = self.v + self.force(self.r)/self.mass*self.dt
        self.v = -next_v


class LJ_Force:
    def __init__(self, epsilon, sigma, boxsize):
        self.epsilon, self.sigma, self.boxsize = epsilon, sigma, boxsize
        self.half_boxsize = self.boxsize / 2
        self.pressure_sum = 0

    # periodical boundary condition
    def modulo_box(self, r):
        for i in range(DIMENSION):
            if r[i] > self.half_boxsize:
                r[i] -= self.boxsize
            elif r[i] < -self.half_boxsize:
                r[i] += self.boxsize
        return r

    def calc_force(self, r1, r2):
        r = r2 - r1
        r = self.modulo_box(r)
        r_len = np.linalg.norm(r)
        assert r_len != 0
        r /= r_len

        arg1 = np.power(self.sigma/r_len, 14)
        arg2 = np.power(self.sigma/r_len, 8)

        return -48*self.epsilon/np.power(self.sigma, 2) * (arg1 - 0.5 * arg2) * r

    def __call__(self, r):
        self.pressure_sum = 0
        n = int(r.shape[0])

        res = np.zeros(r.shape)

        for i in range(n):
            for j in range(i):
                force = self.calc_force(r[i], r[j])
                res[i] += force
                res[j] -= force

                # calculating pressure
                self.pressure_sum += np.dot(force,
                                            self.modulo_box(r[j] - r[i]))

        return res


def init_positions(n, boxsize, offset=0):
    res = np.zeros((n, DIMENSION))

    assert int(np.sqrt(n)) == np.sqrt(n), 'n must be squareable'

    nx = int(np.sqrt(n))
    delta = boxsize/nx

    for i in range(nx):
        for j in range(nx):
            res[i*nx + j] = array((delta * i, delta * j))

    return res + delta/2 + offset


def init_velocities(n, mass, temp):
    res = np.random.rand(n, DIMENSION)
    mean_velocity = np.mean(res, axis=0)
    res -= mean_velocity
    mean_kinetic = np.sum(res**2)/n*mass/2
    scale = np.sqrt(temp/mean_kinetic)

    return res * scale


def draw(particles: Particles, boxsize, radius, step_no, color='b'):
    plt.clf()
    figure = plt.gcf()

    for i in range(particles.n):
        p = particles.get_position(i)

        ax = plt.gca()
        circle = Circle((p[0], p[1]), radius=radius, color=color)
        ax.add_patch(circle)

    plt.plot()
    lims = (0, boxsize)
    plt.xlim(lims)
    plt.ylim(lims)
    figure.set_size_inches(10, 10)

    nStr = str(step_no).rjust(5, '0')

    plt.title('Symulacja gazu Lennarda-Jonesa, krok'+nStr)
    plt.savefig('images/img'+nStr+'.png')


def run_simulation(time, init_positions, n=16,  dt=1e-4, eps=1., sigma=1., radius=0.5, mass=1,
                   boxsize=8., temp=2.5, draw_every_n=100, do_reverse=False):
    particles = Particles(
        init_positions(n, boxsize),
        init_velocities(n, mass, temp),
        LJ_Force(eps, sigma, boxsize),
        boxsize,
        mass=mass,
        dt=dt
    )
    n_steps = int(time / dt)
    temperatures = []
    pressures = []

    for step_count in progress_bar(range(n_steps+1)):
        particles.next_position()

        temperatures.append(particles.get_mean_kinetic_energy().item())

        pressures.append(particles.get_pressure().item())

        if step_count % draw_every_n == 0:
            draw(particles, boxsize, radius, step_count)

    if not do_reverse:
        return temperatures, pressures

    particles.reverse()

    for step_count in progress_bar(range(n_steps+1, 2*n_steps+1)):
        particles.next_position()

        temperatures.append(particles.get_mean_kinetic_energy().item())

        pressures.append(particles.get_pressure().item())

        if step_count % draw_every_n == 0:
            draw(particles, boxsize, radius, step_count, color='r')

    return temperatures, pressures


temperatures, pressures = run_simulation(
    1.,
    init_positions,
    n=16,
    mass=1,
    dt=1e-3,
    draw_every_n=100,
    boxsize=10,
    do_reverse=False
)

plt.figure(figsize=(10, 10))
plt.plot(temperatures)
plt.title('Temperatura')
plt.figure(figsize=(10, 10))
plt.plot(pressures)
plt.title('Ciśnienie')
plt.show()
