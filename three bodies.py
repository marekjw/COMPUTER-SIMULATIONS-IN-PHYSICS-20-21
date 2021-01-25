import numpy as np
from numpy import array
import matplotlib.pyplot as plt


def verlet(F, y0, t_span, m, dt, calc_v=True):
    next_r = y0[:6] - dt * y0[6:]

    t0, t_end = t_span
    t = t0

    solution = [next_r, y0[:6]]
    solution_time = [t-dt, t]

    t = t0 + dt

    while t < t_end:
        r_next = 2*solution[-1] - solution[-2] + F(t, solution[-1])/m * dt**2
        solution_time.append(t)
        solution.append(r_next)
        t += dt

    if not calc_v:
        return np.array(solution)

    solution_with_v = [y0]

    for i, r in enumerate(solution[:-1]):
        if i == 0:
            continue

        v = 1/2/dt * (solution[i+1] - solution[i-1])

        solution_with_v.append(np.concatenate((r, v*m)))

    sol_time = array(solution_time[:-1])
    sol_with_v = array(solution_with_v)

    return sol_time, sol_with_v


def force_n_bodies(R, masses, n, dim=2, G=1e-2):

    def get_n_r(n):
        return R[n*dim:(n+1)*dim]

    forces = []
    for i in range(n):
        r_self = get_n_r(i)

        F = np.zeros(dim)

        for j in range(n):
            if i == j:
                continue

            r_other = get_n_r(j)

            F += (r_other - r_self) * G * \
                masses[i] * masses[j] / np.linalg.norm(r_other - r_self)**3

        forces.append(F)
    return np.concatenate(forces)


def force(_, r): return force_n_bodies(r, (1, 1, 1), 3)


def solve_and_print(consts, y0, method, t_span):
    m, dt = consts

    sol_t, sol = method(force, y0, t_span, m, dt)

    plt.figure(figsize=(10, 10))
    plt.grid(True)
    plt.title('położenie - ' + method.__name__)
    plt.plot(sol[:, 0], sol[:, 1])
    plt.plot(sol[:, 2], sol[:, 3])
    plt.plot(sol[:, 4], sol[:, 5])


m = 1
dt = 1e-3

r1 = array((0.97000436,
            -0.24308753))

r2 = -1*r1

r3 = np.zeros(2)

v3 = array((0.93240737, 0.86473146))
v1 = -2*v3
v2 = -2*v3

y0 = np.concatenate((r1, r2, r3, v1, v2, v3))

solve_and_print((m, dt), y0, verlet, (0, 100))
plt.show()
