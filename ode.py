import math
import matplotlib.pyplot as plt
import numpy as np

c = np.array([0, 1/3, 2/3, 1])
b = np.array([1/8, 3/8, 3/8, 1/8])
a = np.array([
    [0, 0, 0, 0],
    [1/3, 0, 0, 0],
    [-1/3, 1, 0, 0],
    [1, -1, 1, 0]
])

""" inne, możliwe wartości współczynników """
"""c = np.array([0, 0.5, 0.5, 1])
b = np.array([1/6, 1/3, 1/3, 1/6])
a = np.array([
    [0, 0, 0, 0],
    [1/2, 0, 0, 0],
    [0, 1/2, 0, 0],
    [0, 0, 1, 0]
])"""
RungeKuttaOrder = 4


def RungeKutta(xStart, timeStart, f, timeEnd, dt=1e-4):
    resultTime = [timeStart]
    resultX = [xStart]

    t = timeStart
    while t <= timeEnd:
        k = []
        for i in range(RungeKuttaOrder):
            k.append(f(t + c[i]*dt,
                       resultX[-1] + dt * sum(a[i][:i]*np.array(k))
                       ))
        t += dt
        resultTime.append(t)
        resultX.append(resultX[-1] + dt * sum(b * k))

    return resultTime, resultX


def RungeKuttaVector(xStart, timeStart, f, timeEnd, dt=1e-4):
    resultTime = [timeStart]
    resultX = [xStart]

    t = timeStart
    while t <= timeEnd:
        k = [f(t, resultX[-1])]

        for i in range(1, RungeKuttaOrder):

            suma_k = a[i][:i] * np.transpose(np.array(k))
            suma_k = np.transpose(suma_k)
            suma_k = sum(suma_k)

            k.append(f(t + c[i]*dt,
                       resultX[-1] + dt * suma_k))
        t += dt
        resultTime.append(t)

        suma_k = np.transpose(b * np.transpose(np.array(k)))

        resultX.append(resultX[-1] + dt * sum(suma_k))

    return np.array(resultTime), np.array(resultX)
