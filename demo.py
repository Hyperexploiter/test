#!/usr/bin/env python3
"""
Demo of the unified time-frequency cost functional from the README.
Generates cost for a pure cosine trajectory vs. amplitude deviations.
"""
import numpy as np

def unified_cost(epsilon, freq=5.0, duration=1.0, samples=1024, lam=1.0):
    # time-domain trajectory
    t = np.linspace(0, duration, samples, endpoint=False)
    x_des = np.cos(2 * np.pi * freq * t)
    x = (1 + epsilon) * x_des
    dt = t[1] - t[0]
    # time-domain fit term
    J_time = np.sum((x - x_des)**2) * dt
    # frequency-domain fit term via FFT
    X_des = np.fft.fft(x_des)
    X = np.fft.fft(x)
    df = 1.0 / duration
    J_freq = np.sum(np.abs(X - X_des)**2) * df / samples
    # combined functional
    return J_time + lam * J_freq

if __name__ == '__main__':
    # test a range of amplitude deviations
    deviations = np.linspace(0.0, 0.5, 6)
    print("epsilon    cost")
    for eps in deviations:
        cost = unified_cost(eps)
        print(f"{eps:>7.3f}    {cost:.6f}")