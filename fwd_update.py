#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This script shows the Tweedle Update
Any kind of distribution upon addition of noise transforms into a Gaussian
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def tweedie_update(x, beta, num_steps):
    for _ in range(num_steps):
        x = np.sqrt(1 - beta) * x + np.sqrt(beta) * np.random.normal(0, 1)
    return x

# Parameters
beta = 0.1
num_samples = 10000
num_steps = 100

# Generate initial samples from a bimodal distribution
initial_samples = np.concatenate([
    np.random.normal(-3, 1, num_samples // 2),
    np.random.normal(3, 1, num_samples // 2)
])

# Apply Tweedie update
final_samples = np.array([tweedie_update(x, beta, num_steps) for x in initial_samples])

# Plotting
plt.figure(figsize=(12, 6))

# Initial distribution
plt.subplot(121)
plt.hist(initial_samples, bins=50, density=True, alpha=0.7)
plt.title("Initial Distribution")
plt.xlabel("Value")
plt.ylabel("Density")

# Final distribution
plt.subplot(122)
plt.hist(final_samples, bins=50, density=True, alpha=0.7)
plt.title(f"Distribution after {num_steps} Tweedie Updates")
plt.xlabel("Value")
plt.ylabel("Density")

# Add theoretical standard normal PDF
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, 0, 1), 'r-', lw=2, label='Standard Normal PDF')
plt.legend()

plt.tight_layout()
plt.show()
