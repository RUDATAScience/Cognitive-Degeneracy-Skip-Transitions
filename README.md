# Cognitive Degeneracy in Non-Equilibrium Deterioration Processes: A SDE-Markov Simulation Framework

## Overview
This repository provides a computational simulation framework designed to explore **"Cognitive Degeneracy" (Skip Transitions)** in infrastructure deterioration. By combining continuous Stochastic Differential Equations (SDE) with discrete Markov Chain Monte Carlo (MCMC) simulations, this code mathematically models the information asymmetry that occurs when continuous physical degradation is monitored through discrete and imperfect observation intervals.

## Theoretical Architecture
To strictly prevent any structural disconnection between the physical calculations and the simulation parameters, the framework is built upon a tightly coupled three-layer architecture:

### 1. Physical Layer (SDE & First Passage Time)
The true, unobservable physical state is modeled as a continuous random variable $X(t)$ governed by Brownian motion ($dX(t) = \mu dt + \sigma dW(t)$). The baseline probability of a physical state transitioning from Healthy (State A) to Severe Damage (State C) within a given interval $\Delta t$ is rigorously derived using the exact analytical solution of the First Passage Time (FPT) Inverse Gaussian cumulative distribution.

### 2. Observation Layer (Structural Coupling & Beta Noise)
In real-world asset management, subjective expert ratings (Assessor Bias) introduce uncertainty. We model this by drawing the effective transition probability $p'_{eff}$ from a Beta distribution. Crucially, to maintain structural integrity, the shape parameters ($\alpha, \beta$) of this Beta distribution are mathematically constrained to center precisely on the theoretical physical baseline $p_{base}$ derived in Layer 1.

### 3. Macroscopic Layer (Phase Space Exploration)
A large-scale Monte Carlo engine ($N \ge 10^5$ trials) applies these stochastic probabilities to simulate the life cycles of an infrastructure cohort. By sweeping through various values of environmental fluctuation ($\sigma$) and observation intervals ($\Delta t$), the engine generates a Phase Space Heatmap.

## Key Features
* **Identification of Validity Thresholds:** The generated phase space enables researchers and asset managers to quantitatively identify the specific threshold of $\Delta t$ required to prevent the collapse of statistically homogeneous populations under varying environmental noise levels.
* **Vectorized Performance:** The Monte Carlo core is highly optimized using NumPy vectorization, allowing for rapid execution of million-scale simulations directly within Jupyter/Google Colab environments.
