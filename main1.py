import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# ==========================================
# 1. Physical Layer: SDE & First Passage Time
# ==========================================
def calc_fpt_linear_cdf(mu, sigma, L, dt):
    """
    Calculates the theoretical baseline probability of cognitive degeneracy (skip)
    using the Inverse Gaussian CDF derived from the Linear Drift SDE.
    """
    if mu <= 0 or sigma <= 0 or dt <= 0:
        return 0.0
    
    term1 = (mu * dt - L) / (sigma * np.sqrt(dt))
    term2 = (-mu * dt - L) / (sigma * np.sqrt(dt))
    
    # Analytical solution for First Passage Time CDF
    prob = norm.cdf(term1) + np.exp(2 * mu * L / (sigma**2)) * norm.cdf(term2)
    return np.clip(prob, 0.0, 1.0)

# ==========================================
# 2. Observation Layer: Structural Coupling & Noise
# ==========================================
def apply_beta_noise(p_base, K, size):
    """
    Applies assessor bias and local observation uncertainty via Beta distribution.
    Structurally coupled to the physical baseline p_base.
    """
    if p_base <= 0.0:
        return np.zeros(size)
    if p_base >= 1.0:
        return np.ones(size)
        
    alpha = p_base * K
    beta_param = (1.0 - p_base) * K
    return np.random.beta(alpha, beta_param, size)

# ==========================================
# 3. Macroscopic Layer: Monte Carlo Phase Space Exploration
# ==========================================
def explore_phase_space():
    # Dimensionless Constants
    X_A, X_C = 0.0, 0.85
    L_AC = X_C - X_A  # Distance for A -> C Skip Degeneracy
    
    # Simulation Parameters
    mu = 0.1          # Constant drift rate
    K = 100           # Noise concentration (reliability of observation)
    N = 10000         # Number of Monte Carlo trials per grid point
    
    # Grid for Phase Space (X: Observation Interval, Y: Environmental Fluctuation)
    dt_vals = np.linspace(0.01, 1.0, 40)
    sigma_vals = np.linspace(0.01, 0.5, 40)
    
    # Results matrix
    skip_rates = np.zeros((len(sigma_vals), len(dt_vals)))
    
    print("Running Monte Carlo Phase Space Exploration...")
    
    for i, sig in enumerate(sigma_vals):
        for j, dt in enumerate(dt_vals):
            # Step A: Derive physical baseline probability (No structural disconnection)
            p_AC_base = calc_fpt_linear_cdf(mu, sig, L_AC, dt)
            
            # Step B: Superimpose observation noise
            p_AC_eff = apply_beta_noise(p_AC_base, K, N)
            
            # Step C: Macroscopic Monte Carlo evaluation
            # Count how many assets experienced a cognitive skip A -> C
            skips = np.sum(np.random.rand(N) < p_AC_eff)
            skip_rates[i, j] = skips / N

    # ==========================================
    # 4. Visualization (Heatmap)
    # ==========================================
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(skip_rates, 
                     xticklabels=np.round(dt_vals, 2), 
                     yticklabels=np.round(sigma_vals, 2),
                     cmap="viridis", cbar_kws={'label': 'Skip Degeneracy Rate (A->C)'})
    
    ax.invert_yaxis()
    plt.title("Phase Space of Cognitive Degeneracy\nLinear Drift SDE vs Discrete Observation", fontsize=14, pad=15)
    plt.xlabel(r"Observation Interval ($\Delta t$)", fontsize=12)
    plt.ylabel(r"Environmental Fluctuation ($\sigma$)", fontsize=12)
    
    # Simplify axis ticks for readability
    ax.set_xticks(ax.get_xticks()[::4])
    ax.set_yticks(ax.get_yticks()[::4])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    explore_phase_space()
