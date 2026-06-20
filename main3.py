import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import zipfile
from google.colab import files

# ==========================================
# 1. Physical Layer: Non-linear SDE Paths
# ==========================================
def simulate_nonlinear_sde_skips(gamma, dt_obs, sigma=0.1, k=0.1, N_paths=2000):
    """
    Simulates physical deterioration paths using Euler-Maruyama integration
    for the non-linear SDE: dX = k * t^gamma * dt + sigma * dW.
    Returns the baseline probability of A->C cognitive skip.
    """
    X_A, X_B, X_C = 0.0, 0.60, 0.85
    
    # Simulation Time Setup
    T_max = 12.0
    dt_sim = 0.01  # Microscopic physical time step
    n_steps = int(T_max / dt_sim)
    
    # Time array and Drift calculation
    t_array = np.linspace(0, T_max, n_steps)
    mu_array = k * (t_array ** gamma)
    
    # Wiener process (Brownian motion)
    dW = np.random.randn(N_paths, n_steps) * np.sqrt(dt_sim)
    
    # Vectorized SDE Integration
    dX = mu_array * dt_sim + sigma * dW
    X = np.cumsum(dX, axis=1)
    
    # Discrete Observation Sampling
    step_size = max(1, int(np.round(dt_obs / dt_sim)))
    obs_X = X[:, ::step_size]
    
    # Detect Cognitive Degeneracy (Skip A -> C between consecutive observations)
    # State A: X < X_B, State C: X >= X_C
    is_A = obs_X[:, :-1] < X_B
    is_C = obs_X[:, 1:] >= X_C
    skips = is_A & is_C
    
    # Calculate base probability of an asset experiencing at least one A->C skip
    p_base = np.mean(np.any(skips, axis=1))
    return p_base

# ==========================================
# 2. Observation Layer: Structural Coupling
# ==========================================
def apply_beta_noise(p_base, K, size):
    if p_base <= 0.0: return np.zeros(size)
    if p_base >= 1.0: return np.ones(size)
    alpha = p_base * K
    beta_param = (1.0 - p_base) * K
    return np.random.beta(alpha, beta_param, size)

# ==========================================
# 3. Macroscopic Phase Space Engine
# ==========================================
def explore_nonlinear_phase_space():
    output_dir = "nonlinear_results"
    os.makedirs(output_dir, exist_ok=True)

    # Variables for Phase Space
    dt_vals = np.linspace(0.1, 1.5, 30)       # X-axis: Observation Interval
    gamma_vals = np.linspace(0.5, 3.0, 30)    # Y-axis: Acceleration Parameter
    
    K_noise = 100   # Assessor bias variance
    N_macro = 5000  # Cohort size for macroscopic evaluation
    
    skip_rates = np.zeros((len(gamma_vals), len(dt_vals)))
    
    print("Running Non-Linear Monte Carlo Phase Space Exploration...")
    print("This may take a minute due to SDE path integration...")
    
    for i, gamma in enumerate(gamma_vals):
        for j, dt in enumerate(dt_vals):
            # Step A: Derive baseline skip probability from physical SDE paths
            p_base = simulate_nonlinear_sde_skips(gamma, dt)
            
            # Step B: Apply structural beta noise
            p_eff = apply_beta_noise(p_base, K_noise, N_macro)
            
            # Step C: Macroscopic evaluation
            skip_rates[i, j] = np.sum(np.random.rand(N_macro) < p_eff) / N_macro

    # ==========================================
    # 4. Data Export & Visualization
    # ==========================================
    # Save CSV
    df = pd.DataFrame(skip_rates, index=np.round(gamma_vals, 3), columns=np.round(dt_vals, 3))
    df.index.name = "Gamma"
    df.columns.name = "dt"
    csv_path = os.path.join(output_dir, "nonlinear_skip_matrix.csv")
    df.to_csv(csv_path)

    # Figure 1: Heatmap
    plt.figure(figsize=(10, 8))
    ax1 = sns.heatmap(skip_rates, 
                     xticklabels=np.round(dt_vals, 2), 
                     yticklabels=np.round(gamma_vals, 2),
                     cmap="magma", cbar_kws={'label': 'Skip Degeneracy Rate (A->C)'})
    ax1.invert_yaxis()
    plt.title(r"Phase Space of Cognitive Degeneracy vs Acceleration ($\gamma$)", fontsize=14, pad=15)
    plt.xlabel(r"Observation Interval ($\Delta t$)", fontsize=12)
    plt.ylabel(r"Acceleration Parameter ($\gamma$)", fontsize=12)
    ax1.set_xticks(ax1.get_xticks()[::3])
    ax1.set_yticks(ax1.get_yticks()[::3])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_nonlinear_heatmap.png"), dpi=300)
    plt.close()

    # Figure 2: Cross-section Line Plot
    plt.figure(figsize=(10, 6))
    indices = [5, 15, 25]  # Select different gamma levels
    for idx in indices:
        plt.plot(dt_vals, skip_rates[idx, :], marker='o', markersize=4, 
                 label=fr"$\gamma$ = {gamma_vals[idx]:.2f}")
    plt.axhline(0.2, color='red', linestyle='--', alpha=0.6, label="Tolerance Threshold (20%)")
    plt.title(r"Collapse of Homogenität: Skip Rates by $\gamma$", fontsize=14)
    plt.xlabel(r"Observation Interval ($\Delta t$)", fontsize=12)
    plt.ylabel("Skip Degeneracy Rate (A->C)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_nonlinear_lineplot.png"), dpi=300)
    plt.close()

    # Figure 3: 3D Surface
    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(111, projection='3d')
    DT, GAM = np.meshgrid(dt_vals, gamma_vals)
    surf = ax3d.plot_surface(DT, GAM, skip_rates, cmap='magma', edgecolor='none', alpha=0.9)
    ax3d.set_title("3D Boundary of Statistical Homogeneity", fontsize=14)
    ax3d.set_xlabel(r"$\Delta t$", fontsize=12)
    ax3d.set_ylabel(r"$\gamma$", fontsize=12)
    ax3d.set_zlabel("Skip Rate", fontsize=12)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)
    plt.savefig(os.path.join(output_dir, "fig3_nonlinear_surface.png"), dpi=300)
    plt.close()

    print("Graphs generated and saved.")

    # ==========================================
    # 5. Zip and Download
    # ==========================================
    zip_filename = "nonlinear_degeneracy_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files_in_dir in os.walk(output_dir):
            for file in files_in_dir:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.basename(file_path))
                
    print(f"Results zipped successfully into {zip_filename}")
    
    try:
        files.download(zip_filename)
        print("Download triggered automatically.")
    except Exception as e:
        print(f"Error triggering download: {e}")

if __name__ == "__main__":
    explore_nonlinear_phase_space()
