import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import zipfile
from google.colab import files

# ==========================================
# 1. Unified Markov-Jump Engine (Vectorized)
# ==========================================
def simulate_robust_jump_randomization(sigma_AB, lam, gamma=2.5, N=50000):
    """
    Simulates a cohort facing BOTH non-linear accelerated degradation AND 
    sudden Poisson shocks (Jump-Diffusion equivalent).
    Evaluates if initial randomization (sigma_AB) can absorb the shocks.
    """
    T_max = 30.0
    dt = 0.5
    n_steps = int(T_max / dt)
    
    # Continuous Acceleration Drift
    t_array = np.linspace(dt, T_max, n_steps)
    k_AB, k_BC, k_CD = 0.05, 0.15, 0.30 
    p_base_AB = 1.0 - np.exp(-k_AB * (t_array**gamma) * dt)
    p_base_BC = 1.0 - np.exp(-k_BC * (t_array**gamma) * dt)
    p_base_CD = 1.0 - np.exp(-k_CD * (t_array**gamma) * dt)
    
    # Poisson Jump Probability (Sudden Physical Shocks)
    p_jump = 1.0 - np.exp(-lam * dt)
    
    states = np.zeros(N, dtype=np.int8)
    lifespans = np.full(N, T_max)
    
    for step in range(n_steps):
        # 1. Evaluate Sudden Physical Jumps FIRST
        idx_A = (states == 0)
        idx_B = (states == 1)
        
        jumps_A = np.random.rand(np.sum(idx_A)) < p_jump
        states[idx_A] += jumps_A * 2  # A -> C Sudden Skip
        
        jumps_B = np.random.rand(np.sum(idx_B)) < p_jump
        states[idx_B] += jumps_B * 2  # B -> D Sudden Fatal Skip
        
        # Re-evaluate indices after sudden jumps
        idx_A = (states == 0)
        idx_B = (states == 1)
        idx_C = (states == 2)
        
        # 2. Targeted Randomization: Cognitive smoothing on A -> B ONLY
        n_A = np.sum(idx_A)
        if n_A > 0:
            # Apply Gaussian noise based on the discovered winning strategy
            p_eff_AB = np.clip(p_base_AB[step] + np.random.normal(0, sigma_AB, n_A), 0, 1)
            states[idx_A] += (np.random.rand(n_A) < p_eff_AB)
            
        # 3. Strict Determinism on B -> C and C -> D (As proven necessary)
        n_B = np.sum(idx_B)
        if n_B > 0:
            states[idx_B] += (np.random.rand(n_B) < p_base_BC[step])
            
        n_C = np.sum(idx_C)
        if n_C > 0:
            trans_CD = np.random.rand(n_C) < p_base_CD[step]
            states[idx_C] += trans_CD
            
            # Record lifespan
            reached_D = np.zeros(N, dtype=bool)
            np.place(reached_D, idx_C, trans_CD)
            lifespans[reached_D] = t_array[step]
            
    return np.std(lifespans)

# ==========================================
# 2. Phase Space Exploration
# ==========================================
def explore_effective_thresholds():
    output_dir = "robust_threshold_results"
    os.makedirs(output_dir, exist_ok=True)

    # Grid search: Sudden Shock Freq vs Randomization Magnitude
    lambda_vals = np.linspace(0.01, 0.20, 20)      # X-axis: Environmental Severity
    sigma_AB_vals = np.linspace(0.0, 0.40, 25)     # Y-axis: Cognitive Randomization
    N_trials = 50000 
    
    std_matrix = np.zeros((len(sigma_AB_vals), len(lambda_vals)))
    
    print(f"Running Synthesis Phase Space Search: {len(lambda_vals)*len(sigma_AB_vals)} grids...")
    
    for i, sig_AB in enumerate(sigma_AB_vals):
        for j, lam in enumerate(lambda_vals):
            std_matrix[i, j] = simulate_robust_jump_randomization(sig_AB, lam, gamma=2.5, N=N_trials)

    # ==========================================
    # 3. Data Export & Visualization
    # ==========================================
    # CSV
    df_std = pd.DataFrame(std_matrix, index=np.round(sigma_AB_vals, 3), columns=np.round(lambda_vals, 3))
    df_std.index.name = "Sigma_AB"
    df_std.columns.name = "Lambda"
    df_std.to_csv(os.path.join(output_dir, "shock_absorption_matrix.csv"))

    # Fig 1: Heatmap (Finding the stable valley)
    plt.figure(figsize=(10, 8))
    ax1 = sns.heatmap(std_matrix, xticklabels=np.round(lambda_vals, 3), yticklabels=np.round(sigma_AB_vals, 2), cmap="viridis_r", cbar_kws={'label': 'Lifespan Std Dev (Lower = More Stable)'})
    ax1.invert_yaxis()
    plt.title("Effective Thresholds: Absorbing Jump Shocks via Randomization", fontsize=14, pad=15)
    plt.xlabel(r"Frequency of Sudden Shocks ($\lambda$)", fontsize=12)
    plt.ylabel(r"Initial Degradation Randomization ($\sigma_{AB}$)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_effective_threshold_heatmap.png"), dpi=300)
    plt.close()

    # Fig 2: Cross-section (The "U-shape" optimization curve)
    plt.figure(figsize=(10, 6))
    indices = [5, 10, 15] # Different shock levels
    for idx in indices:
        plt.plot(sigma_AB_vals, std_matrix[:, idx], marker='o', label=fr"Shock $\lambda$ = {lambda_vals[idx]:.3f}")
    plt.title("Optimization Curve: Finding the Minimum Variance Threshold", fontsize=14)
    plt.xlabel(r"Magnitude of Randomization ($\sigma_{AB}$)", fontsize=12)
    plt.ylabel("Lifespan Standard Deviation", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_optimization_curve.png"), dpi=300)
    plt.close()

    # Fig 3: 3D Surface
    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(111, projection='3d')
    LAM, SIG = np.meshgrid(lambda_vals, sigma_AB_vals)
    surf = ax3d.plot_surface(LAM, SIG, std_matrix, cmap='viridis_r', edgecolor='none', alpha=0.9)
    ax3d.set_title("3D Boundary of Robust Homogenität", fontsize=14)
    ax3d.set_xlabel(r"$\lambda$ (Shock Freq)", fontsize=12)
    ax3d.set_ylabel(r"$\sigma_{AB}$ (Randomization)", fontsize=12)
    ax3d.set_zlabel("Lifespan Std Dev", fontsize=12)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)
    plt.savefig(os.path.join(output_dir, "fig3_robust_surface.png"), dpi=300)
    plt.close()

    print("Synthesis complete. Graphs generated.")

    # ==========================================
    # 4. Zip and Download
    # ==========================================
    zip_filename = "robust_threshold_synthesis.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files_in_dir in os.walk(output_dir):
            for file in files_in_dir:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.basename(file_path))
                
    files.download(zip_filename)

if __name__ == "__main__":
    explore_effective_thresholds()
