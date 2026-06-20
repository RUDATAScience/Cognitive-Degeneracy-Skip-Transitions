import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import zipfile
from google.colab import files

# ==========================================
# 1. Accelerated Markov Engine (Vectorized)
# ==========================================
def simulate_cohort(sigma_AB, sigma_BC, gamma=2.0, N=100000):
    """
    Simulates a cohort of N assets under non-linear accelerated deterioration.
    Vectorized for high-performance computing (10^5 to 10^6 scale).
    """
    T_max = 30.0
    dt = 0.5
    n_steps = int(T_max / dt)
    
    # Time-dependent base probabilities (Non-linear drift: mu(t) ~ t^gamma)
    t_array = np.linspace(dt, T_max, n_steps)
    k_AB, k_BC, k_CD = 0.05, 0.15, 0.30  # Physical speed constants
    
    p_base_AB = 1.0 - np.exp(-k_AB * (t_array**gamma) * dt)
    p_base_BC = 1.0 - np.exp(-k_BC * (t_array**gamma) * dt)
    p_base_CD = 1.0 - np.exp(-k_CD * (t_array**gamma) * dt)
    
    # State array: 0=A, 1=B, 2=C, 3=D
    states = np.zeros(N, dtype=np.int8)
    lifespans = np.full(N, T_max)
    skips_AC = np.zeros(N, dtype=bool)
    
    for step in range(n_steps):
        # 1. A -> B Transitions
        idx_A = (states == 0)
        n_A = np.sum(idx_A)
        if n_A > 0:
            # Apply targeted randomization (Gaussian noise with clamping)
            p_eff_AB = np.clip(p_base_AB[step] + np.random.normal(0, sigma_AB, n_A), 0, 1)
            trans_AB = np.random.rand(n_A) < p_eff_AB
            states[idx_A] += trans_AB
            
            # Immediately evaluate B->C for those who just transitioned (Cognitive Skip A->C)
            just_transitioned = np.zeros(N, dtype=bool)
            np.place(just_transitioned, idx_A, trans_AB)
            
            n_just = np.sum(just_transitioned)
            if n_just > 0:
                p_eff_BC_skip = np.clip(p_base_BC[step] + np.random.normal(0, sigma_BC, n_just), 0, 1)
                trans_skip = np.random.rand(n_just) < p_eff_BC_skip
                states[just_transitioned] += trans_skip
                
                # Record cognitive skips
                skips_mapped = np.zeros(N, dtype=bool)
                np.place(skips_mapped, just_transitioned, trans_skip)
                skips_AC |= skips_mapped

        # 2. B -> C Transitions (Normal)
        idx_B = (states == 1) & ~skips_AC  # Exclude those who just skipped this step
        n_B = np.sum(idx_B)
        if n_B > 0:
            p_eff_BC = np.clip(p_base_BC[step] + np.random.normal(0, sigma_BC, n_B), 0, 1)
            states[idx_B] += (np.random.rand(n_B) < p_eff_BC)

        # 3. C -> D Transitions
        idx_C = (states == 2)
        n_C = np.sum(idx_C)
        if n_C > 0:
            p_eff_CD = np.clip(p_base_CD[step], 0, 1) # CD is assumed deterministic/stable
            trans_CD = np.random.rand(n_C) < p_eff_CD
            states[idx_C] += trans_CD
            
            # Record lifespan for those who reached D in this step
            reached_D = np.zeros(N, dtype=bool)
            np.place(reached_D, idx_C, trans_CD)
            lifespans[reached_D] = t_array[step]

    # Metrics for Homogeneity
    lifespan_std = np.std(lifespans)
    skip_rate = np.mean(skips_AC)
    
    return lifespan_std, skip_rate

# ==========================================
# 2. Phase Space Exploration
# ==========================================
def explore_randomization_strategies():
    output_dir = "randomization_threshold_results"
    os.makedirs(output_dir, exist_ok=True)

    # Grid search for Randomization parameters
    sigma_AB_vals = np.linspace(0.01, 0.30, 20)
    sigma_BC_vals = np.linspace(0.01, 0.30, 20)
    N_trials = 100000  # 100,000 per grid point
    
    std_matrix = np.zeros((len(sigma_BC_vals), len(sigma_AB_vals)))
    skip_matrix = np.zeros((len(sigma_BC_vals), len(sigma_AB_vals)))
    
    print(f"Running Phase Space Search: {len(sigma_AB_vals)*len(sigma_BC_vals)} grids x {N_trials} sims...")
    
    for i, sig_BC in enumerate(sigma_BC_vals):
        for j, sig_AB in enumerate(sigma_AB_vals):
            std_dev, skip = simulate_cohort(sig_AB, sig_BC, gamma=2.5, N=N_trials)
            std_matrix[i, j] = std_dev
            skip_matrix[i, j] = skip

    # ==========================================
    # 3. Data Export & Visualization
    # ==========================================
    # CSV
    df_std = pd.DataFrame(std_matrix, index=np.round(sigma_BC_vals, 3), columns=np.round(sigma_AB_vals, 3))
    df_std.to_csv(os.path.join(output_dir, "lifespan_std_matrix.csv"))

    # Fig 1: Heatmap of Lifespan StdDev (Stability of Homogenität)
    # Lower StdDev means higher stability/homogeneity
    plt.figure(figsize=(10, 8))
    ax1 = sns.heatmap(std_matrix, xticklabels=np.round(sigma_AB_vals, 2), yticklabels=np.round(sigma_BC_vals, 2),
                      cmap="coolwarm", cbar_kws={'label': 'Lifespan Std Dev (Years)'})
    ax1.invert_yaxis()
    plt.title("Stability of Homogenität via Transition Randomization\n(Lower Std Dev = Higher Stability)", fontsize=14, pad=15)
    plt.xlabel(r"Randomization of Initial Degradation ($\sigma_{A \to B}$)", fontsize=12)
    plt.ylabel(r"Randomization of Critical Transition ($\sigma_{B \to C}$)", fontsize=12)
    ax1.set_xticks(ax1.get_xticks()[::2])
    ax1.set_yticks(ax1.get_yticks()[::2])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_homogeneity_stability.png"), dpi=300)
    plt.close()

    # Fig 2: 3D Surface of Skip Degeneracy
    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(111, projection='3d')
    X_AB, Y_BC = np.meshgrid(sigma_AB_vals, sigma_BC_vals)
    surf = ax3d.plot_surface(X_AB, Y_BC, skip_matrix, cmap='magma', edgecolor='none', alpha=0.9)
    ax3d.set_title("3D Phase Space of Cognitive Skips vs Randomization", fontsize=14)
    ax3d.set_xlabel(r"$\sigma_{A \to B}$", fontsize=12)
    ax3d.set_ylabel(r"$\sigma_{B \to C}$", fontsize=12)
    ax3d.set_zlabel("Skip Rate (A->C)", fontsize=12)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10)
    plt.savefig(os.path.join(output_dir, "fig2_skip_surface.png"), dpi=300)
    plt.close()

    # Fig 3: Cross-sectional Comparison (Which one to randomize?)
    plt.figure(figsize=(10, 6))
    mid_idx = len(sigma_AB_vals) // 2
    plt.plot(sigma_AB_vals, std_matrix[0, :], 'b-o', label=r"Varying $\sigma_{A \to B}$ (with minimal $\sigma_{B \to C}$)")
    plt.plot(sigma_BC_vals, std_matrix[:, 0], 'r-x', label=r"Varying $\sigma_{B \to C}$ (with minimal $\sigma_{A \to B}$)")
    plt.title("Impact of Target Randomization on Lifespan Stability", fontsize=14)
    plt.xlabel("Magnitude of Randomization / Uncertainty", fontsize=12)
    plt.ylabel("Lifespan Standard Deviation (Years)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_strategy_comparison.png"), dpi=300)
    plt.close()

    print("Simulations complete. Graphs generated.")

    # ==========================================
    # 4. Zip and Download
    # ==========================================
    zip_filename = "randomization_strategy_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files_in_dir in os.walk(output_dir):
            for file in files_in_dir:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.basename(file_path))
                
    files.download(zip_filename)

if __name__ == "__main__":
    explore_randomization_strategies()
