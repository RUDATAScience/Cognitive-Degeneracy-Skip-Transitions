import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
import os
import zipfile
from google.colab import files

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
    # 結果保存用のディレクトリ作成
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)

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
            # Step A: Derive physical baseline probability
            p_AC_base = calc_fpt_linear_cdf(mu, sig, L_AC, dt)
            
            # Step B: Superimpose observation noise
            p_AC_eff = apply_beta_noise(p_AC_base, K, N)
            
            # Step C: Macroscopic Monte Carlo evaluation
            skips = np.sum(np.random.rand(N) < p_AC_eff)
            skip_rates[i, j] = skips / N

    # ==========================================
    # 4. Export CSV
    # ==========================================
    df = pd.DataFrame(skip_rates, index=np.round(sigma_vals, 3), columns=np.round(dt_vals, 3))
    df.index.name = "Sigma"
    df.columns.name = "dt"
    csv_path = os.path.join(output_dir, "skip_rates_matrix.csv")
    df.to_csv(csv_path)
    print(f"Saved CSV: {csv_path}")

    # ==========================================
    # 5. Visualization (Multiple Graphs)
    # ==========================================
    # グラフ1: ヒートマップ (Heatmap)
    plt.figure(figsize=(10, 8))
    ax1 = sns.heatmap(skip_rates, 
                     xticklabels=np.round(dt_vals, 2), 
                     yticklabels=np.round(sigma_vals, 2),
                     cmap="viridis", cbar_kws={'label': 'Skip Degeneracy Rate (A->C)'})
    ax1.invert_yaxis()
    plt.title("Phase Space of Cognitive Degeneracy\nLinear Drift SDE vs Discrete Observation", fontsize=14, pad=15)
    plt.xlabel(r"Observation Interval ($\Delta t$)", fontsize=12)
    plt.ylabel(r"Environmental Fluctuation ($\sigma$)", fontsize=12)
    ax1.set_xticks(ax1.get_xticks()[::4])
    ax1.set_yticks(ax1.get_yticks()[::4])
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "fig1_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    # グラフ2: 断面折れ線グラフ (Cross-section Line Plot)
    plt.figure(figsize=(10, 6))
    # 異なるゆらぎ(sigma)の代表値を抽出して描画
    indices_to_plot = [5, 20, 35] 
    for idx in indices_to_plot:
        plt.plot(dt_vals, skip_rates[idx, :], marker='o', markersize=4, 
                 label=fr"$\sigma$ = {sigma_vals[idx]:.2f}")
    plt.title("Skip Degeneracy Rate vs Observation Interval", fontsize=14)
    plt.xlabel(r"Observation Interval ($\Delta t$)", fontsize=12)
    plt.ylabel("Skip Degeneracy Rate (A->C)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    lineplot_path = os.path.join(output_dir, "fig2_lineplot.png")
    plt.savefig(lineplot_path, dpi=300)
    plt.close()

    # グラフ3: 3Dサーフェスプロット (3D Surface Plot)
    fig = plt.figure(figsize=(12, 8))
    ax3d = fig.add_subplot(111, projection='3d')
    DT, SIG = np.meshgrid(dt_vals, sigma_vals)
    surf = ax3d.plot_surface(DT, SIG, skip_rates, cmap='viridis', edgecolor='none', alpha=0.9)
    ax3d.set_title("3D Surface of Phase Space", fontsize=14, pad=15)
    ax3d.set_xlabel(r"$\Delta t$", fontsize=12)
    ax3d.set_ylabel(r"$\sigma$", fontsize=12)
    ax3d.set_zlabel("Skip Rate", fontsize=12)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10, label='Skip Degeneracy Rate')
    surface_path = os.path.join(output_dir, "fig3_surface.png")
    plt.savefig(surface_path, dpi=300)
    plt.close()
    
    print("Graphs generated and saved.")

    # ==========================================
    # 6. Zip and Download
    # ==========================================
    zip_filename = "cognitive_degeneracy_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files_in_dir in os.walk(output_dir):
            for file in files_in_dir:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.basename(file_path))
                
    print(f"Results zipped successfully into {zip_filename}")
    
    # Colabの自動ダウンロードトリガー
    try:
        files.download(zip_filename)
        print("Download triggered automatically.")
    except Exception as e:
        print(f"Error triggering download (Ensure you are running in Google Colab): {e}")

if __name__ == "__main__":
    explore_phase_space()
