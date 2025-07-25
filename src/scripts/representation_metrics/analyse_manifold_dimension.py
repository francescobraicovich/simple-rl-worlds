#!/usr/bin/env python3
"""
Manifold Dimension Analysis: JEPA vs. Encoder-Decoder

This script provides a comprehensive analysis of the intrinsic dimensionality and
complexity of learned representations from models trained with JEPA and 
Encoder-Decoder methods.

**Methodology:**
1.  **Core Comparison:** The analysis directly compares the encoders from the two
    primary training approaches: JEPA and Encoder-Decoder.
2.  **Metrics:** Intrinsic dimensionality is evaluated using two complementary metrics:
    -   **Participation Ratio (PR):** PR(Φ) = (∑ᵢ λᵢ)² / ∑ᵢ λᵢ² where {λᵢ} are 
        the eigenvalues of the empirical covariance of φ(s). Ranges from 1 (1D manifold)
        to D (full dimensionality used uniformly).
    -   **Two-NN Intrinsic Dimension:** Nearest-neighbor distance ratios estimator
        (Facco et al., 2017) that estimates the local intrinsic dimensionality.
3.  **Data:** The analysis uses all states from the validation dataset to extract
    representations and compute manifold properties.
4.  **Analysis Techniques:**
    -   **Participation Ratio:** Measures how uniformly the representation space
        is utilized across all dimensions.
    -   **2NN Estimation:** Provides a robust estimate of the local intrinsic
        dimensionality using nearest neighbor distance ratios.
    -   **Eigenvalue Spectrum:** Visualizes how "power" is distributed across
        representation dimensions.

**Output:**
-   A comprehensive PNG image (`manifold_dimension_analysis.png`) containing multiple
    panels showing participation ratios, intrinsic dimensions, eigenvalue spectra,
    and summary statistics.
-   Key metrics (participation ratio, intrinsic dimension) printed to the console.

**Research Significance:**
-   Higher intrinsic dimension may indicate richer, more expressive representations
-   Lower intrinsic dimension may suggest better compression and generalization
-   Comparing JEPA vs. Encoder-Decoder reveals architectural biases in representation learning
"""

import os
import sys
from pathlib import Path
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_encoder, load_config
from src.scripts.collect_load_data import DataLoadingPipeline
from src.utils.set_device import set_device

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelType:
    JEPA = "jepa"
    ENCODER_DECODER = "encoder_decoder"

def load_model(model_type: str, config_path: str, device: torch.device) -> torch.nn.Module:
    """Loads a pre-trained encoder for a given model type."""
    logging.info(f"Loading {model_type} encoder...")
    
    # Initialize model from config
    encoder = init_encoder(config_path).to(device)
    
    # Define weights path based on convention
    weights_path = project_root / "weights" / model_type / "best_encoder.pth"
    
    if not weights_path.exists():
        logging.error(f"Weights file not found at: {weights_path}")
        raise FileNotFoundError(f"Could not find weights for {model_type} at {weights_path}")
        
    # Load state dict
    encoder.load_state_dict(torch.load(weights_path, map_location=device))
    encoder.eval()
    
    logging.info(f"Successfully loaded {model_type} encoder from {weights_path}")
    return encoder

@torch.no_grad()
def get_reps(encoder: torch.nn.Module, state: torch.Tensor) -> torch.Tensor:
    """
    Computes the L2-normalized latent representation phi(s) for a given state s.
    
    The L2 normalization ensures scale-invariant comparisons between different
    models by projecting all representations onto the unit hypersphere.
    
    Args:
        encoder: The pre-trained encoder model.
        state: The input state tensor [B, T, H, W].
        
    Returns:
        The L2-normalized latent representation tensor [B, Embedding_Dim].
    """
    # The encoder returns [B, T, E], we take the last time step's embedding
    representations = encoder(state)
    # Apply L2 normalization to ensure scale-invariant comparisons
    return torch.nn.functional.normalize(representations, p=2, dim=1)

def compute_participation_ratio(representations: np.ndarray) -> float:
    """
    Computes the Participation Ratio of the representation manifold.
    
    The Participation Ratio measures how uniformly the representation space
    is utilized across all dimensions. A higher PR indicates more uniform
    usage of the available dimensionality.
    
    Args:
        representations: [N, D] array of latent representations
        
    Returns:
        PR value between 1 (1D manifold) and D (full dimensionality used)
    """
    # Center the data
    centered_reps = representations - np.mean(representations, axis=0)
    
    # Compute empirical covariance matrix
    cov_matrix = np.cov(centered_reps.T)
    
    # Get eigenvalues
    eigenvalues = np.linalg.eigvals(cov_matrix)
    eigenvalues = np.real(eigenvalues)  # Remove numerical artifacts
    eigenvalues = eigenvalues[eigenvalues > 1e-12]  # Filter near-zero eigenvalues
    
    if len(eigenvalues) == 0:
        return 1.0  # Degenerate case
    
    # Compute Participation Ratio: PR = (∑λᵢ)² / ∑λᵢ²
    sum_eig = np.sum(eigenvalues)
    sum_eig_squared = np.sum(eigenvalues**2)
    
    if sum_eig_squared == 0:
        return 1.0  # Degenerate case
    
    return (sum_eig**2) / sum_eig_squared

def compute_2nn_dimension(representations: np.ndarray, 
                         n_samples: int = 1000) -> tuple[float, float]:
    """
    Estimates intrinsic dimension using the Two-NN method (Facco et al., 2017).
    
    This method estimates the local intrinsic dimensionality by analyzing
    the ratio of distances to the first and second nearest neighbors.
    
    Args:
        representations: [N, D] array of latent representations
        n_samples: Number of points to sample for computational efficiency
        
    Returns:
        (intrinsic_dimension, estimation_error) tuple
    """
    # Sample for computational efficiency (similar to analyse_smoothness.py)
    if len(representations) > n_samples:
        indices = np.random.choice(len(representations), n_samples, replace=False)
        sample_reps = representations[indices]
    else:
        sample_reps = representations
    
    if len(sample_reps) < 3:
        return np.nan, np.nan
    
    # Find 3 nearest neighbors (self + 2 nearest)
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(sample_reps)
    distances, _ = nbrs.kneighbors(sample_reps)
    
    # Extract first and second nearest neighbor distances
    r1 = distances[:, 1]  # Distance to 1st nearest neighbor
    r2 = distances[:, 2]  # Distance to 2nd nearest neighbor
    
    # Filter out zero distances and compute ratios
    valid_mask = (r1 > 1e-12) & (r2 > 1e-12)
    ratios = r2[valid_mask] / r1[valid_mask]
    
    if len(ratios) == 0:
        return np.nan, np.nan
    
    # Estimate intrinsic dimension: ID = 1 / (log(μ) - ψ(1))
    # where μ is the mean ratio and ψ is the digamma function
    mu = np.mean(ratios)
    
    if mu <= 1.0:
        return np.nan, np.nan
    
    log_mu = np.log(mu)
    digamma_1 = digamma(1)  # ≈ -0.5772 (Euler-Mascheroni constant)
    
    denominator = log_mu - digamma_1
    if denominator <= 0:
        return np.nan, np.nan
    
    intrinsic_dim = 1.0 / denominator
    
    # Bootstrap confidence interval
    n_bootstrap = 100
    bootstrap_dims = []
    for _ in range(n_bootstrap):
        boot_indices = np.random.choice(len(ratios), len(ratios), replace=True)
        boot_ratios = ratios[boot_indices]
        boot_mu = np.mean(boot_ratios)
        
        if boot_mu > 1.0:
            boot_log_mu = np.log(boot_mu)
            boot_denominator = boot_log_mu - digamma_1
            if boot_denominator > 0:
                boot_dim = 1.0 / boot_denominator
                bootstrap_dims.append(boot_dim)
    
    if len(bootstrap_dims) == 0:
        estimation_error = np.nan
    else:
        estimation_error = np.std(bootstrap_dims)
    
    return intrinsic_dim, estimation_error

def extract_all_representations(encoder: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> np.ndarray:
    """
    Extracts all representations from the validation dataset.
    
    Args:
        encoder: The pre-trained encoder model.
        dataloader: The validation data loader.
        device: The device to run computations on.
        
    Returns:
        A NumPy array of shape [N, D] containing all representations.
    """
    all_representations = []
    
    for batch in tqdm(dataloader, desc=f"Extracting {encoder.model_type} representations"):
        state, _, _, _ = batch
        state = state.to(device)
        
        # Extract representations
        reps = get_reps(encoder, state)
        all_representations.append(reps.cpu().numpy())
    
    return np.concatenate(all_representations, axis=0)

def create_plots(results: dict, output_path: Path):
    """
    Generates and saves the manifold dimension analysis plots.
    
    Args:
        results: A dictionary containing computed metrics for each model.
        output_path: The path to save the final PNG image.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Manifold Dimension Analysis: JEPA vs. Encoder-Decoder', fontsize=16, y=0.98)
    
    # Create a 2x3 subplot layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    colors = {ModelType.JEPA: 'royalblue', ModelType.ENCODER_DECODER: 'coral'}
    model_names = list(results.keys())
    
    # Plot 1: Participation Ratio Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Participation Ratio', fontsize=12)
    pr_values = [results[model]['participation_ratio'] for model in model_names]
    bars = ax1.bar(range(len(model_names)), pr_values, 
                   color=[colors[model] for model in model_names])
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels([model.upper() for model in model_names])
    ax1.set_ylabel('Participation Ratio')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(pr_values):
        ax1.text(i, v + max(pr_values) * 0.01, f'{v:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: 2NN Intrinsic Dimension (Bar Chart with Error Bars)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Two-NN Intrinsic Dimension', fontsize=12)
    intrinsic_dims = [results[model]['intrinsic_dimension'] for model in model_names]
    dim_errors = [results[model]['dimension_error'] for model in model_names]
    
    # Handle NaN values for plotting
    plot_dims = [d if not np.isnan(d) else 0 for d in intrinsic_dims]
    plot_errors = [e if not np.isnan(e) else 0 for e in dim_errors]
    
    bars = ax2.bar(range(len(model_names)), plot_dims, 
                   yerr=plot_errors, capsize=5,
                   color=[colors[model] for model in model_names])
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels([model.upper() for model in model_names])
    ax2.set_ylabel('Intrinsic Dimension')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (v, e) in enumerate(zip(intrinsic_dims, dim_errors)):
        if not np.isnan(v):
            ax2.text(i, plot_dims[i] + plot_errors[i] + max(plot_dims) * 0.01, 
                    f'{v:.2f}±{e:.2f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(i, 0.1, 'N/A', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Dimension Usage Efficiency
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('Dimension Usage Efficiency', fontsize=12)
    efficiency = [results[model]['participation_ratio']/results[model]['nominal_dimension']*100 
                  for model in model_names]
    bars = ax3.bar(range(len(model_names)), efficiency,
                   color=[colors[model] for model in model_names])
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels([model.upper() for model in model_names])
    ax3.set_ylabel('Usage Efficiency (%)')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(efficiency):
        ax3.text(i, v + max(efficiency) * 0.01, f'{v:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 4-5: Eigenvalue Spectrum for each model
    for i, (model_type, data) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, i])
        ax.set_title(f'{model_type.upper()} Eigenvalue Spectrum', fontsize=12)
        
        # Compute and plot eigenvalues
        centered_reps = data['representations'] - np.mean(data['representations'], axis=0)
        cov_matrix = np.cov(centered_reps.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        # Plot top 50 eigenvalues or all if fewer
        n_plot = min(50, len(eigenvalues))
        ax.semilogy(range(1, n_plot + 1), eigenvalues[:n_plot], 'o-', 
                   color=colors[model_type], markersize=3)
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Eigenvalue (log scale)')
        ax.grid(True, alpha=0.3)
        
        # Add text annotation with key statistics
        max_eig = eigenvalues[0] if len(eigenvalues) > 0 else 0
        ax.text(0.05, 0.95, f'Max λ: {max_eig:.2e}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 6: Summary comparison table as text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax6.set_title('Summary Statistics', fontsize=12)
    
    summary_text = "Model Comparison:\n\n"
    for model_type, data in results.items():
        summary_text += f"{model_type.upper()}:\n"
        summary_text += f"  Nominal Dim: {data['nominal_dimension']}\n"
        summary_text += f"  Participation Ratio: {data['participation_ratio']:.3f}\n"
        if not np.isnan(data['intrinsic_dimension']):
            summary_text += f"  2NN Intrinsic Dim: {data['intrinsic_dimension']:.3f}\n"
        else:
            summary_text += f"  2NN Intrinsic Dim: N/A\n"
        summary_text += f"  Usage Efficiency: {data['participation_ratio']/data['nominal_dimension']*100:.1f}%\n\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Analysis plots saved to {output_path}")

def save_csv_data(results: dict, output_dir: Path):
    """Saves analysis results as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Summary Statistics CSV ---
    summary_data = []
    for model_type, data in results.items():
        # Calculate eigenvalue statistics
        centered_reps = data['representations'] - np.mean(data['representations'], axis=0)
        cov_matrix = np.cov(centered_reps.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        summary_data.append({
            'Model': model_type.upper(),
            'Nominal_Dimension': data['nominal_dimension'],
            'Participation_Ratio': data['participation_ratio'],
            'Intrinsic_Dimension': data['intrinsic_dimension'] if not np.isnan(data['intrinsic_dimension']) else None,
            'Dimension_Error': data['dimension_error'] if not np.isnan(data['dimension_error']) else None,
            'Usage_Efficiency_Percent': data['participation_ratio']/data['nominal_dimension']*100,
            'Max_Eigenvalue': np.max(eigenvalues) if len(eigenvalues) > 0 else 0,
            'Min_Eigenvalue': np.min(eigenvalues) if len(eigenvalues) > 0 else 0,
            'Eigenvalue_Count': len(eigenvalues),
            'Sample_Count': len(data['representations'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = output_dir / "manifold_dimension_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logging.info(f"Summary statistics saved to {summary_csv_path}")
    
    # --- 2. Eigenvalues CSV ---
    eigenvalue_data = []
    for model_type, data in results.items():
        centered_reps = data['representations'] - np.mean(data['representations'], axis=0)
        cov_matrix = np.cov(centered_reps.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        for i, eigenval in enumerate(eigenvalues):
            eigenvalue_data.append({
                'Model': model_type.upper(),
                'Eigenvalue_Index': i + 1,
                'Eigenvalue': eigenval
            })
    
    eigenval_df = pd.DataFrame(eigenvalue_data)
    eigenval_csv_path = output_dir / "manifold_eigenvalues.csv"
    eigenval_df.to_csv(eigenval_csv_path, index=False)
    logging.info(f"Eigenvalue data saved to {eigenval_csv_path}")
    
    return summary_csv_path, eigenval_csv_path

def main():
    """Main function to run the manifold dimension analysis."""
    config_path = str(project_root / "config.yaml")
    config = load_config(config_path)
    device = set_device()
    
    # --- 1. Load Data ---
    logging.info("Loading validation data...")
    data_pipeline = DataLoadingPipeline(
        batch_size=config['training']['main_loops']['batch_size'],
        config_path=config_path
    )
    _, val_dataloader = data_pipeline.run_pipeline()
    if val_dataloader is None:
        logging.error("Validation dataloader not found. Cannot proceed.")
        return
    
    # --- 2. Extract Representations and Compute Metrics ---
    results = {}
    for model_type in [ModelType.JEPA, ModelType.ENCODER_DECODER]:
        encoder = load_model(model_type, config_path, device)
        encoder.model_type = model_type  # Attach for logging
        
        # Extract all representations
        representations = extract_all_representations(encoder, val_dataloader, device)
        
        # --- 3. Compute Metrics ---
        pr = compute_participation_ratio(representations)
        intrinsic_dim, dim_error = compute_2nn_dimension(representations)
        
        results[model_type] = {
            'representations': representations,
            'participation_ratio': pr,
            'intrinsic_dimension': intrinsic_dim,
            'dimension_error': dim_error,
            'nominal_dimension': representations.shape[1]
        }
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Manifold Analysis Results for: {model_type.upper()}")
        print(f"{'='*50}")
        print(f"  - Nominal Dimension:     {representations.shape[1]}")
        print(f"  - Participation Ratio:   {pr:.4f}")
        if not np.isnan(intrinsic_dim):
            print(f"  - 2NN Intrinsic Dim:     {intrinsic_dim:.4f} ± {dim_error:.4f}")
        else:
            print(f"  - 2NN Intrinsic Dim:     N/A (computation failed)")
        print(f"  - Dimension Usage:       {pr/representations.shape[1]*100:.1f}%")
        print(f"{'='*50}\n")
    
    # --- 4. Generate Plots ---
    output_dir = project_root / "evaluation_plots" / "manifold_dimension"
    output_path = output_dir / "manifold_dimension_analysis.png"
    create_plots(results, output_path)
    
    # --- 5. Save CSV Data ---
    summary_csv, eigenval_csv = save_csv_data(results, output_dir)
    
    print(f"Analysis complete. View results at: {output_path}")
    print(f"Summary data saved to: {summary_csv}")
    print(f"Eigenvalue data saved to: {eigenval_csv}")

if __name__ == "__main__":
    main()
