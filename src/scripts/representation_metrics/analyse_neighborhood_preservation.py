#!/usr/bin/env python3
"""
Representation Neighborhood Preservation Analysis: JEPA vs. Encoder-Decoder

This script evaluates how well the local neighborhood structure of the original
state space is preserved in the learned latent space. It uses the well-established
Trustworthiness and Continuity metrics, now with rigorous statistical validation.

**Methodology:**
1.  **Core Comparison:** The analysis directly compares the encoders from the JEPA
    and Encoder-Decoder training approaches.
2.  **Metrics (Van der Maaten et al., 2008):**
    -   **Trustworthiness (T):** Measures the fraction of points in a latent
        neighborhood that are *not* true neighbors. A high score (near 1.0)
        indicates that the model does not create false, untrustworthy neighbors.
    -   **Continuity (C):** Measures the fraction of points from a true neighborhood
        that are missing from the latent neighborhood. A high score (near 1.0)
        indicates that the model does not break apart the original structure.
3.  **Statistical Rigor:**
    -   **Bootstrapping:** The analysis is performed over multiple bootstrap samples
        of the data. This provides a robust estimate of the mean and standard
        deviation for each metric, ensuring that the comparison is statistically
        meaningful.
4.  **k-NN:** The metrics are computed for multiple neighborhood sizes (k-values)
    using an efficient implementation from scikit-learn.

**Output:**
-   A single PNG image (`neighborhood_preservation.png`) with bar charts for
    Trustworthiness and Continuity scores, including error bars (std. dev.).
-   A summary table of the metrics (mean ± std. dev.) printed to the console.
"""

import sys
from pathlib import Path
import logging
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_encoder
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
    encoder = init_encoder(config_path).to(device)
    weights_path = project_root / "weights" / model_type / "best_encoder.pth"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find weights for {model_type} at {weights_path}")
        
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
    """
    representations = encoder(state)[:, -1, :]
    # Apply L2 normalization to ensure scale-invariant comparisons
    return torch.nn.functional.normalize(representations, p=2, dim=1)

def get_all_states(dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    """Collects all states from the dataloader."""
    all_states = []
    for batch in dataloader:
        state, _, _, _ = batch
        all_states.append(state)
    return torch.cat(all_states, dim=0)

def compute_ranks(dist_matrix: np.ndarray) -> np.ndarray:
    """Computes the rank of each point in the distance matrix."""
    return np.argsort(np.argsort(dist_matrix, axis=1, kind='stable'), axis=1, kind='stable')

def compute_trustworthiness_continuity(
    X_true: np.ndarray, X_latent: np.ndarray, k: int
) -> tuple[float, float]:
    """
    Computes Trustworthiness and Continuity with statistical rigor.
    This implementation is inspired by scikit-learn's internal functions.
    """
    n_samples = X_true.shape[0]
    
    # Find k-NN in both spaces
    nbrs_true = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_true)
    true_knn_indices = nbrs_true.kneighbors(return_distance=False)
    
    nbrs_latent = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_latent)
    latent_knn_indices = nbrs_latent.kneighbors(return_distance=False)

    # Compute full distance matrices and ranks (computationally intensive part)
    true_dists = np.sum((X_true[:, np.newaxis, :] - X_true[np.newaxis, :, :]) ** 2, axis=2)
    latent_dists = np.sum((X_latent[:, np.newaxis, :] - X_latent[np.newaxis, :, :]) ** 2, axis=2)
    
    true_ranks = compute_ranks(true_dists)
    latent_ranks = compute_ranks(latent_dists)

    # --- Trustworthiness ---
    trust_penalty = 0.0
    for i in range(n_samples):
        # Intruders are points in the latent neighborhood but not in the true one
        intruders = np.setdiff1d(latent_knn_indices[i], true_knn_indices[i])
        if intruders.size > 0:
            # Penalty is based on the rank of these intruders in the *true* space
            ranks = true_ranks[i, intruders]
            trust_penalty += np.sum(ranks - k)
            
    # --- Continuity ---
    cont_penalty = 0.0
    for i in range(n_samples):
        # Extrusions are points in the true neighborhood but not in the latent one
        extrusions = np.setdiff1d(true_knn_indices[i], latent_knn_indices[i])
        if extrusions.size > 0:
            # Penalty is based on the rank of these extrusions in the *latent* space
            ranks = latent_ranks[i, extrusions]
            cont_penalty += np.sum(ranks - k)

    # Normalization factor
    norm_factor = (2 / (n_samples * k * (2 * n_samples - 3 * k - 1)))
    
    trustworthiness = 1.0 - (norm_factor * trust_penalty)
    continuity = 1.0 - (norm_factor * cont_penalty)
    
    return trustworthiness, continuity

def run_analysis_for_sample(
    encoder: torch.nn.Module,
    states_sample: torch.Tensor,
    k_values: list[int],
    device: torch.device,
) -> dict:
    """Runs the full T&C analysis for a single sample of data."""
    results = {k: {} for k in k_values}
    
    # Flatten states for true space distance calculation
    states_sample_flat = states_sample.cpu().numpy().reshape(states_sample.size(0), -1)
    
    # Get latent representations
    with torch.no_grad():
        latent_reps = get_reps(encoder, states_sample.to(device)).cpu().numpy()
        
    for k in k_values:
        T, C = compute_trustworthiness_continuity(states_sample_flat, latent_reps, k)
        results[k]['T'] = T
        results[k]['C'] = C
        
    return results

def create_plots(results: dict, k_values: list[int], output_path: Path):
    """Generates and saves the final analysis plots with error bars."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle('Neighborhood Preservation Analysis (with Bootstrap Error)', fontsize=16, y=0.98)
    colors = {ModelType.JEPA: 'royalblue', ModelType.ENCODER_DECODER: 'coral'}

    model_types = list(results.keys())
    x = np.arange(len(k_values))
    width = 0.35

    # --- Plot 1: Trustworthiness ---
    ax1.set_title('Trustworthiness', fontsize=12)
    for i, model_type in enumerate(model_types):
        means = [results[model_type][k]['T_mean'] for k in k_values]
        stds = [results[model_type][k]['T_std'] for k in k_values]
        ax1.bar(x + i * width - width/2, means, width, yerr=stds, capsize=5,
                label=model_type.upper(), color=colors[model_type])
    
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Neighborhood Size (k)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(k_values)
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # --- Plot 2: Continuity ---
    ax2.set_title('Continuity', fontsize=12)
    for i, model_type in enumerate(model_types):
        means = [results[model_type][k]['C_mean'] for k in k_values]
        stds = [results[model_type][k]['C_std'] for k in k_values]
        ax2.bar(x + i * width - width/2, means, width, yerr=stds, capsize=5,
                label=model_type.upper(), color=colors[model_type])

    ax2.set_xlabel('Neighborhood Size (k)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(k_values)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    logging.info(f"Analysis plots saved to {output_path}")

def save_csv_data(results: dict, k_values: list[int], output_dir: Path):
    """Saves analysis results as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Summary Statistics CSV ---
    summary_data = []
    for model_type in results:
        for k in k_values:
            scores = results[model_type][k]
            summary_data.append({
                'Model': model_type.upper(),
                'Neighborhood_Size_k': k,
                'Trustworthiness_Mean': scores['T_mean'],
                'Trustworthiness_Std': scores['T_std'],
                'Continuity_Mean': scores['C_mean'],
                'Continuity_Std': scores['C_std']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = output_dir / "neighborhood_preservation_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logging.info(f"Summary statistics saved to {summary_csv_path}")
    
    return summary_csv_path

def main():
    """Main function to run the analysis."""
    config_path = str(project_root / "config.yaml")
    device = set_device()

    # --- 1. Configuration ---
    k_values = [5, 10, 15] # Adjusted for computational cost
    n_bootstraps = 10 # Number of bootstrap samples
    sample_size = 200 # Sample size for each bootstrap run (critical for performance)

    # --- 2. Load Data ---
    logging.info("Loading validation data...")
    data_pipeline = DataLoadingPipeline(batch_size=128, config_path=config_path)
    _, val_dataloader = data_pipeline.run_pipeline()
    if val_dataloader is None:
        logging.error("Validation dataloader not found. Cannot proceed.")
        return

    logging.info("Collecting all validation states...")
    all_states = get_all_states(val_dataloader)
    logging.info(f"Found {len(all_states)} total states.")

    # --- 3. Analyze Models with Bootstrapping ---
    final_results = {ModelType.JEPA: {}, ModelType.ENCODER_DECODER: {}}

    for model_type in [ModelType.JEPA, ModelType.ENCODER_DECODER]:
        encoder = load_model(model_type, config_path, device)
        
        # Store scores from all bootstrap runs
        bootstrap_scores = defaultdict(lambda: {'T': [], 'C': []})
        
        desc = f"Bootstrapping {model_type.upper()}"
        for i in tqdm(range(n_bootstraps), desc=desc):
            # Resample data for this bootstrap iteration
            states_sample = resample(all_states, n_samples=sample_size, random_state=i)
            
            # Run analysis on the sample
            sample_results = run_analysis_for_sample(encoder, states_sample, k_values, device)
            
            # Collect scores
            for k in k_values:
                bootstrap_scores[k]['T'].append(sample_results[k]['T'])
                bootstrap_scores[k]['C'].append(sample_results[k]['C'])
        
        # Compute mean and std dev from bootstrap runs
        for k in k_values:
            final_results[model_type][k] = {
                'T_mean': np.mean(bootstrap_scores[k]['T']),
                'T_std': np.std(bootstrap_scores[k]['T']),
                'C_mean': np.mean(bootstrap_scores[k]['C']),
                'C_std': np.std(bootstrap_scores[k]['C']),
            }

    # --- 4. Print Summary Table ---
    print("\n" + "="*70)
    print("Neighborhood Preservation Analysis Summary (Mean ± Std. Dev.)".center(70))
    print("="*70)
    header = f"{'k':<5}{'Model':<20}{'Trustworthiness':<25}{'Continuity':<25}"
    print(header)
    print("-"*70)
    for k in k_values:
        for model_type in final_results:
            scores = final_results[model_type][k]
            t_str = f"{scores['T_mean']:.4f} ± {scores['T_std']:.4f}"
            c_str = f"{scores['C_mean']:.4f} ± {scores['C_std']:.4f}"
            row = f"{k:<5}{model_type.upper():<20}{t_str:<25}{c_str:<25}"
            print(row)
        if k != k_values[-1]:
            print("-"*30)
    print("="*70 + "\n")

    # --- 5. Generate and Save Plots ---
    output_dir = project_root / "evaluation_plots" / "neighborhood_preservation"
    output_path = output_dir / "neighborhood_preservation.png"
    create_plots(final_results, k_values, output_path)
    
    # --- 6. Save CSV Data ---
    summary_csv = save_csv_data(final_results, k_values, output_dir)
    
    print(f"Analysis complete. View results at: {output_path}")
    print(f"Summary data saved to: {summary_csv}")

if __name__ == "__main__":
    main()