#!/usr/bin/env python3
"""
Representation Neighborhood Preservation Analysis: JEPA vs. Encoder-Decoder

This script evaluates how well the local neighborhood structure of the original
state space is preserved in the learned latent space. It uses the well-established
Trustworthiness and Continuity metrics.

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
3.  **Data:** A fixed number of states are randomly sampled from the validation
    dataset to form a representative set for neighborhood analysis.
4.  **k-NN:** The metrics are computed for multiple neighborhood sizes (k-values).

**Output:**
-   A single PNG image (`neighborhood_preservation.png`) with bar charts for
    Trustworthiness and Continuity scores.
-   A summary table of the metrics printed to the console.
"""

import os
import sys
from pathlib import Path
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    """Computes the latent representation phi(s) for a given state s."""
    return encoder(state)[:, -1, :]

def get_sample_states(dataloader: torch.utils.data.DataLoader, sample_size: int) -> torch.Tensor:
    """Collects a random sample of states from the dataloader."""
    all_states = []
    for batch in dataloader:
        state, _, _, _ = batch
        all_states.append(state)
    
    states_tensor = torch.cat(all_states, dim=0)
    # Shuffle and select the exact sample size
    indices = torch.randperm(states_tensor.size(0))[:sample_size]
    return states_tensor[indices]

def find_knn(data_matrix: np.ndarray, k: int) -> np.ndarray:
    """Finds the k-nearest neighbors for each point in the data matrix."""
    n_points = data_matrix.shape[0]
    # Compute pairwise squared L2 distances
    dists_sq = np.sum(data_matrix**2, axis=1)[:, np.newaxis] - 2 * np.dot(data_matrix, data_matrix.T) + np.sum(data_matrix**2, axis=1)
    np.fill_diagonal(dists_sq, np.inf) # Exclude self from neighbors
    
    # Get indices of the k smallest distances
    knn_indices = np.argsort(dists_sq, axis=1)[:, :k]
    return knn_indices

def compute_trustworthiness_continuity(true_knn, latent_knn, k):
    """Computes Trustworthiness and Continuity."""
    n_points = true_knn.shape[0]
    
    # --- Trustworthiness ---
    trustworthiness = 0.0
    for i in range(n_points):
        latent_neighbors = set(latent_knn[i])
        true_neighbors = set(true_knn[i])
        
        intruders = latent_neighbors - true_neighbors
        if not intruders:
            continue
            
        penalty = 0.0
        for j in intruders:
            # Find rank of intruder j in true neighborhood of i
            rank = np.where(np.argsort(np.linalg.norm(true_knn - true_knn[i], axis=1)) == j)[0][0]
            penalty += (rank - k)
        trustworthiness += penalty

    # --- Continuity ---
    continuity = 0.0
    for i in range(n_points):
        latent_neighbors = set(latent_knn[i])
        true_neighbors = set(true_knn[i])
        
        extrusions = true_neighbors - latent_neighbors
        if not extrusions:
            continue

        penalty = 0.0
        for j in extrusions:
            # Find rank of extrusion j in latent neighborhood of i
            rank = np.where(np.argsort(np.linalg.norm(latent_knn - latent_knn[i], axis=1)) == j)[0][0]
            penalty += (rank - k)
        continuity += penalty

    # Normalization factor
    norm_factor = (2 / (n_points * k * (2 * n_points - 3 * k - 1)))
    
    trustworthiness = 1.0 - (norm_factor * trustworthiness)
    continuity = 1.0 - (norm_factor * continuity)
    
    return trustworthiness, continuity

def create_plots(results: dict, k_values: list[int], output_path: Path):
    """Generates and saves the final analysis plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle('Neighborhood Preservation Analysis: JEPA vs. Encoder-Decoder', fontsize=16, y=0.98)
    # Use consistent color scheme matching smoothness analysis
    colors = {ModelType.JEPA: 'royalblue', ModelType.ENCODER_DECODER: 'coral'}

    model_types = list(results.keys())
    n_models = len(model_types)
    x = np.arange(len(k_values))
    width = 0.35

    # --- Plot 1: Trustworthiness ---
    ax1.set_title('Trustworthiness', fontsize=12)
    for i, model_type in enumerate(model_types):
        scores = [results[model_type][k]['T'] for k in k_values]
        ax1.bar(x + i * width - width/2, scores, width, label=model_type.upper(), color=colors[model_type])
    
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Neighborhood Size (k)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(k_values)
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # --- Plot 2: Continuity ---
    ax2.set_title('Continuity', fontsize=12)
    for i, model_type in enumerate(model_types):
        scores = [results[model_type][k]['C'] for k in k_values]
        ax2.bar(x + i * width - width/2, scores, width, label=model_type.upper(), color=colors[model_type])

    ax2.set_xlabel('Neighborhood Size (k)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(k_values)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    logging.info(f"Analysis plots saved to {output_path}")

def main():
    """Main function to run the analysis."""
    config_path = str(project_root / "config.yaml")
    config = load_config(config_path)
    device = set_device()

    # --- 1. Configuration ---
    k_values = [5, 10, 20]
    # --- 1. Configuration ---
    k_values = [5, 10, 20]

    # --- 2. Load Data ---
    logging.info("Loading validation data...")
    data_pipeline = DataLoadingPipeline(batch_size=128, config_path=config_path)
    _, val_dataloader = data_pipeline.run_pipeline()
    if val_dataloader is None:
        logging.error("Validation dataloader not found. Cannot proceed.")
        return

    sample_size = len(val_dataloader.dataset)
    logging.info(f"Sampling {sample_size} states from validation data...")
    sample_states = get_sample_states(val_dataloader, sample_size).to(device)

    # --- 3. Analyze Models ---
    all_results = {ModelType.JEPA: {}, ModelType.ENCODER_DECODER: {}}

    # Pre-compute true neighbors
    sample_states_flat = sample_states.cpu().numpy().reshape(sample_size, -1)
    true_knn = {k: find_knn(sample_states_flat, k) for k in k_values}

    for model_type in [ModelType.JEPA, ModelType.ENCODER_DECODER]:
        encoder = load_model(model_type, config_path, device)
        
        with torch.no_grad():
            latent_reps = get_reps(encoder, sample_states).cpu().numpy()
        
        for k in tqdm(k_values, desc=f"Analyzing {model_type.upper()} for k-values"):
            latent_knn_k = find_knn(latent_reps, k)
            true_knn_k = true_knn[k]
            
            T, C = compute_trustworthiness_continuity(true_knn_k, latent_knn_k, k)
            all_results[model_type][k] = {'T': T, 'C': C}

    # --- 4. Print Summary Table ---
    print("\n" + "="*60)
    print("Neighborhood Preservation Analysis Summary".center(60))
    print("="*60)
    header = f"{'k':<5}{'Model':<20}{'Trustworthiness':<20}{'Continuity':<20}"
    print(header)
    print("-"*60)
    for k in k_values:
        for model_type in all_results:
            scores = all_results[model_type][k]
            row = f"{k:<5}{model_type.upper():<20}{scores['T']:.4f}{scores['C']:.4f}"
            print(row)
        if k != k_values[-1]: print("-"*25)
    print("="*60 + "\n")

    # --- 5. Generate and Save Plots ---
    output_dir = project_root / "evaluation_plots" / "neighborhood_preservation"
    output_path = output_dir / "neighborhood_preservation.png"
    create_plots(all_results, k_values, output_path)
    
    print(f"Analysis complete. View results at: {output_path}")

if __name__ == "__main__":
    main()
