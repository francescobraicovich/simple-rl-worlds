#!/usr/bin/env python3
"""
Representation Smoothness Analysis: JEPA vs. Encoder-Decoder

This script provides a statistically rigorous and visually intuitive comparison of
the representation smoothness from models trained with JEPA and Encoder-Decoder
methods.

**Methodology:**
1.  **Core Comparison:** The analysis directly compares the encoders from the two
    primary training approaches: JEPA and Encoder-Decoder.
2.  **Metric:** Smoothness is evaluated by examining the relationship between
    the distance of two consecutive states in the environment's pixel space
    and their distance in the learned latent space.
    -   Pixel Distance: d_s = ||s_t - s_{t+1}||_2
    -   Latent Distance: d_z = ||phi(s_t) - phi(s_{t+1})||_2
    A smoother representation space should exhibit a strong, positive correlation
    between these two distances.
3.  **Data:** The analysis uses pairs of consecutive states (s_t, s_{t+1})
    from the validation dataset.
4.  **Analysis Techniques:**
    -   **Scatter Plot:** Visualizes d_z vs. d_s to show the direct relationship
        for each model.
    -   **Linear Regression:** A line is fit to the scatter plot data for each
        model. The slope of this line serves as a robust, single-number metric
        for smoothness. A lower, well-correlated slope is better.
    -   **Box Plot:** Compares the distribution of the raw smoothness ratio
        (d_z / d_s) for a more traditional view.

**Output:**
-   A single PNG image (`smoothness_comparison.png`) containing a scatter plot
    and a box plot, suitable for inclusion in a research paper.
-   Key metrics (regression slope, correlation coefficient) printed to the console.
"""

import os
import sys
from pathlib import Path
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
    Computes the latent representation phi(s) for a given state s.
    
    Args:
        encoder: The pre-trained encoder model.
        state: The input state tensor [B, C, T, H, W].
        
    Returns:
        The latent representation tensor [B, Embedding_Dim].
    """
    # The encoder returns [B, T, E], we take the last time step's embedding
    return encoder(state)[:, -1, :]

def compute_distances(encoder: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes pixel-space and latent-space distances for all consecutive state pairs.
    
    Args:
        encoder: The pre-trained encoder model.
        dataloader: The validation data loader.
        device: The device to run computations on.
        
    Returns:
        A tuple containing two NumPy arrays: (pixel_distances, latent_distances).
    """
    pixel_distances = []
    latent_distances = []
    
    for batch in tqdm(dataloader, desc=f"Analyzing {encoder.model_type} model"):
        state, next_state, _, _ = batch
        state, next_state = state.to(device), next_state.to(device)
        
        # --- Compute Pixel-space Distance (d_s) ---
        # Flatten the images to compute L2 norm in pixel space
        state_flat = state[:,:,-1,:,:].view(state.size(0), -1)
        next_state_flat = next_state.view(next_state.size(0), -1)
        d_s = torch.norm(state_flat - next_state_flat, dim=1)
        
        # --- Compute Latent-space Distance (d_z) ---
        z_t = get_reps(encoder, state)
        z_t_plus_1 = get_reps(encoder, next_state)
        d_z = torch.norm(z_t - z_t_plus_1, dim=1)
        
        pixel_distances.append(d_s.cpu().numpy())
        latent_distances.append(d_z.cpu().numpy())
        
    return np.concatenate(pixel_distances), np.concatenate(latent_distances)

def create_plots(results: dict, output_path: Path):
    """
    Generates and saves the final analysis plots.
    
    Args:
        results: A dictionary containing the computed distances and stats for each model.
        output_path: The path to save the final PNG image.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Representation Smoothness Analysis: JEPA vs. Encoder-Decoder', fontsize=16, y=0.98)

    colors = {ModelType.JEPA: 'royalblue', ModelType.ENCODER_DECODER: 'coral'}
    
    # --- Plot 1: Scatter Plot with Linear Regression ---
    ax1.set_title('Latent Distance vs. Pixel Distance', fontsize=12)
    
    for model_type, data in results.items():
        d_s, d_z = data['d_s'], data['d_z']
        slope, intercept, r_value, _, _ = data['regression']
        
        # Plot a random subsample for clarity
        indices = np.random.choice(len(d_s), size=min(1000, len(d_s)), replace=False)
        ax1.scatter(d_s[indices], d_z[indices], alpha=0.5, label=f'{model_type.upper()} (Sampled)', color=colors[model_type])
        
        # Plot regression line
        line_x = np.array([0, d_s.max()])
        line_y = slope * line_x + intercept
        ax1.plot(line_x, line_y, color=colors[model_type], linestyle='--', 
                 label=f'{model_type.upper()} Fit (Slope={slope:.2f}, R²={r_value**2:.2f})')

    ax1.set_xlabel('Pixel-space Distance (L2 Norm)')
    ax1.set_ylabel('Latent-space Distance (L2 Norm)')
    ax1.legend()
    
    # --- Plot 2: Box Plot of Smoothness Ratios ---
    ax2.set_title('Distribution of Smoothness Ratios (d_z / d_s)', fontsize=12)
    
    plot_data = []
    labels = []
    for model_type, data in results.items():
        # Filter out near-zero pixel distances to avoid division by zero and instability
        valid_indices = data['d_s'] > 1e-6
        ratios = data['d_z'][valid_indices] / data['d_s'][valid_indices]
        plot_data.append(ratios)
        labels.append(model_type.upper())
        
    sns.boxplot(data=plot_data, ax=ax2, palette=list(colors.values()))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Smoothness Ratio')
    ax2.set_yscale('log') # Use log scale due to potential outliers
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    logging.info(f"Analysis plots saved to {output_path}")

def main():
    """Main function to run the smoothness analysis."""
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
        
    # --- 2. Analyze Models ---
    results = {}
    for model_type in [ModelType.JEPA, ModelType.ENCODER_DECODER]:
        encoder = load_model(model_type, config_path, device)
        encoder.model_type = model_type # Attach for logging
        
        d_s, d_z = compute_distances(encoder, val_dataloader, device)
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(d_s, d_z)
        
        results[model_type] = {
            'd_s': d_s,
            'd_z': d_z,
            'regression': (slope, intercept, r_value, p_value, std_err)
        }
        
        print("\n" + "="*50)
        print(f"Analysis Results for: {model_type.upper()}")
        print("="*50)
        print(f"  - Regression Slope: {slope:.4f}")
        print(f"  - R-squared (R²):   {r_value**2:.4f}")
        print(f"  - Pearson Corr (r): {r_value:.4f}")
        print("="*50 + "\n")

    # --- 3. Generate and Save Plots ---
    output_dir = project_root / "evaluation_plots" / "smoothness_analysis"
    output_path = output_dir / "smoothness_comparison.png"
    create_plots(results, output_path)
    
    print(f"Analysis complete. View results at: {output_path}")

if __name__ == "__main__":
    main()
