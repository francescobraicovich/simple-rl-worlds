#!/usr/bin/env python3
"""
Representation Robustness Analysis: JEPA vs. Encoder-Decoder

This script evaluates the robustness of learned representations to input perturbations,
specifically additive Gaussian noise, for models trained with JEPA and
Encoder-Decoder methods.

**Methodology:**
1.  **Core Comparison:** The analysis directly compares the encoders from the two
    primary training approaches: JEPA and Encoder-Decoder.
2.  **Metric:** Robustness is measured by the L2 distance between the latent
    representation of a clean state `s` and its noisy counterpart `s̃`.
    -   Metric: E||φ_norm(s̃) - φ_norm(s)||_2
    where φ_norm(s) represents the L2-normalized latent representation.
    This normalization ensures scale-invariant comparisons between models
    by projecting all representations onto the unit hypersphere.
    -   A lower value indicates better robustness, meaning the representation is
        more stable under input noise.
3.  **Noise Injection:** Zero-mean Gaussian noise is added to the input states.
    The analysis is performed across a range of noise intensities (standard deviations)
    to observe how robustness degrades.
4.  **Data:** The analysis uses the validation dataset to ensure results are
    representative of the models' generalization capabilities.

**Output:**
-   A single PNG image (`robustness_comparison.png`) containing:
    a) A line plot of Mean Latent Distance vs. Noise Level.
    b) A box plot comparing distance distributions at a fixed noise level.
-   Key metrics printed to the console in a summary table.
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
from tqdm import tqdm
from collections import defaultdict

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
        logging.error(f"Weights file not found at: {weights_path}")
        raise FileNotFoundError(f"Could not find weights for {model_type} at {weights_path}")
        
    encoder.load_state_dict(torch.load(weights_path, map_location=device))
    encoder.eval()
    
    logging.info(f"Successfully loaded {model_type} encoder from {weights_path}")
    return encoder

def add_gaussian_noise(state: torch.Tensor, std_dev: float) -> torch.Tensor:
    """Adds zero-mean Gaussian noise to a state tensor."""
    noise = torch.randn_like(state) * std_dev
    return torch.clamp(state + noise, 0.0, 1.0) # Clamp to valid pixel range

@torch.no_grad()
def get_reps(encoder: torch.nn.Module, state: torch.Tensor) -> torch.Tensor:
    """
    Computes the L2-normalized latent representation phi(s) for a given state s.
    
    The L2 normalization ensures scale-invariant comparisons between different
    models by projecting all representations onto the unit hypersphere.
    """
    representations = encoder(state)
    # Apply L2 normalization to ensure scale-invariant comparisons
    return torch.nn.functional.normalize(representations, p=2, dim=1)

def compute_robustness(encoder: torch.nn.Module, dataloader: torch.utils.data.DataLoader, noise_levels: list[float], device: torch.device) -> dict:
    """
    Computes latent space distances between clean and noisy states across noise levels.
    
    Returns:
        A dictionary mapping noise levels to lists of latent distances.
    """
    results = defaultdict(list)
    
    for batch in tqdm(dataloader, desc=f"Analyzing {encoder.model_type} model"):
        state, _, _, _ = batch
        state = state.to(device)
        
        # Get representation of the clean state
        z_clean = get_reps(encoder, state)
        
        for std_dev in noise_levels:
            # Create noisy state and get its representation
            state_noisy = add_gaussian_noise(state, std_dev)
            z_noisy = get_reps(encoder, state_noisy)
            
            # Compute L2 distance and store it
            d_z = torch.norm(z_clean - z_noisy, dim=1)
            results[std_dev].extend(d_z.cpu().numpy())
            
    return {k: np.array(v) for k, v in results.items()}

def create_plots(results: dict, noise_levels: list[float], fixed_noise_level: float, output_path: Path):
    """Generates and saves the final analysis plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Representation Robustness to Additive Gaussian Noise', fontsize=16, y=0.98)

    colors = {ModelType.JEPA: 'royalblue', ModelType.ENCODER_DECODER: 'coral'}
    
    # --- Plot 1: Mean Latent Distance vs. Noise Level ---
    ax1.set_title('Robustness Curve', fontsize=12)
    
    for model_type, data in results.items():
        mean_distances = [np.mean(data[level]) for level in noise_levels]
        std_err = [np.std(data[level]) / np.sqrt(len(data[level])) for level in noise_levels]
        
        ax1.plot(noise_levels, mean_distances, marker='o', linestyle='-', label=model_type.upper(), color=colors[model_type])
        ax1.fill_between(noise_levels, 
                         np.array(mean_distances) - np.array(std_err), 
                         np.array(mean_distances) + np.array(std_err), 
                         alpha=0.2, color=colors[model_type])

    ax1.set_xlabel('Input Noise Level (Gaussian Std. Dev.)')
    ax1.set_ylabel('Mean Normalized Latent Distance E||φ_norm(s̃) - φ_norm(s)||')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # --- Plot 2: Box Plot of Distances at a Fixed Noise Level ---
    ax2.set_title(f'Distance Distribution at Noise Level {fixed_noise_level}', fontsize=12)
    
    plot_data = [results[model_type][fixed_noise_level] for model_type in results]
    labels = [model_type.upper() for model_type in results]
        
    sns.boxplot(data=plot_data, ax=ax2, palette=list(colors.values()))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Normalized Latent Distance')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    logging.info(f"Analysis plots saved to {output_path}")

def save_csv_data(results: dict, noise_levels: list[float], output_dir: Path):
    """Saves analysis results as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Summary Statistics CSV ---
    summary_data = []
    for model_type in results:
        for level in noise_levels:
            distances = results[model_type][level]
            summary_data.append({
                'Model': model_type.upper(),
                'Noise_Level': level,
                'Mean_Distance': np.mean(distances),
                'Std_Error': np.std(distances) / np.sqrt(len(distances)),
                'Std_Dev': np.std(distances),
                'Min_Distance': np.min(distances),
                'Max_Distance': np.max(distances),
                'Median_Distance': np.median(distances),
                'Sample_Count': len(distances)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = output_dir / "robustness_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logging.info(f"Summary statistics saved to {summary_csv_path}")
    
    # --- 2. Raw Distance Data CSV ---
    raw_data = []
    for model_type in results:
        for level in noise_levels:
            distances = results[model_type][level]
            for i, distance in enumerate(distances):
                raw_data.append({
                    'Model': model_type.upper(),
                    'Noise_Level': level,
                    'Sample_ID': i,
                    'Distance': distance
                })
    
    raw_df = pd.DataFrame(raw_data)
    raw_csv_path = output_dir / "robustness_raw_data.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    logging.info(f"Raw distance data saved to {raw_csv_path}")
    
    return summary_csv_path, raw_csv_path

def main():
    """Main function to run the robustness analysis."""
    config_path = str(project_root / "config.yaml")
    config = load_config(config_path)
    device = set_device()
    
    # --- 1. Configuration ---
    noise_levels = [0.01, 0.05, 0.1, 0.15, 0.2]
    fixed_plot_noise_level = 0.1 # For the boxplot
    
    # --- 2. Load Data ---
    logging.info("Loading validation data...")
    data_pipeline = DataLoadingPipeline(
        batch_size=config['training']['main_loops']['batch_size'],
        config_path=config_path
    )
    _, val_dataloader = data_pipeline.run_pipeline()
    if val_dataloader is None:
        logging.error("Validation dataloader not found. Cannot proceed.")
        return
        
    # --- 3. Analyze Models ---
    all_results = {}
    summary_data = defaultdict(list)

    for model_type in [ModelType.JEPA, ModelType.ENCODER_DECODER]:
        encoder = load_model(model_type, config_path, device)
        encoder.model_type = model_type # Attach for logging
        
        # Compute distances for all noise levels
        model_results = compute_robustness(encoder, val_dataloader, noise_levels, device)
        all_results[model_type] = model_results
        
        # Prepare data for summary table
        for level in noise_levels:
            mean_dist = np.mean(model_results[level])
            std_err = np.std(model_results[level]) / np.sqrt(len(model_results[level]))
            summary_data[level].append((model_type.upper(), mean_dist, std_err))

    # --- 4. Print Summary Table ---
    print("\n" + "="*70)
    print("Robustness Analysis Summary".center(70))
    print("Mean Normalized Latent Distance E||φ_norm(s̃) - φ_norm(s)|| (± Std. Error)".center(70))
    print("="*70)
    print(f"{'Noise Level':<15}{'Model':<20}{'Mean Distance':<25}")
    print("-"*70)
    for level, stats in summary_data.items():
        for model_name, mean_dist, std_err in stats:
            print(f"{level:<15.2f}{model_name:<20}{mean_dist:.4f} ± {std_err:.4f}")
        print("-"*70)
    print()

    # --- 5. Generate and Save Plots ---
    output_dir = project_root / "evaluation_plots" / "robustness_analysis"
    output_path = output_dir / "robustness_comparison.png"
    create_plots(all_results, noise_levels, fixed_plot_noise_level, output_path)
    
    # --- 6. Save CSV Data ---
    summary_csv, raw_csv = save_csv_data(all_results, noise_levels, output_dir)
    
    print(f"Analysis complete. View results at: {output_path}")
    print(f"Summary data saved to: {summary_csv}")
    print(f"Raw data saved to: {raw_csv}")

if __name__ == "__main__":
    main()
