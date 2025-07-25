#!/usr/bin/env python3
"""
Representation Geometry Analysis: JEPA vs. Encoder-Decoder

This script provides geometric analysis of learned representations from models 
trained with JEPA and Encoder-Decoder methods.

**Methodology:**
1.  **Core Comparison:** The analysis directly compares the encoders from the two
    primary training approaches: JEPA and Encoder-Decoder.
2.  **Metrics:** Geometric properties are evaluated using four complementary metrics:
    -   **Uniformity on Hypersphere:** Measures how uniformly distributed the 
        normalized representations are on the unit hypersphere.
    -   **Silhouette Score:** Evaluates clustering quality and separation in the 
        representation space.
    -   **Clustering Quality:** Assesses how well the representations cluster using
        k-means and mutual information metrics.
    -   **k-NN Label Consistency:** Measures neighborhood coherence using k-nearest
        neighbors classification accuracy.
3.  **Data:** The analysis uses states from the validation dataset to extract
    representations and compute geometric properties.

**Output:**
-   A comprehensive PNG image (`geometry_analysis.png`) containing multiple
    panels showing all geometric metrics and comparisons.
-   Key metrics printed to the console in a formatted summary table.

**Research Significance:**
-   Better geometric properties may indicate more structured, learnable representations
-   Comparing JEPA vs. Encoder-Decoder reveals architectural biases in representation geometry
-   These metrics complement other representation quality measures
"""

import torch
import sys
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, Any, Union
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

# Scikit-learn imports
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist, squareform

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


def compute_uniformity_hypersphere(X: np.ndarray, y: Optional[np.ndarray] = None, t: float = 2.0) -> float:
    """
    Compute uniformity on the hypersphere.
    
    Args:
        X: Input representations of shape (n_samples, n_features)
        y: Optional labels (not used for this metric)
        t: Temperature parameter (default: 2.0)
        
    Returns:
        Uniformity score (lower is more uniform)
    """
    if X.shape[0] < 2:
        return float('nan')
    
    # Normalize each row to unit length
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    
    # Compute pairwise squared Euclidean distances
    distances_sq = pdist(X_norm, metric='sqeuclidean')
    
    if len(distances_sq) == 0:
        return float('nan')
    
    # Apply the uniformity formula
    # uniformity = log(mean(exp(-t * D2_offdiag)))
    exp_terms = np.exp(-t * distances_sq)
    uniformity = np.log(np.mean(exp_terms) + 1e-8)
    
    return uniformity


def compute_silhouette_score(X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> float:
    """
    Compute silhouette score for clustering quality.
    
    Args:
        X: Input representations of shape (n_samples, n_features)
        y: Optional labels for supervised silhouette computation
        **kwargs: Additional arguments (unused)
        
    Returns:
        Silhouette score (higher is better, range [-1, 1])
    """
    if X.shape[0] < 2:
        return float('nan')
    
    try:
        if y is not None and len(np.unique(y)) > 1:
            # Use provided labels
            return silhouette_score(X, y, metric='euclidean')
        else:
            # No labels provided, use unsupervised clustering
            # Use a reasonable number of clusters based on data size
            n_clusters = min(max(2, X.shape[0] // 10), 10)
            if X.shape[0] < n_clusters:
                n_clusters = 2
            
            # Fit k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Check if we have more than one cluster
            if len(np.unique(labels)) < 2:
                return float('nan')
            
            return silhouette_score(X, labels, metric='euclidean')
    except Exception as e:
        logging.warning(f"Silhouette score computation failed: {e}")
        return float('nan')


def compute_clustering_quality(X: np.ndarray, y: Optional[np.ndarray] = None, 
                             n_clusters: int = 5, **kwargs) -> Dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Args:
        X: Input representations of shape (n_samples, n_features)
        y: Optional true labels for supervised evaluation
        n_clusters: Number of clusters for k-means (default: 5)
        **kwargs: Additional arguments (unused)
        
    Returns:
        Dictionary containing clustering quality metrics
    """
    if X.shape[0] < n_clusters:
        n_clusters = min(2, X.shape[0])
    
    if X.shape[0] < 2:
        return {"nmi_score": float('nan'), "inertia": float('nan'), "cluster_balance": float('nan')}
    
    try:
        # Fit k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        results = {}
        
        if y is not None and len(np.unique(y)) > 1:
            # Compute normalized mutual information score
            try:
                results["nmi_score"] = normalized_mutual_info_score(y, cluster_labels)
            except Exception:
                results["nmi_score"] = float('nan')
        else:
            results["nmi_score"] = float('nan')
        
        # Compute inertia (within-cluster sum of squares)
        results["inertia"] = kmeans.inertia_
        
        # Compute cluster balance (how evenly distributed the clusters are)
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        if len(counts) > 1:
            # Use coefficient of variation (std/mean) as balance metric
            # Lower values indicate more balanced clusters
            results["cluster_balance"] = np.std(counts) / np.mean(counts)
        else:
            results["cluster_balance"] = float('nan')
        
        return results
        
    except Exception as e:
        logging.warning(f"Clustering quality computation failed: {e}")
        return {"nmi_score": float('nan'), "inertia": float('nan'), "cluster_balance": float('nan')}


def compute_knn_label_consistency(X: np.ndarray, y: Optional[np.ndarray] = None, 
                                k: int = 5, test_size: float = 0.3, **kwargs) -> float:
    """
    Compute k-NN label consistency (neighborhood hit rate).
    
    Args:
        X: Input representations of shape (n_samples, n_features)
        y: Labels for classification task
        k: Number of neighbors for k-NN (default: 5)
        test_size: Fraction of data to use for testing (default: 0.3)
        **kwargs: Additional arguments (unused)
        
    Returns:
        k-NN classification accuracy (higher is better)
    """
    if y is None or X.shape[0] < 10:
        return float('nan')
    
    # Ensure we have enough samples and multiple classes
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return float('nan')
    
    try:
        # Encode labels if they're not numeric
        if not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Ensure k is not larger than training set
        k_adjusted = min(k, X_train.shape[0] - 1)
        if k_adjusted < 1:
            return float('nan')
        
        # Train k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=k_adjusted, metric='euclidean')
        knn.fit(X_train, y_train)
        
        # Evaluate on test set
        accuracy = knn.score(X_test, y_test)
        
        return accuracy
        
    except Exception as e:
        logging.warning(f"k-NN label consistency computation failed: {e}")
        return float('nan')


def load_model(model_type: str, config_path: str, device: str):
    """Load encoder model based on type."""
    try:
        if model_type == ModelType.JEPA:
            weights_path = project_root / "weights" / "jepa" / "best_encoder.pth"
        elif model_type == ModelType.ENCODER_DECODER:
            weights_path = project_root / "weights" / "encoder_decoder" / "best_encoder.pth"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
        # Initialize model
        encoder = init_encoder(config_path).to(device)
        
        # Load weights
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        encoder.load_state_dict(state_dict)
        encoder.eval()
        
        logging.info(f"Loaded {model_type} encoder from {weights_path}")
        return encoder
        
    except Exception as e:
        logging.error(f"Failed to load {model_type} model: {e}")
        raise


def extract_all_representations(encoder, dataloader, device: str, max_samples: int = 1000):
    """Extract representations from all samples in the dataloader."""
    import torch
    
    representations = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Extracting representations")):
            if len(representations) >= max_samples:
                break
                
            state, _, _, _ = batch
            state = state.to(device)
            
            # Get representations
            z = encoder(state)  # [B, E]

            representations.append(z.cpu().numpy())

    if not representations:
        raise ValueError("No representations extracted")
    
    # Concatenate all representations
    all_representations = np.concatenate(representations, axis=0)
    
    logging.info(f"Extracted {all_representations.shape[0]} representations of dimension {all_representations.shape[1]}")
    return all_representations


def create_plots(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Generate and save the final analysis plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Representation Geometry Analysis: JEPA vs. Encoder-Decoder', 
                 fontsize=16, y=0.98)
    
    colors = {ModelType.JEPA: 'royalblue', ModelType.ENCODER_DECODER: 'coral'}
    model_types = list(results.keys())
    
    # Plot 1: Uniformity Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Uniformity on Hypersphere', fontsize=12)
    uniformity_values = [results[model]['uniformity'] for model in model_types]
    bars = ax1.bar(range(len(model_types)), uniformity_values, 
                   color=[colors[model] for model in model_types])
    ax1.set_xticks(range(len(model_types)))
    ax1.set_xticklabels([model.upper() for model in model_types])
    ax1.set_ylabel('Uniformity Score (lower = more uniform)')
    
    # Add value labels on bars
    for bar, value in zip(bars, uniformity_values):
        if not np.isnan(value):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Silhouette Score Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Silhouette Score', fontsize=12)
    silhouette_values = [results[model]['silhouette'] for model in model_types]
    bars = ax2.bar(range(len(model_types)), silhouette_values,
                   color=[colors[model] for model in model_types])
    ax2.set_xticks(range(len(model_types)))
    ax2.set_xticklabels([model.upper() for model in model_types])
    ax2.set_ylabel('Silhouette Score (higher = better)')
    
    # Add value labels on bars
    for bar, value in zip(bars, silhouette_values):
        if not np.isnan(value):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Clustering Quality - NMI Score
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('Clustering Quality (NMI)', fontsize=12)
    nmi_values = [results[model]['clustering']['nmi_score'] for model in model_types]
    bars = ax3.bar(range(len(model_types)), nmi_values,
                   color=[colors[model] for model in model_types])
    ax3.set_xticks(range(len(model_types)))
    ax3.set_xticklabels([model.upper() for model in model_types])
    ax3.set_ylabel('Normalized Mutual Info (higher = better)')
    
    # Add value labels on bars
    for bar, value in zip(bars, nmi_values):
        if not np.isnan(value):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Clustering Inertia
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title('Clustering Inertia', fontsize=12)
    inertia_values = [results[model]['clustering']['inertia'] for model in model_types]
    bars = ax4.bar(range(len(model_types)), inertia_values,
                   color=[colors[model] for model in model_types])
    ax4.set_xticks(range(len(model_types)))
    ax4.set_xticklabels([model.upper() for model in model_types])
    ax4.set_ylabel('Inertia (lower = tighter clusters)')
    
    # Add value labels on bars
    for bar, value in zip(bars, inertia_values):
        if not np.isnan(value):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 5: Cluster Balance
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title('Cluster Balance', fontsize=12)
    balance_values = [results[model]['clustering']['cluster_balance'] for model in model_types]
    bars = ax5.bar(range(len(model_types)), balance_values,
                   color=[colors[model] for model in model_types])
    ax5.set_xticks(range(len(model_types)))
    ax5.set_xticklabels([model.upper() for model in model_types])
    ax5.set_ylabel('Balance (lower = more balanced)')
    
    # Add value labels on bars
    for bar, value in zip(bars, balance_values):
        if not np.isnan(value):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 6: k-NN Accuracy
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title('k-NN Label Consistency', fontsize=12)
    knn_values = [results[model]['knn_accuracy'] for model in model_types]
    bars = ax6.bar(range(len(model_types)), knn_values,
                   color=[colors[model] for model in model_types])
    ax6.set_xticks(range(len(model_types)))
    ax6.set_xticklabels([model.upper() for model in model_types])
    ax6.set_ylabel('k-NN Accuracy (higher = better)')
    
    # Add value labels on bars
    for bar, value in zip(bars, knn_values):
        if not np.isnan(value):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 7: Summary Table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    ax7.set_title('Geometry Analysis Summary', fontsize=14, pad=20)
    
    # Create summary table
    table_data = []
    headers = ['Metric', 'JEPA', 'Encoder-Decoder', 'Better']
    
    metrics_info = [
        ('Uniformity', 'uniformity', 'Lower'),
        ('Silhouette Score', 'silhouette', 'Higher'),
        ('NMI Score', 'clustering.nmi_score', 'Higher'),
        ('Clustering Inertia', 'clustering.inertia', 'Lower'),
        ('Cluster Balance', 'clustering.cluster_balance', 'Lower'),
        ('k-NN Accuracy', 'knn_accuracy', 'Higher')
    ]
    
    for metric_name, metric_key, better_direction in metrics_info:
        # Get values using nested key access
        if '.' in metric_key:
            main_key, sub_key = metric_key.split('.')
            jepa_val = results[ModelType.JEPA][main_key][sub_key]
            ed_val = results[ModelType.ENCODER_DECODER][main_key][sub_key]
        else:
            jepa_val = results[ModelType.JEPA][metric_key]
            ed_val = results[ModelType.ENCODER_DECODER][metric_key]
        
        # Determine which is better
        if np.isnan(jepa_val) or np.isnan(ed_val):
            better = 'N/A'
        elif better_direction == 'Higher':
            better = 'JEPA' if jepa_val > ed_val else 'Encoder-Decoder'
        else:  # Lower is better
            better = 'JEPA' if jepa_val < ed_val else 'Encoder-Decoder'
        
        jepa_str = f'{jepa_val:.3f}' if not np.isnan(jepa_val) else 'N/A'
        ed_str = f'{ed_val:.3f}' if not np.isnan(ed_val) else 'N/A'
        
        table_data.append([metric_name, jepa_str, ed_str, better])
    
    # Create table
    table = ax7.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colColours=['lightgray']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Analysis plots saved to {output_path}")


def main():
    """Main function to run the geometry analysis."""
    import torch
    
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
        
        # --- 3. Compute Geometric Metrics ---
        logging.info(f"Computing geometric metrics for {model_type.upper()}...")
        
        # Note: We don't have labels in this RL context, so y=None for most metrics
        uniformity = compute_uniformity_hypersphere(representations)
        silhouette = compute_silhouette_score(representations)
        clustering_results = compute_clustering_quality(representations, n_clusters=5)
        knn_accuracy = compute_knn_label_consistency(representations)  # Will return NaN without labels
        
        results[model_type] = {
            'representations': representations,
            'uniformity': uniformity,
            'silhouette': silhouette,
            'clustering': clustering_results,
            'knn_accuracy': knn_accuracy
        }
        
        # Print results for this model
        print(f"\n{'='*50}")
        print(f"Geometry Analysis Results for: {model_type.upper()}")
        print(f"{'='*50}")
        print(f"  - Representation Shape:   {representations.shape}")
        print(f"  - Uniformity Score:       {uniformity:.4f}")
        print(f"  - Silhouette Score:       {silhouette:.4f}")
        print(f"  - NMI Score:              {clustering_results['nmi_score']:.4f}" if not np.isnan(clustering_results['nmi_score']) else "  - NMI Score:              N/A")
        print(f"  - Clustering Inertia:     {clustering_results['inertia']:.2f}")
        print(f"  - Cluster Balance:        {clustering_results['cluster_balance']:.4f}")
        print(f"  - k-NN Accuracy:          {knn_accuracy:.4f}" if not np.isnan(knn_accuracy) else "  - k-NN Accuracy:          N/A")
        print(f"{'='*50}")
    
    # --- 4. Generate and Save Plots ---
    output_dir = project_root / "evaluation_plots" / "geometry_analysis"
    output_path = output_dir / "geometry_analysis.png"
    create_plots(results, output_path)
    
    print(f"\nAnalysis complete. View results at: {output_path}")


# --- Unit Tests ---
def test_uniformity_hypersphere():
    """Test uniformity computation with synthetic data."""
    # Test case 1: Perfect uniform distribution on 2D unit circle
    n_samples = 100
    angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    X_uniform = np.column_stack([np.cos(angles), np.sin(angles)])
    
    uniformity_uniform = compute_uniformity_hypersphere(X_uniform)
    
    # Test case 2: Clustered points (should be less uniform)
    X_clustered = np.random.normal(0, 0.1, (n_samples, 2))
    X_clustered = X_clustered / np.linalg.norm(X_clustered, axis=1, keepdims=True)
    
    uniformity_clustered = compute_uniformity_hypersphere(X_clustered)
    
    # Uniform distribution should have lower (better) uniformity score
    assert uniformity_uniform < uniformity_clustered, "Uniform distribution should have better uniformity score"
    
    print("✓ Uniformity test passed")


def test_silhouette_score():
    """Test silhouette score computation."""
    # Generate well-separated clusters
    np.random.seed(42)
    cluster1 = np.random.normal([-2, -2], 0.5, (50, 2))
    cluster2 = np.random.normal([2, 2], 0.5, (50, 2))
    X_good = np.vstack([cluster1, cluster2])
    
    # Generate overlapping clusters
    cluster3 = np.random.normal([0, 0], 2, (50, 2))
    cluster4 = np.random.normal([0.5, 0.5], 2, (50, 2))
    X_bad = np.vstack([cluster3, cluster4])
    
    sil_good = compute_silhouette_score(X_good)
    sil_bad = compute_silhouette_score(X_bad)
    
    # Well-separated clusters should have higher silhouette score
    assert sil_good > sil_bad, "Well-separated clusters should have higher silhouette score"
    
    print("✓ Silhouette score test passed")


def test_clustering_quality():
    """Test clustering quality metrics."""
    # Generate data with known clusters
    np.random.seed(42)
    cluster1 = np.random.normal([0, 0], 0.5, (30, 2))
    cluster2 = np.random.normal([3, 3], 0.5, (30, 2))
    cluster3 = np.random.normal([-3, 3], 0.5, (30, 2))
    X = np.vstack([cluster1, cluster2, cluster3])
    y = np.array([0]*30 + [1]*30 + [2]*30)
    
    results = compute_clustering_quality(X, y, n_clusters=3)
    
    # Should detect good clustering
    assert not np.isnan(results['nmi_score']), "NMI score should not be NaN"
    assert results['nmi_score'] > 0.5, "NMI score should be reasonably high for well-separated clusters"
    assert results['inertia'] > 0, "Inertia should be positive"
    
    print("✓ Clustering quality test passed")


def test_knn_label_consistency():
    """Test k-NN label consistency."""
    # Generate linearly separable data
    np.random.seed(42)
    X1 = np.random.normal([0, 0], 0.5, (50, 2))
    X2 = np.random.normal([3, 3], 0.5, (50, 2))
    X = np.vstack([X1, X2])
    y = np.array([0]*50 + [1]*50)
    
    accuracy = compute_knn_label_consistency(X, y, k=5)
    
    # Should achieve high accuracy on well-separated data
    assert not np.isnan(accuracy), "Accuracy should not be NaN"
    assert accuracy > 0.8, "Accuracy should be high for well-separated data"
    
    print("✓ k-NN label consistency test passed")


def run_unit_tests():
    """Run all unit tests."""
    print("Running unit tests...")
    test_uniformity_hypersphere()
    test_silhouette_score()
    test_clustering_quality()
    test_knn_label_consistency()
    print("All unit tests passed! ✓")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze geometric properties of learned representations')
    parser.add_argument('--test', action='store_true', help='Run unit tests only')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    if args.test:
        run_unit_tests()
    else:
        main()
