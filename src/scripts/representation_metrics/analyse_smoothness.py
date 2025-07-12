#!/usr/bin/env python3
"""
Representation Smoothness Analysis

This script analyzes the smoothness of learned representations across 8 different cases:
1. JEPA encoder representations (single frame)
2. JEPA encoder representations (clip)
3. JEPA predictor representations (single frame)
4. JEPA predictor representations (clip)
5. Encoder-Decoder encoder representations (single frame)
6. Encoder-Decoder encoder representations (clip)
7. Encoder-Decoder predictor representations (single frame)
8. Encoder-Decoder predictor representations (clip)

Smoothness metric: E[(||φ(s) - φ(s')||) / (||s - s'||_env)]
Lower values indicate better local continuity in the learned representation space.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_encoder, init_predictor, load_config
from src.scripts.collect_load_data import DataCollectionPipeline
from src.utils.set_device import set_device


class TrainingApproach(Enum):
    """Training approach used for the models."""
    JEPA = "jepa"
    ENCODER_DECODER = "encoder_decoder"


class RepresentationType(Enum):
    """Type of representation to analyze."""
    ENCODER = "encoder"
    PREDICTOR = "predictor"


class InputType(Enum):
    """Input processing type."""
    SINGLE_FRAME = "single_frame"
    CLIP = "clip"


@dataclass
class SmoothnessBenchmarkCase:
    """Configuration for a single smoothness benchmark case."""
    training_approach: TrainingApproach
    representation_type: RepresentationType
    input_type: InputType
    model_weights_path: str
    
    def get_case_name(self) -> str:
        """Generate a descriptive name for this benchmark case."""
        return f"{self.training_approach.value}_{self.representation_type.value}_{self.input_type.value}"


class SmoothnessAnalyzer:
    """
    Analyzes representation smoothness across different model configurations.
    
    This class handles:
    - Loading pre-trained models from different training approaches
    - Computing smoothness metrics for encoder and predictor representations
    - Supporting both single-frame and clip-based analysis
    - Statistical analysis and visualization of results
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the smoothness analyzer.
        
        Args:
            config_path: Path to configuration file. Uses project config.yaml if None.
        """
        # Load configuration
        if config_path is None:
            config_path = project_root / "config.yaml"
        self.config = load_config(config_path)
        
        # Setup device
        self.device = set_device()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize data pipeline with batch size for analysis
        batch_size = self.config.get('training', {}).get('batch_size', 32)
        self.data_pipeline = DataCollectionPipeline(
            batch_size=batch_size,
            config_path=str(config_path)
        )
        
        # Run the full pipeline to get dataloaders
        self.train_dataloader, self.val_dataloader = self.data_pipeline.run_full_pipeline()
        
        # Extract validation data for analysis
        if self.val_dataloader is None:
            raise ValueError("No validation data available for smoothness analysis")
        
        # Model storage
        self.models: Dict[str, Dict[str, torch.nn.Module]] = {}
        
        # Results storage
        self.smoothness_results: Dict[str, Dict[int, List[float]]] = {}
        
        # Configuration extraction
        self.frame_stack_size = self.config['environment']['frame_stack_size']
        self.sequence_length = self.config['data_and_patching']['sequence_length']
        
    def _setup_logging(self):
        """Configure logging for the analyzer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _get_model_paths(self) -> Dict[str, str]:
        """
        Get paths to pre-trained model weights.
        
        Returns:
            Dictionary mapping case names to model weight paths
            
        Based on the repository structure, models are typically saved in:
        - weights/jepa/ directory for JEPA models
        - weights/encoder_decoder/ directory for Encoder-Decoder models
        """
        paths = {}
        
        # JEPA model paths (from train_jepa.py and train_jepa_decoder.py)
        jepa_encoder_path = project_root / "weights" / "jepa" / "best_encoder.pth"
        jepa_predictor_path = project_root / "weights" / "jepa" / "best_predictor.pth"
        
        # Encoder-Decoder model paths (from train_encoder_decoder.py)
        enc_dec_encoder_path = project_root / "weights" / "encoder_decoder" / "best_encoder.pth"
        enc_dec_predictor_path = project_root / "weights" / "encoder_decoder" / "best_predictor.pth"
        
        paths.update({
            f"{TrainingApproach.JEPA.value}_encoder": str(jepa_encoder_path),
            f"{TrainingApproach.JEPA.value}_predictor": str(jepa_predictor_path),
            f"{TrainingApproach.ENCODER_DECODER.value}_encoder": str(enc_dec_encoder_path),
            f"{TrainingApproach.ENCODER_DECODER.value}_predictor": str(enc_dec_predictor_path),
        })
        
        return paths
        
    def _generate_benchmark_cases(self) -> List[SmoothnessBenchmarkCase]:
        """
        Generate all 8 benchmark cases for smoothness analysis.
        
        Returns:
            List of benchmark cases covering all combinations
        """
        model_paths = self._get_model_paths()
        cases = []
        
        for training_approach in TrainingApproach:
            for representation_type in RepresentationType:
                for input_type in InputType:
                    # Get appropriate model weights path
                    path_key = f"{training_approach.value}_{representation_type.value}"
                    weights_path = model_paths.get(path_key)
                    
                    if weights_path and Path(weights_path).exists():
                        case = SmoothnessBenchmarkCase(
                            training_approach=training_approach,
                            representation_type=representation_type,
                            input_type=input_type,
                            model_weights_path=weights_path
                        )
                        cases.append(case)
                    else:
                        self.logger.warning(f"Model weights not found for {path_key}: {weights_path}")
        
        return cases
        
    def _load_models_for_case(self, case: SmoothnessBenchmarkCase) -> Dict[str, torch.nn.Module]:
        """
        Load encoder and predictor models for a specific benchmark case.
        
        Args:
            case: Benchmark case configuration
            
        Returns:
            Dictionary containing loaded models
            
        Uses init_encoder and init_predictor from src/utils/init_models.py
        """
        models = {}
        
        # Initialize encoder (based on VideoViT architecture)
        encoder = init_encoder(str(project_root / "config.yaml"))
        
        # Initialize predictor (based on LatentDynamicsPredictor architecture)
        predictor = init_predictor(str(project_root / "config.yaml"))
        
        # Load appropriate weights based on training approach
        if case.training_approach == TrainingApproach.JEPA:
            # For JEPA, load weights from JEPA training
            encoder_path = self._get_model_paths()[f"{TrainingApproach.JEPA.value}_encoder"]
            predictor_path = self._get_model_paths()[f"{TrainingApproach.JEPA.value}_predictor"]
        else:
            # For Encoder-Decoder, load weights from encoder-decoder training
            encoder_path = self._get_model_paths()[f"{TrainingApproach.ENCODER_DECODER.value}_encoder"]
            predictor_path = self._get_model_paths()[f"{TrainingApproach.ENCODER_DECODER.value}_predictor"]
        
        # Load encoder weights
        if Path(encoder_path).exists():
            encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.logger.info(f"Loaded encoder weights from {encoder_path}")
        else:
            self.logger.warning(f"Encoder weights not found: {encoder_path}")
            
        # Load predictor weights
        if Path(predictor_path).exists():
            predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
            self.logger.info(f"Loaded predictor weights from {predictor_path}")
        else:
            self.logger.warning(f"Predictor weights not found: {predictor_path}")
        
        # Move to device and set to eval mode
        encoder = encoder.to(self.device).eval()
        predictor = predictor.to(self.device).eval()
        
        models['encoder'] = encoder
        models['predictor'] = predictor
        
        return models
        
    def _compute_state_distance(self, state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
        """
        Compute environment-specific state distance.
        
        Args:
            state1, state2: State tensors to compare
            
        Returns:
            Distance tensor
            
        For visual environments, we use pixel-wise MSE as the state distance.
        """
        # Flatten spatial dimensions for pixel-wise comparison
        state1_flat = state1.view(state1.size(0), -1)
        state2_flat = state2.view(state2.size(0), -1)
        
        # Compute L2 distance in pixel space
        distance = torch.norm(state1_flat - state2_flat, dim=1)
        return distance
        
    def _extract_representation(self, 
                              models: Dict[str, torch.nn.Module], 
                              state: torch.Tensor, 
                              representation_type: RepresentationType,
                              input_type: InputType,
                              action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract representation from a model.
        
        Args:
            models: Dictionary containing encoder and predictor models
            state: Input state tensor
            representation_type: Whether to use encoder or predictor
            input_type: Whether to process single frame or clip
            action: Action tensor (required for predictor representations)
            
        Returns:
            Extracted representation tensor with consistent shape [B, D]
        """
        with torch.no_grad():
            if representation_type == RepresentationType.ENCODER:
                # Extract encoder representation
                if input_type == InputType.SINGLE_FRAME:
                    # Use only the last frame from the sequence
                    if state.dim() == 5:  # [B, T, C, H, W]
                        single_frame = state[:, :, -1:, :, :]  # [B, T, 1, H, W] - last frame
                    else:  # [B, C, H, W]
                        single_frame = state.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H, W]
                    representation = models['encoder'](single_frame)  # [B, T, E]
                else:  # CLIP
                    representation = models['encoder'](state)  # [B, T, E]
                    
            else:  # PREDICTOR
                # First get encoder representation
                if input_type == InputType.SINGLE_FRAME:
                    if state.dim() == 5:
                        single_frame = state[:, :, -1:, :, :]  # [B, T, 1, H, W]
                    else:
                        single_frame = state.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H, W]
                    encoder_repr = models['encoder'](single_frame)  # [B, T, E]
                else:  # CLIP
                    encoder_repr = models['encoder'](state)  # [B, T, E]
                
                # Then get predictor representation
                if action is not None:
                    # Ensure action is in the right format for predictor [B]
                    if action.dim() > 1:
                        action = action[:, -1]  # Use last action [B]
                    action = action.long()  # Ensure correct dtype
                    representation = models['predictor'](encoder_repr, action)  # [B, 1, E]
                else:
                    raise ValueError("Action required for predictor representations")
        
        # Flatten to [B, D] for consistent distance computation
        batch_size = representation.size(0)
        representation = representation.view(batch_size, -1)
        
        return representation
        
    def _compute_smoothness_for_horizon(self, 
                                      case: SmoothnessBenchmarkCase, 
                                      models: Dict[str, torch.nn.Module],
                                      horizon: int = 1,
                                      num_samples: int = 1000) -> List[float]:
        """
        Compute smoothness metric for a specific temporal horizon.
        
        Args:
            case: Benchmark case configuration
            models: Loaded models dictionary
            horizon: Temporal distance between states (k in s_t, s_{t+k})
            num_samples: Number of state pairs to sample
            
        Returns:
            List of smoothness ratios for sampled pairs
        """
        smoothness_ratios = []
        sample_count = 0
        
        for batch in self.val_dataloader:
            if sample_count >= num_samples:
                break
                
            state, next_state, action, reward = batch
            batch_size = state.size(0)
            seq_len = state.size(1) if state.dim() == 5 else 1
            
            # Move to device
            state = state.to(self.device)
            next_state = next_state.to(self.device)
            action = action.to(self.device)
            
            # Sample pairs with temporal distance = horizon
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                    
                # For horizon > 1, we need sequential states
                if horizon == 1:
                    state1 = state[i:i+1]
                    state2 = next_state[i:i+1]
                    action1 = action[i:i+1] if action.dim() > 1 else action[i:i+1]
                else:
                    # For multi-step, we need to ensure we have enough sequence length
                    if seq_len <= horizon:
                        continue
                    state1 = state[i:i+1, :-horizon]
                    state2 = state[i:i+1, horizon:]
                    action1 = action[i:i+1] if action.dim() <= 1 else action[i:i+1, :-horizon]
                
                # Extract representations
                try:
                    repr1 = self._extract_representation(
                        models, state1, case.representation_type, case.input_type, action1
                    )
                    repr2 = self._extract_representation(
                        models, state2, case.representation_type, case.input_type, action1
                    )
                    
                    # Compute representation distance
                    repr_distance = torch.norm(repr1 - repr2, dim=-1).mean().item()
                    
                    # Compute state distance
                    state_distance = self._compute_state_distance(
                        state1.squeeze() if state1.dim() > 4 else state1,
                        state2.squeeze() if state2.dim() > 4 else state2
                    ).mean().item()
                    
                    # Avoid division by zero
                    if state_distance > 1e-8:
                        smoothness_ratio = repr_distance / state_distance
                        smoothness_ratios.append(smoothness_ratio)
                        
                    sample_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error processing sample {sample_count}: {e}")
                    continue
        
        return smoothness_ratios
        
    def analyze_case(self, case: SmoothnessBenchmarkCase, horizons: List[int] = [1, 5, 10]) -> Dict[int, List[float]]:
        """
        Analyze smoothness for a single benchmark case across multiple horizons.
        
        Args:
            case: Benchmark case to analyze
            horizons: List of temporal horizons to evaluate
            
        Returns:
            Dictionary mapping horizons to smoothness ratio lists
        """
        self.logger.info(f"Analyzing case: {case.get_case_name()}")
        
        # Load models for this case
        models = self._load_models_for_case(case)
        
        case_results = {}
        
        for horizon in horizons:
            self.logger.info(f"Computing smoothness for horizon {horizon}")
            smoothness_ratios = self._compute_smoothness_for_horizon(case, models, horizon)
            case_results[horizon] = smoothness_ratios
            
            # Log basic statistics
            if smoothness_ratios:
                mean_smoothness = np.mean(smoothness_ratios)
                std_smoothness = np.std(smoothness_ratios)
                self.logger.info(f"Horizon {horizon}: Mean={mean_smoothness:.4f}, Std={std_smoothness:.4f}, N={len(smoothness_ratios)}")
            else:
                self.logger.warning(f"No valid smoothness ratios computed for horizon {horizon}")
        
        return case_results
        
    def run_full_analysis(self, horizons: List[int] = [1, 5, 10]) -> Dict[str, Dict[int, List[float]]]:
        """
        Run smoothness analysis for all 8 benchmark cases.
        
        Args:
            horizons: List of temporal horizons to evaluate
            
        Returns:
            Complete results dictionary
        """
        self.logger.info("Starting full smoothness analysis")
        
        # Generate all benchmark cases
        cases = self._generate_benchmark_cases()
        self.logger.info(f"Generated {len(cases)} benchmark cases")
        
        # Analyze each case
        for case in tqdm(cases, desc="Analyzing smoothness cases"):
            case_name = case.get_case_name()
            try:
                case_results = self.analyze_case(case, horizons)
                self.smoothness_results[case_name] = case_results
            except Exception as e:
                self.logger.error(f"Failed to analyze case {case_name}: {e}")
                continue
        
        return self.smoothness_results
        
    def compute_statistical_significance(self) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise statistical significance tests between different cases.
        
        Returns:
            Dictionary of p-values for pairwise comparisons
        """
        significance_results = {}
        case_names = list(self.smoothness_results.keys())
        
        for horizon in [1, 5, 10]:
            significance_results[f"horizon_{horizon}"] = {}
            
            for i, case1 in enumerate(case_names):
                for j, case2 in enumerate(case_names[i+1:], i+1):
                    ratios1 = self.smoothness_results[case1].get(horizon, [])
                    ratios2 = self.smoothness_results[case2].get(horizon, [])
                    
                    if len(ratios1) > 0 and len(ratios2) > 0:
                        # Perform Mann-Whitney U test (non-parametric alternative to t-test)
                        statistic, p_value = stats.mannwhitneyu(ratios1, ratios2, alternative='two-sided')
                        comparison_key = f"{case1}_vs_{case2}"
                        significance_results[f"horizon_{horizon}"][comparison_key] = p_value
                        
        return significance_results
        
    def plot_smoothness_analysis(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of smoothness analysis results.
        
        Args:
            save_path: Path to save the plot. If None, displays the plot.
        """
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Representation Smoothness Analysis', fontsize=16)
        
        # Plot 1: Smoothness vs Horizon for all cases
        ax1 = axes[0, 0]
        horizons = [1, 5, 10]
        
        for case_name, results in self.smoothness_results.items():
            mean_smoothness = []
            std_smoothness = []
            
            for horizon in horizons:
                ratios = results.get(horizon, [])
                if ratios:
                    mean_smoothness.append(np.mean(ratios))
                    std_smoothness.append(np.std(ratios))
                else:
                    mean_smoothness.append(np.nan)
                    std_smoothness.append(np.nan)
            
            ax1.errorbar(horizons, mean_smoothness, yerr=std_smoothness, 
                        label=case_name, marker='o', capsize=5)
        
        ax1.set_xlabel('Temporal Horizon')
        ax1.set_ylabel('Smoothness Ratio')
        ax1.set_title('Smoothness vs Temporal Horizon')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training Approach Comparison
        ax2 = axes[0, 1]
        jepa_cases = [k for k in self.smoothness_results.keys() if 'jepa' in k]
        enc_dec_cases = [k for k in self.smoothness_results.keys() if 'encoder_decoder' in k]
        
        jepa_means = []
        enc_dec_means = []
        
        for horizon in horizons:
            jepa_horizon_ratios = []
            enc_dec_horizon_ratios = []
            
            for case in jepa_cases:
                ratios = self.smoothness_results[case].get(horizon, [])
                jepa_horizon_ratios.extend(ratios)
                
            for case in enc_dec_cases:
                ratios = self.smoothness_results[case].get(horizon, [])
                enc_dec_horizon_ratios.extend(ratios)
            
            jepa_means.append(np.mean(jepa_horizon_ratios) if jepa_horizon_ratios else np.nan)
            enc_dec_means.append(np.mean(enc_dec_horizon_ratios) if enc_dec_horizon_ratios else np.nan)
        
        width = 0.35
        x = np.arange(len(horizons))
        ax2.bar(x - width/2, jepa_means, width, label='JEPA', alpha=0.7)
        ax2.bar(x + width/2, enc_dec_means, width, label='Encoder-Decoder', alpha=0.7)
        
        ax2.set_xlabel('Temporal Horizon')
        ax2.set_ylabel('Mean Smoothness Ratio')
        ax2.set_title('Training Approach Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(horizons)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Representation Type Comparison
        ax3 = axes[1, 0]
        encoder_cases = [k for k in self.smoothness_results.keys() if 'encoder' in k and 'encoder_decoder' not in k]
        predictor_cases = [k for k in self.smoothness_results.keys() if 'predictor' in k]
        
        encoder_means = []
        predictor_means = []
        
        for horizon in horizons:
            encoder_horizon_ratios = []
            predictor_horizon_ratios = []
            
            for case in encoder_cases:
                ratios = self.smoothness_results[case].get(horizon, [])
                encoder_horizon_ratios.extend(ratios)
                
            for case in predictor_cases:
                ratios = self.smoothness_results[case].get(horizon, [])
                predictor_horizon_ratios.extend(ratios)
            
            encoder_means.append(np.mean(encoder_horizon_ratios) if encoder_horizon_ratios else np.nan)
            predictor_means.append(np.mean(predictor_horizon_ratios) if predictor_horizon_ratios else np.nan)
        
        ax3.bar(x - width/2, encoder_means, width, label='Encoder', alpha=0.7)
        ax3.bar(x + width/2, predictor_means, width, label='Predictor', alpha=0.7)
        
        ax3.set_xlabel('Temporal Horizon')
        ax3.set_ylabel('Mean Smoothness Ratio')
        ax3.set_title('Representation Type Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(horizons)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Input Type Comparison
        ax4 = axes[1, 1]
        single_frame_cases = [k for k in self.smoothness_results.keys() if 'single_frame' in k]
        clip_cases = [k for k in self.smoothness_results.keys() if 'clip' in k]
        
        single_frame_means = []
        clip_means = []
        
        for horizon in horizons:
            single_frame_horizon_ratios = []
            clip_horizon_ratios = []
            
            for case in single_frame_cases:
                ratios = self.smoothness_results[case].get(horizon, [])
                single_frame_horizon_ratios.extend(ratios)
                
            for case in clip_cases:
                ratios = self.smoothness_results[case].get(horizon, [])
                clip_horizon_ratios.extend(ratios)
            
            single_frame_means.append(np.mean(single_frame_horizon_ratios) if single_frame_horizon_ratios else np.nan)
            clip_means.append(np.mean(clip_horizon_ratios) if clip_horizon_ratios else np.nan)
        
        ax4.bar(x - width/2, single_frame_means, width, label='Single Frame', alpha=0.7)
        ax4.bar(x + width/2, clip_means, width, label='Clip', alpha=0.7)
        
        ax4.set_xlabel('Temporal Horizon')
        ax4.set_ylabel('Mean Smoothness Ratio')
        ax4.set_title('Input Type Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(horizons)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Smoothness analysis plot saved to {save_path}")
        else:
            plt.show()
            
    def save_results(self, save_path: str):
        """
        Save analysis results to file.
        
        Args:
            save_path: Path to save results
        """
        results_dict = {
            'smoothness_results': self.smoothness_results,
            'statistical_significance': self.compute_statistical_significance(),
            'config': self.config
        }
        
        torch.save(results_dict, save_path)
        self.logger.info(f"Results saved to {save_path}")


def main():
    """Main function to run smoothness analysis."""
    analyzer = SmoothnessAnalyzer()
    
    # Run full analysis
    results = analyzer.run_full_analysis(horizons=[1, 5, 10])
    
    # Create visualizations
    plots_dir = project_root / "evaluation_plots" / "smoothness_analysis"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer.plot_smoothness_analysis(
        save_path=str(plots_dir / "smoothness_analysis_comprehensive.png")
    )
    
    # Save results
    results_dir = project_root / "evaluation_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer.save_results(
        str(results_dir / "smoothness_analysis_results.pth")
    )
    
    print("Smoothness analysis completed successfully!")


if __name__ == "__main__":
    main()