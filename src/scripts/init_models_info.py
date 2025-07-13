#!/usr/bin/env python3
"""
Model Initialization and Parameter Info Script

This script initializes the encoder, predictor, and decoder models and prints
information about each model including their parameter counts. This is useful
for understanding model complexity and ensuring proper model initialization
before training begins.

This script serves as the first stage in the full training pipeline to validate
that all models can be properly instantiated and to provide visibility into
model architectures and parameter counts.
"""

import os
import sys
import argparse
from pathlib import Path

# Set MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_models import init_encoder, init_predictor, init_decoder, init_reward_predictor
from src.utils.set_device import set_device


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_parameter_count(count):
    """
    Format parameter count in human-readable format (e.g., 1.2M, 3.4K).
    
    Args:
        count: Parameter count as integer
        
    Returns:
        Formatted string with appropriate unit
    """
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.2f}K"
    else:
        return str(count)


def print_model_info(model, model_name):
    """
    Print detailed information about a model including parameter counts.
    
    Args:
        model: PyTorch model
        model_name: Name of the model for display
    """
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} MODEL INFORMATION")
    print(f"{'='*60}")
    print(f"Model Class: {model.__class__.__name__}")
    print(f"Total Parameters: {format_parameter_count(total_params)} ({total_params:,})")
    print(f"Trainable Parameters: {format_parameter_count(trainable_params)} ({trainable_params:,})")
    print(f"Non-trainable Parameters: {format_parameter_count(total_params - trainable_params)} ({total_params - trainable_params:,})")
    
    # Print basic model architecture info
    print(f"\nModel Architecture Summary:")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    
    # Count different types of layers
    layer_counts = {}
    for name, module in model.named_modules():
        module_type = module.__class__.__name__
        if module_type not in layer_counts:
            layer_counts[module_type] = 0
        layer_counts[module_type] += 1
    
    print(f"  Layer Types:")
    for layer_type, count in sorted(layer_counts.items()):
        if count > 1 and layer_type != model.__class__.__name__:  # Skip the root module
            print(f"    {layer_type}: {count}")


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description='Initialize models and display parameter information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script initializes all models (encoder, predictor, decoder, reward predictor)
and displays detailed information about their parameter counts and architecture.
This is useful for:
- Validating model initialization
- Understanding model complexity
- Comparing different architectures
- Debugging configuration issues

Examples:
  # Initialize models with default config
  python init_models_info.py
  
  # Initialize models with custom config
  python init_models_info.py --config my_config.yaml
        """
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml file')
    
    args = parser.parse_args()
    
    print("🚀 MODEL INITIALIZATION AND PARAMETER ANALYSIS")
    print("=" * 80)
    
    try:
        # Set device
        device = set_device()
        print(f"Using device: {device}")
        
        # Initialize all models
        print("\n📊 Initializing models...")
        
        print("\n1. Initializing Encoder...")
        encoder = init_encoder(args.config).to(device)
        
        print("2. Initializing Predictor...")
        predictor = init_predictor(args.config).to(device)
        
        print("3. Initializing Decoder...")
        decoder = init_decoder(args.config).to(device)
        
        print("4. Initializing Reward Predictor...")
        reward_predictor = init_reward_predictor(args.config).to(device)
        
        print("\n✅ All models initialized successfully!")
        
        # Print detailed information for each model
        print_model_info(encoder, "Encoder")
        print_model_info(predictor, "Predictor") 
        print_model_info(decoder, "Decoder")
        print_model_info(reward_predictor, "Reward Predictor")
        
        # Print summary
        encoder_total, encoder_trainable = count_parameters(encoder)
        predictor_total, predictor_trainable = count_parameters(predictor)
        decoder_total, decoder_trainable = count_parameters(decoder)
        reward_predictor_total, reward_predictor_trainable = count_parameters(reward_predictor)
        
        total_all_models = encoder_total + predictor_total + decoder_total + reward_predictor_total
        trainable_all_models = encoder_trainable + predictor_trainable + decoder_trainable + reward_predictor_trainable
        
        print(f"\n{'='*60}")
        print("SUMMARY - ALL MODELS COMBINED")
        print(f"{'='*60}")
        print(f"Total Parameters (All Models): {format_parameter_count(total_all_models)} ({total_all_models:,})")
        print(f"Trainable Parameters (All Models): {format_parameter_count(trainable_all_models)} ({trainable_all_models:,})")
        print(f"Non-trainable Parameters (All Models): {format_parameter_count(total_all_models - trainable_all_models)} ({total_all_models - trainable_all_models:,})")
        
        print(f"\nParameter Distribution:")
        print(f"  Encoder: {format_parameter_count(encoder_total)} ({encoder_total/total_all_models*100:.1f}%)")
        print(f"  Predictor: {format_parameter_count(predictor_total)} ({predictor_total/total_all_models*100:.1f}%)")
        print(f"  Decoder: {format_parameter_count(decoder_total)} ({decoder_total/total_all_models*100:.1f}%)")
        print(f"  Reward Predictor: {format_parameter_count(reward_predictor_total)} ({reward_predictor_total/total_all_models*100:.1f}%)")
        
        print(f"\n🎉 Model initialization completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during model initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
