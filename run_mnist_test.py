#!/usr/bin/env python3
"""
Run ViTMAE pretraining on MNIST data for testing purposes.
This script uses MNIST data instead of the original video data to test if ViTMAE can learn.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.scripts.vitmae_pretrain import ViTMAETrainer

def main():
    """Run ViTMAE training on MNIST data."""
    print("🧪 Testing ViTMAE learning on MNIST data")
    print("=" * 50)
    
    # Create trainer with a simple config for testing
    trainer = ViTMAETrainer()
    
    # Reduce epochs for quick testing
    trainer.num_epochs = 3
    print(f"Modified epochs to {trainer.num_epochs} for quick testing")
    
    # Initialize everything
    print("Initializing models...")
    trainer.initialize_models()
    
    print("Initializing optimizer...")
    trainer.initialize_optimizer()
    
    print("Loading MNIST data...")
    trainer.load_data(use_mnist=True)
    
    print("Starting training...")
    trainer.train()
    
    print("Training completed! 🎉")

if __name__ == "__main__":
    main()
