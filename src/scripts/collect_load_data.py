#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader

# Add project root to path for imports (two levels up from scripts)
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_utils import load_config
from src.data.data_utils import collect_ppo_episodes, _load_existing_dataset
from src.data.dataset import ExperienceDataset


class DataCollectionPipeline:
    """
    High-level pipeline for RL data collection and DataLoader creation.
    
    This class encapsulates the entire data collection workflow, providing
    a clean interface for both standalone usage and integration into larger
    training systems.
    """
    
    def __init__(self, batch_size: int, config_path: str = None):
        """
        Initialize the data collection pipeline.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # If no config path provided, use the default from config_utils
        self.config_path = config_path
        self.config = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None  
        self.batch_size = batch_size

        # Removed logging setup
        
    def load_configuration(self) -> dict:
        """
        Load and validate the configuration file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        self.config = load_config(self.config_path)

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            self.batch_size = self.config['training']['main_loops']['batch_size']
        
        # Validate critical configuration parameters
        self._validate_config()
        
        return self.config
    
    def _validate_config(self):
        """Validate that all required configuration parameters are present."""
        required_sections = ['environment', 'data_collection', 'training']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate environment config
        env_config = self.config['environment']
        required_env_params = ['name']
        for param in required_env_params:
            if param not in env_config:
                raise ValueError(f"Missing required environment parameter: {param}")
            
        # Validate data and patching config
        data_and_patching_config = self.config['data_and_patching']
        required_data_and_patching_params = ['image_height', 'image_width']
        for param in required_data_and_patching_params:
            if param not in data_and_patching_config:
                raise ValueError(f"Missing required data and patching parameter: {param}")

        # Validate data collection config
        data_config = self.config['data_collection']
        required_data_params = ['num_episodes', 'max_steps_per_episode', 'validation_split']
        for param in required_data_params:
            if param not in data_config:
                raise ValueError(f"Missing required data collection parameter: {param}")
    
    def collect_data(self) -> Tuple[ExperienceDataset, ExperienceDataset]:
        """
        Collect or load experience data using PPO agents.
        
        This method handles the complete data collection workflow:
        1. Attempts to load existing datasets
        2. If unavailable, trains PPO agent and collects new data
        3. Preprocesses and validates the collected data
        4. Saves datasets for future use
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_configuration() first.")
        
        # Collect or load data using the PPO pipeline
        self.train_dataset, self.val_dataset = collect_ppo_episodes(self.config)

        print(f"Data loaded: {len(self.train_dataset)} training samples, {len(self.val_dataset) if self.val_dataset else 0} validation samples")
        
        # Validate collected data
        self._validate_datasets()
        
        return self.train_dataset, self.val_dataset
    
    def _validate_datasets(self):
        """Validate that the collected datasets are properly formatted."""
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise ValueError("No training data was collected")
        
        # Check data integrity by sampling a few items
        sample_state, sample_action, sample_reward, sample_next_state = self.train_dataset[0]
        
        # Validate tensor shapes and types
        expected_shape = (
            224, 224  # ResNet expects 224x224 input
        )
        
        if sample_state.shape[-2:] != expected_shape:
            raise ValueError(f"State shape mismatch. Expected {expected_shape}, got {sample_state.shape[-2:]}")
            
        if not isinstance(sample_action, torch.Tensor):
            raise ValueError("Actions must be torch tensors")
            
        if not isinstance(sample_reward, torch.Tensor):
            raise ValueError("Rewards must be torch tensors")
    
    def create_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create optimized PyTorch DataLoaders from the collected datasets.
        
        This method creates high-performance DataLoaders with:
        - Proper device memory pinning for GPU training
        - Optimized batch sizes and worker configurations
        - Appropriate shuffling strategies
        
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        if self.train_dataset is None:
            raise RuntimeError("No datasets available. Call collect_data() first.")
        
        # Extract training configuration

        batch_size = self.batch_size
        num_workers = 0
        
        # Determine optimal DataLoader settings
        pin_memory = torch.cuda.is_available() 
        persistent_workers = num_workers > 0

        print('len(train_dataset): ', len(self.train_dataset))
        
        # Create training DataLoader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Important for training stability
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True  # Ensures consistent batch sizes
        )
        
        # Create validation DataLoader if validation data exists
        self.val_dataloader = None
        if self.val_dataset and len(self.val_dataset) > 0:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,  # No shuffling for validation
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                drop_last=False  # Use all validation data
            )
        
        return self.train_dataloader, self.val_dataloader
    
    def run_full_pipeline(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Execute the complete data collection and loading pipeline.
        
        This is the main entry point for the pipeline that executes:
        1. Configuration loading and validation
        2. Data collection or loading
        3. DataLoader creation
        
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        
        # Step 1: Load configuration
        self.load_configuration()
        
        # Step 2: Collect data
        self.collect_data()        
        
        # Step 3: Create DataLoaders
        train_dataloader, val_dataloader = self.create_dataloaders()

        
        return train_dataloader, val_dataloader
    
    def get_dataset_info(self) -> dict:
        """
        Get comprehensive information about the collected datasets.
        
        Returns:
            Dictionary containing dataset statistics and metadata
        """
        if self.train_dataset is None:
            return {"error": "No datasets available"}
        
        # Sample a batch to get tensor shapes
        sample_batch = next(iter(self.train_dataloader))
        state_batch, action_batch, reward_batch, next_state_batch = sample_batch
        
        info = {
            "train_dataset_size": len(self.train_dataset),
            "val_dataset_size": len(self.val_dataset) if self.val_dataset else 0,
            "state_shape": list(state_batch.shape[1:]),  # Exclude batch dimension
            "action_shape": list(action_batch.shape[1:]) if action_batch.dim() > 1 else [1],
            "reward_shape": list(reward_batch.shape[1:]) if reward_batch.dim() > 1 else [1],
            "batch_size": state_batch.shape[0],
            "num_train_batches": len(self.train_dataloader),
            "num_val_batches": len(self.val_dataloader) if self.val_dataloader else 0,
            "environment_name": self.config['environment']['name'],
            "collection_method": "PPO"
        }
        
        return info


class DataLoadingPipeline:
    """
    Pipeline for loading existing datasets without triggering new data collection.
    
    This class is designed to be used by training scripts that assume data has
    already been collected and just need to load and create dataloaders from
    existing datasets.
    """
    
    def __init__(self, batch_size: int, config_path: str = None):
        """
        Initialize the data loading pipeline.
        
        Args:
            batch_size: Batch size for DataLoaders
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None  
        self.batch_size = batch_size
        
        # Removed logging setup
        
    def load_configuration(self) -> dict:
        """
        Load and validate the configuration file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        self.config = load_config(self.config_path)
        
        # Validate critical configuration parameters
        self._validate_config()
        
        return self.config
        
    def _validate_config(self):
        """Validate that all required configuration parameters are present."""
        required_sections = ['environment', 'data_collection', 'training']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate environment config
        env_config = self.config['environment']
        required_env_params = ['name']
        for param in required_env_params:
            if param not in env_config:
                raise ValueError(f"Missing required environment parameter: {param}")
            
        # Validate data and patching config
        data_and_patching_config = self.config['data_and_patching']
        required_data_and_patching_params = ['image_height', 'image_width']
        for param in required_data_and_patching_params:
            if param not in data_and_patching_config:
                raise ValueError(f"Missing required data and patching parameter: {param}")

        # Validate data collection config
        data_config = self.config['data_collection']
        required_data_params = ['validation_split']
        for param in required_data_params:
            if param not in data_config:
                raise ValueError(f"Missing required data collection parameter: {param}")
                
    def load_existing_data(self) -> Tuple[ExperienceDataset, ExperienceDataset]:
        """
        Load existing datasets without triggering new data collection.
        
        Returns:
            Tuple of (train_dataset, val_dataset)
            
        Raises:
            RuntimeError: If no existing datasets are found
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_configuration() first.")
        
        # Try to load existing data
        loaded_data = _load_existing_dataset(self.config)
        
        if loaded_data is None:
            raise RuntimeError(
                "No existing datasets found. Please run data collection first using "
                "the data_collection stage of the pipeline or run collect_load_data.py standalone."
            )
        
        self.train_dataset, self.val_dataset = loaded_data
        
        # Validate loaded data
        self._validate_datasets()
        
        print(f"Data loaded: {len(self.train_dataset)} training samples, {len(self.val_dataset) if self.val_dataset else 0} validation samples")
        
        return self.train_dataset, self.val_dataset
        
    def _validate_datasets(self):
        """Validate that the loaded datasets are properly formatted."""
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise ValueError("No training data was loaded")
        
        # Check data integrity by sampling a few items
        sample_state, sample_action, sample_reward, sample_next_state = self.train_dataset[0]
        
        # Validate tensor shapes and types
        expected_shape = (
            224, 224  # ResNet expects 224x224 input
        )
        
        if sample_state.shape[-2:] != expected_shape:
            raise ValueError(f"State shape mismatch. Expected {expected_shape}, got {sample_state.shape[-2:]}")
            
        if not isinstance(sample_action, torch.Tensor):
            raise ValueError("Actions must be torch tensors")
            
        if not isinstance(sample_reward, torch.Tensor):
            raise ValueError("Rewards must be torch tensors")
            
    def create_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create optimized PyTorch DataLoaders from the loaded datasets.
        
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        if self.train_dataset is None:
            raise RuntimeError("No datasets available. Call load_existing_data() first.")
        
        # Extract training configuration
        batch_size = self.batch_size
        num_workers = 0
        
        # Determine optimal DataLoader settings
        pin_memory = torch.cuda.is_available() 
        persistent_workers = num_workers > 0
        
        # Create training DataLoader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Important for training stability
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True  # Ensures consistent batch sizes
        )
        
        # Create validation DataLoader if validation data exists
        self.val_dataloader = None
        if self.val_dataset and len(self.val_dataset) > 0:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,  # No shuffling for validation
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                drop_last=False  # Use all validation data
            )
        
        return self.train_dataloader, self.val_dataloader
        
    def run_pipeline(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Execute the data loading pipeline (load existing data only).
        
        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        
        # Step 1: Load configuration
        self.load_configuration()
        
        # Step 2: Load existing data
        self.load_existing_data()        
        
        # Step 3: Create DataLoaders
        train_dataloader, val_dataloader = self.create_dataloaders()

        return train_dataloader, val_dataloader


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description="RL Data Collection and Loading Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file"
    )
    
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only display dataset information without training"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the configuration and datasets"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = DataCollectionPipeline(batch_size=128, config_path=args.config)
    
    if args.validate_only:
        # Only validate configuration
        pipeline.load_configuration()
        print("✓ Configuration validation passed")
        return
    
    # Run full pipeline
    train_dataloader, val_dataloader = pipeline.run_full_pipeline()
    
    if args.info_only:
        # Display dataset information
        info = pipeline.get_dataset_info()
        print("\n" + "=" * 50)
        print("DATASET INFORMATION")
        print("=" * 50)
        for key, value in info.items():
            print(f"{key:25}: {value}")
    else:
        print("\n✓ Data collection and loading completed successfully!")
        print(f"✓ Training DataLoader ready with {len(train_dataloader)} batches")
        if val_dataloader:
            print(f"✓ Validation DataLoader ready with {len(val_dataloader)} batches")


if __name__ == "__main__":
    main()
