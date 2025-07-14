#!/usr/bin/env python3
"""
Full Training Pipeline Script

This script orchestrates the complete training pipeline by running all training scripts
in the correct order:

1. train_encoder_decoder.py - End-to-end training of encoder, predictor, and decoder
2. train_jepa.py - Self-supervised training of encoder and predictor using JEPA
3. train_jepa_decoder.py - Training decoder using pre-trained JEPA encoder/predictor
4. train_reward_predictor.py - Training reward predictors for both approaches
5. train_dynamics_reward_predictor.py - Training dynamics-based reward predictors
6. run_representation_metrics.py - Comprehensive representation metrics analysis

Each script is executed as a subprocess with proper error handling and logging.
The pipeline can be configured to run specific stages or the complete sequence.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List

# Set MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class FullPipelineRunner:
    """
    Orchestrates the complete training pipeline by running all training scripts
    in the correct order with proper error handling and logging.
    """
    
    def __init__(self, config_path: str = None, skip_stages: List[str] = None, 
                 only_stages: List[str] = None):
        """
        Initialize the full pipeline runner.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
            skip_stages: List of stage names to skip during execution.
            only_stages: List of stage names to run exclusively (ignores skip_stages).
        """
        self.config_path = config_path
        self.skip_stages = skip_stages or []
        self.only_stages = only_stages or []
        
        # Define the training pipeline stages in order
        self.pipeline_stages = [
            {
                'name': 'model_init',
                'script': 'init_models_info.py',
                'description': 'Initialize models and display parameter information'
            },
            {
                'name': 'data_collection',
                'script': 'collect_load_data.py',
                'description': 'Collect or load training data using PPO agents'
            },
            {
                'name': 'encoder_decoder',
                'script': 'train_encoder_decoder.py',
                'description': 'End-to-end training of encoder, predictor, and decoder'
            },
            {
                'name': 'jepa',
                'script': 'train_jepa.py',
                'description': 'Self-supervised training of encoder and predictor using JEPA'
            },
            {
                'name': 'jepa_decoder',
                'script': 'train_jepa_decoder.py',
                'description': 'Training decoder using pre-trained JEPA encoder/predictor'
            },
            {
                'name': 'reward_predictor',
                'script': 'train_reward_predictor.py',
                'description': 'Training reward predictors for both approaches'
            },
            {
                'name': 'dynamics_reward_predictor',
                'script': 'train_dynamics_reward_predictor.py',
                'description': 'Training dynamics-based reward predictors'
            },
            {
                'name': 'representation_metrics',
                'script': 'run_representation_metrics.py',
                'description': 'Comprehensive representation metrics analysis (neighborhood preservation, manifold dimension, smoothness, robustness)'
            }
        ]
        
        # Get the scripts directory path
        self.scripts_dir = Path(__file__).parent
        
    def _should_run_stage(self, stage_name: str) -> bool:
        """
        Determine if a stage should be run based on skip_stages and only_stages.
        
        Args:
            stage_name: Name of the stage to check
            
        Returns:
            True if the stage should be run, False otherwise
        """
        if self.only_stages:
            return stage_name in self.only_stages
        
        return stage_name not in self.skip_stages
        
    def _run_script(self, script_name: str, stage_name: str) -> bool:
        """
        Run a training script as a subprocess.
        
        Args:
            script_name: Name of the script file to run
            stage_name: Name of the stage for logging purposes
            
        Returns:
            True if script executed successfully, False otherwise
        """
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            return False
            
        # Prepare command
        cmd = [sys.executable, str(script_path)]
        if self.config_path:
            cmd.extend(['--config', self.config_path])
            
        start_time = time.time()
        
        try:
            # Run the script with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,  # Line buffering
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},  # Force unbuffered output
                cwd=project_root
            )
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # Print each line to the console
                sys.stdout.flush()  # Force immediate output
                
            process.stdout.close()
            return_code = process.wait()
            
            end_time = time.time()
            duration = end_time - start_time
            
            if return_code == 0:
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error while running script {script_name}: {e}")
            return False
            
    def run_pipeline(self, fail_fast: bool = True) -> bool:
        """
        Run the complete training pipeline.
        
        Args:
            fail_fast: If True, stop pipeline on first failure. If False, continue with remaining stages.
            
        Returns:
            True if all stages completed successfully, False if any stage failed
        """
        if self.only_stages:
            pass
        elif self.skip_stages:
            pass
        else:
            pass
            
        pipeline_start_time = time.time()
        successful_stages = []
        failed_stages = []
        
        for stage in self.pipeline_stages:

            print(f"\nRunning stage: {stage['name']}")
            stage_name = stage['name']
            script_name = stage['script']
            description = stage['description']
            
            if not self._should_run_stage(stage_name):
                continue
                
            success = self._run_script(script_name, stage_name)
            
            if success:
                successful_stages.append(stage_name)
            else:
                failed_stages.append(stage_name)
                
                if fail_fast:
                    break
                    
        pipeline_end_time = time.time()
        total_duration = pipeline_end_time - pipeline_start_time
        
        if failed_stages:
            return False
        else:
            return True
            
    def list_stages(self):
        """Print information about all available pipeline stages."""
        print("\nAvailable Pipeline Stages:")
        print("="*50)
        for i, stage in enumerate(self.pipeline_stages, 1):
            print(f"{i}. {stage['name']}")
            print(f"   Script: {stage['script']}")
            print(f"   Description: {stage['description']}")


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description='Run the complete training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (including model initialization and data collection)
  python full_pipeline.py
  
  # Run with custom config
  python full_pipeline.py --config my_config.yaml
  
  # Run only specific stages
  python full_pipeline.py --only jepa jepa_decoder
  
  # Skip model initialization if you just want to see training
  python full_pipeline.py --skip model_init
  
  # Skip data collection if data already exists
  python full_pipeline.py --skip data_collection
  
  # Run only model initialization to check parameter counts
  python full_pipeline.py --only model_init
  
  # Run only data collection
  python full_pipeline.py --only data_collection
  
  # Run only representation metrics analysis
  python full_pipeline.py --only representation_metrics
  
  # Skip specific stages
  python full_pipeline.py --skip encoder_decoder reward_predictor
  
  # Continue on failures instead of stopping
  python full_pipeline.py --no-fail-fast
  
  # List available stages
  python full_pipeline.py --list-stages
        """
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml file')
    
    parser.add_argument('--skip', nargs='+', default=None,
                       choices=['model_init', 'data_collection', 'encoder_decoder', 'jepa', 'jepa_decoder', 
                               'reward_predictor', 'dynamics_reward_predictor', 'representation_metrics'],
                       help='Stages to skip during execution')
    
    parser.add_argument('--only', nargs='+', default=None,
                       choices=['model_init', 'data_collection', 'encoder_decoder', 'jepa', 'jepa_decoder', 
                               'reward_predictor', 'dynamics_reward_predictor', 'representation_metrics'],
                       help='Run only these stages (ignores --skip)')
    
    parser.add_argument('--no-fail-fast', action='store_true',
                       help='Continue pipeline execution even if a stage fails')
    
    parser.add_argument('--list-stages', action='store_true',
                       help='List all available pipeline stages and exit')
    
    args = parser.parse_args()
    
    # Create pipeline runner
    runner = FullPipelineRunner(
        config_path=args.config,
        skip_stages=args.skip,
        only_stages=args.only
    )
    
    if args.list_stages:
        runner.list_stages()
        return
        
    # Run the pipeline
    fail_fast = not args.no_fail_fast
    success = runner.run_pipeline(fail_fast=fail_fast)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
