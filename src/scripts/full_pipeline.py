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

Each script is executed as a subprocess with proper error handling and logging.
The pipeline can be configured to run specific stages or the complete sequence.
"""

import os
import sys
import subprocess
import argparse
import logging
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
            }
        ]
        
        # Setup logging
        self._setup_logging()
        
        # Get the scripts directory path
        self.scripts_dir = Path(__file__).parent
        
    def _setup_logging(self):
        """Configure logging for the pipeline runner."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('full_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
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
            self.logger.error(f"Script not found: {script_path}")
            return False
            
        # Prepare command
        cmd = [sys.executable, str(script_path)]
        if self.config_path:
            cmd.extend(['--config', self.config_path])
            
        self.logger.info(f"Starting stage '{stage_name}': {script_name}")
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # Run the script with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=project_root
            )
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    self.logger.info(f"[{stage_name}] {line.strip()}")
                    
            process.stdout.close()
            return_code = process.wait()
            
            end_time = time.time()
            duration = end_time - start_time
            
            if return_code == 0:
                self.logger.info(f"Stage '{stage_name}' completed successfully in {duration:.2f} seconds")
                return True
            else:
                self.logger.error(f"Stage '{stage_name}' failed with return code {return_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running stage '{stage_name}': {str(e)}")
            return False
            
    def run_pipeline(self, fail_fast: bool = True) -> bool:
        """
        Run the complete training pipeline.
        
        Args:
            fail_fast: If True, stop pipeline on first failure. If False, continue with remaining stages.
            
        Returns:
            True if all stages completed successfully, False if any stage failed
        """
        self.logger.info("="*80)
        self.logger.info("Starting Full Training Pipeline")
        self.logger.info("="*80)
        
        if self.config_path:
            self.logger.info(f"Using config file: {self.config_path}")
        else:
            self.logger.info("Using default configuration")
            
        if self.only_stages:
            self.logger.info(f"Running only stages: {', '.join(self.only_stages)}")
        elif self.skip_stages:
            self.logger.info(f"Skipping stages: {', '.join(self.skip_stages)}")
        else:
            self.logger.info("Running all stages")
            
        pipeline_start_time = time.time()
        successful_stages = []
        failed_stages = []
        
        for stage in self.pipeline_stages:
            stage_name = stage['name']
            script_name = stage['script']
            description = stage['description']
            
            if not self._should_run_stage(stage_name):
                self.logger.info(f"Skipping stage '{stage_name}': {description}")
                continue
                
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Stage: {stage_name}")
            self.logger.info(f"Description: {description}")
            self.logger.info(f"Script: {script_name}")
            self.logger.info(f"{'='*60}")
            
            success = self._run_script(script_name, stage_name)
            
            if success:
                successful_stages.append(stage_name)
                self.logger.info(f"✅ Stage '{stage_name}' completed successfully")
            else:
                failed_stages.append(stage_name)
                self.logger.error(f"❌ Stage '{stage_name}' failed")
                
                if fail_fast:
                    self.logger.error(f"Stopping pipeline due to failure in stage '{stage_name}'")
                    break
                    
        pipeline_end_time = time.time()
        total_duration = pipeline_end_time - pipeline_start_time
        
        # Summary
        self.logger.info("\n" + "="*80)
        self.logger.info("Pipeline Execution Summary")
        self.logger.info("="*80)
        self.logger.info(f"Total execution time: {total_duration:.2f} seconds")
        self.logger.info(f"Successful stages ({len(successful_stages)}): {', '.join(successful_stages) if successful_stages else 'None'}")
        
        if failed_stages:
            self.logger.error(f"Failed stages ({len(failed_stages)}): {', '.join(failed_stages)}")
            self.logger.error("❌ Pipeline completed with failures")
            return False
        else:
            self.logger.info("✅ Pipeline completed successfully!")
            return True
            
    def list_stages(self):
        """Print information about all available pipeline stages."""
        print("\nAvailable Pipeline Stages:")
        print("="*50)
        for i, stage in enumerate(self.pipeline_stages, 1):
            print(f"{i}. {stage['name']}")
            print(f"   Script: {stage['script']}")
            print(f"   Description: {stage['description']}")
            print()


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description='Run the complete training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python full_pipeline.py
  
  # Run with custom config
  python full_pipeline.py --config my_config.yaml
  
  # Run only specific stages
  python full_pipeline.py --only jepa jepa_decoder
  
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
                       choices=['encoder_decoder', 'jepa', 'jepa_decoder', 
                               'reward_predictor', 'dynamics_reward_predictor'],
                       help='Stages to skip during execution')
    
    parser.add_argument('--only', nargs='+', default=None,
                       choices=['encoder_decoder', 'jepa', 'jepa_decoder', 
                               'reward_predictor', 'dynamics_reward_predictor'],
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
