#!/usr/bin/env python3
"""
Run All Representation Metrics Analysis

This script orchestrates the execution of all representation metrics analysis scripts
to provide a comprehensive evaluation of learned representations from JEPA and
Encoder-Decoder training approaches.

The analysis suite includes:
1. Neighborhood Preservation - Evaluates local structure preservation using
   Trustworthiness and Continuity metrics
2. Manifold Dimension - Analyzes intrinsic dimensionality using Participation
   Ratio and Two-NN estimation
3. Smoothness Analysis - Measures representation smoothness by correlating
   pixel-space and latent-space distances
4. Robustness Analysis - Tests stability under Gaussian noise perturbations

Each analysis generates publication-ready visualizations and detailed metrics.
"""

import os
import sys
import subprocess
import argparse
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Set MPS fallback for unsupported operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepresentationMetricsRunner:
    """
    Orchestrates the execution of all representation metrics analysis scripts
    with proper error handling and logging.
    """
    
    def __init__(self, config_path: str = None, skip_analyses: List[str] = None, 
                 only_analyses: List[str] = None):
        """
        Initialize the representation metrics runner.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
            skip_analyses: List of analysis names to skip during execution.
            only_analyses: List of analysis names to run exclusively (ignores skip_analyses).
        """
        self.config_path = config_path
        self.skip_analyses = skip_analyses or []
        self.only_analyses = only_analyses or []
        
        # Define the analysis scripts in order
        self.analysis_scripts = [
            {
                'name': 'neighborhood_preservation',
                'script': 'representation_metrics/analyse_neighborhood_preservation.py',
                'description': 'Analyze local neighborhood structure preservation using Trustworthiness and Continuity'
            },
            {
                'name': 'manifold_dimension',
                'script': 'representation_metrics/analyse_manifold_dimension.py',
                'description': 'Evaluate intrinsic dimensionality using Participation Ratio and Two-NN estimation'
            },
            {
                'name': 'smoothness',
                'script': 'representation_metrics/analyse_smoothness.py',
                'description': 'Measure representation smoothness by correlating pixel and latent distances'
            },
            {
                'name': 'robustness',
                'script': 'representation_metrics/analyse_robustness.py',
                'description': 'Test representation stability under Gaussian noise perturbations'
            }
        ]
        
        # Get the scripts directory path
        self.scripts_dir = Path(__file__).parent
        
    def _should_run_analysis(self, analysis_name: str) -> bool:
        """
        Determine if an analysis should be run based on skip_analyses and only_analyses.
        
        Args:
            analysis_name: Name of the analysis to check
            
        Returns:
            True if the analysis should be run, False otherwise
        """
        if self.only_analyses:
            return analysis_name in self.only_analyses
        
        return analysis_name not in self.skip_analyses
        
    def _run_analysis_script(self, script_name: str, analysis_name: str) -> bool:
        """
        Run an analysis script as a subprocess.
        
        Args:
            script_name: Name of the script file to run
            analysis_name: Name of the analysis for logging purposes
            
        Returns:
            True if script executed successfully, False otherwise
        """
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
            
        # Prepare command
        cmd = [sys.executable, str(script_path)]
        if self.config_path:
            cmd.extend(['--config', self.config_path])
            
        logger.info(f"Starting {analysis_name} analysis...")
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
                print(line, end='')  # Print each line to the console
                
            process.stdout.close()
            return_code = process.wait()
            
            end_time = time.time()
            duration = end_time - start_time
            
            if return_code == 0:
                logger.info(f"✅ {analysis_name} analysis completed successfully in {duration:.2f}s")
                return True
            else:
                logger.error(f"❌ {analysis_name} analysis failed with return code {return_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error while running {analysis_name} analysis: {e}")
            return False
            
    def run_all_analyses(self, fail_fast: bool = True) -> bool:
        """
        Run all representation metrics analyses.
        
        Args:
            fail_fast: If True, stop on first failure. If False, continue with remaining analyses.
            
        Returns:
            True if all analyses completed successfully, False if any analysis failed
        """
        logger.info("🚀 Starting comprehensive representation metrics analysis...")
        
        if self.only_analyses:
            logger.info(f"Running only selected analyses: {', '.join(self.only_analyses)}")
        elif self.skip_analyses:
            logger.info(f"Skipping analyses: {', '.join(self.skip_analyses)}")
        else:
            logger.info("Running all available analyses")
            
        pipeline_start_time = time.time()
        successful_analyses = []
        failed_analyses = []
        
        for analysis in self.analysis_scripts:
            analysis_name = analysis['name']
            script_name = analysis['script']
            description = analysis['description']
            
            if not self._should_run_analysis(analysis_name):
                logger.info(f"⏭️  Skipping {analysis_name} analysis")
                continue
                
            logger.info(f"📊 {description}")
            success = self._run_analysis_script(script_name, analysis_name)
            
            if success:
                successful_analyses.append(analysis_name)
            else:
                failed_analyses.append(analysis_name)
                
                if fail_fast:
                    logger.error(f"💥 Stopping pipeline due to failure in {analysis_name}")
                    break
                    
        pipeline_end_time = time.time()
        total_duration = pipeline_end_time - pipeline_start_time
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("🏁 REPRESENTATION METRICS ANALYSIS SUMMARY")
        logger.info("="*60)
        logger.info(f"⏱️  Total duration: {total_duration:.2f}s")
        logger.info(f"✅ Successful analyses ({len(successful_analyses)}): {', '.join(successful_analyses) if successful_analyses else 'None'}")
        
        if failed_analyses:
            logger.error(f"❌ Failed analyses ({len(failed_analyses)}): {', '.join(failed_analyses)}")
            logger.info("\n📁 Check the evaluation_plots/ directory for generated visualizations")
            logger.info("📊 Generated plots can be found in:")
            logger.info("   • evaluation_plots/neighborhood_preservation/")
            logger.info("   • evaluation_plots/manifold_dimension/")
            logger.info("   • evaluation_plots/smoothness_analysis/")
            logger.info("   • evaluation_plots/robustness_analysis/")
            return False
        else:
            logger.info("🎉 All analyses completed successfully!")
            logger.info("\n📁 Check the evaluation_plots/ directory for generated visualizations")
            logger.info("📊 Generated plots can be found in:")
            logger.info("   • evaluation_plots/neighborhood_preservation/")
            logger.info("   • evaluation_plots/manifold_dimension/")
            logger.info("   • evaluation_plots/smoothness_analysis/")
            logger.info("   • evaluation_plots/robustness_analysis/")
            return True
            
    def list_analyses(self):
        """Print information about all available analyses."""
        print("\n📊 Available Representation Metrics Analyses:")
        print("="*60)
        for i, analysis in enumerate(self.analysis_scripts, 1):
            print(f"{i}. {analysis['name']}")
            print(f"   Script: {analysis['script']}")
            print(f"   Description: {analysis['description']}")
            print()


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive representation metrics analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all representation metrics analyses
  python run_representation_metrics.py
  
  # Run with custom config
  python run_representation_metrics.py --config my_config.yaml
  
  # Run only specific analyses
  python run_representation_metrics.py --only neighborhood_preservation smoothness
  
  # Skip specific analyses
  python run_representation_metrics.py --skip manifold_dimension robustness
  
  # Continue on failures instead of stopping
  python run_representation_metrics.py --no-fail-fast
  
  # List available analyses
  python run_representation_metrics.py --list-analyses
        """
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.yaml file')
    
    parser.add_argument('--skip', nargs='+', default=None,
                       choices=['neighborhood_preservation', 'manifold_dimension', 'smoothness', 'robustness'],
                       help='Analyses to skip during execution')
    
    parser.add_argument('--only', nargs='+', default=None,
                       choices=['neighborhood_preservation', 'manifold_dimension', 'smoothness', 'robustness'],
                       help='Run only these analyses (ignores --skip)')
    
    parser.add_argument('--no-fail-fast', action='store_true',
                       help='Continue execution even if an analysis fails')
    
    parser.add_argument('--list-analyses', action='store_true',
                       help='List all available analyses and exit')
    
    args = parser.parse_args()
    
    # Create runner
    runner = RepresentationMetricsRunner(
        config_path=args.config,
        skip_analyses=args.skip,
        only_analyses=args.only
    )
    
    if args.list_analyses:
        runner.list_analyses()
        return
        
    # Run all analyses
    fail_fast = not args.no_fail_fast
    success = runner.run_all_analyses(fail_fast=fail_fast)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
