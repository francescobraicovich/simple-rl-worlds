#!/bin/bash
#SBATCH --job-name="train_ga"
#SBATCH --partition=gpu
#SBATCH --gpus=2                      # Request 1 GPU
#SBATCH --mem=50G                      # Request 16GB of RAM
#SBATCH --cpus-per-task=4            #  Request 3 CPU cores
#SBATCH --mail-type=END                # Notify when job ends
#SBATCH --mail-user=francesco.braicovich@studbocconi.it  # Your email

source /software/miniconda3/etc/profile.d/conda.sh
conda  activate rlworlds310                  # Activate Conda environment

echo "Running on node: $(hostname)"
echo "Using $(nproc) CPU cores"
echo "CUDA devices available:"
nvidia-smi  # Show available GPUs

export WANDB_API_KEY="adec5ee374cb589d5a323a9bc720b4292243deb5"  # Set API key manually (alternative method)

python -u src/scripts/full_pipeline.py
