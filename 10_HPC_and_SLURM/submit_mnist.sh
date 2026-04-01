#!/bin/bash
#SBATCH --job-name=mnist-train
#SBATCH --output=mnist_%j.out
#SBATCH --error=mnist_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@northeastern.edu

# ============================================
# MNIST Training Job - Machine Learning 101
# Explorer HPC Cluster (Northeastern University)
# ============================================

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load required modules
module load anaconda3/2024.06
module load cuda/12.1

# Activate Conda environment
conda activate /home/$USER/envs/ml101

# Print environment info for debugging
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
nvidia-smi

# Run training
python train_mnist.py \
    --epochs 10 \
    --lr 0.001 \
    --batch-size 64 \
    --output-dir /scratch/$USER/ml-101/results/$SLURM_JOB_ID

echo "Job finished at $(date)"
