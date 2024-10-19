#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust the path as necessary
conda activate rlgpu  # Replace with your conda environment name

# Set the base log directory
LOG_DIR="/home/chz/Workspace/_rl_ws/legged_gym/logs/rough_bhr8tcphase"

# Find the latest log directory
LATEST_LOG=$(ls -td $LOG_DIR/*/ | grep -v 'exported' | head -n 1)

echo "Log Directory: $LATEST_LOG"

# Launch TensorBoard
tensorboard --logdir="$LATEST_LOG"

