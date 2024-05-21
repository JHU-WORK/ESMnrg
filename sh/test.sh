#!/bin/bash

#SBATCH -A "danielk_gpu"                # Set the account name, assuming this is correct
#SBATCH --partition=a100                # Use the A100 GPU partition
#SBATCH --gres=gpu:1                    # Request one GPU
#SBATCH --nodes=1                       # Request one node
#SBATCH --ntasks-per-node=1             # Run one task per node
#SBATCH --mem-per-gpu=40G               # Memory per CPU
#SBATCH --time=2:00:0                   # set job time
#SBATCH --job-name="gpu test"           # Set the job name
#SBATCH --output=./slurm/%j.out         # output specification

#SBATCH --mail-user=asandhu9@jhu.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

source ~/.bashrc

module load anaconda
conda activate esm      # activate the Python environment

nvidia-smi

ps

python test.py