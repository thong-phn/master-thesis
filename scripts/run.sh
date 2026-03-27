#!/bin/bash
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --exclude=gpu-node[001-004],gpu-node[009-010]
#SBATCH --mem=32G                    
#SBATCH --output=job_%j.out          
#SBATCH --error=job_%j.err           

# Activate conda
source ~/.bashrc
conda deactivate
conda activate py38

# Activate slurm
module load slurm

# Main
python wear_main_loso_four_stage.py --single_subject_only