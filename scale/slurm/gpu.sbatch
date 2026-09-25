#!/bin/bash
#SBATCH -J pgg-gpu
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -o /mnt/beegfs/projects/pangenome_models/pangramgraph/logs/%x-%j.out
# ESM-2 embeddings and Bacformer calls for extracted species, one at a time
source /mnt/beegfs/projects/pangenome_models/pangramgraph/code/slurm/common.sh
python gpu.py --loop
