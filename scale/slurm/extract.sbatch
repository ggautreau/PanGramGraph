#!/bin/bash
#SBATCH -J pgg-extract
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -o /mnt/beegfs/projects/pangenome_models/pangramgraph/logs/%x-%j.out
# takes downloaded files one at a time, keeps the compact tables, deletes the PanGBank file
source /mnt/beegfs/projects/pangenome_models/pangramgraph/code/slurm/common.sh
python extract.py --loop --delete
