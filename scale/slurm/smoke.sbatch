#!/bin/bash
#SBATCH -J pgg-smoke
#SBATCH --constraint=a6000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -o /mnt/beegfs/projects/pangenome_models/pangramgraph/logs/%x-%j.out
# one species end to end: download, extract (file deleted), embed and call, with the 8-bit check
set -eo pipefail
source /mnt/beegfs/projects/pangenome_models/pangramgraph/code/slurm/common.sh
ID=$1
echo "node $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), $(date)"
time python download.py --only $ID
time python extract.py $PGG_ROOT/raw/$ID.h5 --delete
rm -f $PGG_ROOT/raw/$ID.ok
time python gpu.py $PGG_ROOT/out/$ID --check 20
ls -la $PGG_ROOT/out/$ID
