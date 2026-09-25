#!/bin/bash
#SBATCH -J pgg-download
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -o /mnt/beegfs/projects/pangenome_models/pangramgraph/logs/%x-%j.out
# the only process that talks to PanGBank (one request per 31 s); pauses while 20 GB wait for extraction
source /mnt/beegfs/projects/pangenome_models/pangramgraph/code/slurm/common.sh
python download.py "$@"
