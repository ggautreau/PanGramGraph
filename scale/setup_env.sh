#!/bin/bash
# The environment of the PanGBank-wide run on LaBIA, entirely under the project space (the home is
# over quota): micromamba's Python 3.11, then PyTorch (CUDA 12.4, as the evo_env proven on the A6000
# nodes), transformers 4.53 (Bacformer's code breaks on 5.x), PyTables, Biopython. Caches are removed.
set -euo pipefail
ROOT=${PGG_ROOT:-/mnt/beegfs/projects/pangenome_models/pangramgraph}
MM=/mnt/beegfs/projects/pangenome_models/suite_stage_aliou/micromamba/bin/micromamba
export http_proxy=http://129.175.8.243:8080 https_proxy=http://129.175.8.243:8080   # micromamba does not resolve the proxy's name
export MAMBA_ROOT_PREFIX=$ROOT/.mamba PIP_CACHE_DIR=$ROOT/.pipcache
[ -x $ROOT/env/bin/python ] || $MM create -y -q -p $ROOT/env -c conda-forge python=3.11 pip
$ROOT/env/bin/pip install -q --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# wheels only: the cluster's compiler is too old to build numpy
$ROOT/env/bin/pip install -q --no-cache-dir --only-binary=:all: transformers==4.53.3 numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 \
    tables==3.10.1 biopython==1.84 huggingface_hub safetensors
rm -rf $ROOT/.mamba/pkgs $ROOT/.pipcache
$ROOT/env/bin/python -c "import torch, transformers, tables, Bio, pandas; print('torch', torch.__version__, '| transformers', transformers.__version__)"
