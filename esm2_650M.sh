#!/bin/bash
#SBATCH --job-name=esm2_650M_train
#SBATCH --output=esm2_650M_train.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

source /home/b6dd/hb0501.b6dd/miniforge3/etc/profile.d/conda.sh
conda activate esm2_environment

python - <<'PY'
import torch

print(torch.__version__) 

PY

python ESM2_pipeline_Isambard/workflow_ismb.py



