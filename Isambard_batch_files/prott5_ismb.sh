#!/bin/bash
#SBATCH --job-name=prott5_train
#SBATCH --output=prott5_train.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00

source /home/b6dd/hb0501.b6dd/miniforge3/etc/profile.d/conda.sh
conda activate esm2_environment

python - <<'PY'
import torch

print(torch.__version__) 

PY

python ProtT5_pipeline/workflow_prott5.py

