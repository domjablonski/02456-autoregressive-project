#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/train.out
#BSUB -e hpc/logs/train.err

# Submit your training run to a GPU queue:   bsub < hpc/train.sh
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
##BSUB -u <EMAIL_ADDRESS>
##BSUB -B
##BSUB -N

<<<<<<< HEAD
set -eu
cd /dtu/blackhole/0e/234190/02456-autoregressive-project
=======
#set -eu
#cd "$(dirname "$0")/.."
>>>>>>> e10d445092b76a832f3ecced66b4a492cd748a86
source hpc/env.sh   # uv on the PATH, caches off the home quota

# Your training command, e.g.:
uv run python src/protgen/models/mlp.py
