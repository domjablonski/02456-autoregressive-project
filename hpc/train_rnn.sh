#!/bin/bash
#BSUB -q gpuv100
#BSUB -J train_rnn
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/logs/train_rnn.out
#BSUB -e hpc/logs/train_rnn.err

# Train the GRU and write results/sequences/rnn.csv:   bsub < hpc/train_rnn.sh
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
##BSUB -u <EMAIL_ADDRESS>
##BSUB -B
##BSUB -N

set -eu
source hpc/env.sh   # uv on the PATH, caches off the home quota

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run python src/protgen/models/rnn.py
