#!/usr/bin/env bash
#BSUB -q gpuv100
#BSUB -J eval_cfg
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/eval_cfg.out
#BSUB -e hpc/logs/eval_cfg.err

set -eu

cd /dtu/blackhole/0e/234190/02456-autoregressive-project
source hpc/env.sh

echo "Starting CFG evaluation"

uv run --group eval python -u eval/eval.py \
    --samples results/sequences/rnn_cfg.csv \
    --fold

echo "CFG evaluation finished"
