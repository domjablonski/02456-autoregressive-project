#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train_guided
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/logs/train_guided.out
#BSUB -e hpc/logs/train_guided.err

# Week 3, classifier guidance, end to end:   bsub < hpc/train_guided.sh
#
# Needs results/checkpoints/rnn.pt already - run hpc/train_rnn.sh first.
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
##BSUB -u <EMAIL_ADDRESS>
##BSUB -B
##BSUB -N

set -eu
source hpc/env.sh   # uv on the PATH, caches off the home quota

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The noisy predictor that guides, then the independent judge that scores the result.
uv run python src/protgen/models/classifier.py
uv run python src/protgen/models/classifier.py --judge

# The lambda sweep. --family picks the target: AcyP, HPr or HU.
uv run python src/protgen/models/guided.py --family HPr
