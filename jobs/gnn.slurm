#!/bin/bash

# Time estimate:
#   ~22 min per 1000 episodes  →  40,000 episodes ≈ 880 min ≈ 14.7 h
#   Wall time set to 22 h to leave a safety buffer.
#
# Resume behaviour:
#   If the job times out or crashes, just resubmit this same script.
#   Training will automatically resume from the latest checkpoint in
#   checkpoints/gnn/.

#SBATCH --nodes=1
#SBATCH --time=22:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=tangram-gnn
#SBATCH --output=tangram-gnn.%j.out

module purge
module load anaconda

conda activate amfa-custom-env

python3 run_gnn.py \
    --episodes 40000 \
    --checkpoint-dir checkpoints/gnn \
    --checkpoint-interval 500
