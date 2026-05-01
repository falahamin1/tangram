#!/bin/bash

# Time estimate:
#   ~20 min per 1000 episodes  →  40,000 episodes ≈ 800 min ≈ 13.3 h
#   Wall time set to 20 h to leave a safety buffer.
#
# Resume behaviour:
#   If the job times out or crashes, just resubmit this same script.
#   Training will automatically resume from the latest checkpoint in
#   checkpoints/vrep/.

#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=tangram-vrep
#SBATCH --output=tangram-vrep.%j.out
#SBATCH --qos=normal

module purge
module load anaconda

conda activate amfa-custom-env

python3 run_vrep.py \
    --episodes 40000 \
    --checkpoint-dir checkpoints/vrep \
    --checkpoint-interval 500
