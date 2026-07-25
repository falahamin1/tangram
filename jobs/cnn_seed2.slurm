#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=tangram-cnn-s2
#SBATCH --output=tangram-cnn-s2.%j.out
#SBATCH --qos=long

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

python3 train_single.py \
    --method cnn \
    --seed 2 \
    --episodes 40000 \
    --checkpoint-interval 500
