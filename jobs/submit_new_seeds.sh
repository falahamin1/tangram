#!/bin/bash
# Submit only the NEW seed-scaling jobs added to reach 10 seeds per method:
#   - hrep/vrep/gnn/mlp: seeds 3-9 (7 new each, seeds 0-2 already have
#     policies/*.pth and would just no-op through train_single.py's
#     existence check if resubmitted)
#   - cnn: seeds 0-9 (all 10, regenerated at --time=48:00:00 -- CNN never
#     finished at seeds 0-2 under the old 24:00:00 walltime, capping out at
#     ep~30500-33000 of 40000; this resubmission resumes those same
#     checkpoints with enough wall-time to actually finish, not restart them)
# Run this from tangram-git/ (sbatch's working directory must be tangram-git/).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for seed in 3 4 5 6 7 8 9; do
    for method in hrep vrep gnn mlp; do
        f="$SCRIPT_DIR/${method}_seed${seed}.slurm"
        if [ -f "$f" ]; then
            sbatch "$f"
        fi
    done
done

for seed in 0 1 2 3 4 5 6 7 8 9; do
    f="$SCRIPT_DIR/cnn_seed${seed}.slurm"
    if [ -f "$f" ]; then
        sbatch "$f"
    fi
done

echo "New seed jobs submitted (28 for hrep/vrep/gnn/mlp seeds 3-9, 10 for cnn seeds 0-9 = 38 total)."
echo "Check status with: squeue -u \$USER"
