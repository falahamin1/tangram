#!/bin/bash
# Submit all three training jobs to the Alpine cluster.
# Run this from tangram-git/ or tangram-git/jobs/.
# If a job was already partially run, it will resume from its last checkpoint.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

sbatch "$SCRIPT_DIR/hrep.slurm"
sbatch "$SCRIPT_DIR/vrep.slurm"
sbatch "$SCRIPT_DIR/gnn.slurm"

echo "All three jobs submitted. Check status with: squeue -u \$USER"
