#!/bin/bash
#SBATCH --job-name=eval-rubble-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/rubble_citygs_merged_v2_pruned/eval_slurm_%j.log
#SBATCH --error=/work/pi_rsitaram_umass_edu/tungi/lstvgs/results/rubble_citygs_merged_v2_pruned/eval_slurm_%j.log

WORK_DIR="/work/pi_rsitaram_umass_edu/tungi"
CONDA_ENV="$WORK_DIR/conda/envs/gsplat"
CUDA_DIR="$WORK_DIR/cuda-13.0"

unset CC CXX
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"
export CUDA_HOME="$CUDA_DIR"
export PATH="$CONDA_ENV/bin:$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$CUDA_DIR/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK_DIR/lstvgs/gsplat:$PYTHONPATH"

cd "$WORK_DIR/lstvgs"

echo "=== Eval rubble v2 pruned merge: $(date) ==="
python citygs/eval_citygs.py \
    --ply-path results/rubble_citygs_merged_v2_pruned/splat_merged.ply \
    --data-dir /work/pi_rsitaram_umass_edu/tungi/datasets/rubble \
    --data-factor 4 \
    --test-every 8 \
    --output-dir results/rubble_citygs_merged_v2_pruned/eval \
    --save-images

echo "=== Results: ==="
cat results/rubble_citygs_merged_v2_pruned/eval/metrics.json
