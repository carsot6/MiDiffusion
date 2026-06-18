#!/bin/bash
set -e

# 1. Wait for export process to finish
echo "Waiting for background GPC export (PID 42112) to complete..."
while kill -0 42112 2>/dev/null; do
    sleep 30
done
echo "GPC raw export completed successfully."

# 2. Make Splits
echo "Creating splits..."
/Users/carlos.soto1/Projects/BLINKA/MiDiffusion/.venv/bin/python scripts/make_gpc_splits.py \
  ../ThreedFront/output/3d_front_processed/gpc_real --out gpc_real_splits.csv
cp gpc_real_splits.csv ../ThreedFront/dataset_files/gpc_real_splits.csv

# 3. Generate dataset stats
echo "Generating dataset stats..."
/Users/carlos.soto1/Projects/BLINKA/MiDiffusion/.venv/bin/python scripts/generate_dataset_stats.py \
  ../ThreedFront/output/3d_front_processed/gpc_real \
  --frozen-vocab config/gpc_categories.json \
  --splits ../ThreedFront/dataset_files/gpc_real_splits.csv

# 4. Start Training
echo "Starting MiDiffusion training on real GPC dataset (pointing to bedroom and living room)..."
PYTHONPATH=. /Users/carlos.soto1/Projects/BLINKA/MiDiffusion/.venv/bin/python scripts/train_diffusion.py \
  config/gpc_mixed_real.yaml \
  --experiment_tag gpc_real_p99_train \
  > "/var/folders/q1/xbc76nwj5sn7jw0vk24trrk00000gp/T/opencode/gpc_real_train.log" 2>&1

echo "Pipeline script finished."
