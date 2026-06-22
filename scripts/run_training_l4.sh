#!/bin/bash
set -euo pipefail

# ============================================================
# MiDiffusion L4 Training Bootstrap
# ============================================================
# Usage:
#   1. Fill in BUCKET and GH_TOKEN below
#   2. bash scripts/run_training_l4.sh
# ============================================================

# ─── CONFIG ───
BUCKET="gs://ingka-b2b-da-bci-test-rec2b-research"
EXPERIMENT_TAG="gpc_all_paper_v1"
CONFIG_FILE="config/gpc_l4.yaml"
GH_TOKEN="<tu-github-personal-access-token>"
GH_USER="carsot6"
# ─────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
echo "[$(date)] Starting MiDiffusion L4 bootstrap in $REPO_ROOT"

# ─── 1. Ensure sibling repos exist ──────────────────────────
clone_if_missing() {
  local dir="$1" url="$2"
  if [ ! -d "$dir" ]; then
    echo "[$(date)] Cloning $url"
    git clone "https://${GH_TOKEN}@${url#https://}" "$dir"
  else
    echo "[$(date)] $dir already exists, pulling latest"
    git -C "$dir" pull
  fi
}

clone_if_missing "$REPO_ROOT/../ml-diffusion-data-pipeline" \
  "https://github.com/ingka-group-digital/ml-diffusion-data-pipeline.git"

# ─── 2. Install dependencies ────────────────────────────────
echo "[$(date)] Installing MiDiffusion"
pip install -e "$REPO_ROOT" --quiet
echo "[$(date)] Installing ml-diffusion-data-pipeline"
pip install -e "$REPO_ROOT/../ml-diffusion-data-pipeline" --quiet

# ─── 3. Download data from bucket ────────────────────────────
echo "[$(date)] Downloading training data from $BUCKET"
mkdir -p "$REPO_ROOT/../ThreedFront/output/3d_front_processed"
mkdir -p "$REPO_ROOT/../ThreedFront/dataset_files"
gcloud storage cp -r \
  "$BUCKET/training-data/gpc_all" \
  "$REPO_ROOT/../ThreedFront/output/3d_front_processed/gpc_all"
gcloud storage cp \
  "$BUCKET/training-data/gpc_all_splits.csv" \
  "$REPO_ROOT/../ThreedFront/dataset_files/"
gcloud storage cp \
  "$BUCKET/training-data/dataset_stats.txt" \
  "$REPO_ROOT/../ThreedFront/output/3d_front_processed/gpc_all/"

# ─── 4. Download previous checkpoint if exists ──────────────
RESUME_FLAGS=""
mkdir -p "$REPO_ROOT/output/log/$EXPERIMENT_TAG"

if gcloud storage cp \
  "$BUCKET/checkpoints/$EXPERIMENT_TAG/best_model.pt" \
  "$REPO_ROOT/output/log/$EXPERIMENT_TAG/best_model.pt" 2>/dev/null; then
  # Also download optimizer state if available
  if gcloud storage cp \
    "$BUCKET/checkpoints/$EXPERIMENT_TAG/optimizer_state.pt" \
    "$REPO_ROOT/output/log/$EXPERIMENT_TAG/optimizer_state.pt" 2>/dev/null; then
    # Was exact resume possible? Only if there's a periodic checkpoint.
    # best_model.pt alone means we resume from epoch 100.
    echo "[$(date)] Found best_model.pt, resuming from epoch 100"
    RESUME_FLAGS="--weight_file $REPO_ROOT/output/log/$EXPERIMENT_TAG/best_model.pt --continue_from_epoch 100"
  else
    echo "[$(date)] Found best_model.pt only, resuming from epoch 100"
    RESUME_FLAGS="--weight_file $REPO_ROOT/output/log/$EXPERIMENT_TAG/best_model.pt --continue_from_epoch 100"
  fi
else
  echo "[$(date)] No checkpoint found, starting from scratch"
fi

# If periodic checkpoints exist, they take priority (load_checkpoints handles them)
# Just keep --weight_file for weights and let load_checkpoints find model_XXXXX

# ─── 5. Launch training ─────────────────────────────────────
echo "[$(date)] Launching training with config=$CONFIG_FILE tag=$EXPERIMENT_TAG"
echo "[$(date)] Effective batch: check yaml (batch_size * grad_accum_steps)"
echo "[$(date)] GPU mode: --gpu 0 --use_amp"

nohup env PYTHONPATH="$REPO_ROOT" python "$REPO_ROOT/scripts/train_diffusion.py" \
  "$REPO_ROOT/$CONFIG_FILE" \
  --experiment_tag "$EXPERIMENT_TAG" \
  --gpu 0 \
  --use_amp \
  $RESUME_FLAGS \
  > "$REPO_ROOT/output/log/$EXPERIMENT_TAG/training_stdout.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$REPO_ROOT/output/log/$EXPERIMENT_TAG/training.pid"
echo "[$(date)] Training PID: $TRAIN_PID"

# ─── 6. Sync checkpoints every 5 min ─────────────────────────
nohup bash -c "
  set -euo pipefail
  LOG_DIR=\"$REPO_ROOT/output/log/$EXPERIMENT_TAG\"
  BKT=\"$BUCKET/checkpoints/$EXPERIMENT_TAG\"
  while true; do
    sleep 300

    # best_model.pt → atomic upload via tmp then rename
    if [ -f \"\$LOG_DIR/best_model.pt\" ]; then
      gcloud storage cp \"\$LOG_DIR/best_model.pt\" \"\$BKT/best_model.tmp.pt\" && \
      gcloud storage mv \"\$BKT/best_model.tmp.pt\" \"\$BKT/best_model.pt\" 2>/dev/null
    fi

    # Periodic model checkpoints (model_XXXXX.pt + opt_XXXXX.pt)
    for f in \"\$LOG_DIR\"/model_*.pt; do
      [ -f \"\$f\" ] || continue
      base=\$(basename \"\$f\")
      remote=\"\$BKT/\$base\"
      gcloud storage cp \"\$f\" \"\${remote}.tmp\" && \
      gcloud storage mv \"\${remote}.tmp\" \"\$remote\" 2>/dev/null
    done
    for f in \"\$LOG_DIR\"/opt_*.pt; do
      [ -f \"\$f\" ] || continue
      base=\$(basename \"\$f\")
      remote=\"\$BKT/\$base\"
      gcloud storage cp \"\$f\" \"\${remote}.tmp\" && \
      gcloud storage mv \"\${remote}.tmp\" \"\$remote\" 2>/dev/null
    done

    # Optimizer state for exact resume
    if [ -f \"\$LOG_DIR/optimizer_state.pt\" ]; then
      gcloud storage cp \"\$LOG_DIR/optimizer_state.pt\" \"\$BKT/optimizer_state.pt\"
    fi

    # Training stdout log
    if [ -f \"\$LOG_DIR/training_stdout.log\" ]; then
      gcloud storage cp \"\$LOG_DIR/training_stdout.log\" \"\$BKT/training_stdout.log\"
    fi

    echo \"[sync] \$(date) - checkpoints synced\"
  done
" > "$REPO_ROOT/output/log/$EXPERIMENT_TAG/sync.log" 2>&1 &
SYNC_PID=$!
echo "[$(date)] Sync PID: $SYNC_PID"

echo ""
echo "============================================"
echo "  Training launched successfully!"
echo "  PID: $TRAIN_PID"
echo "  Log: output/log/$EXPERIMENT_TAG/training_stdout.log"
echo "  Sync: output/log/$EXPERIMENT_TAG/sync.log"
echo "============================================"
echo ""
echo "Monitor with:"
echo "  tail -f output/log/$EXPERIMENT_TAG/training_stdout.log"
echo "  nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv -l 5"
