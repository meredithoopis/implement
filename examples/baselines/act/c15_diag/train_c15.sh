#!/bin/bash
# Train ACT/DINOv2 on C15 only (15 tasks x 4 levels x 50 sim demos, ~3000 episodes).
#
#   RECIPE=paper (default)  spec section 3 / paper App. F.7 = the trainer's own defaults:
#                           4 DINOv2 frames, batch 256, 100 epochs
#   RECIPE=repo             exp_scripts/act/run_exp_act.sh (the launcher the README says
#                           produced the paper numbers): 10 frames, batch 128, 10 epochs
#
# Env vars: DATA (demos/ig10k), TE_CACHE (runs/c15/te_cache_f<frames>), EXP_NAME, SEED,
#           WORKERS (16), FRAME_CACHE (pre-decoded frames dir, optional),
#           SIM_CFG / HUMAN_CFG (default: the C15 training configs; few-shot runs use the
#           *_test_config_unseen.json configs = 10 demos per unseen task-level pair)
# Extra args go to the trainer, e.g.  --max-train-minutes 10  (throughput check) or
#           --init-from <ckpt>  (P+FT: weights-only init, fresh optimizer and LR schedule).
set -euo pipefail

RECIPE="${RECIPE:-paper}"
case "$RECIPE" in
  paper) FRAMES=4;  BATCH=256; EPOCHS=100; WARMUP=5 ;;
  repo)  FRAMES=10; BATCH=128; EPOCHS=10;  WARMUP=5 ;;   # launcher leaves warmup at the default 5
  *) echo "RECIPE must be 'paper' or 'repo'"; exit 1 ;;
esac

DATA="${DATA:-demos/ig10k}"
TE_CACHE="${TE_CACHE:-runs/c15/te_cache_f${FRAMES}}"
EXP_NAME="${EXP_NAME:-act_dinov2_c15_${RECIPE}}"
WORKERS="${WORKERS:-16}"
CFG=examples/baselines/lerobot_dataset/config/exp_configs
EXTRA=()
[[ -n "${FRAME_CACHE:-}" ]] && EXTRA+=(--sim-frame-cache-dir "$FRAME_CACHE")

if [[ ! -f "$TE_CACHE/backbone_dinov2_vitl14/raw_features.pt" ]]; then
  echo "missing $TE_CACHE/backbone_dinov2_vitl14/raw_features.pt"
  echo "run: python -m examples.baselines.act.c15_diag.build_task_bank --frames $FRAMES --te-cache-root $TE_CACHE ..."
  exit 1
fi

echo "RECIPE=$RECIPE frames=$FRAMES batch=$BATCH epochs=$EPOCHS workers=$WORKERS te_cache=$TE_CACHE"
echo "sim config: ${SIM_CFG:-$CFG/sim_train_config_15.json}"
python -m examples.baselines.act.train_act_imitator \
  --exp-name "$EXP_NAME" \
  --seed "${SEED:-1}" \
  --human-root "$DATA/imitator_human_v1" \
  --sim-root "$DATA/imitator_sim_v1_zed2i" \
  --human-dataset-file "${HUMAN_CFG:-$CFG/human_train_config_15.json}" \
  --sim-dataset-file "${SIM_CFG:-$CFG/sim_train_config_15.json}" \
  --task-mapping-file examples/baselines/lerobot_dataset/task_mapping.json \
  --human-task-description-file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --sim-task-description-file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --input-mode video_only \
  --task-encoder-type frozen_backbone \
  --frozen-backbone-type dinov2_vitl14 \
  --frozen-backbone-num-frames "$FRAMES" \
  --frozen-backbone-adapter-layers 1 \
  --frozen-backbone-seq-patches 32 \
  --te-cache-root "$TE_CACHE" \
  --pred-horizon 24 \
  --batch-size "$BATCH" \
  --total-epochs "$EPOCHS" \
  --warmup-epochs "$WARMUP" \
  --lr 1e-4 \
  --kl-weight 10 \
  --num-dataload-workers "$WORKERS" \
  --save-epoch-freq 10 \
  --control-mode pd_joint_pos \
  --env-id TwoRobotStirSpoon-v1 \
  --max-episode-steps 500 \
  --no-include-depth \
  ${EXTRA[@]+"${EXTRA[@]}"} "$@"
