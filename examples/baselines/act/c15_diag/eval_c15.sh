#!/bin/bash
# Run eval_diag in parallel shards and wait. Re-running resumes (finished rollouts are skipped).
#
#   CKPT=runs/<run>/checkpoints/final_model.pt   (required)
#   BANK=runs/c15/bank_f4/bank.pt  OUT=runs/c15/eval/seen  GPUS="0 1"  PER_GPU=4
#   bash examples/baselines/act/c15_diag/eval_c15.sh --tasks seen --conditions original zero
#
# Shards split the (task, level) pairs, so use at most 20 shards for the 5 seen tasks.
set -euo pipefail
: "${CKPT:?set CKPT to the checkpoint (final_model.pt)}"
BANK="${BANK:-runs/c15/bank_f4/bank.pt}"
OUT="${OUT:-runs/c15/eval/seen}"
read -r -a GPU_LIST <<< "${GPUS:-0}"
PER_GPU="${PER_GPU:-4}"
N=$(( ${#GPU_LIST[@]} * PER_GPU ))
mkdir -p "$OUT/logs"

pids=()
for ((i = 0; i < N; i++)); do
  gpu=${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}
  CUDA_VISIBLE_DEVICES=$gpu python -m examples.baselines.act.c15_diag.eval_diag \
    --checkpoint "$CKPT" --bank "$BANK" --out "$OUT" --shard "$i" --num-shards "$N" "$@" \
    > "$OUT/logs/shard$i.log" 2>&1 &
  pids+=($!)
done
echo "launched $N shards on GPUs ${GPU_LIST[*]}; follow with: tail -f $OUT/logs/shard0.log"

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
done_n=$(cat "$OUT"/episodes_shard*.jsonl 2>/dev/null | wc -l)
if [[ $fail == 0 ]]; then
  echo "all shards finished; $done_n rollouts in $OUT"
else
  echo "some shards failed (see $OUT/logs); $done_n rollouts saved. Re-run the same command to resume."
  exit 1
fi
