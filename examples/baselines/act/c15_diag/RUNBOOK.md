# ACT/DINOv2 on C15 (simulation): end-to-end runbook

Reproduces ACT/DINOv2 trained on C15 only, then runs the diagnostics: no-demo, demo swap (in-distribution vs OOD donors), held-out L3 substitutes, oracle demo, and failure-phase labelling. Every command runs on the GPU server from the repo root:

```bash
cd /mnt/data/hanhnth/The-Imitator-Game
```

Run long steps inside `tmux`. Times and sizes below are estimates; the throughput check (step 5) gives the real training time.

| Step | What | Output |
|---|---|---|
| 1 | One-time setup | `.venv`, env vars |
| 2 | Download data + assets | `demos/ig10k/`, `$MS_ASSET_DIR/data` |
| 3 | Simulator sanity | `runs/c15/sanity/` |
| 4 | Cache DINOv2 task embeddings | `runs/c15/bank_f4/`, `runs/c15/te_cache_f4/` |
| 5 | Throughput check (10 min) | steps/s, data-wait % |
| 6 | Train on C15 | `runs/act_dinov2_c15_paper-<date>/checkpoints/` |
| 7 | Evaluate seen tasks + diagnostics | `runs/c15/eval/seen/` |
| 8 | Report (failure phases, tables, videos) | `runs/c15/report/report.md` |
| 9 | Optional: ZS, Scr, P+FT | `runs/c15/eval/unseen_*` |

New code lives in `examples/baselines/act/c15_diag/`. Repo files changed (all backward compatible):

- `act/train_act_imitator.py`: throughput logging, `--max-train-minutes`, `--sim-frame-cache-dir`, `--init-from`.
- `lerobot_dataset/lerobot_sim_dataset.py`, `lerobot_dataloader.py` and `lerobot_paired_dataset.py`: optional pre-decoded frames.

---

## 1. One-time setup

Copy the changed and new files to the server (or push and pull them), then:

```bash
# caches must not land on the full root disk (UV_CACHE_DIR / HF_HOME / PIP_CACHE_DIR are already in ~/.bashrc)
cat >> ~/.bashrc <<'EOF'
export MS_ASSET_DIR=/mnt/data/hanhnth/.maniskill
export TORCH_HOME=/mnt/data/hanhnth/.cache/torch
EOF
source ~/.bashrc

deactivate 2>/dev/null                      # leave the phase-1 .venv-explore if active
uv sync                                     # Python 3.12 env in .venv (pyproject requires 3.12)
source .venv/bin/activate
# the README's lerobot patch; note python3.12, not 3.11 as the README says
curl -sSL https://raw.githubusercontent.com/huggingface/lerobot/0e81a275fcdbf03d74f78aa69eaa28c172a9f256/src/lerobot/datasets/lerobot_dataset.py \
  -o .venv/lib/python3.12/site-packages/lerobot/datasets/lerobot_dataset.py
uv pip install zstandard tabulate           # asset extraction fallback, report tables
export PYTHONPATH=$PWD:$PYTHONPATH          # every command below needs this

python -c "import torch, sapien, mani_skill, lerobot; print(torch.__version__, torch.cuda.is_available())"
python -c "from torchcodec.decoders import VideoDecoder; print('torchcodec ok')"
```

If torchcodec fails to load FFmpeg, see Troubleshooting. Vulkan is checked in step 3.

## 2. Download (~15 GB data + 20 GB assets)

```bash
python -m examples.baselines.act.c15_diag.download --unseen --oracle --assets
python -m mani_skill.utils.download_asset ycb -y      # YCB models (StirSpoon L2 uses a YCB fork); not in IG-10K-Assets
```

- **Data:** sim C15 (60 folders), human zed2i clips for C15 + the 4 OOD swap donors, the 5 unseen tasks (for step 9), and `human_H14_L1-L3` (oracle).
- **Assets:** go to `$MS_ASSET_DIR/data`, and the archives are deleted after extraction.
- **To skip step 9:** drop `--unseen`.

## 3. Simulator sanity (spec step 1)

```bash
mkdir -p runs/c15/sanity
python -m examples.baselines.act.c15_diag.env_sanity --task StirSpoon --out runs/c15/sanity 2>&1 | tee runs/c15/sanity/StirSpoon.log
python -m examples.baselines.act.c15_diag.env_sanity --task PlaceMugRack --levels L2 --out runs/c15/sanity
for V in StirSpoon_markpen PlaceMugRack_coaster PlaceFileFolder_basket; do
  python -m examples.baselines.act.c15_diag.env_sanity --render-variant $V --out runs/c15/sanity
done
```

Check the following:
- **Rendering:** `runs/c15/sanity/*_policy_cam.png` exist and show the scene.
  - A Vulkan error means rendering is broken; see Troubleshooting.
  - If only `rt-fast` fails, rerun with `--shader default`. That changes the images relative to the training data, so treat results as indicative.
- **Observations:** `obs: state (1, D), rgb (1, 224, 224, 3)`.
- **Replays:** lines like `demo 0: seed 12 (fingerprint err 0.00000), 412 actions -> SUCCESS at step 380`.
  - The sim data stores no seeds, so each demo's seed is recovered from its first-frame joint angles before its actions are replayed. L0, L1 and L3 demos should succeed.
  - If L2 demos fail while the others succeed, the official L2 protocol (scene mirrored, robots not mirrored) doesn't reproduce how the data was collected. Rerun with `--robot-mirror on` to confirm, and note it in the report.
- **Arm used at L2:** `data joint path: agent-0/left …, agent-1/right …` shows which arm moves in the data at each level.
  - StirSpoon, PlaceMugRack and PlaceFileFolder swap arms at L2.
  - FoldTowel and PlacePlateRack opt out of the mirror, so they don't.
- **Held-out substitutes:** compare `L3_*_original_*` with `L3_*_<variant>_*` and drop any variant whose object is wrong-sized, floating or intersecting.
  - `PlaceFileFolder_basket` needs this check most: the task code fixes that object's scale.

## 4. Cache DINOv2 task embeddings (spec step 3)

```bash
python -m examples.baselines.act.c15_diag.build_task_bank --frames 4 \
    --out runs/c15/bank_f4 --te-cache-root runs/c15/te_cache_f4
cat runs/c15/bank_f4/bank_report.md
```

- **What it encodes:** every human clip of the C15 tasks, the swap donors, the unseen tasks and the oracle clips.
  - Each clip: 4 frames uniformly over the clip → DINOv2-L CLS token per frame → mean over frames (1024-d), exactly the repo's encoder code.
  - It also writes the trainer's cache, one clip per C15 task.
- **Checks in the report:**
  - feature shapes `cls [1024]`, `seq [32, 1024]`;
  - `adapter(bank cls) vs FrozenVideoBackbone.forward` max abs diff below 1e-4 (proves the pooling matches the repo);
  - 1-NN task identification from a single clip, which shows how much the human clips' scenes alone identify the task;
  - whether each "similar" swap donor really is closer in DINOv2 space than the "unrelated" one.
- **For `RECIPE=repo`** (step 6), build a second bank with `--frames 10 --out runs/c15/bank_f10 --te-cache-root runs/c15/te_cache_f10`.

## 5. Throughput check (spec step 4)

```bash
nproc                                        # CPU cores available for DataLoader workers
EXP_NAME=throughput_check WORKERS=16 bash examples/baselines/act/c15_diag/train_c15.sh --max-train-minutes 10
```

Read the `[throughput]` lines; ignore the first one, which includes warm-up:

```
[throughput] iter 400: 6.10 steps/s, 1562 samples/s, data-wait 35.0%
[throughput] projected epoch time: 11.0 min (4037 steps/epoch)
```

- **Data-wait under ~20%:** go to step 6.
- **Data-wait above ~20%:**
  1. Retry with more workers (`WORKERS=32`, up to `nproc`).
  2. If it's still high, pre-decode the frames (156 GB; RAM-backed `/tmp` is fastest but is lost on restart, `/mnt/data` survives but is slower):

     ```bash
     python -m examples.baselines.act.c15_diag.build_frame_cache --out /tmp/c15_frames --workers 16
     FRAME_CACHE=/tmp/c15_frames EXP_NAME=throughput_check bash examples/baselines/act/c15_diag/train_c15.sh --max-train-minutes 10
     ```

     The builder checks random rows against direct decoding (`verify mismatches 0/32` expected), and frames are bit-identical to the decode path.

## 6. Train on C15 (spec step 5)

C15 = 60 datasets, 3,000 episodes, 1,033,711 frames. Choose the recipe:

| `RECIPE` | Source | DINOv2 frames | Batch | Epochs | Steps |
|---|---|---|---|---|---|
| `paper` (default) | your spec §3 = trainer defaults | 4 | 256 | 100 | ~404K |
| `repo` | `exp_scripts/act/run_exp_act.sh` (the launcher the README says produced the paper numbers) | 10 | 128 | 10 | ~81K (~1/10 the samples) |

Training time is roughly steps divided by the steps/s from step 5. If `paper` projects much more than a day, run `repo` first; it's the cheaper reference.

```bash
mkdir -p runs/c15
RECIPE=paper bash examples/baselines/act/c15_diag/train_c15.sh 2>&1 | tee runs/c15/train_paper.log
# add FRAME_CACHE=/tmp/c15_frames if step 5 needed it;  for the repo recipe:
# RECIPE=repo TE_CACHE=runs/c15/te_cache_f10 bash examples/baselines/act/c15_diag/train_c15.sh

CKPT=$(ls -t runs/act_dinov2_c15_paper-*/checkpoints/final_model.pt | head -1); echo $CKPT
tensorboard --logdir runs --port 6006       # optional: loss, perf/data_wait_frac
```

Checkpoints are written every 10 epochs and at the end. The log should show the cache in use (`RawFeat: 256/256 hits`) and `skip_video: True`.

## 7. Evaluate the seen tasks + diagnostics (spec steps 5–6)

Every rollout uses `reset(seed=1000+k)` for trial k = 0–9, so all conditions face identical initial states and are compared pairwise. Rollouts stop at first success; the horizon is 500 steps. Use at most 20 shards (5 tasks × 4 levels) and about one shard per free CPU core.

**7a. Main run: reproduction, no-demo and demo swap (1,200 rollouts)**

```bash
CKPT=$CKPT BANK=runs/c15/bank_f4/bank.pt OUT=runs/c15/eval/seen GPUS="0" PER_GPU=10 \
  bash examples/baselines/act/c15_diag/eval_c15.sh --tasks seen \
  --conditions original train_clip swap_similar swap_unrelated zero mean
```

| Condition | z (task embedding) comes from |
|---|---|
| `original` | the task's own human clip; trial k uses clip k of episodes 0–49. This is the Seen SR. |
| `train_clip` | the exact clip the trainer cached |
| `swap_similar` / `swap_unrelated` | the paper's Table 10 donors; scene unchanged |
| `zero` | z = 0 after the task LayerNorm |
| `mean` | training-set mean z, weighted by each task's training frames |

**7b. StirSpoon L3 in its training scene.** The official ACT eval turns on a global L3 flag that swaps StirSpoon's L3 bowl for a plastic box the data never had:

```bash
python -m examples.baselines.act.c15_diag.eval_diag --checkpoint $CKPT --bank runs/c15/bank_f4/bank.pt \
  --out runs/c15/eval/seen --tasks StirSpoon --levels L3 --conditions original zero --l3-flag off --tag l3off
```

**7c. Held-out L3 substitutes (spec diagnostic 3).** Only run the variants that passed the render check in step 3:

```bash
for V in StirSpoon_markpen PlaceMugRack_coaster PlaceFileFolder_basket; do
  python -m examples.baselines.act.c15_diag.eval_diag --checkpoint $CKPT --bank runs/c15/bank_f4/bank.pt \
    --out runs/c15/eval/seen --l3-variant $V --l3-flag off --conditions original zero
done
```

Each variant swaps the L3 substitute for an asset no C15 task uses at any level: wooden-block stirrer → marker; plate under the mug → coaster; tray → basket.

**7d. Optional: oracle demo (spec diagnostic 4)**

```bash
python -m examples.baselines.act.c15_diag.eval_diag --checkpoint $CKPT --bank runs/c15/bank_f4/bank.pt \
  --out runs/c15/eval/seen --tasks PlacePlateRack --levels L1 L2 L3 --conditions oracle
```

The demo is a human clip filmed in the L1–L3 scene itself. Only PlacePlateRack (`human_H14_L*`) among the seen tasks has these clips, and they were filmed with a different camera (`zed`, not `zed2i`), so a gain mixes "goal visible" with "camera changed".

**7e. Optional: parity with the official evaluator (2 environments, ~20 min).** The official evaluator samples random clips and seeds, so expect agreement within noise:

```bash
printf "L0_TwoRobotStirSpoon-v1\nL1_TwoRobotFoldTowel-v1\n" > runs/c15/parity_envs.txt
python -m examples.baselines.act.eval_act_imitator --eval-config runs/c15/parity_envs.txt --checkpoint $CKPT \
  --output-dir runs/c15/eval/official_parity \
  --human-root demos/ig10k/imitator_human_v1 --sim-root demos/ig10k/imitator_sim_v1_zed2i \
  --human-config examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_seen.json \
  --sim-config examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_seen.json \
  --frozen-backbone-num-frames 4 --num-episodes 10 --max-episode-steps 500
```

Don't use `exp_scripts/act/run_eval_act.sh` / `parallel_eval_act.py`; they crash (see Repo issues).

## 8. Failure phases + report (spec steps 7 and 9)

```bash
python -m examples.baselines.act.c15_diag.analyze --eval-dirs runs/c15/eval/seen \
  --bank-report runs/c15/bank_f4/bank_report.md --out runs/c15/report
```

`runs/c15/report/report.md` contains:

- Seen SR / Sub-SR vs the paper (0.81 / 0.93; C15 should land around 0.75–0.85), with a 95% CI.
- SR and Sub-SR per level for every condition, plus SR per task × level.
- The demo-swap test in the paper's grouping, then split into in-distribution donors (C15 tasks) vs OOD donors.
- The L3-flag and held-out-substitute table, each against the right training-scene baseline.
- Paired comparisons on identical seeds: P(success with this demo | success with the right demo). A high value under a wrong or empty demo means the scene alone drives the policy.
- Which arm grasped at each level (did the policy switch arms at mirrored L2?).
- Failure categories per level and condition, a per-phase CSV (`failed_phases.csv`), and 2–3 example videos per category.

How failures are labelled:
- The label is the first phase whose peak sub-reward never reached 0.99. Phases snap to 1.0 at their milestones; the repo defines no thresholds, so the report also shows Sub-SR at 0.5 and 0.9.
- Reach and grasp come from arm-agnostic probes, because the tasks score them on agent-0's gripper only.
- "Goal binding" means the robot held the right object to the end but never brought it to the target, or never performed the task action (stir, fold, …).
- At a mirrored L2, post-grasp phases are measured on the idle arm, so those failures are reported as a separate "post-grasp" bucket.

Change thresholds without re-running rollouts: `--phase-threshold 0.9 --reach-radius 0.08`. Videos are kept for failed rollouts only (`--save-video all` keeps every one).

## 9. Optional: ZS, Scr, P+FT (spec step 8)

```bash
# ZS: the C15 checkpoint on the unseen tasks, no update (eval only)
CKPT=$CKPT OUT=runs/c15/eval/unseen_zs GPUS="0" PER_GPU=10 \
  bash examples/baselines/act/c15_diag/eval_c15.sh --tasks unseen --conditions original zero

# trainer cache for the unseen tasks (clips from episodes 0-9, as in human_test_config_unseen.json)
python -m examples.baselines.act.c15_diag.build_task_bank --frames 4 --cache-tasks unseen \
  --from-bank runs/c15/bank_f4/bank.pt --te-cache-root runs/c15/te_cache_unseen_f4

# few-shot data = sim_test_config_unseen.json: episodes 0-9 of every unseen task-level pair (10 per pair)
FEW="SIM_CFG=examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_unseen.json HUMAN_CFG=examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_unseen.json TE_CACHE=runs/c15/te_cache_unseen_f4"
env $FEW EXP_NAME=act_dinov2_unseen_scr bash examples/baselines/act/c15_diag/train_c15.sh                    # Scr
env $FEW EXP_NAME=act_dinov2_unseen_pft bash examples/baselines/act/c15_diag/train_c15.sh --init-from $CKPT  # P+FT

for S in scr pft; do
  C=$(ls -t runs/act_dinov2_unseen_${S}-*/checkpoints/final_model.pt | head -1)
  SETTING=$([ $S = scr ] && echo Scr || echo P+FT)
  CKPT=$C OUT=runs/c15/eval/unseen_$S GPUS="0" PER_GPU=10 \
    bash examples/baselines/act/c15_diag/eval_c15.sh --tasks unseen --conditions original --setting $SETTING
done

python -m examples.baselines.act.c15_diag.analyze --eval-dirs runs/c15/eval/seen runs/c15/eval/unseen_zs \
  runs/c15/eval/unseen_scr runs/c15/eval/unseen_pft --bank-report runs/c15/bank_f4/bank_report.md --out runs/c15/report
```

Each few-shot run is about 69K frames (≈270 steps/epoch at batch 256), well under an hour at typical speeds. PourKettle at L2 can't succeed under this protocol (see Repo issues), and FoldBox uses a different success threshold at L2.

---

## Reference: what the repo actually does (the spec's [CHECK REPO] items)

| Item | Repo behaviour (used here unless noted) |
|---|---|
| Sim camera | `zed2i` sensor, 224×224, vertical FOV 0.55π, eye (−0.4, 0, 0.6) → target (0, 0, 0.2); the only sensor, the same as the 224×224 sim training video. Shader `rt-fast` in data and eval. |
| Proprioception | 18-d qpos: per arm 7 joints + 2 finger positions (m), agent-0 first. Eval takes it from the flat obs as `state[0:9] + state[18:27]`. |
| Actions | 16-d `pd_joint_pos`: per arm 7 absolute joint targets + gripper in [−1, 1] (1 = open), agent-0 first; 20 Hz control. |
| Normalisation | state and action → [−1, 1] using the q01/q99 bounds of each task × level dataset, chosen at eval by env id (see Repo issues). |
| Frame pooling | K frames at linspace(0, L−1, K) → DINOv2-L CLS per frame → mean over frames = 1024-d. Paper / trainer default K = 4; the launcher uses 10. |
| 1024 → 256 | trainable `Linear(1024,256)+LayerNorm` adapter, then a task LayerNorm. Only the frozen 1024-d features are cached. |
| Cache granularity | **one clip per task** (repo: a random, colour-jittered episode; here: seeded, no augmentation). So during training z is effectively a learned task code shared by all 4 levels and 50 robot episodes. |
| z injection | one extra token `[z, latent, proprio]` prepended to the ResNet image tokens at the input of the ACT transformer encoder. "Decoder token" in the paper = ACT's "CVAE decoder", i.e. the policy transformer. |
| Image backbone | ResNet18, ImageNet-pretrained, frozen BatchNorm, lr 1e-5; 224×224 input in [0, 1] with no ImageNet mean/std. Augmentation: ShiftScaleRotate (p 0.5) + ColorJitter (p 0.5). |
| Loss / optimiser | L1 over the 24-step chunk (episode-end padding included) + 10·KL; AdamW lr 1e-4, wd 1e-4; cosine schedule with 5 warm-up epochs; grad clip 1.0 (0.1 for the backbone); bf16 autocast; seed 1. |
| Inference | light temporal aggregation: query every 4 steps, average the last 4 chunks with weights exp(−0.1·age); CVAE latent = 0. |
| Few-shot budget | `sim_test_config_unseen.json` = episodes 0–9 of each unseen task × level, i.e. **10 per task-level pair** (40 per task). |
| Initial states | object xy + U[0, 2 cm) for most tasks (PickWash ±2 cm, GrindFood none); robot joints + N(0, 0.02), the same vector for both arms; L1 = fixed per-object xy shift. The official eval never seeds `reset()`, so its placements aren't reproducible; here `reset(seed=1000+k)`. |
| Horizon | tasks register 100–200 steps; the ACT eval overrides every task to 500 (used here). |
| SR / Sub-SR | SR = success at any step. Sub-SR: phases come from each task's RewardTracker (`info["R{i}_<phase>"]`); thresholds aren't in the repo (0.99 used here). |
| L2 in sim | scene objects mirrored, robots not (eval sets robot mirroring off), so the other arm must act. FoldTowel and PlacePlateRack opt out of the mirror, so their L2 is an appearance change only. |

## Repo issues found (and how this pipeline handles them)

1. **The official parallel eval crashes.** `parallel_eval_act.py` passes about 15 arguments `eval_act_imitator.py` doesn't accept (`--task-semantic-dim`, `--qwen-vl-model`, …), so every worker exits with code 2 and `run_eval_act.sh` produces nothing. → `eval_diag.py`; parity via `eval_act_imitator.py` directly (7e).
2. **The embedding precompute is noisy and never invalidates.** It encodes one random episode per task with colour jitter (it turns augmentation off on the wrong object), and the cache is keyed by repo id only, so changing the frame count silently reuses stale features. → `build_task_bank.py` (seeded, no augmentation, one cache dir per frame count, refuses to overwrite).
3. **Eval can read the previous episode's frame.** The eval human clip starts at `int(from_timestamp*fps)` (training uses `round`), so the first frame can belong to the previous episode. → the bank uses `round`.
4. **Training data loading is single-process.** `num_dataload_workers` defaults to 0 and the launcher doesn't set it. → `WORKERS=16`.
5. **`--resume-from` can't fine-tune.** It restores the finished LR schedule and epoch counter, and `reset_lr_scheduler` is never used. → `--init-from` (weights only).
6. **The official eval doesn't seed `reset()`.** → seeded, paired trials.
7. **The global L3 flag differs between data and eval.** It's off in data collection and on in the ACT eval; for StirSpoon L3 this swaps the bowl for a plastic box. → both settings evaluated (7b).
8. **Phase metrics and some success checks assume arm-0.** Reach/grasp sub-rewards use agent-0's gripper only, so they're wrong at mirrored L2. PourKettle and CutFruit success requires an agent-0 grasp, so L2 PourKettle is effectively unscoreable. → arm-agnostic probes; flagged in the report.
9. **Normalisation reveals task and level.** Stats are picked by env id, so the normalised state and the action scale carry task-and-level information regardless of z (and ZS uses the unseen task's own demo statistics). Not changed, since removing it needs retraining with pooled stats. Keep it in mind when reading the swap and no-demo results.
10. **Docs:**
    - the README's lerobot patch path says python3.11 (the env is 3.12);
    - `mani_skill/README.md` says `hi_res`/`wrist_sensor` default to True (they're False);
    - IG-10K-Assets has no YCB models (step 2 adds them).

## Troubleshooting

- **Vulkan / "failed to find a rendering device":**
  - Check `nvidia-smi` and `ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/`.
  - The container needs the NVIDIA graphics capability (`NVIDIA_DRIVER_CAPABILITIES` including `graphics` or `all`) and `libvulkan1`.
  - Without an ICD file, ask the cluster admin; simulation can't run headless without it.
- **`rt-fast` errors but `default` works:** your GPU lacks Vulkan ray tracing. Use `--shader default` in `env_sanity` / `eval_diag` and report it; images then differ from the training data.
- **torchcodec can't load FFmpeg:** install FFmpeg 4–7 shared libraries (`apt-get install -y ffmpeg`, or `conda install -c conda-forge ffmpeg`). As a last resort, train with `--video-backend pyav` (extra arg to `train_c15.sh`). The frame cache builder falls back to PyAV automatically.
- **`unexpected sensor image` in eval:** the env has more than one camera; don't pass `hi_res` or `wrist_sensor`.
- **A shard died:** rerun the same `eval_c15.sh` command; finished rollouts are skipped.
- **`no frame cache for …` / `stale frame cache`:** rebuild with `build_frame_cache.py` for the same sim config.
- **`checkpoint weight … did not load`:** the checkpoint isn't from `train_act_imitator.py`, or the architecture flags differ.
