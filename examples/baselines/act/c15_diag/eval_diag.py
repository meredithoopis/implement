"""Evaluate an ACT/DINOv2 checkpoint under task-conditioning diagnostics.

Same policy, same seeded initial states, only the task embedding z changes:

  original        z from a clip of the task's own human demos (clip k for trial k)
  train_clip      z from the exact clip the trainer cached for this task
  heldout         z from the task's clips never used anywhere (episodes >= 50), if any
  swap_similar    z from the paper's "similar" donor task (scene unchanged)
  swap_unrelated  z from the paper's "unrelated" donor task
  zero            z = 0 (after the task LayerNorm)
  mean            training-set mean z over the 15 C15 tasks (sample-weighted)
  oracle          z from a human demo filmed in the evaluated L1-L3 scene (PlacePlateRack only;
                  different camera 'zed', so camera change and goal visibility are confounded)

Rollout = the official ACT eval loop (eval_act_imitator.py): per-task-level state/action
normalisation, light temporal aggregation (query every 4 steps, last 4 chunks weighted
exp(-0.1*age)), pd_joint_pos, 500-step horizon. Differences: every trial is seeded
(reset(seed)), z comes from the precomputed bank (identical encoder), and by default a
rollout stops at the first success (SR = success_once is unchanged by this).

One process evaluates one shard of (task, level) pairs and appends one JSON line per rollout
to <out>/episodes_shard<k>.jsonl. Re-running skips rollouts already recorded.

  python -m examples.baselines.act.c15_diag.eval_diag --checkpoint <final_model.pt> \
      --bank runs/c15/bank_f4/bank.pt --out runs/c15/eval/seen --tasks seen \
      --conditions original swap_similar swap_unrelated zero mean --shard 0 --num-shards 4
"""

import argparse
import gc
import json
import os
import socket
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

# Importing the official eval module also sets L0_L3_utils.set_lr_mirror_robot_pose_enabled(False),
# exactly as the official evaluation does.
from examples.baselines.act.eval_act_imitator import (
    clear_l_level, extract_base_env_name, load_agent_from_checkpoint, set_l_level,
)
from examples.baselines.act.c15_diag import phases
from examples.baselines.act.c15_diag.common import (
    C15, LEVELS, ORACLE_TASKS, SEEN, SIM_ROOT, SWAP, UNSEEN, append_jsonl, human_repo, read_json,
    read_jsonl, sim_env_id,
)
from examples.baselines.lerobot_dataset.evaluate_processor import (
    HumanVideoSimEvaluateProcessor, HumanVideoSimEvaluateProcessorConfig,
)
from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer

CONDITIONS = ["original", "train_clip", "heldout", "swap_similar", "swap_unrelated", "zero", "mean", "oracle"]


# ── normalisation: reuse the official processor's code, without loading human videos ──────────
def make_normaliser(env_ids, sim_root: str, out_dir: Path):
    cfg_path = out_dir / "_norm_sim_config.json"
    cfg_path.write_text(json.dumps([{"repo_id": e, "root": e, "train": "0:50"} for e in env_ids], indent=2))
    proc = HumanVideoSimEvaluateProcessor.__new__(HumanVideoSimEvaluateProcessor)
    proc.config = HumanVideoSimEvaluateProcessorConfig(
        sim_root=sim_root, sim_dataset_file=str(cfg_path), sim_state_type="qpos",
        sim_single_arm=False, normalization_method="bounds_q99")
    proc.normalizer = ActionNormalizer()
    proc.repo_id_to_dataset_idx, proc.sim_task_to_dataset_idx = {}, {}
    proc._load_sim_normalizer()
    return proc


# ── agent ─────────────────────────────────────────────────────────────────────────────────────
def load_agent(ckpt_path: str, device):
    defaults = SimpleNamespace(  # same defaults as eval_act_imitator.parse_args; the checkpoint's
        action_dim=16, state_dim=18, pred_horizon=24, obs_horizon=1,  # saved args take priority
        include_depth=False, cameras=["zed2i"], image_size=[224, 224], state_type="qpos",
        single_arm=False, backbone="resnet18", enc_layers=2, dec_layers=4, dim_feedforward=512,
        hidden_dim=256, nheads=8, task_encoder_type="frozen_backbone",
        frozen_backbone_type="dinov2_vitl14", frozen_backbone_adapter_layers=1,
        frozen_backbone_seq_patches=32, frozen_backbone_num_frames=4, frozen_backbone_lora_rank=0,
        frozen_backbone_lora_alpha=16.0, task_latent_dim=256, task_num_frames=10,
        hf_cache_dir=os.environ.get("HF_HUB_CACHE"), use_ema=False)
    agent, model_cfg = load_agent_from_checkpoint(ckpt_path, defaults, device)

    # load_agent_from_checkpoint uses strict=False; make sure the weights that matter really loaded
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = {k.replace("task_encoder.backbone.", "task_encoder."): v for k, v in ckpt["agent_state_dict"].items()}
    live = agent.state_dict()
    for key in ("task_encoder.cls_adapter.0.weight", "ftask_norm.weight", "model.action_head.weight",
                "model.video_feature_proj.weight"):
        if key not in raw or not torch.equal(live[key].cpu(), raw[key].cpu()):
            raise RuntimeError(f"checkpoint weight {key} did not load into the agent")
    saved = ckpt.get("args", {})
    return agent, model_cfg, saved


@torch.no_grad()
def z_from_raw(agent, raw_cls: torch.Tensor, device) -> torch.Tensor:
    """Bank feature (1024) -> trained adapter -> task LayerNorm, as ACTAgent._encode_task."""
    return agent.ftask_norm(agent.task_encoder.cls_adapter(raw_cls.to(device)[None].float()))


# ── which clip / embedding each condition uses ────────────────────────────────────────────────
class DemoSource:
    def __init__(self, bank_path: str, agent, device, sim_root: str):
        self.bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        meta_path = Path(bank_path).with_name("bank_meta.json")
        self.meta = read_json(meta_path) if meta_path.exists() else {}
        self.agent, self.device = agent, device
        self.train_clip = self.meta.get("train_clip", {})
        self.mean_z, self.mean_info = self._mean_z(sim_root)

    def _pool(self, repo: str, which: str):
        eps = self.bank[repo]["episodes"]
        if which == "train":
            return [e for e in eps if e < 50]
        if which == "heldout":
            return [e for e in eps if e >= 50]
        return eps

    def _clip_z(self, repo: str, episode: int):
        i = self.bank[repo]["episodes"].index(episode)
        return z_from_raw(self.agent, self.bank[repo]["cls"][i], self.device)

    def _mean_z(self, sim_root: str):
        zs, ws, missing = [], [], []
        for t in C15:
            repo = human_repo(t)
            ep = self.train_clip.get(repo)
            if repo not in self.bank or ep is None:
                missing.append(t)
                continue
            zs.append(self._clip_z(repo, ep))
            frames = 0
            for lv in LEVELS:
                info = Path(sim_root) / sim_env_id(t, lv) / "meta" / "info.json"
                frames += read_json(info)["total_frames"] if info.exists() else 0
            ws.append(frames)
        if not zs:
            return None, {"missing": missing}
        w = torch.tensor(ws, dtype=torch.float32, device=self.device)
        if w.sum() == 0:
            w = torch.ones_like(w)
        z = (torch.cat(zs) * (w / w.sum())[:, None]).sum(0, keepdim=True)
        return z, {"weighting": "training frames per task", "tasks": len(zs), "missing": missing}

    def get(self, cond: str, task: str, level: str, trial: int, pool: str):
        """-> (z [1,256], record fields) or (None, reason) if the condition does not apply."""
        if cond in ("zero", "mean"):
            z = torch.zeros(1, self.agent.ftask_norm.normalized_shape[0], device=self.device) \
                if cond == "zero" else self.mean_z
            return z, {"demo_task": None, "demo_repo": None, "demo_episode": None}
        if cond == "oracle":
            if level not in ORACLE_TASKS.get(task, []):
                return None, "no human demo filmed in this scene"
            repo, demo_task = human_repo(task, level), task
            eps = self._pool(repo, "all") if repo in self.bank else []
        elif cond.startswith("swap_"):
            if task not in SWAP:
                return None, "no swap donor defined"
            demo_task = SWAP[task][cond.split("_", 1)[1]]
            repo = human_repo(demo_task)
            eps = self._pool(repo, pool) if repo in self.bank else []
        else:
            demo_task, repo = task, human_repo(task)
            if repo not in self.bank:
                return None, f"{repo} not in bank"
            if cond == "train_clip":
                eps = [self.train_clip[repo]] if repo in self.train_clip else []
            else:
                eps = self._pool(repo, "heldout" if cond == "heldout" else pool)
        if not eps:
            return None, f"no clips for {repo}"
        ep = eps[trial % len(eps)]
        return self._clip_z(repo, ep), {"demo_task": demo_task, "demo_repo": repo, "demo_episode": ep,
                                        "donor_in_c15": demo_task in C15}


# ── environment ───────────────────────────────────────────────────────────────────────────────
def apply_level(level: str, l3_flag: str) -> None:
    """Official ACT eval: set_l_level turns the global L3 flag on for L3, which swaps some L3
    assets (e.g. StirSpoon's bowl -> plasticbox). The data was collected with it off."""
    set_l_level(level)
    if level == "L3" and l3_flag == "off":
        from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
        os.environ.pop("MANI_SKILL_L3", None)
        L0_L3_utils.set_l3_enabled(False)


def apply_variant(base_env, variant: dict) -> None:
    """Held-out L3 substitute: overwrite asset attributes; the next reset() reconfigures."""
    for k, v in variant["attrs"].items():
        if not hasattr(base_env, k):
            raise AttributeError(f"{type(base_env).__name__} has no attribute {k!r}")
        setattr(base_env, k, tuple(v) if isinstance(v, list) else v)


def make_env(env_id: str, max_steps: int, video_dir, obs_horizon: int, l3_flag: str = "on",
             shader: str = "rt-fast"):
    """Single in-process env with the official wrapper stack (make_env.make_eval_envs, CPU path).
    shader: 'rt-fast' as in data collection and the official eval; 'default' only as a fallback on
    GPUs without Vulkan ray tracing (the images then differ from the training data)."""
    from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    import mani_skill.envs  # noqa: F401  (registers the tasks)

    apply_level(env_id.split("_", 1)[0], l3_flag)
    env = gym.make(extract_base_env_name(env_id), reconfiguration_freq=1, control_mode="pd_joint_pos",
                   reward_mode="dense", obs_mode="rgb", render_mode="rgb_array", max_episode_steps=max_steps,
                   sensor_configs=dict(shader_pack=shader),
                   human_render_camera_configs=dict(shader_pack=shader))
    env = FlattenRGBDObservationWrapper(env)
    env = FrameStack(env, num_stack=obs_horizon)
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
    if video_dir is not None:
        env = RecordEpisode(env, output_dir=str(video_dir), save_trajectory=False, save_video=True,
                            save_on_reset=False, info_on_video=True, source_type="act",
                            source_desc="c15_diag rollout")
    return env


def to_batch(obs: dict, device):
    return {k: torch.as_tensor(np.asarray(v)).unsqueeze(0).to(device) for k, v in obs.items()}


@torch.no_grad()
def rollout(env, agent, proc, env_id: str, z, seed: int, args, tracker):
    obs, info = env.reset(seed=seed)
    agent._cached_task_z = z
    window = deque(maxlen=args.tagg_window)
    first_success, success_once, last_info = -1, False, info
    t0 = time.time()
    for t in range(args.max_steps):
        o = to_batch(obs, args.device)
        if t == 0 and (o["rgb"].shape[-1] != 3 or tuple(o["rgb"].shape[-3:-1]) != (224, 224)):
            raise RuntimeError(f"unexpected sensor image {tuple(o['rgb'].shape)}; ACT expects one 224x224 RGB camera")
        state, rgb = proc.normalize_state_rgb(o["state"], o["rgb"], env_id)
        if args.temporal_agg == "light":
            if t % args.tagg_window == 0:
                window.append(agent.get_action({"state": state, "rgb": rgb}))
            step = t % args.tagg_window
            acts, ages = [], []
            for age, pred in enumerate(reversed(window)):
                idx = step + age * args.tagg_window
                if idx < pred.shape[1]:
                    acts.append(pred[:, idx])
                    ages.append(age)
            w = torch.exp(-0.1 * torch.tensor(ages, dtype=torch.float32, device=args.device))
            raw = (torch.stack(acts, 1) * (w / w.sum())[None, :, None]).sum(1)
        else:  # plain chunking: execute the whole 24-step chunk open loop
            if t % agent.pred_horizon == 0:
                chunk = agent.get_action({"state": state, "rgb": rgb})
            raw = chunk[:, t % agent.pred_horizon]
        act = proc.denormalize_action(raw, env_id)[0].cpu().numpy()
        obs, _, _, truncated, info = env.step({"panda_wristcam-0": act[:8], "panda_wristcam-1": act[8:16]})
        last_info = info
        tracker.update(info)
        if bool(np.asarray(info.get("success", False))) and first_success < 0:
            first_success = t + 1
            success_once = True
            if args.stop_on_success:
                break
        if bool(np.asarray(truncated)):
            break
    ep = last_info.get("episode", {})
    return {
        "success_once": bool(success_once or bool(np.asarray(ep.get("success_once", False)))),
        "success_at_end": bool(np.asarray(last_info.get("success", False))),
        "first_success_step": first_success,
        "steps": t + 1,
        "return": float(np.asarray(ep.get("return", 0.0))),
        "wall_s": round(time.time() - t0, 2),
        **tracker.result(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bank", required=True, help="bank.pt from build_task_bank (same --frames as training)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=["seen"], help="'seen', 'unseen', or task names")
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--conditions", nargs="+", default=["original"], choices=CONDITIONS)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=1000, help="trial k uses reset(seed=seed_base+k)")
    ap.add_argument("--demo-pool", default="train", choices=["train", "heldout", "all"],
                    help="clips for original/swap: train = episodes 0-49 (what the official eval samples)")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--temporal-agg", default="light", choices=["light", "none"])
    ap.add_argument("--tagg-window", type=int, default=4)
    ap.add_argument("--no-stop-on-success", dest="stop_on_success", action="store_false")
    ap.add_argument("--save-video", default="failures", choices=["all", "failures", "none"])
    ap.add_argument("--sim-root", default=SIM_ROOT)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--setting", default=None, help="label stored in records (default: Seen / ZS)")
    ap.add_argument("--l3-flag", default="on", choices=["on", "off"],
                    help="on = official ACT eval (global L3 flag set); off = scene used to collect the data")
    ap.add_argument("--l3-variant", default=None,
                    help="held-out L3 substitute name from heldout_l3.json (evaluates L3 only)")
    ap.add_argument("--tag", default="", help="free label stored in records and part of the resume key")
    ap.add_argument("--shader", default="rt-fast", help="rt-fast (official); 'default' only if RT is unsupported")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    variant = None
    if args.l3_variant:
        variant = read_json(Path(__file__).with_name("heldout_l3.json"))[args.l3_variant]
        args.tasks, args.levels = [variant["task"]], ["L3"]
        args.tag = args.tag or args.l3_variant

    tasks = []
    for t in args.tasks:
        tasks += SEEN if t == "seen" else UNSEEN if t == "unseen" else [t]
    pairs = [(t, lv) for t in tasks for lv in args.levels]
    mine = pairs[args.shard::args.num_shards]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"episodes_shard{args.shard}.jsonl"
    done = {(r["env_id"], r["condition"], r["trial"], r.get("tag", "")) for p in out.glob("episodes_shard*.jsonl")
            for r in read_jsonl(p)}

    device = torch.device(args.device)
    agent, model_cfg, saved = load_agent(args.checkpoint, device)
    proc = make_normaliser([sim_env_id(t, lv) for t, lv in mine], args.sim_root, out)
    demos = DemoSource(args.bank, agent, device, args.sim_root)
    frames_bank, frames_ckpt = demos.meta.get("frames"), saved.get("frozen_backbone_num_frames")
    if frames_bank and frames_ckpt and frames_bank != frames_ckpt:
        raise SystemExit(f"bank was built with {frames_bank} frames but the checkpoint used {frames_ckpt}")
    run_info = {"checkpoint": str(Path(args.checkpoint).resolve()), "bank": str(Path(args.bank).resolve()),
                "frames": frames_ckpt, "mean_z": demos.mean_info, "host": socket.gethostname(),
                "args": vars(args)}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out / f"run_shard{args.shard}_{args.tag or 'main'}_{stamp}.json").write_text(
        json.dumps(run_info, indent=2, default=str))
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} task-level pairs, conditions {args.conditions}")

    for task, level in mine:
        env_id = sim_env_id(task, level)
        todo = [(c, k) for c in args.conditions for k in range(args.trials)
                if (env_id, c, k, args.tag) not in done]
        if not todo:
            continue
        video_dir = None if args.save_video == "none" else out / "videos" / env_id
        env = make_env(env_id, args.max_steps, video_dir, model_cfg["obs_horizon"], args.l3_flag, args.shader)
        if variant is not None:
            apply_variant(env.unwrapped, variant)
        try:
            for cond, k in todo:
                z, demo = demos.get(cond, task, level, k, args.demo_pool)
                if z is None:
                    continue
                seed = args.seed_base + k
                tracker = phases.make_tracker(env.unwrapped, task, level)
                res = rollout(env, agent, proc, env_id, z, seed, args, tracker)
                name = f"{cond}{'_' + args.tag if args.tag else ''}__t{k}_s{seed}"
                keep = args.save_video == "all" or (args.save_video == "failures" and not res["success_once"])
                if video_dir is not None:
                    env.flush_video(name=name, save=keep)
                rec = {"env_id": env_id, "task": task, "level": level, "condition": cond, "trial": k,
                       "seed": seed, "tag": args.tag, "l3_flag": args.l3_flag if level == "L3" else None,
                       "shader": args.shader,
                       "setting": args.setting or ("Seen" if task in C15 else "ZS"),
                       "corpus": "C15", **demo, **res,
                       "video": str(Path("videos") / env_id / f"{name}.mp4") if keep and video_dir else None}
                append_jsonl(log_path, rec)
                print(f"{env_id:34s} {cond:15s} t{k} seed {seed}: "
                      f"{'SUCCESS' if res['success_once'] else 'fail   '} sub-SR {res.get('sub_sr')} "
                      f"({res['steps']} steps, {res['wall_s']}s)")
        finally:
            env.close()
            del env
            gc.collect()
            torch.cuda.empty_cache()
            clear_l_level()


if __name__ == "__main__":
    main()
