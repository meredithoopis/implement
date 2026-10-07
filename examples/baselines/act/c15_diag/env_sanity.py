"""Step 1 sanity checks: rendering works, all 4 levels load, dataset demos replay to success.

For one task at L0-L3 (same wrapper stack and level switches as eval_diag):
  - render the policy camera and the 512x512 video camera to PNGs (Vulkan check);
  - print obs shapes, mirror flag, reward phases and info keys;
  - which arm moves in the released sim data at each level (the sim-L2 arm-swap claim);
  - replay dataset demos open loop. The sim data stores no seeds, so each demo's seed is
    recovered from its first-frame joint angles (nominal pose + seeded N(0, 0.02) noise,
    identical on both arms), then its 16-d actions are stepped and success is checked.

  python -m examples.baselines.act.c15_diag.env_sanity --task StirSpoon --out runs/c15/sanity
  python -m examples.baselines.act.c15_diag.env_sanity --render-variant PlaceMugRack_coaster --out runs/c15/sanity
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from examples.baselines.act.c15_diag.common import LEVELS, SIM_ROOT, read_json, sim_env_id
from examples.baselines.act.c15_diag.eval_diag import apply_variant, make_env

STATE_KEY = "observation.qpos_gripper_states"
ACTION_KEY = "action.qpos_gripper_actions"


def save_png(arr, path: Path):
    from PIL import Image
    a = np.asarray(arr)
    a = a[-1] if a.ndim == 4 else a  # frame-stacked obs
    Image.fromarray(a[..., :3].astype(np.uint8)).save(path)


def env_qpos18(obs) -> np.ndarray:
    s = np.asarray(obs["state"])[-1]
    return np.concatenate([s[0:9], s[18:27]])


def fingerprint_err(q_env: np.ndarray, q_data: np.ndarray) -> float:
    """Min over arm pairings of mean | |q| - |q| | on the 7 arm joints (robust to the L2 mirror,
    swapped arm order and tasks that override one arm's start pose)."""
    arms_e = [np.abs(q_env[0:7]), np.abs(q_env[9:16])]
    arms_d = [np.abs(q_data[0:7]), np.abs(q_data[9:16])]
    return min(float(np.mean(np.abs(a - b))) for a in arms_e for b in arms_d)


def load_demos(sim_dir: Path) -> pd.DataFrame:
    files = sorted((sim_dir / "data").glob("*/*.parquet"))
    return pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", STATE_KEY, ACTION_KEY])
                     for f in files)


def arm_activity(df: pd.DataFrame, names) -> str:
    """Mean joint-space path length per arm, by feature names (sim stores left first)."""
    q = np.stack(df.sort_values(["episode_index", "frame_index"])[STATE_KEY].to_numpy())
    ep = df.sort_values(["episode_index", "frame_index"])["episode_index"].to_numpy()
    out = {}
    for side in ("left", "right"):
        idx = [i for i, n in enumerate(names) if n.startswith(f"{side}_joint")]
        steps = np.abs(np.diff(q[:, idx], axis=0)).sum(1)
        steps[ep[1:] != ep[:-1]] = 0
        out[side] = steps.sum() / len(np.unique(ep))
    return f"data joint path: agent-0/left {out['left']:.2f}, agent-1/right {out['right']:.2f}"


def check_level(task: str, level: str, args, out: Path):
    env_id = sim_env_id(task, level)
    sim_dir = Path(args.sim_root) / env_id
    print(f"\n=== {env_id}")
    if args.robot_mirror == "on":
        from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
        L0_L3_utils.set_lr_mirror_robot_pose_enabled(True)
    env = make_env(env_id, args.max_steps, None, 1, args.l3_flag, args.shader)
    try:
        base = env.unwrapped
        obs, info = env.reset(seed=0)
        print(f"  env class {type(base).__name__}; mirrored this reset: "
              f"{getattr(base, '_lr_mirror_applied_this_reset', None)}; L3 flag {args.l3_flag}")
        print(f"  obs: state {np.asarray(obs['state']).shape}, rgb {np.asarray(obs['rgb']).shape} "
              f"(expect one 224x224x3 camera)")
        save_png(obs["rgb"], out / f"{env_id}_policy_cam.png")
        save_png(env.render(), out / f"{env_id}_video_cam.png")
        tracker = getattr(base, "reward_tracker", None)
        if tracker is not None:
            print(f"  reward phases: {tracker.phase_names}")

        if not (sim_dir / "meta" / "info.json").exists():
            print(f"  no dataset at {sim_dir}; skipping replay")
            return
        names = read_json(sim_dir / "meta" / "info.json")["features"][STATE_KEY]["names"]
        demos = load_demos(sim_dir)
        print(f"  {arm_activity(demos, names)}")

        # seed fingerprints of this level
        table = {}
        for s in range(args.max_seed):
            o, _ = env.reset(seed=s)
            table[s] = env_qpos18(o)
        for e in sorted(demos["episode_index"].unique())[:args.demos]:
            d = demos[demos["episode_index"] == e].sort_values("frame_index")
            q0 = np.asarray(d[STATE_KEY].iloc[0], dtype=np.float64)
            errs = {s: fingerprint_err(q, q0) for s, q in table.items()}
            seed = min(errs, key=errs.get)
            if errs[seed] > args.match_tol:
                print(f"  demo {e}: no seed < {args.max_seed} matches (best {seed}, err {errs[seed]:.4f})")
                continue
            o, _ = env.reset(seed=seed)
            acts = np.stack(d[ACTION_KEY].to_numpy()).astype(np.float32)
            if args.robot_mirror == "on" and getattr(base, "_lr_mirror_applied_this_reset", False):
                acts = np.concatenate([acts[:, 8:], acts[:, :8]], axis=1)  # data is ordered physically
            first, info = -1, {}
            for t, a in enumerate(acts):
                o, _, _, _, info = env.step({"panda_wristcam-0": a[:8], "panda_wristcam-1": a[8:16]})
                if first < 0 and bool(np.asarray(info.get("success", False))):
                    first = t + 1
            keys = sorted(k for k in info if not k.startswith(("R", "peak_r_", "cur_r_", "episode")))
            print(f"  demo {e}: seed {seed} (fingerprint err {errs[seed]:.5f}), {len(acts)} actions -> "
                  f"{'SUCCESS at step ' + str(first) if first > 0 else 'no success'}")
            if e == sorted(demos["episode_index"].unique())[0]:
                print(f"  info keys: {keys}")
    finally:
        env.close()


def render_variant(name: str, args, out: Path):
    variant = read_json(Path(__file__).with_name("heldout_l3.json"))[name]
    env_id = sim_env_id(variant["task"], "L3")
    for tag in ("original", name):
        env = make_env(env_id, args.max_steps, None, 1, args.l3_flag, args.shader)
        try:
            if tag != "original":
                apply_variant(env.unwrapped, variant)
            obs, _ = env.reset(seed=0)
            save_png(obs["rgb"], out / f"{env_id}_{tag}_policy_cam.png")
            save_png(env.render(), out / f"{env_id}_{tag}_video_cam.png")
            print(f"  wrote {env_id}_{tag}_*.png")
        finally:
            env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="StirSpoon")
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--out", default="runs/c15/sanity")
    ap.add_argument("--sim-root", default=SIM_ROOT)
    ap.add_argument("--demos", type=int, default=3, help="demos to replay per level")
    ap.add_argument("--max-seed", type=int, default=60, help="seeds scanned to recover each demo's seed")
    ap.add_argument("--match-tol", type=float, default=2e-3)
    ap.add_argument("--max-steps", type=int, default=2000, help="env time limit; above any demo length")
    ap.add_argument("--l3-flag", default="off", choices=["on", "off"],
                    help="off reproduces the data-collection scene (default here); eval_diag defaults to on")
    ap.add_argument("--robot-mirror", default="off", choices=["off", "on"],
                    help="off = official eval setting; on = data-collection setting")
    ap.add_argument("--render-variant", default=None, help="only render a heldout_l3.json variant")
    ap.add_argument("--shader", default="rt-fast", help="try 'default' if rt-fast fails (no Vulkan ray tracing)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.render_variant:
        render_variant(args.render_variant, args, out)
        return
    for level in args.levels:
        check_level(args.task, level, args, out)
    print(f"\nimages in {out}")


if __name__ == "__main__":
    main()
