"""Encode human clips with the frozen DINOv2 task encoder, exactly as ACT/DINOv2 does.

Per clip: K frames at linspace(0, L-1, K) of the zed2i stream -> Resize 224x224 -> [0,1]
-> FrozenVideoBackbone._encode_dino (HF processor, CLS token per frame, mean over frames).
The trainable 1024->256 adapter is NOT applied here; it is trained with the policy.

Writes
  <out>/bank.pt           {repo: {"episodes": [...], "cls": [N,1024] f32, "seq": [N,32,1024] f16}}
  <out>/bank_meta.json    encoder settings, frame rule, which clip the trainer uses per task
  <out>/bank_report.md    shape/pooling checks, within- vs between-task similarity, donor distances
  <te-cache-root>/backbone_<type>/raw_features.pt
                          the trainer's cache: one clip per C15 task, keyed by human repo id.
                          Pre-filling it replaces the repo's precompute, which picks a random
                          clip with colour jitter and never invalidates when --frames changes.

  python -m examples.baselines.act.c15_diag.build_task_bank --frames 4 \
      --out runs/c15/bank_f4 --te-cache-root runs/c15/te_cache_f4

  # few-shot (Scr / P+FT) trainer cache for the unseen tasks, from the bank built above:
  python -m examples.baselines.act.c15_diag.build_task_bank --frames 4 --cache-tasks unseen \
      --from-bank runs/c15/bank_f4/bank.pt --te-cache-root runs/c15/te_cache_unseen_f4
"""

import argparse
import time
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch

from examples.baselines.act.c15_diag.common import (
    C15, HUMAN_LEVELS_ROOT, HUMAN_ROOT, ORACLE_CAMERA, ORACLE_TASKS, SWAP, SWAP_DONORS, UNSEEN,
    human_repo, read_json, seeded_choice, write_json,
)
from examples.baselines.encoders.task_encoder.video_backbone import build_video_backbone
from examples.baselines.lerobot_dataset.lerobot_human_dataset import VideoFrameReader

RESIZE = A.Compose([A.Resize(height=224, width=224, p=1.0)])
TO_UNIT = A.Compose([A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0)])


def load_episodes(repo_dir: Path) -> pd.DataFrame:
    files = sorted((repo_dir / "meta" / "episodes").glob("*/*.parquet"))
    return pd.concat(pd.read_parquet(f) for f in files).sort_values("episode_index").reset_index(drop=True)


def clip_frames(repo_dir: Path, info: dict, row, cam: str, k: int, reader) -> np.ndarray:
    """K frames of one episode as float32 [K, 224, 224, 3] in [0, 1] (augmentation off)."""
    key = f"observation.images.{cam}"
    path = repo_dir / info["video_path"].format(
        video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
        file_index=int(row[f"videos/{key}/file_index"]))
    length = int(row["length"])
    idx = np.linspace(0, length - 1, k).astype(int) if length > k else np.arange(length)
    start = round(float(row[f"videos/{key}/from_timestamp"]) * info["fps"])  # training uses round()
    frames = reader.read_frames(str(path), [start + int(i) for i in idx])
    out = [TO_UNIT(image=RESIZE(image=f)["image"])["image"] for f in frames]
    return np.stack(out).astype(np.float32)


@torch.no_grad()
def encode_repo(bb, repo_dir: Path, cam: str, k: int, reader, batch: int):
    info = read_json(repo_dir / "meta" / "info.json")
    if f"observation.images.{cam}" not in info["features"]:
        raise ValueError(f"{repo_dir.name} has no {cam} stream")
    eps = load_episodes(repo_dir)
    cls_all, seq_all = [], []
    for s in range(0, len(eps), batch):
        clips = [clip_frames(repo_dir, info, r, cam, k, reader) for _, r in eps.iloc[s:s + batch].iterrows()]
        video = torch.from_numpy(np.stack(clips))  # [B, K, H, W, C]
        raw_cls, raw_seq = bb._encode_dino(video)
        cls_all.append(raw_cls.float().cpu())
        seq_all.append(raw_seq.half().cpu())
    return {"episodes": eps["episode_index"].astype(int).tolist(),
            "cls": torch.cat(cls_all), "seq": torch.cat(seq_all)}


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def similarity_report(bank: dict, train_clip: dict) -> list:
    """Within/between-task cosine on raw CLS features, 1-NN task id, swap-donor distances."""
    repos = [r for r in bank if "_L" not in r]  # drop the oracle level folders
    feats = unit(np.concatenate([bank[r]["cls"].numpy() for r in repos]))
    labels = np.concatenate([[i] * len(bank[r]["cls"]) for i, r in enumerate(repos)])
    sim = feats @ feats.T
    within = np.mean([sim[np.ix_(labels == i, labels == i)][~np.eye((labels == i).sum(), dtype=bool)].mean()
                      for i in range(len(repos))])
    cent = unit(np.stack([feats[labels == i].mean(0) for i in range(len(repos))]))
    csim = cent @ cent.T
    between = csim[~np.eye(len(repos), dtype=bool)].mean()
    np.fill_diagonal(sim, -2)
    nn_acc = float((labels[sim.argmax(1)] == labels).mean())

    by_task = {human_repo(t): t for t in set(C15) | set(SWAP_DONORS) | set(UNSEEN)}
    lines = [
        "## Similarity of raw DINOv2 CLS features (cosine)",
        "",
        f"- clips: {len(labels)} from {len(repos)} human repos",
        f"- mean within-task cosine (clip pairs): {within:.3f}",
        f"- mean between-task cosine (task centroids): {between:.3f}",
        f"- leave-one-out 1-NN task identification from a single clip: {nn_acc:.3f} "
        f"(chance {1 / len(repos):.3f})",
        "",
        "## Swap donors in DINOv2 space (centroid cosine to the original task; rank among all other tasks)",
        "",
        "| task | similar donor | cos | rank | unrelated donor | cos | rank |",
        "|---|---|---|---|---|---|---|",
    ]
    idx = {r: i for i, r in enumerate(repos)}
    for t, kinds in SWAP.items():
        i = idx.get(human_repo(t))
        if i is None:
            continue
        order = [j for j in np.argsort(-csim[i]).tolist() if j != i]
        cells = []
        for k in ("similar", "unrelated"):
            j = idx.get(human_repo(kinds[k]))
            cells += [kinds[k], f"{csim[i, j]:.3f}" if j is not None else "n/a",
                      str(order.index(j) + 1) if j is not None else "n/a"]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", "## Nearest other task per seen task", ""]
    for t in SWAP:
        i = idx.get(human_repo(t))
        if i is not None:
            j = [j for j in np.argsort(-csim[i]).tolist() if j != i][0]
            lines.append(f"- {t}: {by_task.get(repos[j], repos[j])} ({csim[i, j]:.3f})")
    lines += ["", "## Training clip per task (episode index used by the trainer)", ""]
    lines += [f"- {by_task.get(r, r)} ({r}): episode {e}" for r, e in sorted(train_clip.items())]
    return lines


def write_trainer_cache(bank: dict, tasks, pool: str, seed: int, cache_path: Path) -> dict:
    """One clip per task (seeded pick from `pool`), keyed by human repo id, in RawFeatureCache format."""
    lo, hi = map(int, pool.split(":"))
    train_clip, cache = {}, {}
    for t in tasks:
        repo = human_repo(t)
        if repo not in bank:
            raise SystemExit(f"{repo} ({t}) is not in the bank; download it and rebuild the bank")
        eps = [e for e in bank[repo]["episodes"] if lo <= e < hi]
        ep = seeded_choice(eps, repo, seed)
        i = bank[repo]["episodes"].index(ep)
        train_clip[repo] = ep
        cache[repo] = {"cls": bank[repo]["cls"][i].clone(), "seq": bank[repo]["seq"][i].float().clone()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return train_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, required=True, help="frames per clip = --frozen-backbone-num-frames")
    ap.add_argument("--out", default=None, help="bank output dir (not needed with --from-bank)")
    ap.add_argument("--te-cache-root", required=True, help="pass the same path to the trainer")
    ap.add_argument("--human-root", default=HUMAN_ROOT)
    ap.add_argument("--human-levels-root", default=HUMAN_LEVELS_ROOT)
    ap.add_argument("--camera", default="zed2i")
    ap.add_argument("--backbone", default="dinov2_vitl14")
    ap.add_argument("--seq-patches", type=int, default=32)
    ap.add_argument("--cache-tasks", default="c15", choices=["c15", "unseen"],
                    help="which tasks the trainer cache covers: C15 pretraining or the unseen few-shot tasks")
    ap.add_argument("--train-pool", default=None,
                    help="episodes the trainer's one clip is drawn from (default 0:50 for c15, 0:10 for unseen, "
                         "matching human_train_config_15.json / human_test_config_unseen.json)")
    ap.add_argument("--train-clip-seed", type=int, default=1)
    ap.add_argument("--from-bank", default=None, help="reuse an existing bank.pt; only write the trainer cache")
    ap.add_argument("--batch", type=int, default=16, help="clips per DINOv2 call")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite-cache", action="store_true")
    args = ap.parse_args()

    cache_tasks = C15 if args.cache_tasks == "c15" else UNSEEN
    pool = args.train_pool or ("0:50" if args.cache_tasks == "c15" else "0:10")
    cache_path = Path(args.te_cache_root) / f"backbone_{args.backbone}" / "raw_features.pt"
    if cache_path.exists() and not args.overwrite_cache:
        raise SystemExit(f"{cache_path} exists; it is keyed only by repo id, so a stale cache would be "
                         "reused silently. Use a fresh --te-cache-root or pass --overwrite-cache.")

    if args.from_bank:
        meta = read_json(Path(args.from_bank).with_name("bank_meta.json"))
        if meta["frames"] != args.frames:
            raise SystemExit(f"bank was built with {meta['frames']} frames, not {args.frames}")
        bank = torch.load(args.from_bank, map_location="cpu", weights_only=False)
        clips = write_trainer_cache(bank, cache_tasks, pool, args.train_clip_seed, cache_path)
        write_json(cache_path.parent / "train_clip.json", {"pool": pool, "train_clip": clips})
        print(f"wrote {cache_path} ({len(clips)} tasks, pool {pool}): {clips}")
        return

    if not args.out:
        raise SystemExit("--out is required when building a bank")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bb = build_video_backbone(backbone_type=args.backbone, latent_dim=256, max_seq_patches=args.seq_patches,
                              adapter_layers=1, num_sampled_frames=args.frames).to(args.device)
    bb._load_backbone()
    bb.eval()
    reader = VideoFrameReader("torchcodec")

    jobs = [(human_repo(t), Path(args.human_root) / human_repo(t), args.camera)
            for t in list(C15) + [d for d in SWAP_DONORS if d not in C15] + list(UNSEEN)]
    for t, levels in ORACLE_TASKS.items():
        jobs += [(human_repo(t, lv), Path(args.human_levels_root) / human_repo(t, lv), ORACLE_CAMERA)
                 for lv in levels]

    bank, skipped = {}, []
    for repo, repo_dir, cam in jobs:
        if not (repo_dir / "meta" / "info.json").exists():
            skipped.append(repo)
            continue
        t0 = time.time()
        bank[repo] = encode_repo(bb, repo_dir, cam, args.frames, reader, args.batch)
        print(f"  {repo:16s} {len(bank[repo]['episodes']):3d} clips  {time.time() - t0:5.1f}s")
    missing = [human_repo(t) for t in cache_tasks if human_repo(t) not in bank]
    if missing:
        raise SystemExit(f"missing human repos for the trainer cache: {missing} (run download.py first)")

    train_clip = write_trainer_cache(bank, cache_tasks, pool, args.train_clip_seed, cache_path)
    torch.save(bank, out / "bank.pt")

    # ── checks: shapes, and adapter(bank feature) == FrozenVideoBackbone.forward on the same clips.
    # Re-encode exactly the first bank batch (same batch shape -> same fp16 kernels).
    repo0 = human_repo(C15[0])
    repo_dir = Path(args.human_root) / repo0
    info = read_json(repo_dir / "meta" / "info.json")
    first = load_episodes(repo_dir).iloc[:args.batch]
    video = torch.from_numpy(np.stack([clip_frames(repo_dir, info, r, args.camera, args.frames, reader)
                                       for _, r in first.iterrows()]))
    with torch.no_grad():
        cls_fwd, _ = bb(video)
        cls_bank = bb.cls_adapter(bank[repo0]["cls"][:len(first)].to(args.device))
    pool_err = (cls_fwd - cls_bank).abs().max().item()
    shapes = {"cls": list(bank[repo0]["cls"].shape[1:]), "seq": list(bank[repo0]["seq"].shape[1:])}

    meta = {
        "backbone": args.backbone, "frames": args.frames, "camera": args.camera, "image_size": [224, 224],
        "frame_rule": "linspace(0, L-1, K).astype(int); start=round(from_timestamp*fps); no augmentation",
        "pooling": "CLS token per frame, mean over K frames (raw 1024-d); adapter applied later",
        "seq_patches": args.seq_patches, "train_pool": pool, "train_clip_seed": args.train_clip_seed,
        "train_clip": train_clip, "te_cache": str(cache_path),
        "repos": {r: len(v["episodes"]) for r, v in bank.items()}, "skipped": skipped,
        "check_shapes": shapes, "check_adapter_vs_forward_max_abs_diff": pool_err,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(out / "bank_meta.json", meta)

    lines = ["# Task bank report", "",
             f"- encoder: {args.backbone}, frames per clip: {args.frames}, camera: {args.camera}",
             f"- raw feature shapes: cls {shapes['cls']}, seq {shapes['seq']} (expected [1024], [{args.seq_patches}, 1024])",
             f"- adapter(bank cls) vs FrozenVideoBackbone.forward on the same clips: max abs diff {pool_err:.2e} "
             "(expect < 1e-4; larger means the frame pipeline differs from the encoder's)",
             f"- training cache: {cache_path} ({len(train_clip)} entries, pool {pool})",
             f"- repos skipped (not downloaded): {skipped or 'none'}", ""]
    lines += similarity_report(bank, train_clip)
    (out / "bank_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"\nwrote {out / 'bank.pt'}, {out / 'bank_meta.json'}, {out / 'bank_report.md'}, {cache_path}")


if __name__ == "__main__":
    main()
