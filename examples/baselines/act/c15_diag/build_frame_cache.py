"""Pre-decode sim RGB frames into uint8 memmaps so training stops decoding video.

Only needed if the throughput check shows the trainer waiting on the DataLoader
(> ~20% data-wait). Then train with FRAME_CACHE=<out> (train_c15.sh) or --sim-frame-cache-dir.

Layout per sim dataset:
  <out>/<repo_id>/observation.images.<cam>.u8    raw uint8, shape [total_frames, H, W, 3]
  <out>/<repo_id>/observation.images.<cam>.json  {"shape": [...], "verify_mismatch": n, ...}
Row i is the frame whose dataset 'index' == i, i.e. what LeRobotDataset decodes for that row.

  python -m examples.baselines.act.c15_diag.build_frame_cache --out runs/c15/frame_cache
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from examples.baselines.act.c15_diag.common import SIM_ROOT, SIM_TRAIN_CONFIG_15, read_json


def load_episodes(repo_dir: Path) -> pd.DataFrame:
    files = sorted((repo_dir / "meta" / "episodes").glob("*/*.parquet"))
    return pd.concat(pd.read_parquet(f) for f in files).sort_values("episode_index").reset_index(drop=True)


def iter_frames(path: Path, chunk: int = 256):
    """Yield uint8 [H, W, 3] frames of a video in order (torchcodec, else PyAV)."""
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError:
        VideoDecoder = None
    if VideoDecoder is not None:
        dec = VideoDecoder(str(path))
        n = dec.metadata.num_frames
        for s in range(0, n, chunk):
            yield from dec.get_frames_in_range(start=s, stop=min(s + chunk, n)).data.permute(0, 2, 3, 1).numpy()
    else:
        import av
        with av.open(str(path)) as c:
            for f in c.decode(video=0):
                yield f.to_ndarray(format="rgb24")


def frame_at(path: Path, index: int):
    """Single frame by index, the way LeRobot's torchcodec backend reads a row (None without torchcodec)."""
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError:
        return None
    return VideoDecoder(str(path)).get_frames_at(indices=[index]).data[0].permute(1, 2, 0).numpy()


def build_one(repo_dir: str, repo_id: str, cam: str, out_root: str, n_verify: int) -> str:
    repo_dir, out_dir = Path(repo_dir), Path(out_root) / repo_id
    key = f"observation.images.{cam}"
    final, spec_path = out_dir / f"{key}.u8", out_dir / f"{key}.json"
    info = read_json(repo_dir / "meta" / "info.json")
    fps, total = info["fps"], info["total_frames"]
    if spec_path.exists() and read_json(spec_path)["shape"][0] == total:
        return f"{repo_id}: cached"
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = load_episodes(repo_dir)
    ccol, fcol, tcol = (f"videos/{key}/chunk_index", f"videos/{key}/file_index", f"videos/{key}/from_timestamp")

    def video_path(ci, fi):
        return repo_dir / info["video_path"].format(video_key=key, chunk_index=int(ci), file_index=int(fi))

    tmp = out_dir / f"{key}.u8.tmp"
    mm, shape = None, None
    filled = np.zeros(total, dtype=bool)
    for (ci, fi), grp in eps.groupby([ccol, fcol]):
        # video frame k of this file -> dataset row, from the episode table
        vid2row = {}
        for _, r in grp.iterrows():
            start, lo = round(float(r[tcol]) * fps), int(r["dataset_from_index"])
            for j in range(int(r["length"])):
                vid2row[start + j] = lo + j
        for k, frame in enumerate(iter_frames(video_path(ci, fi))):
            if mm is None:
                shape = (total, *frame.shape)
                mm = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=shape)
            row = vid2row.get(k)
            if row is not None:
                mm[row] = frame
                filled[row] = True
    short = int((~filled).sum())
    if short:  # rows past the end of a video: repeat the episode's last available frame
        for _, r in eps.iterrows():
            lo, n = int(r["dataset_from_index"]), int(r["length"])
            got = np.flatnonzero(filled[lo:lo + n])
            if len(got) and len(got) < n:
                mm[lo + got[-1] + 1:lo + n] = mm[lo + got[-1]]
                filled[lo:lo + n] = True
    if not filled.all():
        raise RuntimeError(f"{repo_id}: {int((~filled).sum())} rows have no frame")

    # verify random rows against index-based decoding at the row's timestamp (LeRobot's read path)
    rng = np.random.default_rng(0)
    starts = eps["dataset_from_index"].to_numpy()
    mismatch, checked = 0, 0
    for i in rng.integers(0, total, size=min(n_verify, total)):
        r = eps.iloc[int(np.searchsorted(starts, i, side="right") - 1)]
        ts = float(r[tcol]) + (int(i) - int(r["dataset_from_index"])) / fps
        ref = frame_at(video_path(r[ccol], r[fcol]), round(ts * fps))
        if ref is not None:
            checked += 1
            mismatch += int(not np.array_equal(ref, mm[i]))
    mm.flush()
    del mm
    tmp.replace(final)
    spec = {"shape": list(shape), "dtype": "uint8", "repo_id": repo_id, "key": key,
            "verify_samples": checked, "verify_mismatch": mismatch, "short_frames": short}
    spec_path.write_text(json.dumps(spec, indent=2))
    return f"{repo_id}: {total} frames, verify mismatches {mismatch}/{checked}, short {short}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim-root", default=SIM_ROOT)
    ap.add_argument("--sim-config", default=str(SIM_TRAIN_CONFIG_15))
    ap.add_argument("--camera", default="zed2i")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", type=int, default=32, help="random rows per dataset to check")
    args = ap.parse_args()

    cfgs = read_json(args.sim_config)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(build_one, str(Path(args.sim_root) / c["root"]), c["repo_id"], args.camera,
                            args.out, args.verify) for c in cfgs]
        for f in as_completed(futs):
            print(" ", f.result())
    total_gb = sum(p.stat().st_size for p in Path(args.out).rglob("*.u8")) / 1e9
    print(f"done: {len(cfgs)} datasets, {total_gb:.1f} GB in {args.out}")
