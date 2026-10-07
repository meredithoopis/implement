"""Download the 7-task IG-10K exploration subset from the Hugging Face hub.

Tiers are cumulative:
  meta   meta/ + data/ parquet only (proprio, gripper, actions, mask labels)  ~0.5 GB
  video  + RGB/depth videos of --cams, no masks
  mask   + segmentation mask videos of --cams

Sizes for all 7 tasks (human + real), per camera: cam1/2/3 ~11 GB RGB + ~12 GB depth + ~0.7 GB mask,
zed2i ~18 GB RGB + ~1.7 GB mask (no real/human zed2i depth). Sim only has zed2i (~2 GB, always fetched).
Default --cams cam1 at tier mask is ~26 GB in total.

  python explore/download_subset.py --tier meta
  python explore/download_subset.py --tier mask                    # cam1 rgb+depth+mask, ~26 GB
  python explore/download_subset.py --tier video --cams cam1 zed2i # + zed2i overview, ~+18 GB
"""

import argparse

from huggingface_hub import snapshot_download

from subset import TASKS, folders

TIERS = ["meta", "video", "mask"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=TIERS, default="meta")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--cams", nargs="+", default=["cam1"], help="cam1 cam2 cam3 zed2i, or 'all'")
    ap.add_argument("--local-dir", default="demos/ig10k")
    ap.add_argument("--with-extra", action="store_true", help="also fetch robot_*_e / _e_d variants")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    allow, ignore = [], []
    for hid in args.tasks:
        for domain, _, rel in folders(hid, args.with_extra):
            allow += [f"{rel}/meta/*", f"{rel}/data/*"]
            if args.tier == "meta":
                continue
            if domain == "sim" or "all" in args.cams:
                allow.append(f"{rel}/videos/*")
            else:
                for cam in args.cams:
                    allow += [f"{rel}/videos/observation.images.{cam}/*",
                              f"{rel}/videos/observation.images.{cam}_depth/*",
                              f"{rel}/videos/observation.images.{cam}_mask/*"]
    if args.tier != "mask":
        ignore.append("*/videos/*_mask/*")  # keep meta/mask_labels/*_mask/global.json

    path = snapshot_download(
        repo_id="imitator-game/IG-10K-Dataset",
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=allow,
        ignore_patterns=ignore,
        max_workers=args.workers,
    )
    print("downloaded to", path)
