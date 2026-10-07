"""Download the 7-task IG-10K exploration subset from the Hugging Face hub.

Tiers are cumulative:
  meta   meta/ + data/ parquet only (proprio, gripper, actions, mask labels)  ~ a few hundred MB
  video  + RGB/depth videos, no masks
  mask   + segmentation mask videos                                           ~ 93 GB total

  python explore/download_subset.py --tier meta
  python explore/download_subset.py --tier mask --tasks H20 H4
"""

import argparse

from huggingface_hub import snapshot_download

from subset import TASKS, folders

TIERS = ["meta", "video", "mask"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=TIERS, default="meta")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--local-dir", default="demos/ig10k")
    ap.add_argument("--with-extra", action="store_true", help="also fetch robot_*_e / _e_d variants")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    allow, ignore = [], []
    for hid in args.tasks:
        for _, _, rel in folders(hid, args.with_extra):
            allow += [f"{rel}/meta/*", f"{rel}/data/*"]
            if args.tier != "meta":
                allow.append(f"{rel}/videos/*")
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
