"""Download what the C15 ACT/DINOv2 run needs, nothing more.

  required   sim C15 (15 tasks x 4 levels, full folders)            ~6 GB
             human clips (zed2i only) for C15 + the 4 OOD swap donors ~5 GB
  --unseen   sim + human for the 5 unseen tasks (ZS / Scr / P+FT)
  --oracle   human demos filmed in the L1-L3 scenes of PlacePlateRack (camera 'zed')
  --assets   IG-10K-Assets into $MS_ASSET_DIR/data and extract (20 GB compressed)

  python -m examples.baselines.act.c15_diag.download --assets
"""

import argparse
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download

from examples.baselines.act.c15_diag.common import (
    C15, DATA_ROOT, LEVELS, OOD_DONORS, ORACLE_CAMERA, ORACLE_TASKS, UNSEEN,
    human_repo, sim_env_id,
)

DATA_REPO = "imitator-game/IG-10K-Dataset"
ASSET_REPO = "imitator-game/IG-10K-Assets"


def human_patterns(folder: str, cam: str):
    return [f"{folder}/meta/*", f"{folder}/data/*", f"{folder}/videos/observation.images.{cam}/*"]


def data_patterns(args):
    sim_tasks = list(C15) + (list(UNSEEN) if args.unseen else [])
    human_tasks = list(C15) + list(OOD_DONORS) + (list(UNSEEN) if args.unseen else [])
    allow = [f"imitator_sim_v1_zed2i/{sim_env_id(t, lv)}/*" for t in sim_tasks for lv in LEVELS]
    for t in human_tasks:
        allow += human_patterns(f"imitator_human_v1/{human_repo(t)}", "zed2i")
    if args.oracle:
        for t, levels in ORACLE_TASKS.items():
            for lv in levels:
                allow += human_patterns(f"imitator_human_v1_levels/{human_repo(t, lv)}", ORACLE_CAMERA)
    return allow


def extract_archives(asset_dir: Path, keep: bool) -> None:
    for archive in sorted(asset_dir.glob("*.tar.zst")):
        print(f"extracting {archive.name} ...")
        if shutil.which("unzstd"):
            subprocess.run(["tar", "--use-compress-program=unzstd", "-xf", str(archive), "-C", str(asset_dir)],
                           check=True)
        else:  # fall back to the python zstandard package
            import zstandard
            with open(archive, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as reader, \
                    tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(asset_dir)
        if not keep:
            archive.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default=DATA_ROOT)
    ap.add_argument("--unseen", action="store_true", help="also fetch the 5 unseen tasks")
    ap.add_argument("--oracle", action="store_true", help="also fetch human_H14_L1-L3 (oracle condition)")
    ap.add_argument("--assets", action="store_true", help="also fetch and extract IG-10K-Assets")
    ap.add_argument("--no-data", action="store_true", help="skip the dataset (assets only)")
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.no_data:
        allow = data_patterns(args)
        print(f"fetching {len(allow)} patterns into {args.local_dir}")
        snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=args.local_dir,
                          allow_patterns=allow, max_workers=args.workers)

    if args.assets:
        root = os.environ.get("MS_ASSET_DIR")
        if not root:
            raise SystemExit("set MS_ASSET_DIR first (the root disk is full); assets go to $MS_ASSET_DIR/data")
        asset_dir = Path(root) / "data"
        asset_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=ASSET_REPO, repo_type="dataset", local_dir=str(asset_dir),
                          allow_patterns=["*.tar.zst"], max_workers=args.workers)
        extract_archives(asset_dir, args.keep_archives)
        print("assets ready in", asset_dir)
