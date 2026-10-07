"""The IG-10K subset used for dataset exploration, and helpers to read it.

Folder layout follows the Hugging Face repo (imitator-game/IG-10K-Dataset):
  imitator_human_v1/human_H20
  imitator_human_v1_levels/human_H15_L1
  imitator_robot_v1/robot_H20_L3
  imitator_sim_v1_zed2i/L3_TwoRobotPourKettle-v1
"""

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DESC = REPO_ROOT / "examples" / "baselines" / "lerobot_dataset" / "task_desc"

# human id -> (sim task name, real-robot levels, human-level folders)
TASKS = {
    "H20": ("PourKettle", ["L0", "L1", "L2", "L3", "L3_1"], []),
    "H4": ("PickRemoteControl", ["L0", "L1", "L2", "L3"], []),
    "H3": ("PlaceClothBasket", ["L0", "L1", "L2", "L3"], []),
    "H27": ("ScanPillBottle", ["L0", "L1", "L2", "L3"], []),
    "H15": ("CutFruit", ["L0", "L1", "L2", "L3", "L3_1"], ["L1", "L2", "L3", "L3_1"]),
    "H29": ("OpenBox", ["L0", "L1", "L2"], []),  # no real-robot L3 released
    "H30": ("FoldBox", ["L0", "L1", "L2", "L3"], []),
}
# extra real-robot variants on the hub (purpose undocumented), only fetched with --with-extra
EXTRA_ROBOT = {"H3": ["L0_e", "L1_e"], "H4": ["L0_e", "L1_e", "L2_e", "L3_e", "L0_e_d"]}
SIM_LEVELS = ["L0", "L1", "L2", "L3"]


def folders(hid, with_extra=False):
    """(domain, level, relative folder) for every folder of one task."""
    name, robot_levels, human_levels = TASKS[hid]
    out = [("human", "-", f"imitator_human_v1/human_{hid}")]
    out += [("human", lv, f"imitator_human_v1_levels/human_{hid}_{lv}") for lv in human_levels]
    robot_levels = robot_levels + (EXTRA_ROBOT.get(hid, []) if with_extra else [])
    out += [("real", lv, f"imitator_robot_v1/robot_{hid}_{lv}") for lv in robot_levels]
    out += [("sim", lv, f"imitator_sim_v1_zed2i/{lv}_TwoRobot{name}-v1") for lv in SIM_LEVELS]
    return out


def desc_key(domain, level, hid):
    name = TASKS[hid][0]
    if domain == "human":
        return f"human_{hid}" if level == "-" else f"human_{hid}_{level}"
    if domain == "real":
        return f"robot_{hid}_{level}"
    return f"{level}_TwoRobot{name}-v1"


def load_descs():
    """Paraphrased annotations: {key: ["task ; scene ; (a) step, (b) step ...", ...]}."""
    out = {}
    for f in ["human_desc.json", "robot_desc.json", "sim_desc.json"]:
        out.update(json.loads((TASK_DESC / f).read_text(encoding="utf-8-sig")))
    return out


class Folder:
    """One LeRobot v3 dataset folder: metadata, numeric data and video frames."""

    def __init__(self, root):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = self.info["fps"]
        self.episodes = pd.concat(
            pd.read_parquet(p) for p in sorted((self.root / "meta" / "episodes").glob("*/*.parquet"))
        ).set_index("episode_index").sort_index()
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = pd.concat(pd.read_parquet(p) for p in sorted((self.root / "data").glob("*/*.parquet")))
        return self._data

    def has(self, key):
        return key in self.info["features"]

    def names(self, key):
        return self.info["features"][key].get("names") or []

    def series(self, key, ep):
        """[T, D] array of a numeric feature for one episode."""
        d = self.data[self.data["episode_index"] == ep].sort_values("frame_index")
        return np.stack(d[key].to_numpy()).astype(np.float32)

    def video_path(self, key, ep):
        row = self.episodes.loc[ep]
        ci, fi = row[f"videos/{key}/chunk_index"], row[f"videos/{key}/file_index"]
        return self.root / "videos" / key / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"

    def has_video(self, key):
        return self.has(key) and (self.root / "videos" / key).exists()

    def frames(self, key, ep, frame_idx):
        """Decode frames (ascending frame indices) of one episode as uint8 [H, W, 3]."""
        t0 = self.episodes.loc[ep, f"videos/{key}/from_timestamp"]
        return read_frames(self.video_path(key, ep), [t0 + i / self.fps for i in frame_idx], 0.5 / self.fps)

    def mask_labels(self, key):
        """{id: label} for a mask feature, e.g. key='observation.images.cam1_mask'."""
        paths = list((self.root / "meta" / "mask_labels").glob(f"*/{key}/global.json"))
        return {int(k): v for k, v in json.loads(paths[0].read_text()).items()} if paths else {}


def read_frames(path, times, tol):
    """Decode the frames at ascending `times` (seconds) from one mp4; seeks only on large gaps."""
    out, last = [], None
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        it = None
        for t in times:
            if last is None or t - last > 2.0:
                c.seek(int(max(t - 0.1, 0) / s.time_base), stream=s, backward=True)
                it = c.decode(s)
            for f in it:
                last = f.time
                if f.time >= t - tol:
                    out.append(f.to_ndarray(format="rgb24"))
                    break
    return out


def decode_depth(rgb, zrange):
    """Hue-encoded depth video -> metres. Mirrors decode_depth in lerobot_robot_dataset.py."""
    from matplotlib.colors import rgb_to_hsv

    hsv = rgb_to_hsv(rgb.astype(np.float32) / 255.0)
    hue = hsv[..., 0] * 360.0
    ok = (hsv[..., 1] >= 0.1) & (hsv[..., 2] >= 0.1) & (hue <= 300.0)
    return np.where(ok, hue / 300.0 * (zrange[1] - zrange[0]) + zrange[0], 0.0).astype(np.float32)


def depth_range(domain, cam):
    # zranges used at encode time, see lerobot_robot_dataset.py / lerobot_human_dataset.py
    if domain == "sim":
        return (0.0, 2.0) if "cam2" in cam else (0.0, 3.0)
    return (0.0, 0.5) if "zed" in cam else (0.0, 4.0)
