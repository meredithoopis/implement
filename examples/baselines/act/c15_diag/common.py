"""Shared registry and helpers for the C15 ACT/DINOv2 reproduction and its diagnostics.

Run every script in this folder from the repository root, e.g.
    python -m examples.baselines.act.c15_diag.build_task_bank ...
"""

import functools
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
LEROBOT_DIR = REPO_ROOT / "examples" / "baselines" / "lerobot_dataset"
TASK_MAPPING = LEROBOT_DIR / "task_mapping.json"
EXP_CONFIGS = LEROBOT_DIR / "config" / "exp_configs"
SIM_TRAIN_CONFIG_15 = EXP_CONFIGS / "sim_train_config_15.json"
HUMAN_TRAIN_CONFIG_15 = EXP_CONFIGS / "human_train_config_15.json"
HUMAN_DESC = LEROBOT_DIR / "task_desc" / "human_desc.json"
SIM_DESC = LEROBOT_DIR / "task_desc" / "sim_desc.json"

# Local layout produced by download.py (same folder names as the HF repo).
DATA_ROOT = "demos/ig10k"
HUMAN_ROOT = f"{DATA_ROOT}/imitator_human_v1"
HUMAN_LEVELS_ROOT = f"{DATA_ROOT}/imitator_human_v1_levels"
SIM_ROOT = f"{DATA_ROOT}/imitator_sim_v1_zed2i"

LEVELS = ["L0", "L1", "L2", "L3"]

# Paper Table 7 (C15 pretraining corpus) and the 5+5 evaluation tasks.
C15 = [
    "StirSpoon", "PlaceClothBasket", "PlaceMagazineFolder", "PickWash", "PlaceChipsRack",
    "PlaceFruitBox", "PlacePlateRack", "CutFruit", "PlaceFileFolder", "PlaceBrushRest",
    "CleanCup", "GrindFood", "LiftLidFromSkillet", "FoldTowel", "PlaceMugRack",
]
SEEN = ["StirSpoon", "FoldTowel", "PlaceMugRack", "PlaceFileFolder", "PlacePlateRack"]
UNSEEN = ["PickRemoteControl", "ScanMilkBox", "PourKettle", "PickFood", "FoldBox"]

# Paper Table 10 demo-swap donors (scene unchanged, only the demo is replaced).
SWAP = {
    "StirSpoon": {"similar": "GrindFood", "unrelated": "OpenBox"},
    "FoldTowel": {"similar": "PlaceClothBasket", "unrelated": "PickWash"},
    "PlaceMugRack": {"similar": "PlaceCommodityRack", "unrelated": "PlaceBrushRest"},
    "PlaceFileFolder": {"similar": "PlaceMagazineFolder", "unrelated": "PlaceScrewdriver"},
    "PlacePlateRack": {"similar": "PlaceBookBookcase", "unrelated": "ScanPillBottle"},
}
SWAP_DONORS = sorted({d for v in SWAP.values() for d in v.values()})
OOD_DONORS = [d for d in SWAP_DONORS if d not in C15]

# Human demos filmed in the L1-L3 robot scenes exist only for a few tasks; among the seen
# tasks only PlacePlateRack (human_H14). They use other cameras ("zed", not "zed2i").
ORACLE_TASKS = {"PlacePlateRack": ["L1", "L2", "L3"]}
ORACLE_CAMERA = "zed"

# Reference numbers quoted in the spec (paper Tables 3 and 10), for the report.
PAPER_TABLE3 = {"Seen": (0.81, 0.93), "ZS": (0.02, 0.13), "Scr": (0.76, 0.83), "P+FT": (0.84, 0.88)}
PAPER_TABLE10 = {
    "original": [0.87, 0.77, 0.72, 0.88],
    "similar": [0.32, 0.29, 0.32, 0.32],
    "unrelated": [0.34, 0.28, 0.39, 0.19],
}


def sim_env_id(task: str, level: str) -> str:
    return f"{level}_TwoRobot{task}-v1"


def split_env_id(env_id: str):
    """'L2_TwoRobotStirSpoon-v1' -> ('StirSpoon', 'L2')."""
    level, rest = env_id.split("_", 1)
    return rest.replace("TwoRobot", "").replace("-v1", ""), level


@functools.lru_cache(maxsize=None)
def task_to_human_id() -> Dict[str, str]:
    """Task name (e.g. 'StirSpoon') -> human repo id (e.g. 'human_H1')."""
    out = {}
    for m in json.loads(TASK_MAPPING.read_text())["task_mappings"]:
        sims = [s for s in m["sim_task_id"] if s]
        if sims:
            out[split_env_id(sims[0])[0]] = m["human_task_id"]
    return out


def human_repo(task: str, level: Optional[str] = None) -> str:
    hid = task_to_human_id()[task]
    return hid if level is None else f"{hid}_{level}"


def swap_rows() -> List[dict]:
    """One row per (task, swap kind) with the donor and whether it was a C15 training task."""
    return [
        {"task": t, "kind": k, "donor": d, "donor_in_c15": d in C15}
        for t, kinds in SWAP.items() for k, d in kinds.items()
    ]


def seeded_choice(items: List, key: str, seed: int):
    """Deterministic pick that does not depend on global RNG state."""
    return random.Random(f"{seed}:{key}").choice(items)


def read_json(path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path, rec: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
