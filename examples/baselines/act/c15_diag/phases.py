"""Per-rollout phase tracking for Sub-SR and failure labelling.

Env side (every task has a RewardTracker): with reward_mode="dense", each step writes the
current value of every shaped phase into info["R{i}_{phase}"] (1-indexed, phase order).
Peak = running max over the rollout. Milestones snap a phase to exactly 1.0, and the repo
defines no Sub-SR thresholds, so "done" is decided later in analyze.py (default 0.99).

Arm-agnostic probes: the seen tasks' reach/grasp phases use agent-0's TCP only, so at a
mirrored L2 (the other arm acts) they are measured on the idle arm. The probes record the
closest any gripper got to the task's main object and which arm(s) ever grasped it.
"""

import re

import numpy as np

# main manipulated object per task (first attribute that exists on the env)
PROBE_OBJECTS = {
    "StirSpoon": ["spoon"],
    "PlaceMugRack": ["mug"],
    "FoldTowel": ["partnet_link", "cloth"],
    "PlaceFileFolder": ["file"],
    "PlacePlateRack": ["plate", "bowl"],  # L0-L2 move the plate; L3 moves a bowl onto a plate
    "PickRemoteControl": ["remote", "remotecontrol", "remote_control"],
    "ScanMilkBox": ["milkbox", "milk_box"],
    "PourKettle": ["kettle"],
    "PickFood": ["food_0", "fruit_0", "obj_0"],
    "FoldBox": ["box_link", "partnet_link"],
}
L3_PROBE_OVERRIDE = {"PlacePlateRack": ["bowl"]}

PHASE_RULES = [  # (regex on the phase name, category)
    (r"^(open_)?reach", "localization / reach"),
    (r"^grasp", "grasp"),
    (r"^(lift|transport|approach|bring|move)", "transport"),
    (r"^(place|release|return|put)", "release / place"),
]
GOAL = "goal binding"  # task-specific action: stir, pour, fold, scan, cut, clean, grind, open, close


def phase_category(name: str) -> str:
    for pat, cat in PHASE_RULES:
        if re.match(pat, name):
            return cat
    return GOAL


def _scalar(v) -> float:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(a[0]) if a.size else float("nan")


class PhaseTracker:
    def __init__(self, base_env, task: str, level: str):
        self.env, self.task, self.level = base_env, task, level
        self.peaks = {}       # phase index -> peak value
        self.names = {}       # phase index -> name
        self.obj = None
        self.obj_name = None
        self.resolved = False
        self.min_dist = float("inf")
        self.grasp_arms = set()
        self.held = False
        self.ever_held = False
        self.lost_after_hold = False
        self.mirrored = None

    def _resolve(self):
        self.resolved = True
        self.mirrored = bool(getattr(self.env, "_lr_mirror_applied_this_reset", False))
        names = (L3_PROBE_OVERRIDE.get(self.task) if self.level == "L3" else None) or PROBE_OBJECTS.get(self.task, [])
        for n in names:
            obj = getattr(self.env, n, None)
            if obj is not None and hasattr(obj, "pose"):
                self.obj, self.obj_name = obj, n
                return

    def update(self, info: dict):
        for k, v in info.items():
            m = re.match(r"^R(\d+)_(.+)$", k)
            if m:
                i = int(m.group(1))
                self.names[i] = m.group(2)
                self.peaks[i] = max(self.peaks.get(i, -np.inf), _scalar(v))
        if not self.resolved:
            self._resolve()
        if self.obj is None:
            return
        try:
            agents = self.env.agent.agents
            p = self.obj.pose.p
            d = min(float((a.tcp.pose.p - p).norm(dim=-1).min()) for a in agents)
            self.min_dist = min(self.min_dist, d)
            held_now = False
            for i, a in enumerate(agents):
                if bool(np.asarray(a.is_grasping(self.obj).cpu()).reshape(-1)[0]):
                    self.grasp_arms.add(i)
                    held_now = True
            if self.ever_held and self.held and not held_now:
                self.lost_after_hold = True
            self.held = held_now
            self.ever_held |= held_now
        except Exception:  # probe is best-effort; never break a rollout
            self.obj = None

    def result(self, threshold: float = 0.99) -> dict:
        order = sorted(self.peaks)
        names = [self.names[i] for i in order]
        peaks = [round(self.peaks[i], 4) for i in order]
        done = [p >= threshold for p in peaks]
        return {
            "phase_names": names,
            "phase_peaks": peaks,
            "phase_categories": [phase_category(n) for n in names],
            "sub_sr": round(sum(done) / len(done), 4) if done else None,
            "mirrored": self.mirrored,
            "probe_object": self.obj_name,
            "probe_min_tcp_dist": None if self.min_dist == float("inf") else round(self.min_dist, 4),
            "probe_grasp_arms": sorted(self.grasp_arms),
            "probe_held_at_end": self.held,
            "probe_lost_after_grasp": self.lost_after_hold,
        }


def make_tracker(base_env, task: str, level: str) -> PhaseTracker:
    return PhaseTracker(base_env, task, level)
