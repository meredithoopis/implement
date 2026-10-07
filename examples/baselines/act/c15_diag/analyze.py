"""Aggregate eval_diag rollouts into the report: SR / Sub-SR per level for every condition,
the demo-swap test split into in-distribution vs out-of-distribution donors, paired
comparisons on identical seeds, failure-phase categories with example videos, and which
arm actually grasped at each level.

  python -m examples.baselines.act.c15_diag.analyze --eval-dirs runs/c15/eval/seen \
      --bank-report runs/c15/bank_f4/bank_report.md --out runs/c15/report
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from examples.baselines.act.c15_diag.common import LEVELS, PAPER_TABLE3, PAPER_TABLE10, read_json, read_jsonl

CATEGORIES = ["localization / reach", "grasp", "transport", "release / place", "goal binding",
              "post-grasp (mirrored L2: phases measured on the idle arm)",
              "all phases done, goal predicate false", "no phase data"]
POST_GRASP = ("transport", "release / place", "goal binding")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def phase_done(r: dict, thr: float):
    return [p >= thr for p in (r.get("phase_peaks") or [])]


def failure_label(r: dict, thr: float, reach_radius: float):
    """-> (category, first failed phase). Env phases, with reach/grasp replaced by the
    arm-agnostic probes when they exist (the seen tasks score reach/grasp on agent-0 only)."""
    if r["success_once"]:
        return "", ""
    names, cats, done = r.get("phase_names") or [], r.get("phase_categories") or [], phase_done(r, thr)
    probe = r.get("probe_object") is not None and r.get("probe_min_tcp_dist") is not None
    if probe:
        if not r.get("probe_grasp_arms"):
            cat = "grasp" if r["probe_min_tcp_dist"] <= reach_radius else "localization / reach"
            return cat, f"(probe) {cat}"
        if r.get("mirrored"):
            return CATEGORIES[5], "(probe) grasped, then failed"
        for ok, cat, name in zip(done, cats, names):
            if not ok and cat in POST_GRASP:
                if cat == "transport" and r.get("probe_held_at_end") and not r.get("probe_lost_after_grasp"):
                    return "goal binding", name  # still holding the right object, never got it to the target
                return cat, name
        return ("all phases done, goal predicate false", "(none)") if names else ("no phase data", "")
    if not names:
        return "no phase data", ""
    for ok, cat, name in zip(done, cats, names):
        if not ok:
            return cat, name
    return "all phases done, goal predicate false", "(none)"


def level_table(df: pd.DataFrame, value: str) -> pd.DataFrame:
    t = df.pivot_table(index="label", columns="level", values=value, aggfunc="mean")
    t = t.reindex(columns=[lv for lv in LEVELS if lv in t.columns])
    t["Avg"] = df.groupby("label")[value].mean()
    t["n"] = df.groupby("label")[value].size()
    return t


def md(df: pd.DataFrame, digits: int = 2) -> str:
    return df.round(digits).to_markdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank-report", default=None, help="bank_report.md to quote donor similarities")
    ap.add_argument("--phase-threshold", type=float, default=0.99,
                    help="a phase is done if its peak sub-reward >= this (milestones snap to 1.0)")
    ap.add_argument("--reach-radius", type=float, default=0.06,
                    help="probe: closest gripper distance (m) that counts as having reached the object")
    ap.add_argument("--examples", type=int, default=3, help="example videos per failure category")
    args = ap.parse_args()

    recs = []
    for d in args.eval_dirs:
        for p in sorted(Path(d).glob("episodes_shard*.jsonl")):
            for r in read_jsonl(p):
                r["eval_dir"] = d
                recs.append(r)
    if not recs:
        raise SystemExit("no episodes found")
    df = pd.DataFrame(recs)
    if "tag" not in df:
        df["tag"] = ""
    df["tag"] = df["tag"].fillna("")
    df = df.drop_duplicates(subset=["eval_dir", "env_id", "condition", "trial", "tag"], keep="last")
    df["label"] = [c + (f"[{t}]" if t else "") for c, t in zip(df["condition"], df["tag"])]
    df["sr"] = df["success_once"].astype(float)
    rows = df.to_dict("records")
    df["sub_sr"] = [(sum(d) / len(d)) if d else float("nan") for d in (phase_done(r, args.phase_threshold) for r in rows)]
    labels = [failure_label(r, args.phase_threshold, args.reach_radius) for r in rows]
    df["failure_category"] = [c for c, _ in labels]
    df["first_failed_phase"] = [p for _, p in labels]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "episodes.csv", index=False)

    L = ["# ACT/DINOv2, C15 pretraining: reproduction and task-conditioning diagnostics", ""]
    n_trials = df.groupby(["env_id", "label"]).size()
    L += [f"- rollouts: {len(df)}; trials per task-level-condition: {n_trials.min()}-{n_trials.max()}; "
          f"seeds {df['seed'].min()}-{df['seed'].max()} (identical across conditions)",
          f"- tasks: {', '.join(sorted(df['task'].unique()))}",
          f"- conditions: {', '.join(sorted(df['label'].unique()))}",
          f"- SR = success at any step (rollouts stop at first success). Sub-SR = mean fraction of phases "
          f"whose peak sub-reward reached {args.phase_threshold} (the repo defines no thresholds; milestones "
          "snap to 1.0). With 10 trials one cell carries +/-0.10-0.15 of noise; compare averages.", ""]

    # ── headline vs paper ───────────────────────────────────────────────────────────────────
    L += ["## Headline vs. paper (paper numbers average the 15/30/45 scales; C15 should land ~0.75-0.85)", "",
          "| setting | SR (ours) | 95% CI | Sub-SR (ours) | SR (paper) | Sub-SR (paper) |",
          "|---|---|---|---|---|---|"]
    for setting in ("Seen", "ZS", "Scr", "P+FT"):
        s = df[(df["setting"] == setting) & (df["label"] == "original")]
        if len(s):
            lo, hi = wilson(int(s["sr"].sum()), len(s))
            ref = PAPER_TABLE3[setting]
            L.append(f"| {setting} | {s['sr'].mean():.2f} | {lo:.2f}-{hi:.2f} | {s['sub_sr'].mean():.2f} | "
                     f"{ref[0]:.2f} | {ref[1]:.2f} |")
    sens = []
    for thr in (0.5, 0.9, 0.99):
        o = df[(df["label"] == "original") & (df["setting"] == "Seen")]
        v = [sum(d) / len(d) for d in (phase_done(r, thr) for r in o.to_dict("records")) if d]
        if v:
            sens.append(f"{thr}: {sum(v) / len(v):.2f}")
    L += ["", f"Seen Sub-SR (original) at other thresholds: {', '.join(sens)}", ""]
    seen = df[df["setting"] == "Seen"]

    for setting, s in df.groupby("setting"):
        L += [f"## {setting}: SR per level, every condition", "", md(level_table(s, "sr")), "",
              f"## {setting}: Sub-SR per level, every condition", "", md(level_table(s, "sub_sr")), "",
              f"## {setting}: SR per task x level (condition = original)", ""]
        o = s[s["label"] == "original"]
        if len(o):
            t = o.pivot_table(index="task", columns="level", values="sr", aggfunc="mean")
            t["Avg"] = o.groupby("task")["sr"].mean()
            L += [md(t), ""]

    # ── demo-swap test ──────────────────────────────────────────────────────────────────────
    sw = seen[seen["condition"].str.startswith("swap_") & (seen["tag"] == "")].copy()
    if len(sw):
        sw["kind"] = sw["condition"].str.replace("swap_", "", regex=False)
        sw["donor"] = sw["donor_in_c15"].map({True: "in-distribution (C15 task)", False: "OOD (never trained)"})
        L += ["## Demo-swap test (scene unchanged, demo replaced)", "",
              "Paper Table 10 (45-task checkpoint): "
              + "; ".join(f"{k} L0-L3 = {v}" for k, v in PAPER_TABLE10.items()), ""]
        both = pd.concat([seen[seen["label"] == "original"].assign(row="original")]
                         + [sw[sw["kind"] == k].assign(row=k) for k in ("similar", "unrelated")])
        t = both.pivot_table(index="row", columns="level", values="sr", aggfunc="mean")
        t["Avg"] = both.groupby("row")["sr"].mean()
        L += ["### Paper grouping (similar vs unrelated)", "", md(t), ""]
        t2 = sw.pivot_table(index=["kind", "donor"], columns="level", values="sr", aggfunc="mean")
        t2["Avg"] = sw.groupby(["kind", "donor"])["sr"].mean()
        t2["n"] = sw.groupby(["kind", "donor"])["sr"].size()
        t3 = sw.groupby("donor")["sr"].agg(["mean", "size"]).rename(columns={"mean": "SR", "size": "n"})
        L += ["### Split by whether the donor task was in the C15 training set", "", md(t2), "", md(t3), "",
              "In-distribution pairs: StirSpoon<-GrindFood, FoldTowel<-PlaceClothBasket, FoldTowel<-PickWash, "
              "PlaceMugRack<-PlaceBrushRest, PlaceFileFolder<-PlaceMagazineFolder. OOD: the other five.", ""]

    # ── L3 flag and held-out L3 substitutes (same seeds, L3 only) ───────────────────────────
    variants = read_json(Path(__file__).with_name("heldout_l3.json"))
    l3 = seen[seen["level"] == "L3"]
    rows_l3 = []
    for env_id, s in l3.groupby("env_id"):
        def sr(lab):
            v = s[s["label"] == lab]
            return (v["sr"].mean(), len(v)) if len(v) else (float("nan"), 0)
        on, off = sr("original"), sr("original[l3off]")
        var_tags = sorted(t for t in s["tag"].unique() if t in variants)
        if off[1]:
            rows_l3.append((env_id, "original, L3 flag on (official ACT eval)", *on))
            rows_l3.append((env_id, "original, L3 flag off (training scene)", *off))
        elif var_tags and on[1]:
            rows_l3.append((env_id, "original (training L3 scene; the L3 flag does not change this task)", *on))
        for tag in var_tags:
            for cond in sorted(s[s["tag"] == tag]["condition"].unique()):
                rows_l3.append((env_id, f"{cond}, held-out substitute {tag}", *sr(f"{cond}[{tag}]")))
    if rows_l3:
        L += ["## L3 scene checks: L3 flag and held-out substitutes", "",
              "Baseline for a held-out substitute is the training L3 scene: 'L3 flag off' for StirSpoon "
              "(the flag swaps its bowl), plain 'original' for tasks the flag does not touch.", "",
              "| env | condition | SR | n |", "|---|---|---|---|"]
        L += [f"| {e} | {c} | {v:.2f} | {n} |" for e, c, v, n in rows_l3] + [""]
        for name, v in variants.items():
            if not name.startswith("_") and any(t == name for t in l3["tag"].unique()):
                L.append(f"- {name}: {v['replaces']}")
        L.append("")

    # ── paired comparison on identical seeds (within each setting) ────────────────────────────
    for setting, sd in df.groupby("setting"):
        base = sd[sd["label"] == "original"].set_index(["env_id", "trial"])["success_once"]
        others = sd[sd["label"] != "original"]
        if not len(base) or not len(others):
            continue
        L += [f"## {setting}: paired comparison with the original demo (same env, same seed)", "",
              "| condition | both succeed | only original | only this condition | both fail | "
              "P(success here given success with right demo) |", "|---|---|---|---|---|---|"]
        for lab, s in others.groupby("label"):
            j = s.set_index(["env_id", "trial"])["success_once"].to_frame("c").join(base.rename("o"), how="inner")
            if not len(j):
                continue
            c = Counter(zip(j["o"], j["c"]))
            both_s, only_o, only_c, none = c[(True, True)], c[(True, False)], c[(False, True)], c[(False, False)]
            L.append(f"| {lab} | {both_s} | {only_o} | {only_c} | {none} | {both_s / max(both_s + only_o, 1):.2f} |")
        L += ["", "A high last column under a wrong or empty demo means the policy solves the scene without "
              "using the demo. Tagged L3 variants compare against the original L3 rollouts with the same seed.", ""]

    # ── which arm grasped ───────────────────────────────────────────────────────────────────
    if "probe_grasp_arms" in df:
        g = seen[seen["label"] == "original"].copy()
        g["arm"] = g["probe_grasp_arms"].map(lambda a: "none" if not isinstance(a, list) or not a
                                             else "both" if len(a) > 1 else f"agent-{a[0]}")
        t = g.pivot_table(index=["task", "level"], columns="arm", values="trial", aggfunc="count", fill_value=0)
        if "mirrored" in g:
            t["mirrored"] = g.groupby(["task", "level"])["mirrored"].first()
        L += ["## Which arm grasped the main object (condition = original)", "",
              "At a mirrored L2 the objects sit on the other side; a policy that still reaches with agent-0 "
              "has not adapted to the arm swap.", "", t.to_markdown(), ""]

    # ── failure phases ──────────────────────────────────────────────────────────────────────
    fails = df[~df["success_once"]]
    if len(fails):
        L += ["## Failure categories", "",
              "First phase that never reached the threshold. Reach/grasp come from the arm-agnostic probes; "
              "'goal binding' = held the right object but never brought it to the target, or never performed "
              "the task action (stir/pour/fold/...).", ""]
        for (setting, lab), s in fails.groupby(["setting", "label"]):
            t = s.pivot_table(index="failure_category", columns="level", values="trial", aggfunc="count",
                              fill_value=0)
            t = t.reindex([c for c in CATEGORIES if c in t.index])
            t["total"] = t.sum(axis=1)
            L += [f"### {setting} / {lab} ({len(s)} failures)", "", t.to_markdown(), ""]
        fails.groupby(["setting", "label", "task", "level", "first_failed_phase"]).size().rename("failures") \
             .reset_index().to_csv(out / "failed_phases.csv", index=False)
        L += ["Per task / level first failed phase: failed_phases.csv", "", "## Example failure videos", ""]
        for cat, s in fails[fails["video"].notna()].groupby("failure_category"):
            L.append(f"- **{cat}**")
            for _, r in s.sort_values(["level", "task"]).groupby("level").head(1).head(args.examples).iterrows():
                L.append(f"  - {r['env_id']} / {r['label']} / trial {r['trial']}: "
                         f"`{Path(r['eval_dir']) / r['video']}` (first failed: {r['first_failed_phase']})")
        L.append("")

    if args.bank_report and Path(args.bank_report).exists():
        L += ["## Task-embedding similarity (from build_task_bank)", "",
              Path(args.bank_report).read_text(encoding="utf-8").split("\n", 2)[-1]]

    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for setting, s in df.groupby("setting"):
        level_table(s, "sr").to_csv(out / f"sr_by_condition_level_{setting}.csv")
        level_table(s, "sub_sr").to_csv(out / f"subsr_by_condition_level_{setting}.csv")
    print(f"wrote {out / 'report.md'} ({len(df)} rollouts)")


if __name__ == "__main__":
    main()
