"""Phase-1 exploration of the IG-10K subset (see subset.py for the 7 tasks).

  python explore/explore.py levels    # L0-L3 substitution: contact sheets + arm/gripper activity table
  python explore/explore.py variety   # within task x level scene variety + first-frame leakage test
  python explore/explore.py masks     # mask label quality + 2.5D object trajectories (mask + depth)
  python explore/explore.py grippers  # left/right gripper command vs state on real data
  python explore/explore.py episode --tasks H20 --ep 0   # full-length mp4s of one episode, every folder

Every command works on whatever has been downloaded; missing folders/videos are skipped.
"""

import argparse
import re
import textwrap
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subset import TASKS, Folder, decode_depth, depth_range, desc_key, folders, load_descs

STATE_KEY = "observation.qpos_gripper_states"
ACTION_KEY = "action.qpos_gripper_actions"
VIEW_KEYS = ["observation.images.zed2i", "observation.images.cam1"]


def open_folders(args, hids=None):
    for hid in hids or args.tasks:
        for domain, level, rel in folders(hid, args.with_extra):
            if (Path(args.root) / rel / "meta" / "info.json").exists():
                yield hid, domain, level, Folder(Path(args.root) / rel)
            else:
                print(f"  skip (not downloaded): {rel}")


def side_idx(f, side):
    """Joint and gripper column indices of one arm in STATE/ACTION, by feature names (sim is left-first)."""
    out = {}
    for key in (STATE_KEY, ACTION_KEY):
        names = f.names(key)
        joints = [i for i, n in enumerate(names) if n.startswith(f"{side}_joint")]
        grip = [i for i, n in enumerate(names) if n.startswith(f"{side}_gripper")][:1]
        out[key] = (joints, grip)
    return out


def view_key(f):
    return next((k for k in VIEW_KEYS if f.has_video(k)), None)


def parse_desc(texts):
    """Arms the annotation says act / stay idle (majority over paraphrases), plus the scene clause."""
    act, idle = {"left": 0, "right": 0}, {"left": 0, "right": 0}
    for s in texts:
        for m in re.finditer(r"(left|right) arm[^,(]*", s.split(";", 2)[-1].lower()):
            still = re.search(r"still|stationary|remain|idle|motionless|does not move|stay", m.group(0))
            (idle if still else act)[m.group(1)] += 1
    active = [a for a in ("left", "right") if act[a] > idle[a]]
    parts = texts[0].split(";") if texts else ["", ""]
    mojibake = sum(bool(re.search("[âÃ]", s)) for s in texts)
    return active, [a for a in ("left", "right") if a not in active], parts[1].strip() if len(parts) > 1 else "", mojibake


# --------------------------------------------------------------------------------------------- levels
def cmd_levels(args, out):
    descs = load_descs()
    rows = []
    for hid in args.tasks:
        sheet = []
        for _, domain, level, f in open_folders(args, [hid]):
            texts = descs.get(desc_key(domain, level, hid), [])
            act, idle, scene, moji = parse_desc(texts)
            row = dict(task=TASKS[hid][0], hid=hid, domain=domain, level=level,
                       episodes=len(f.episodes), mean_len=round(f.episodes["length"].mean()),
                       text_active=",".join(act), text_idle=",".join(idle),
                       n_paraphrases=len(texts), mojibake=moji, scene=scene)
            if f.has(STATE_KEY):  # robot proprio: which arm actually moves, and do grippers toggle
                for side in ("right", "left"):
                    joints, grip = side_idx(f, side)[STATE_KEY]
                    path, toggles = [], []
                    for ep in f.episodes.index:
                        s = f.series(STATE_KEY, ep)
                        path.append(np.abs(np.diff(s[:, joints], axis=0)).sum())
                        toggles.append(count_toggles(s[:, grip[0]]) if grip else 0)
                    row[f"{side}_joint_path"] = round(float(np.mean(path)), 2)
                    row[f"{side}_grip_toggles"] = round(float(np.mean(toggles)), 2)
                tot = row["right_joint_path"] + row["left_joint_path"]
                row["proprio_active"] = ",".join(
                    s for s in ("left", "right") if tot and row[f"{s}_joint_path"] / tot > 0.15)
            if domain == "human":
                for side in ("left", "right"):
                    k = f"observation.hand.{side}.cam1.is_right"
                    if f.has(k):
                        row[f"{side}_hand_detect"] = round(float((np.stack(f.data[k].to_numpy()) != -1).mean()), 2)
            rows.append(row)
            vk = view_key(f)
            if vk:
                n = int(f.episodes.loc[f.episodes.index[0], "length"])
                frames = f.frames(vk, f.episodes.index[0], [0, n // 3, 2 * n // 3, n - 1])
                sheet.append((f"{domain} {level}\n" + textwrap.fill(scene, 60), frames))
        if sheet:
            save_sheet(sheet, out / f"levels_{hid}_{TASKS[hid][0]}.png")
    df = pd.DataFrame(rows)
    df.to_csv(out / "levels.csv", index=False)
    cols = [c for c in ["task", "domain", "level", "episodes", "mean_len", "text_active", "proprio_active",
                        "right_joint_path", "left_joint_path", "right_grip_toggles", "left_grip_toggles",
                        "left_hand_detect", "right_hand_detect", "mojibake"] if c in df]
    (out / "levels.md").write_text(df[cols].to_markdown(index=False) + "\n\n" +
                                   df[["task", "domain", "level", "scene"]].to_markdown(index=False))
    print(df[cols].to_string(index=False))


def count_toggles(g):
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    if hi - lo < 1e-3:
        return 0
    b = g > (lo + hi) / 2
    return int(np.count_nonzero(b[1:] != b[:-1]))


def save_sheet(sheet, path):
    rows, cols = len(sheet), len(sheet[0][1])
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for r, (title, frames) in enumerate(sheet):
        for c in range(cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(frames):
                ax.imshow(frames[c])
        axes[r][0].set_title(title, fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print("  wrote", path)


# --------------------------------------------------------------------------------------------- variety
def cmd_variety(args, out):
    feats = []  # (hid, domain, level, ep, feature vector)
    for hid, domain, level, f in open_folders(args):
        vk = "observation.images.zed2i" if f.has_video("observation.images.zed2i") else view_key(f)
        if not vk:
            continue
        eps = list(f.episodes.index)
        thumbs = []
        for ep in eps:  # one seek per episode; first frame only
            img = f.frames(vk, ep, [0])[0]
            thumbs.append(img[:: max(1, img.shape[0] // 120), :: max(1, img.shape[1] // 160)])
            g = small_gray(img)
            feats.append((hid, domain, level, ep, (g - g.mean()) / (g.std() + 1e-6)))
        grid(thumbs, out / f"variety_{hid}_{domain}_{level}.png", f"{TASKS[hid][0]} {domain} {level} first frames")
    if not feats:
        return
    meta = pd.DataFrame([x[:4] for x in feats], columns=["hid", "domain", "level", "ep"])
    X = np.stack([x[4].ravel() for x in feats])
    D = ((X[:, None] - X[None]) ** 2).mean(-1)
    np.fill_diagonal(D, np.inf)
    lines = ["# First-frame leakage (1-NN, leave-one-out, 24x32 grayscale thumbnails)\n"]
    for domain in meta["domain"].unique():
        m = (meta["domain"] == domain).to_numpy()
        idx = np.where(m)[0]
        nn = idx[np.argmin(D[np.ix_(idx, idx)], axis=1)]
        acc_task = (meta["hid"].to_numpy()[nn] == meta["hid"].to_numpy()[idx]).mean()
        same = (meta["hid"].to_numpy()[nn] == meta["hid"].to_numpy()[idx]) & \
               (meta["level"].to_numpy()[nn] == meta["level"].to_numpy()[idx])
        lines.append(f"- {domain}: task-id acc {acc_task:.2f} (chance {1 / meta[m]['hid'].nunique():.2f}), "
                     f"task+level acc {same.mean():.2f} (chance {1 / meta[m].groupby(['hid', 'level']).ngroups:.2f})")
    lines.append("\n# Mean pairwise first-frame distance (lower = more uniform scenes)\n")
    np.fill_diagonal(D, 0)
    for (hid, domain, level), g in meta.groupby(["hid", "domain", "level"]):
        i = g.index.to_numpy()
        within = D[np.ix_(i, i)].sum() / max(1, len(i) * (len(i) - 1))
        others = meta[(meta["hid"] == hid) & (meta["domain"] == domain) & (meta["level"] != level)].index.to_numpy()
        between = D[np.ix_(i, others)].mean() if len(others) else float("nan")
        lines.append(f"- {TASKS[hid][0]:18s} {domain:5s} {level:5s} within={within:.3f} vs other levels={between:.3f}")
    (out / "variety.md").write_text("\n".join(lines))
    print("\n".join(lines))


def small_gray(img, hw=(24, 32)):
    g = img.astype(np.float32).mean(-1)
    h, w = g.shape
    g = g[: h - h % hw[0], : w - w % hw[1]]
    return g.reshape(hw[0], g.shape[0] // hw[0], hw[1], g.shape[1] // hw[1]).mean((1, 3))


def grid(imgs, path, title, cols=10):
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.3 * rows + 0.4), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < len(imgs):
            ax.imshow(imgs[i])
            ax.set_title(str(i), fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- masks
def cmd_masks(args, out):
    rows = []
    for hid, domain, level, f in open_folders(args):
        mk, dk = f"observation.images.{args.cam}_mask", f"observation.images.{args.cam}_depth"
        if not f.has_video(mk):
            rows.append(dict(task=TASKS[hid][0], domain=domain, level=level,
                             label="(not downloaded)" if f.has(mk) else "(no masks in dataset)"))
            continue
        labels = f.mask_labels(mk)
        recs = []
        for ep in list(f.episodes.index)[: args.episodes]:
            n = int(f.episodes.loc[ep, "length"])
            fi = list(range(0, n, args.stride))
            masks = [m[..., 0] for m in f.frames(mk, ep, fi)]
            depths = [decode_depth(d, depth_range(domain, args.cam)) for d in f.frames(dk, ep, fi)] \
                if f.has_video(dk) else [None] * len(masks)
            for t, m, d in zip(fi, masks, depths):
                for lid, name in labels.items():
                    ys, xs = np.nonzero(m == lid)
                    r = dict(ep=ep, frame=t, label=name, area=len(xs))
                    if len(xs):
                        r.update(u=xs.mean(), v=ys.mean())
                        if d is not None:
                            z = d[ys, xs]
                            r.update(z=float(np.median(z[z > 0])) if (z > 0).any() else np.nan,
                                     depth_valid=float((z > 0).mean()))
                    recs.append(r)
        df = pd.DataFrame(recs)
        tag = f"{hid}_{domain}_{level}"
        df.to_csv(out / f"traj_{tag}_{args.cam}.csv", index=False)
        plot_traj(df, f, mk, out / f"traj_{tag}_{args.cam}.png", f"{TASKS[hid][0]} {domain} {level} {args.cam}")
        for name, g in df.groupby("label"):
            p = g[g["area"] > 0].sort_values(["ep", "frame"])
            jumps = np.hypot(p.groupby("ep")["u"].diff(), p.groupby("ep")["v"].diff())
            rows.append(dict(task=TASKS[hid][0], domain=domain, level=level, label=name,
                             present=round((g["area"] > 0).mean(), 2),
                             median_area=int(p["area"].median()) if len(p) else 0,
                             jump_gt50px=round(float((jumps > 50).mean()), 3) if len(p) else np.nan,
                             depth_valid=round(float(p["depth_valid"].mean()), 2) if "depth_valid" in p else np.nan,
                             z_median_m=round(float(p["z"].median()), 3) if "z" in p else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(out / f"masks_{args.cam}.csv", index=False)
    (out / f"masks_{args.cam}.md").write_text(df.to_markdown(index=False))
    print(df.to_string(index=False))


def plot_traj(df, f, mk, path, title):
    ep = df["ep"].iloc[0] if len(df) else None
    if ep is None:
        return
    d = df[(df["ep"] == ep) & (df["area"] > 0)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    n = int(f.episodes.loc[ep, "length"])
    rgb_key = mk.replace("_mask", "")
    rgb, m = f.frames(rgb_key, ep, [n // 2])[0], f.frames(mk, ep, [n // 2])[0][..., 0]
    axes[0].imshow(rgb)
    axes[0].imshow(np.ma.masked_equal(m, 0), alpha=0.5, cmap="tab10", vmin=0, vmax=9)
    axes[0].axis("off")
    for name, g in d.groupby("label"):
        axes[1].plot(g["u"], g["v"], ".-", ms=2, label=name)
        axes[2].plot(g["frame"], g["area"], label=name)
        if "z" in g:
            axes[3].plot(g["frame"], g["z"], label=name)
    axes[1].invert_yaxis()
    axes[1].set_title("centroid (px)")
    axes[2].set_title("area (px)")
    axes[3].set_title("median depth in mask (m)")
    axes[1].legend(fontsize=6)
    fig.suptitle(f"{title} ep{ep}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- grippers
def cmd_grippers(args, out):
    rows = []
    for hid, domain, level, f in open_folders(args):
        if domain == "human" or not f.has(STATE_KEY):
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 5), sharex=True)
        for c, side in enumerate(("right", "left")):
            idx = side_idx(f, side)
            (_, gs), (_, ga) = idx[STATE_KEY], idx[ACTION_KEY]
            for ep in f.episodes.index:
                s, a = f.series(STATE_KEY, ep)[:, gs[0]], f.series(ACTION_KEY, ep)[:, ga[0]]
                t = np.linspace(0, 1, len(s))
                axes[0][c].plot(t, a, lw=0.6, alpha=0.5)
                axes[1][c].plot(t, s, lw=0.6, alpha=0.5)
                rows.append(dict(task=TASKS[hid][0], domain=domain, level=level, ep=ep, side=side,
                                 act_min=a.min(), act_max=a.max(), state_min=s.min(), state_max=s.max(),
                                 act_toggles=count_toggles(a), state_toggles=count_toggles(s)))
            axes[0][c].set_title(f"{side} gripper ACTION")
            axes[1][c].set_title(f"{side} gripper STATE")
        fig.suptitle(f"{TASKS[hid][0]} {domain} {level} (all episodes, normalised time)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"grippers_{hid}_{domain}_{level}.png", dpi=80)
        plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(out / "grippers_per_episode.csv", index=False)
    s = df.groupby(["task", "domain", "level", "side"]).agg(
        eps=("ep", "count"), act_min=("act_min", "min"), act_max=("act_max", "max"),
        state_min=("state_min", "min"), state_max=("state_max", "max"),
        eps_cmd_toggle=("act_toggles", lambda x: (x > 0).mean()),
        eps_state_toggle=("state_toggles", lambda x: (x > 0).mean())).round(3).reset_index()
    # commanded to toggle but measured state never toggles -> candidate "non-closure"
    df["cmd_no_follow"] = (df["act_toggles"] > 0) & (df["state_toggles"] == 0)
    s = s.merge(df.groupby(["task", "domain", "level", "side"])["cmd_no_follow"].mean().round(3).reset_index())
    (out / "grippers.md").write_text(s.to_markdown(index=False))
    print(s.to_string(index=False))


# --------------------------------------------------------------------------------------------- episode
TILE = (320, 240)  # w, h of one camera tile
PALETTE = (plt.get_cmap("tab10")(np.arange(10))[:, :3] * 255).astype(np.uint8)


def cmd_episode(args, out):
    """Per folder: one mp4 with every downloaded stream (rows rgb / depth / mask, columns cameras),
    plus overview.mp4 with the main camera of every folder side by side. Episode i of each folder is
    just its i-th episode: human and robot episodes are not paired."""
    from PIL import Image, ImageDraw

    descs = load_descs()
    for hid in args.tasks:
        d = out / f"episode_{hid}_{TASKS[hid][0]}_ep{args.ep}"
        d.mkdir(parents=True, exist_ok=True)
        overview, notes = [], []
        for _, domain, level, f in open_folders(args, [hid]):
            ep = f.episodes.index[min(args.ep, len(f.episodes) - 1)]
            keys = [k for k, v in f.info["features"].items() if v.get("dtype") == "video" and f.has_video(k)]
            if not keys:
                continue
            n = int(f.episodes.loc[ep, "length"])
            cams = sorted({k.split(".")[-1].replace("_depth", "").replace("_mask", "") for k in keys})
            kinds = [s for s in ("", "_depth", "_mask") if any(k.endswith(c + s) for k in keys for c in cams)]
            tiles = {k: decode_tiles(f, k, ep, n) for k in keys}
            grips = gripper_text(f, ep) if f.has(STATE_KEY) else [""] * n
            text = descs.get(desc_key(domain, level, hid), [""])[0]
            name = f"{domain}_{level}".replace("-", "base")
            notes.append(f"## {name}  (episode {ep}, {n} frames, {n / f.fps:.1f}s)\n{text}\n")
            frames = []
            for t in range(n):
                canvas = Image.new("RGB", (TILE[0] * len(cams), TILE[1] * len(kinds) + 40))
                for r, kind in enumerate(kinds):
                    for c, cam in enumerate(cams):
                        k = f"observation.images.{cam}{kind}"
                        if k in tiles:
                            canvas.paste(Image.fromarray(tiles[k][t]), (c * TILE[0], 40 + r * TILE[1]))
                            ImageDraw.Draw(canvas).text((c * TILE[0] + 4, 44 + r * TILE[1]), cam + kind, fill="yellow")
                draw = ImageDraw.Draw(canvas)
                draw.text((4, 2), f"{TASKS[hid][0]} {domain} {level}  ep{ep}  frame {t}/{n - 1}  {grips[t]}", fill="white")
                draw.text((4, 20), text.split(";")[0][:150], fill="gray")
                frames.append(np.asarray(canvas))
            write_mp4(frames, d / f"{name}.mp4", f.fps)
            main = next((k for k in VIEW_KEYS if k in tiles), keys[0])
            overview.append((f"{domain} {level} ({n / f.fps:.0f}s)", tiles[main], f.fps))
        if overview:
            write_overview(overview, d / "overview.mp4")
        (d / "annotations.md").write_text("\n".join(notes), encoding="utf-8")
        print("  wrote", d)


def decode_tiles(f, key, ep, n, chunk=64):
    """All frames of one stream, letterboxed to TILE; masks are colourised."""
    from PIL import Image

    out = []
    for s in range(0, n, chunk):  # chunked so full-res 720p frames never pile up in memory
        for img in f.frames(key, ep, list(range(s, min(s + chunk, n)))):
            if key.endswith("_mask"):
                ids = img[..., 0]
                img = np.where(ids[..., None] > 0, PALETTE[ids % 10], 0).astype(np.uint8)
            im = Image.fromarray(img)
            im.thumbnail(TILE, Image.NEAREST if key.endswith("_mask") else Image.BILINEAR)
            tile = Image.new("RGB", TILE)
            tile.paste(im, ((TILE[0] - im.width) // 2, (TILE[1] - im.height) // 2))
            out.append(np.asarray(tile))
    return out + [out[-1]] * (n - len(out))  # guard against a short decode at the file end


def gripper_text(f, ep):
    s = f.series(STATE_KEY, ep)
    r, l = side_idx(f, "right")[STATE_KEY][1][0], side_idx(f, "left")[STATE_KEY][1][0]
    return [f"grip state R={row[r]:.2f} L={row[l]:.2f}" for row in s]


def write_overview(items, path, cols=3):
    """Main camera of every folder in a grid, each at its own speed, holding its last frame."""
    from PIL import Image, ImageDraw

    rows = int(np.ceil(len(items) / cols))
    fps = items[0][2]
    n = max(len(tiles) for _, tiles, _ in items)
    frames = []
    for t in range(n):
        canvas = Image.new("RGB", (TILE[0] * cols, TILE[1] * rows))
        for i, (label, tiles, _) in enumerate(items):
            x, y = (i % cols) * TILE[0], (i // cols) * TILE[1]
            canvas.paste(Image.fromarray(tiles[min(t, len(tiles) - 1)]), (x, y))
            ImageDraw.Draw(canvas).text((x + 4, y + 4), label + (" (done)" if t >= len(tiles) else ""), fill="yellow")
        frames.append(np.asarray(canvas))
    write_mp4(frames, path, fps)


def write_mp4(frames, path, fps):
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=int(round(fps)))
        s.height, s.width = frames[0].shape[0] // 2 * 2, frames[0].shape[1] // 2 * 2
        s.pix_fmt = "yuv420p"
        s.options = {"crf": "23", "preset": "veryfast"}
        for fr in frames:
            c.mux(s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(fr[: s.height, : s.width]), format="rgb24")))
        c.mux(s.encode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["levels", "variety", "masks", "grippers", "episode"])
    ap.add_argument("--ep", type=int, default=0, help="episode: index within each folder")
    ap.add_argument("--root", default="demos/ig10k")
    ap.add_argument("--out", default="explore_out")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--with-extra", action="store_true")
    ap.add_argument("--cam", default="cam1", help="masks: cam1|cam2|cam3|zed2i (depth exists for cam1-3)")
    ap.add_argument("--episodes", type=int, default=5, help="masks: episodes per folder")
    ap.add_argument("--stride", type=int, default=3, help="masks: frame stride")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    {"levels": cmd_levels, "variety": cmd_variety, "masks": cmd_masks, "grippers": cmd_grippers,
     "episode": cmd_episode}[args.cmd](args, out)
