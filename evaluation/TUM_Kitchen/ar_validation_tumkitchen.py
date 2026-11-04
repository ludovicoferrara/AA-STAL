"""
Frame-level validation of the Action Recognition stage on TUM Kitchen.

Differences with respect to the PennAction version:
  * No bbox / track matching (bboxes are validated on PennAction). The person track is
    simply taken from the AR files of the video.
  * The GT has one label per hand (lefthand / righthand) at 25 Hz. Both hands are flattened on the
    person as a multi-label SET per frame (duplicates collapse: Reaching + Reaching -> {Reaching}).
    The trunk column is not used.
  * The evaluation unit is the GT frame (25 Hz). Every GT frame f25 is mapped to the
    nearest pipeline frame (30 fps):  pipe_idx = floor(f25 * PIPELINE_FPS / GT_FPS + 0.5)
  * The model predicts once at the fine level (L0). Predictions and GT are then projected to the
    coarse level (L1) and the metrics are computed at both levels:
        L0: OpeningADoor, OpeningADrawer, ClosingADoor, ClosingADrawer, Reaching, TakingSomething,
            LoweringAnObject, ReleasingGraspOfSomething, CarryingWhileLocomoting
        L1: OpeningADoor/Drawer -> OpeningAContainer, ClosingADoor/Drawer -> ClosingAContainer
  * Macro / micro averages are reported with and without CarryingWhileLocomoting.

Naming:  labels.csv, labels_2.csv, ... labels_19.csv  <->  cam0, cam0_2, ... (and cam1, cam2, ...)
         The suffix is read as an index (no suffix = 1). Videos without a label file are skipped.
         The pipeline works on PARTS of at most 300 frames: cam<N>_<X>_scene<Y>_part<Z> (frame folders in
         DECODE_DIR, AR files cam<N>_<X>_scene<Y>_part<Z>_person_<id>_actions.json). The parts of a video are
         put back on one timeline (windows shifted by the frames of the previous parts) and evaluated together.
         The person track is chosen independently in every part (track ids restart in each part).

Usage:
    python ar_validation_tumkitchen.py --inspect   # check labels and the video <-> label pairing first
    python ar_validation_tumkitchen.py
"""
import os
import re
import json
import glob
import argparse
import numpy as np
import pandas as pd
from collections import Counter

# PATH CONFIG
DATA_ROOT = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT"
AR_DIR = os.path.join(DATA_ROOT, "action_recognition_finished")
GT_DIR = os.path.join(DATA_ROOT, "groundtruth/TUMKitchen")
DECODE_DIR = os.path.join(DATA_ROOT, "Videos_crop_decode")   # folders of extracted frames, one per video

OUTPUT_DIR = "/home/ludovico/workspace/AA-STAL/evaluation/TUMKitchen/action_recognition_validation"
OUTPUT_CSV_PER_VIDEO = os.path.join(OUTPUT_DIR, "ar_validation_per_video.csv")
OUTPUT_CSV_PER_CLASS = os.path.join(OUTPUT_DIR, "ar_validation_per_class_prf1.csv")
OUTPUT_CSV_COOC = os.path.join(OUTPUT_DIR, "ar_validation_cooccurrence_{level}.csv")

# FILE NAMES
GT_FILE_GLOB = "**/*.csv"
GT_FILE_RE = re.compile(r"^labels?(?:_(\d+))?\.csv$", re.IGNORECASE)   # labels.csv -> 1, labels_17.csv -> 17
VIDEO_NAME_RE = re.compile(r"^cam(\d+)(?:_(\d+))?$", re.IGNORECASE)     # cam0 -> (0, 1), cam2_17 -> (2, 17)
# The "_N" suffixes look like the automatic rename of duplicate file names at download time: if so
# they follow the DOWNLOAD ORDER, not the real episode id, and videos and labels could be paired wrongly
# (e.g. if one episode in the middle has no labels). Check the pairing with --inspect; if it is shifted,
# fix it here: {video_index: label_index}.
# Episodio 19 (1-6) non ha label (-1 -> skippato); Episodio 20 (1-7) usa labels_19.csv
VIDEO_TO_LABEL_INDEX = {
    19: -1,
    20: 19,
}

# startframe (a 25 Hz) per ciascun file labels_1.csv .. labels_19.csv
GT_ROW_OFFSET = {
    1: 240,   # 0-0
    2: 255,   # 0-1
    3: 234,   # 0-2
    4: 306,   # 0-3
    5: 256,   # 0-4
    6: 281,   # 0-6
    7: 240,   # 0-7
    8: 237,   # 0-8
    9: 254,   # 0-9
    10: 215,  # 0-10
    11: 290,  # 0-11
    12: 265,  # 0-12
    13: 309,  # 1-0
    14: 195,  # 1-1
    15: 228,  # 1-2
    16: 255,  # 1-3
    17: 242,  # 1-4
    18: 205,  # 1-5
    19: 200,  # 1-7 (saved as labels_19.csv)
}
DEFAULT_GT_ROW_OFFSET = 0

# GT FORMAT: columns  instance,lefthand,righthand,trunk  (column names are matched case-insensitively)
GT_CSV_KWARGS = dict(sep=None, engine="python")   # sniffs the separator; add header=None if no header
GT_COLUMNS = {"left": None, "right": None, "index": None}   # force names (or int indices) if autodetect fails
GT_COLUMN_CANDIDATES = {
    "left": ["lefthand", "lefthandlabel", "lefthandaction", "lhand", "left"],
    "right": ["righthand", "righthandlabel", "righthandaction", "rhand", "right"],
    "index": ["instance", "frame", "frameid", "framenumber", "videoframe", "frameno"],
}

# TEMPORAL ALIGNMENT
PIPELINE_FPS = 30   # frames on which the Action Recognition is run
GT_FPS = 25         # TUM Kitchen labels / videos
# Video frame at 25 Hz of a label row = value of the "instance" column (row number if missing) + offset.
# IMPORTANT: the dataset page says that the first row of the pose files of episode 0-0 is video frame 240
# (tracking starts at "startframe"), while the JPG sequences start at frame 0. The docs do not say whether the
# labels start at video frame 0 or at startframe. --inspect compares the video length with the number of label
# rows; if the labels turn out to start later, set the offset per label index: GT_ROW_OFFSET = {1: 240, ...}
# DEFAULT_GT_ROW_OFFSET = 0
# GT_ROW_OFFSET = {}
# Warn if the video length and the end of the GT differ by more than this (wrong offset or wrong pairing).
ALIGNMENT_WARN_SEC = 2.0

# ACTION VOCABULARY AND HIERARCHY
CARRY = "CarryingWhileLocomoting"
CLASSES_L0 = [
    "OpeningADoor", "OpeningADrawer", "ClosingADoor", "ClosingADrawer",
    "Reaching", "TakingSomething", "LoweringAnObject", "ReleasingGraspOfSomething", CARRY,
]
PARENT = {
    "OpeningADoor": "OpeningAContainer", "OpeningADrawer": "OpeningAContainer",
    "ClosingADoor": "ClosingAContainer", "ClosingADrawer": "ClosingAContainer",
}
CLASSES_L1 = list(dict.fromkeys(PARENT.get(c, c) for c in CLASSES_L0))
LEVELS = {"L0_fine": CLASSES_L0, "L1_coarse": CLASSES_L1}

# CarryingWhileLocomoting is the catch-all of the dataset: when one hand does a specific action the
# other is often "carrying". If True, CARRY is removed from any set (GT and predictions) that also
# contains a specific action, so the model is not asked to say "carrying" next to the salient action.
CARRY_AS_BACKGROUND = False

# AR JSON: keys searched (in order) inside every window. The value can be a string
# ("Reaching, OpeningADrawer"), a list, or a null (-> window skipped, as in the PennAction script).
AR_ACTION_KEYS = ("actions", "action")
# Window frame indices ("frames_10_to_40") are relative to the part (0 = first frame of the part).
# Set False if they are already global to the whole video.
AR_FRAMES_RELATIVE_TO_PART = True
PART_FRAMES = 300   # nominal frames per part: used for the offsets only if a frame folder is missing
PART_NAME_RE = re.compile(r"^(cam\d+(?:_\d+)?)_scene(\d+)_part(\d+)$", re.IGNORECASE)
NULL_TOKENS = {"", "none", "noaction", "noactions", "null", "nan"}   # empty prediction, not an error

# LABEL NORMALISATION: any case / spacing / ontology prefix ("knowrob:") is accepted
def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_CANONICAL = {norm(c): c for c in CLASSES_L0}
_CANONICAL.update({"releasinggrasp": "ReleasingGraspOfSomething", "carrying": CARRY, "carryingidle": CARRY})

UNKNOWN_GT_LABELS = Counter()
UNKNOWN_PRED_LABELS = Counter()
GT_CACHE = {}


def canonical_label(raw):
    """Raw label -> canonical class name, or None if it is missing / not in the vocabulary."""
    if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
        return None
    return _CANONICAL.get(norm(re.split(r"[:#/]", str(raw).strip())[-1]))


def apply_carry_rule(labels):
    if CARRY_AS_BACKGROUND and CARRY in labels and len(labels) > 1:
        return labels - {CARRY}
    return labels


# GROUND TRUTH
def parse_gt_file_index(path):
    m = GT_FILE_RE.match(os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)) if m.group(1) else 1


def parse_video_name(name):
    """-> (camera, video_index) or None if the name is not a TUM video."""
    m = VIDEO_NAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else 1)


def label_index_for_video(video_index):
    return VIDEO_TO_LABEL_INDEX.get(video_index, video_index)


def build_gt_index():
    index = {}
    for path in sorted(glob.glob(os.path.join(GT_DIR, GT_FILE_GLOB), recursive=True)):
        k = parse_gt_file_index(path)
        if k is None:
            continue
        if k in index:
            print(f"[GT] more than one label file with index {k}: keeping {index[k]}, ignoring {path}")
            continue
        index[k] = path
    return index


def read_gt_csv(path):
    return pd.read_csv(path, **GT_CSV_KWARGS)


def resolve_columns(df):
    normalized = {norm(c): c for c in df.columns}
    cols = {}
    for key, candidates in GT_COLUMN_CANDIDATES.items():
        if GT_COLUMNS.get(key) is not None:
            cols[key] = GT_COLUMNS[key]
        else:
            cols[key] = next((normalized[c] for c in candidates if c in normalized), None)
    missing = [k for k in ("left", "right") if cols[k] is None]
    if missing:
        raise ValueError(f"Cannot find the {missing} hand column(s). Columns found: {list(df.columns)}. "
                         f"Set GT_COLUMNS (or GT_CSV_KWARGS['header']=None and use integer indices).")
    return cols


def gt_frame25(df, cols, label_index):
    """Video frame (25 Hz) of every label row."""
    offset = GT_ROW_OFFSET.get(label_index, DEFAULT_GT_ROW_OFFSET)
    if cols["index"] is not None:
        base = df[cols["index"]].to_numpy(dtype=np.int64)
    else:
        base = np.arange(len(df), dtype=np.int64)
    return base + offset


def resolve_gt_pair(raw_left, raw_right):
    """Flatten left/right hand labels into one set. Returns (valid, labels)."""
    labels = set()
    for raw in (raw_left, raw_right):
        c = canonical_label(raw)
        if c is None:
            UNKNOWN_GT_LABELS[str(raw)] += 1
            return False, None
        labels.add(c)
    return True, apply_carry_rule(labels)


def build_gt_rows(df, cols, label_index):
    idx = {c: j for j, c in enumerate(CLASSES_L0)}
    n = len(df)
    Y = np.zeros((n, len(CLASSES_L0)), dtype=bool)
    valid = np.zeros(n, dtype=bool)
    for r, (raw_l, raw_r) in enumerate(zip(df[cols["left"]].tolist(), df[cols["right"]].tolist())):
        ok, labels = resolve_gt_pair(raw_l, raw_r)
        if ok:
            valid[r] = True
            for c in labels:
                Y[r, idx[c]] = True
    return {"Y": Y, "valid": valid, "frame25": gt_frame25(df, cols, label_index),
            "n_rows": n, "n_invalid": int((~valid).sum())}


def load_gt(label_index, gt_index):
    if label_index not in GT_CACHE:
        df = read_gt_csv(gt_index[label_index])
        GT_CACHE[label_index] = build_gt_rows(df, resolve_columns(df), label_index)
    return GT_CACHE[label_index]


# TEMPORAL ALIGNMENT GT 25Hz -> 30 fps pipeline frames
def count_decoded_frames(video_name):
    d = os.path.join(DECODE_DIR, video_name)
    if not os.path.isdir(d):
        return None
    return sum(1 for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def parse_part_name(name):
    """cam0_5_scene1_part2 -> ("cam0_5", 1, 2), or None."""
    m = PART_NAME_RE.match(name)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


def part_offsets(parts):
    """parts sorted by (scene, part). Returns (offset of every part in video frames, total frames or None
    if the last frame folder is missing). Returns (None, None) if an intermediate part is missing."""
    offsets, pos, known = [], 0, True
    for i, p in enumerate(parts):
        offsets.append(pos)
        n = count_decoded_frames(p)
        if n is None:
            if i < len(parts) - 1:
                return None, None
            known, n = False, PART_FRAMES
        pos += n
    return offsets, (pos if known else None)


def map_gt_to_pipeline_frames(frame25, n_decoded):
    """For every 25 Hz GT frame, find the nearest 30 fps pipeline frame and keep the frames
    that fall within the decoded video length. Returns (pipeline_frame_idx, gt_row_idx)."""
    frame25 = np.asarray(frame25)
    rows_all = np.arange(len(frame25))
    ok = frame25 >= 0
    frame25, rows_all = frame25[ok], rows_all[ok]
    if len(frame25) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    pipe_idx = np.floor(frame25 * PIPELINE_FPS / GT_FPS + 0.5).astype(np.int64)

    if n_decoded is not None:
        valid_mask = (pipe_idx >= 0) & (pipe_idx < n_decoded)
    else:
        valid_mask = pipe_idx >= 0

    return pipe_idx[valid_mask], rows_all[valid_mask]


# ACTION RECOGNITION PREDICTIONS
def extract_raw_labels(window_data):
    """Raw label strings of a window, or None if the window has no prediction (null)."""
    if isinstance(window_data, dict):
        found, value = False, None
        for key in AR_ACTION_KEYS:
            if key in window_data:
                found, value = True, window_data[key]
                break
        if not found or value is None:
            return None
    else:
        value = window_data
        if value is None:
            return None

    if isinstance(value, str):
        parts = re.split(r"[,;|]", value)
    elif isinstance(value, (list, tuple)):
        parts = [str(v) for v in value if v is not None]
    else:
        parts = [str(value)]
    parts = [p.strip() for p in parts]
    return [p for p in parts if norm(p) not in NULL_TOKENS]


def resolve_pred_labels(raw_labels):
    out = set()
    for raw in raw_labels:
        c = canonical_label(raw)
        if c is None:
            UNKNOWN_PRED_LABELS[raw] += 1     # outside the 9-class vocabulary: removed from the prediction
            continue
        out.add(c)
    return frozenset(apply_carry_rule(out))


def load_ar_predictions(ar_path):
    """List of (start, end, frozenset(labels)); end is exclusive."""
    with open(ar_path, "r") as f:
        ar_data = json.load(f)

    windows = []
    for window_key, window_data in ar_data.items():
        match = re.match(r"frames_(\d+)_to_(\d+)", window_key)
        if not match:
            continue
        raw = extract_raw_labels(window_data)
        if raw is None:
            continue
        windows.append((int(match.group(1)), int(match.group(2)), resolve_pred_labels(raw)))
    return windows


def n_covered_frames(windows):
    if not windows:
        return 0
    mask = np.zeros(max(e for _, e, _ in windows) + 1, dtype=bool)
    for s, e, _ in windows:
        mask[s:e] = True
    return int(mask.sum())


def select_ar_track(video_name):
    """No bbox here: take the person track with the largest temporal coverage
    (the video has one subject; extra tracks are spurious detections)."""
    pattern = os.path.join(AR_DIR, f"{glob.escape(video_name)}_person_*_actions.json")
    candidates = []
    for path in sorted(glob.glob(pattern)):
        windows = load_ar_predictions(path)
        track_id = re.search(r"_person_([^_]+)_actions\.json$", os.path.basename(path)).group(1)
        candidates.append((n_covered_frames(windows), len(windows), path, track_id, windows))
    if not candidates:
        return None, None, None
    if len(candidates) > 1:
        print(f"[{video_name}] {len(candidates)} person tracks, keeping the one with the largest coverage.")
    best = max(candidates, key=lambda c: (c[0], c[1]))
    return best[2], best[3], best[4]


def assign_frame_windows(windows, n_frames):
    """For every frame index < n_frames, index of the window used for it (-1 if none).
    Among overlapping windows the nearest center wins; ties -> smaller start (same rule as PennAction)."""
    best_dist = np.full(n_frames, np.inf)
    best_start = np.full(n_frames, np.inf)
    sel = np.full(n_frames, -1, dtype=np.int64)
    for w_idx, (s, e, _) in enumerate(windows):
        lo, hi = max(s, 0), min(e, n_frames)
        if hi <= lo:
            continue
        idx = np.arange(lo, hi)
        dist = np.abs((s + e) / 2.0 - idx)
        better = (dist < best_dist[idx]) | ((dist == best_dist[idx]) & (s < best_start[idx]))
        upd = idx[better]
        best_dist[upd] = dist[better]
        best_start[upd] = s
        sel[upd] = w_idx
    return sel


# HIERARCHY
def project_to_l1(Y):
    out = np.zeros((Y.shape[0], len(CLASSES_L1)), dtype=bool)
    for j, c in enumerate(CLASSES_L0):
        out[:, CLASSES_L1.index(PARENT.get(c, c))] |= Y[:, j]
    return out


# METRICS
def set_metrics(Y_true, Y_pred):
    """Subset accuracy and mean Jaccard of the label sets (uncovered frames = empty prediction)."""
    inter = (Y_true & Y_pred).sum(axis=1)
    union = (Y_true | Y_pred).sum(axis=1)
    jacc = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return float((Y_true == Y_pred).all(axis=1).mean()), float(jacc.mean())


def _prf1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_per_class_prf1(Y_true, Y_pred, covered, classes, level):
    """Per-class P/R/F1 over all evaluated frames (each class is a binary decision per frame).
    Frames with no covering window have an empty prediction, so they count as false negatives.
    MACRO: classes with support > 0.  MICRO: sum of tp/fp/fn over all classes.
    *_NO_CARRY rows exclude CarryingWhileLocomoting from both."""
    rows = []
    for j, cls in enumerate(classes):
        t, p = Y_true[:, j], Y_pred[:, j]
        tp = int(np.sum(t & p))
        fp = int(np.sum(~t & p))
        fn = int(np.sum(t & ~p))
        support = int(t.sum())
        prec, rec, f1 = _prf1(tp, fp, fn)
        rows.append({
            "level": level, "action": cls, "precision": prec, "recall": rec, "f1": f1,
            "support": support, "tp": tp, "fp": fp, "fn": fn,
            "coverage_ratio": int(np.sum(t & covered)) / support if support > 0 else np.nan,
        })
    df = pd.DataFrame(rows)

    extra = []
    for suffix, excluded in (("", set()), ("_NO_CARRY", {CARRY})):
        sub = df[~df["action"].isin(excluded)]
        valid = sub[sub["support"] > 0]
        if valid.empty:
            continue
        tp_t, fp_t, fn_t = int(sub["tp"].sum()), int(sub["fp"].sum()), int(sub["fn"].sum())
        mp, mr, mf = _prf1(tp_t, fp_t, fn_t)
        extra.append({
            "level": level, "action": f"MACRO_AVG{suffix}",
            "precision": valid["precision"].mean(), "recall": valid["recall"].mean(), "f1": valid["f1"].mean(),
            "support": int(valid["support"].sum()), "tp": np.nan, "fp": np.nan, "fn": np.nan,
            "coverage_ratio": valid["coverage_ratio"].mean(),
        })
        extra.append({
            "level": level, "action": f"MICRO_AVG{suffix}",
            "precision": mp, "recall": mr, "f1": mf,
            "support": int(valid["support"].sum()), "tp": tp_t, "fp": fp_t, "fn": fn_t,
            "coverage_ratio": float(np.average(valid["coverage_ratio"], weights=valid["support"])),
        })
    out = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    out[["tp", "fp", "fn"]] = out[["tp", "fp", "fn"]].astype("Int64")
    return out


def compute_cooccurrence(Y_true, Y_pred, classes):
    """Row = GT class, column = predicted class: fraction of the GT frames of that class in which the
    column class was predicted (diagonal = recall). '(no_label)' = frames with an empty prediction."""
    t = Y_true.astype(np.int64)
    p = Y_pred.astype(np.int64)
    support = t.sum(axis=0)
    denom = np.where(support > 0, support, np.nan)
    co = (t.T @ p) / denom[:, None]
    none_pred = ~Y_pred.any(axis=1)
    no_label = (Y_true & none_pred[:, None]).sum(axis=0) / denom
    df = pd.DataFrame(co, index=classes, columns=classes)
    df["(no_label)"] = no_label
    df.insert(0, "support", support)
    df.index.name = "gt_class"
    return df


# MAIN SCRIPT
def discover_videos():
    """TUM videos (camX / camX_N) with their parts, found among the decoded-frame folders and the AR files.
    Returns (video, camera, video_index, [part names sorted by scene, part])."""
    names = set()   # part names
    if os.path.isdir(DECODE_DIR):
        names.update(d for d in os.listdir(DECODE_DIR) if os.path.isdir(os.path.join(DECODE_DIR, d)))
    if os.path.isdir(AR_DIR):
        for f in os.listdir(AR_DIR):
            m = re.match(r"(.+)_person_[^_]+_actions\.json$", f)
            if m:
                names.add(m.group(1))
    groups = {}
    for n in names:
        p = parse_part_name(n)
        if p:
            groups.setdefault(p[0], []).append((p[1], p[2], n))
    videos = []
    for base, items in groups.items():
        parsed = parse_video_name(base)
        if parsed:
            videos.append((base, parsed[0], parsed[1], [n for _, _, n in sorted(items)]))
    return sorted(videos, key=lambda v: (v[1], v[2]))


def empty_video_result(meta, reason):
    return {
        **meta, "skip_reason": reason, "track_id": None,
        "n_eval_frames": 0, "n_covered_frames": 0, "coverage_ratio": 0.0, "n_frames_excluded": None,
        "subset_acc_L0": np.nan, "jaccard_L0": np.nan, "subset_acc_L1": np.nan, "jaccard_L1": np.nan,
        "video_sec": None, "gt_end_sec": None,
    }


def evaluate_video(meta, parts, gt):
    """Returns (result_dict, (Y_true_L0, Y_pred_L0, covered) or None)."""
    video_name = meta["video"]
    offsets, n_decoded = part_offsets(parts)
    
    if offsets is None:
        return empty_video_result(meta, "missing_intermediate_part"), None

    windows, track_ids = [], []
    for part, off in zip(parts, offsets):
        ar_path, track_id, part_windows = select_ar_track(part)
        if ar_path is None:
            print(f"[{part}] no AR file: its frames stay uncovered.")
            continue
        track_ids.append(track_id)
        shift = off if AR_FRAMES_RELATIVE_TO_PART else 0
        windows += [(s + shift, e + shift, lab) for s, e, lab in part_windows]
    if not track_ids:
        return empty_video_result(meta, "no_ar_file"), None

    video_sec = n_decoded / PIPELINE_FPS if n_decoded is not None else None
    gt_end_sec = (int(gt["frame25"].max()) + 1) / GT_FPS
    if video_sec is not None and abs(video_sec - gt_end_sec) > ALIGNMENT_WARN_SEC:
        print(f"[{video_name}] WARNING: video lasts {video_sec:.1f}s but labels_{meta['label_index']} end at "
              f"{gt_end_sec:.1f}s. Check GT_ROW_OFFSET and VIDEO_TO_LABEL_INDEX (run --inspect).")

    pipe_idx, rows = map_gt_to_pipeline_frames(gt["frame25"], n_decoded)
    keep = gt["valid"][rows] if len(rows) else np.array([], dtype=bool)
    n_excluded = int((~keep).sum())          # GT frames with a missing / unknown label
    pipe_idx, rows = pipe_idx[keep], rows[keep]
    if len(pipe_idx) == 0:
        return empty_video_result(meta, "no_overlap"), None

    max_pipe = int(pipe_idx.max()) + 1 if len(pipe_idx) > 0 else 0
    sel = assign_frame_windows(windows, max_pipe)[pipe_idx]
    covered = sel >= 0
    idx = {c: j for j, c in enumerate(CLASSES_L0)}
    W = np.zeros((len(windows), len(CLASSES_L0)), dtype=bool)
    for w, (_, _, labels) in enumerate(windows):
        for c in labels:
            W[w, idx[c]] = True
    Y_pred = np.zeros((len(pipe_idx), len(CLASSES_L0)), dtype=bool)
    Y_pred[covered] = W[sel[covered]]
    Y_true = gt["Y"][rows]

    acc0, jac0 = set_metrics(Y_true, Y_pred)
    acc1, jac1 = set_metrics(project_to_l1(Y_true), project_to_l1(Y_pred))
    n = len(pipe_idx)
    result = {
        **meta, "skip_reason": None, "track_id": ",".join(track_ids),
        "n_eval_frames": n, "n_covered_frames": int(covered.sum()), "coverage_ratio": covered.sum() / n,
        "n_frames_excluded": n_excluded,
        "subset_acc_L0": acc0, "jaccard_L0": jac0, "subset_acc_L1": acc1, "jaccard_L1": jac1,
        "video_sec": video_sec, "gt_end_sec": gt_end_sec,
    }
    return result, (Y_true, Y_pred, covered)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gt_index = build_gt_index()
    videos = discover_videos()
    print(f"Found {len(gt_index)} label files and {len(videos)} videos. Starting Action Recognition validation...")

    video_results, pooled = [], []
    skipped = {"no_gt_file": 0, "no_ar_file": 0, "invalid_gt": 0, "no_overlap": 0, "missing_intermediate_part": 0}

    for video_name, camera, video_index, parts in videos:
        label_index = label_index_for_video(video_index)
        meta = {"video": video_name, "camera": camera, "video_index": video_index,
                "label_index": label_index, "n_parts": len(parts)}

        if label_index not in gt_index:
            print(f"[{video_name}] no label file with index {label_index}. Skipping.")
            skipped["no_gt_file"] += 1
            video_results.append(empty_video_result(meta, "no_gt_file"))
            continue

        gt = load_gt(label_index, gt_index)
        if not gt["valid"].any():
            print(f"[{video_name}] No valid GT rows in label file {label_index}. Skipping.")
            skipped["invalid_gt"] += 1
            video_results.append(empty_video_result(meta, "invalid_gt"))
            continue

        result, arrays = evaluate_video(meta, parts, gt)
        video_results.append(result)
        if arrays is None:
            print(f"[{video_name}] {result['skip_reason']}. Skipping.")
            skipped[result["skip_reason"]] += 1
            continue
        pooled.append(arrays)
        print(
            f"[{video_name}] labels={label_index} track={result['track_id']} | frames={result['n_eval_frames']} | "
            f"coverage={result['coverage_ratio']:.2f} | "
            f"subset_acc L0={result['subset_acc_L0']:.3f} L1={result['subset_acc_L1']:.3f} | "
            f"jaccard L0={result['jaccard_L0']:.3f} L1={result['jaccard_L1']:.3f}"
        )

    print_and_save_summary(video_results, skipped, pooled)


def print_and_save_summary(video_results, skipped, pooled):
    print("\n" + "=" * 60)
    print("SKIP RESUME")
    print("=" * 60)
    for reason, count in skipped.items():
        print(f"  {reason}: {count}")
    if UNKNOWN_GT_LABELS:
        print(f"  GT labels not recognised (rows excluded): {dict(UNKNOWN_GT_LABELS)}")
    if UNKNOWN_PRED_LABELS:
        print(f"  predicted labels outside the vocabulary (removed): {dict(UNKNOWN_PRED_LABELS)}")

    if not video_results:
        print("\nNo videos found.")
        return

    df_video = pd.DataFrame(video_results)
    df_video.to_csv(OUTPUT_CSV_PER_VIDEO, index=False)
    df_valid = df_video[df_video["skip_reason"].isna()]

    print("\n" + "=" * 60)
    print("FINAL RESULTS ACTION RECOGNITION (frame level, 25 Hz GT frames)")
    print("=" * 60)
    print(f"Total videos found:                      {len(df_video)}")
    print(f"Videos actually evaluated:               {len(df_valid)}")
    if not df_valid.empty:
        print(f"Average coverage (macro, per video):     {df_valid['coverage_ratio'].mean():.4f}")
        for lvl in ("L0", "L1"):
            print(f"Subset accuracy {lvl} (macro, per video):  {df_valid[f'subset_acc_{lvl}'].mean():.4f}   "
                  f"Jaccard {lvl}: {df_valid[f'jaccard_{lvl}'].mean():.4f}")
    print(f"\nResults per-video saved in: {OUTPUT_CSV_PER_VIDEO}")

    if not pooled:
        print("\nNo evaluable frames for Precision/Recall/F1 calculation.")
        return

    Y_true = np.concatenate([a[0] for a in pooled])
    Y_pred = np.concatenate([a[1] for a in pooled])
    covered = np.concatenate([a[2] for a in pooled])
    per_level = {
        "L0_fine": (Y_true, Y_pred),
        "L1_coarse": (project_to_l1(Y_true), project_to_l1(Y_pred)),
    }

    tables = []
    for level, (yt, yp) in per_level.items():
        classes = LEVELS[level]
        df = compute_per_class_prf1(yt, yp, covered, classes, level)
        tables.append(df)
        compute_cooccurrence(yt, yp, classes).to_csv(OUTPUT_CSV_COOC.format(level=level))
        print(f"\nPrecision / Recall / F1 per class - {level} (pooled over {len(yt)} frames):")
        print(df.drop(columns="level").to_string(
            index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "n/a"))

    df_all = pd.concat(tables, ignore_index=True)
    df_all.to_csv(OUTPUT_CSV_PER_CLASS, index=False)
    print(f"\nPrecision/Recall/F1 per class saved in: {OUTPUT_CSV_PER_CLASS}")
    print("GT x predicted co-occurrence matrices saved in: " + OUTPUT_CSV_COOC.format(level="<level>"))

    print("\nHEADLINE (macro-F1):")
    for level in per_level:
        d = df_all[df_all["level"] == level].set_index("action")["f1"]
        print(f"  {level}: all classes {d['MACRO_AVG']:.4f} | without {CARRY} {d['MACRO_AVG_NO_CARRY']:.4f}"
              f" | micro-F1 {d['MICRO_AVG']:.4f}")


# INSPECTION: GT format + video <-> label pairing
def inspect_gt():
    gt_index = build_gt_index()
    print(f"{len(gt_index)} label files in {GT_DIR}: "
          + ", ".join(f"{k}={os.path.basename(p)}" for k, p in sorted(gt_index.items())))

    raw_values = {"left": Counter(), "right": Counter()}
    gt_info = {}   # label index -> (rows, end of the labels in seconds)
    for n, (k, path) in enumerate(sorted(gt_index.items())):
        df = read_gt_csv(path)
        cols = resolve_columns(df)
        if n == 0:
            print(f"\nFirst file: {path}\nColumns: {list(df.columns)}\nResolved: {cols}\n{df.head(5).to_string()}")
        gt_info[k] = (len(df), (int(gt_frame25(df, cols, k).max()) + 1) / GT_FPS)
        for side in ("left", "right"):
            raw_values[side].update(df[cols[side]].astype(str).tolist())

    for side, counter in raw_values.items():
        print(f"\nRaw labels, {side} hand:")
        for raw, count in counter.most_common():
            print(f"  {raw!r:40} {count:8d}  ->  {canonical_label(raw) or 'UNMAPPED'}")

    videos = discover_videos()
    print(f"\nVIDEO <-> LABEL PAIRING  ({len(videos)} videos; video length = decoded frames / {PIPELINE_FPS} fps, "
          f"label length = end of labels at {GT_FPS} Hz)")
    print(f"{'video':14}{'cam':>4}{'labels':>8}{'video_s':>9}{'labels_s':>10}{'diff_s':>8}  status")
    durations, n_pairs, n_ok, no_label = {}, 0, 0, []
    for name, camera, vidx, parts in videos:
        lidx = label_index_for_video(vidx)
        offsets, n_dec = part_offsets(parts)
        vsec = n_dec / PIPELINE_FPS if n_dec is not None else None
        if vsec is not None:
            durations.setdefault(vidx, []).append(vsec)
        if lidx not in gt_info:
            no_label.append(name)
            print(f"{name:14}{camera:>4}{lidx:>8}{'' if vsec is None else f'{vsec:9.1f}':>9}{'-':>10}{'-':>8}  NO LABEL FILE")
            continue
        gsec = gt_info[lidx][1]
        if vsec is None:
            print(f"{name:14}{camera:>4}{lidx:>8}{'-':>9}{gsec:10.1f}{'-':>8}  no decoded frames in DECODE_DIR or missing intermediate part")
            continue
        diff = vsec - gsec
        n_pairs += 1
        n_ok += abs(diff) <= ALIGNMENT_WARN_SEC
        print(f"{name:14}{camera:>4}{lidx:>8}{vsec:9.1f}{gsec:10.1f}{diff:8.1f}  "
              + ("ok" if abs(diff) <= ALIGNMENT_WARN_SEC else "CHECK"))

    print(f"\nPairs with |video - labels| <= {ALIGNMENT_WARN_SEC}s: {n_ok}/{n_pairs}")
    inconsistent = {v: d for v, d in durations.items() if max(d) - min(d) > 1.0}
    print("Same index, different length across cameras (the suffix is not consistent between cameras): "
          + (", ".join(f"{v} ({min(d):.1f}-{max(d):.1f}s)" for v, d in sorted(inconsistent.items())) or "none"))
    print(f"Videos without label file: {', '.join(no_label) or 'none'}")
    used = {label_index_for_video(v[2]) for v in videos}
    print(f"Label files without videos: {', '.join(str(k) for k in sorted(set(gt_info) - used)) or 'none'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true", help="inspect the GT files and the pairing, then exit")
    args = parser.parse_args()
    if args.inspect:
        inspect_gt()
    else:
        main()