import os
import re
import json
import glob
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter, defaultdict

# PATH CONFIG
DATA_ROOT = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT"
DET_TRACK_DIR = os.path.join(DATA_ROOT, "video_general_obj_det_finished")
AR_DIR = os.path.join(DATA_ROOT, "action_recognition_finished")
GT_DIR = os.path.join(DATA_ROOT, "groundtruth/PennAction")
DECODE_DIR = os.path.join(DATA_ROOT, "Videos_crop_decode")

OUTPUT_DIR = "/home/ludovico/workspace/AA-STAL/evaluation/PennAction/action_recognition_validation"
OUTPUT_CSV_PER_VIDEO = os.path.join(OUTPUT_DIR, "ar_validation_per_video.csv")
OUTPUT_CSV_PER_CLASS = os.path.join(OUTPUT_DIR, "ar_validation_per_class_prf1.csv")

REQUIRED_GT_COLS = ["frame_id", "action", "x_min", "y_min", "x_max", "y_max"]

# Minimum IoU threshold (across the entire GT) to accept a track as "the right person".
# Below this threshold, tracking is too unreliable to sensibly associate Action Recognition predictions with the GT.
MIN_MEAN_IOU_FOR_TRACK_MATCH = 0.1

# Label used for GT frames not covered by any AR window. It is
# treated as a true negative class in the precision/recall calculation per class,
# so that the recall also reflects the gaps in the coverage of the Action Recognition model,
# not just the classification errors on the actually predicted frames.
NO_PRED_LABEL = "no_prediction"

ACTIONS = [
    "baseball_pitch", "clean_and_jerk", "pull_up", "strum_guitar",
    "baseball_swing", "golf_swing", "push_up", "tennis_forehand",
    "bench_press", "jumping_jacks", "sit_up", "tennis_serve", "bowl",
    "jump_rope", "squat"
]
ACTIONS_SET = set(ACTIONS)


_warned_labels = set()


# UTILITY BBOX / MATCHING PERSON-GT
def compute_iou_batch(boxes_a, boxes_b):
    xA = np.maximum(boxes_a[:, 0], boxes_b[:, 0])
    yA = np.maximum(boxes_a[:, 1], boxes_b[:, 1])
    xB = np.minimum(boxes_a[:, 2], boxes_b[:, 2])
    yB = np.minimum(boxes_a[:, 3], boxes_b[:, 3])

    inter_w = np.clip(xB - xA, 0, None)
    inter_h = np.clip(yB - yA, 0, None)
    inter_area = inter_w * inter_h

    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)

    union = area_a + area_b - inter_area
    return np.where(union > 0, inter_area / (union + 1e-8), 0.0)


def is_valid_box_array(boxes):
    return (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])


def get_frame_dimensions(video_name):
    video_frame_dir = os.path.join(DECODE_DIR, video_name)
    if not os.path.isdir(video_frame_dir):
        return None, None
    frames = sorted(glob.glob(os.path.join(video_frame_dir, "*")))
    if not frames:
        return None, None
    try:
        with Image.open(frames[0]) as img:
            return img.size  # (width, height)
    except Exception:
        return None, None


def evaluate_track_against_gt(pred_bboxes, gt_frame_idx, gt_boxes, width, height):
    """
    Evaluate a track respect to all frames requested by the GT.
    """
    n = len(gt_frame_idx)
    pred_len = len(pred_bboxes)

    detected_mask = np.zeros(n, dtype=bool)
    pred_boxes_px = np.zeros((n, 4), dtype=np.float64)

    within_range = gt_frame_idx < pred_len
    for pos in np.where(within_range)[0]:
        frame_i = gt_frame_idx[pos]
        box = pred_bboxes[frame_i]

        if box is None:
            continue
        if len(box) != 4:
            continue

        box_px = np.asarray(box, dtype=np.float64)
        box_px[[0, 2]] *= width
        box_px[[1, 3]] *= height
        pred_boxes_px[pos] = box_px
        detected_mask[pos] = True

    if detected_mask.any():
        valid_box_mask = is_valid_box_array(pred_boxes_px)
        detected_mask = detected_mask & valid_box_mask

    iou = np.zeros(n, dtype=np.float64)
    if detected_mask.any():
        iou[detected_mask] = compute_iou_batch(gt_boxes[detected_mask], pred_boxes_px[detected_mask])

    return iou


def find_best_person_track(video_name, gt_frame_idx, gt_boxes, width, height, min_mean_iou):
    """
    Find best person ID track (higher avg IoU). 
    Returns (track_id, fail_reason, mean_iou):
    fail_reason is None in case of success, otherwise indicates the reason for skipping
    ("no_prediction", "no_person_tracks", "no_valid_track").
    """
    json_path = os.path.join(DET_TRACK_DIR, video_name, f"{video_name}.json")
    if not os.path.exists(json_path):
        return None, "no_prediction", None

    with open(json_path, "r") as f:
        pred_data = json.load(f)

    person_tracks = {
        tid: info for tid, info in pred_data.get("detected_objects", {}).items()
        if info.get("class_name") == "person"
    }
    if not person_tracks:
        return None, "no_person_tracks", None

    best_track_id, best_mean_iou = None, -1.0
    for track_id, track_info in person_tracks.items():
        iou = evaluate_track_against_gt(track_info.get("bbox", []), gt_frame_idx, gt_boxes, width, height)
        mean_iou = float(iou.mean())
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_track_id = track_id

    if best_mean_iou < min_mean_iou:
        return None, "no_valid_track", best_mean_iou

    return best_track_id, None, best_mean_iou


# LOAD ACTION RECOGNITION PREDICTIONS
def load_ar_predictions(video_name, track_id):
    """Load action predictions for a given track. 
    Returns 
    (None, filename) if the file does not exist, 
    (list of (start, end, action), filename) otherwise."""
    clean_track_id = str(track_id).split("_")[-1] if "_" in str(track_id) else str(track_id)
    ar_filename = f"{video_name}_person_{clean_track_id}_actions.json"
    ar_path = os.path.join(AR_DIR, ar_filename)

    if not os.path.exists(ar_path):
        return None, ar_filename

    with open(ar_path, "r") as f:
        ar_data = json.load(f)

    windows = []
    for window_key, window_data in ar_data.items():
        match = re.match(r"frames_(\d+)_to_(\d+)", window_key)
        if not match:
            continue
        start_f, end_f = int(match.group(1)), int(match.group(2))
        action = window_data.get("action", "")
        if action is None:
            continue
        windows.append((start_f, end_f, action))

    return windows, ar_filename


def assign_frame_actions(windows):
    """
    Assign each frame the nearest window.
    The assignment for "nearest center" is deterministic and dynamic.
    In case of tie in the distance, the window with the smaller start
    value wins.
    """
    frame_to_windows = defaultdict(list)
    for start_f, end_f, action in windows:
        for f_idx in range(start_f, end_f):
            frame_to_windows[f_idx].append((start_f, end_f, action))

    frame_action = {}
    for f_idx, candidates in frame_to_windows.items():
        best = min(candidates, key=lambda w: (abs((w[0] + w[1]) / 2.0 - f_idx), w[0]))
        frame_action[f_idx] = best[2]

    return frame_action


# PRECISION / RECALL / F1 CLASS-WISE
def compute_per_class_prf1(y_true, y_pred):
    """
    Precision/Recall/F1 for each class, calculated on
    all valid GT frames: the frame non covered are treated as predicted with the label NO_PRED_LABEL.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rows = []
    for cls in ACTIONS:
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        support = int(np.sum(y_true == cls))
        covered = int(np.sum((y_true == cls) & (y_pred != NO_PRED_LABEL)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({
            "action": cls,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "coverage_ratio": (covered / support) if support > 0 else np.nan,
        })

    df = pd.DataFrame(rows)
    valid = df[df["support"] > 0]

    if valid.empty:
        return df

    macro = {
        "action": "MACRO_AVG",
        "precision": valid["precision"].mean(),
        "recall": valid["recall"].mean(),
        "f1": valid["f1"].mean(),
        "support": int(valid["support"].sum()),
        "coverage_ratio": valid["coverage_ratio"].mean(),
    }
    weighted = {
        "action": "WEIGHTED_AVG",
        "precision": np.average(valid["precision"], weights=valid["support"]),
        "recall": np.average(valid["recall"], weights=valid["support"]),
        "f1": np.average(valid["f1"], weights=valid["support"]),
        "support": int(valid["support"].sum()),
        "coverage_ratio": np.average(valid["coverage_ratio"], weights=valid["support"]),
    }

    tp_tot = int(sum(np.sum((y_true == c) & (y_pred == c)) for c in ACTIONS))
    fp_tot = int(sum(np.sum((y_true != c) & (y_pred == c)) for c in ACTIONS))
    fn_tot = int(sum(np.sum((y_true == c) & (y_pred != c)) for c in ACTIONS))
    micro_p = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
    micro_r = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    micro = {
        "action": "MICRO_AVG",
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f1,
        "support": int(valid["support"].sum()),
        "coverage_ratio": np.nan,
    }

    return pd.concat([df, pd.DataFrame([macro, weighted, micro])], ignore_index=True)


# MAIN SCRIPT 

def empty_video_result(video_name, gt_action, reason, n_gt_frames, track_mean_iou=None):
    return {
        "video": video_name,
        "gt_action": gt_action,
        "skip_reason": reason,
        "track_id": None,
        "track_mean_iou": track_mean_iou,
        "n_gt_frames": n_gt_frames,
        "n_covered_frames": 0,
        "coverage_ratio": 0.0,
        "correct_frames": 0,
        "frame_accuracy_covered": np.nan,
        "frame_accuracy_all": 0.0,
        "video_pred_action": None,
        "video_match": 0,
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.csv")))

    video_results = []
    all_y_true = []
    all_y_pred = []

    skipped = {
        "no_prediction": 0, "no_frame_dims": 0, "invalid_gt": 0,
        "no_person_tracks": 0, "no_valid_track": 0, "no_ar_file": 0,
    }

    print(f"Found {len(gt_files)} ground truth files. Starting Action Recognition validation...")

    for gt_path in gt_files:
        video_name = os.path.splitext(os.path.basename(gt_path))[0]

        width, height = get_frame_dimensions(video_name)
        if width is None or height is None:
            print(f"[{video_name}] Impossible to recover frame dimensions. Skipping.")
            skipped["no_frame_dims"] += 1
            continue

        df_gt = pd.read_csv(gt_path)
        missing_cols = [c for c in REQUIRED_GT_COLS if c not in df_gt.columns]
        if missing_cols:
            print(f"[{video_name}] CSV GT missing columns {missing_cols}. Skipping.")
            skipped["invalid_gt"] += 1
            continue

        df_gt = df_gt.sort_values("frame_id").reset_index(drop=True)

        gt_boxes_raw = df_gt[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=np.float64)
        valid_box_mask = is_valid_box_array(gt_boxes_raw)
        n_invalid = int((~valid_box_mask).sum())
        if n_invalid > 0:
            print(f"[{video_name}] {n_invalid} GT rows with invalid bounding boxes, excluded.")
        df_gt = df_gt[valid_box_mask].reset_index(drop=True)

        if df_gt.empty:
            print(f"[{video_name}] No valid GT rows after filtering. Skipping.")
            skipped["invalid_gt"] += 1
            continue

        gt_frame_idx = df_gt["frame_id"].to_numpy(dtype=int) - 1
        gt_boxes = df_gt[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=np.float64)

        video_action_raw = df_gt["action"].mode().iat[0] if df_gt["action"].notna().any() else None
        video_gt_action = video_action_raw if video_action_raw is not None else None

        gt_actions = df_gt["action"]
        action_valid_mask = gt_actions.notna().to_numpy()
        n_invalid_action = int((~action_valid_mask).sum())
        if n_invalid_action > 0:
            print(f"[{video_name}] {n_invalid_action} GT rows with missing or unmappable actions, "
                  f"excluded from AR metrics denominator.")

        # 1. Find the best person track (if any) that matches the GT frames, based on mean IoU.
        best_track_id, fail_reason, track_mean_iou = find_best_person_track(
            video_name, gt_frame_idx, gt_boxes, width, height, MIN_MEAN_IOU_FOR_TRACK_MATCH
        )
        if best_track_id is None:
            miou_str = f"{track_mean_iou:.3f}" if track_mean_iou is not None else "n/a"
            print(f"[{video_name}] {fail_reason} (mean_iou={miou_str}). Skipping.")
            skipped[fail_reason] += 1
            video_results.append(empty_video_result(
                video_name, video_gt_action, fail_reason, int(action_valid_mask.sum()), track_mean_iou
            ))
            continue

        # Frame subset with valid GT action
        eval_frame_idx = gt_frame_idx[action_valid_mask]
        eval_gt_actions = gt_actions.to_numpy()[action_valid_mask]

        if len(eval_frame_idx) == 0:
            print(f"[{video_name}] No valid GT rows with action. Skipping.")
            skipped["invalid_gt"] += 1
            video_results.append(empty_video_result(
                video_name, video_gt_action, "invalid_gt", 0, track_mean_iou
            ))
            continue

        # 2. Load AR predictions
        windows, ar_filename = load_ar_predictions(video_name, best_track_id)
        if windows is None:
            print(f"[{video_name}] Missing AR file {ar_filename}. Skipping.")
            skipped["no_ar_file"] += 1
            video_results.append(empty_video_result(
                video_name, video_gt_action, "no_ar_file", len(eval_frame_idx), track_mean_iou
            ))
            continue

        frame_action = assign_frame_actions(windows)

        # 3. Frame-level evaluation
        correct_frames = 0
        covered_frames = 0
        video_pred_pool = [] 

        for f_idx, gt_action in zip(eval_frame_idx, eval_gt_actions):
            pred_action = frame_action.get(int(f_idx))

            all_y_true.append(gt_action)
            all_y_pred.append(pred_action if pred_action is not None else NO_PRED_LABEL)

            if pred_action is not None:
                covered_frames += 1
                video_pred_pool.append(pred_action)
                if pred_action == gt_action:
                    correct_frames += 1

        n_gt_frames = len(eval_frame_idx)
        coverage_ratio = covered_frames / n_gt_frames if n_gt_frames > 0 else 0.0
        frame_acc_covered = correct_frames / covered_frames if covered_frames > 0 else np.nan
        frame_acc_all = correct_frames / n_gt_frames if n_gt_frames > 0 else 0.0

        video_pred_action = Counter(video_pred_pool).most_common(1)[0][0] if video_pred_pool else None
        video_match = int(video_pred_action is not None and video_pred_action == video_gt_action)

        result = {
            "video": video_name,
            "gt_action": video_gt_action,
            "skip_reason": None,
            "track_id": best_track_id,
            "track_mean_iou": track_mean_iou,
            "n_gt_frames": n_gt_frames,
            "n_covered_frames": covered_frames,
            "coverage_ratio": coverage_ratio,
            "correct_frames": correct_frames,
            "frame_accuracy_covered": frame_acc_covered,
            "frame_accuracy_all": frame_acc_all,
            "video_pred_action": video_pred_action,
            "video_match": video_match,
        }
        video_results.append(result)

        cov_str = f"{frame_acc_covered:.3f}" if not np.isnan(frame_acc_covered) else "n/a"
        print(
            f"[{video_name}] track={best_track_id} (mIoU={track_mean_iou:.3f}) | "
            f"GT={video_gt_action} | Pred={video_pred_action} | "
            f"coverage={coverage_ratio:.2f} | acc(coperti)={cov_str} | "
            f"acc(tutti)={frame_acc_all:.3f}"
        )

    print_and_save_summary(video_results, skipped, all_y_true, all_y_pred)


def print_and_save_summary(video_results, skipped, all_y_true, all_y_pred):
    print("\n" + "=" * 60)
    print("SKIP RESUME")
    print("=" * 60)
    for reason, count in skipped.items():
        print(f"  {reason}: {count}")

    if not video_results:
        print("\nNo evaluated videos (no match between GT and predictions available).")
        return

    df_video = pd.DataFrame(video_results)
    df_video.to_csv(OUTPUT_CSV_PER_VIDEO, index=False)

    df_valid = df_video[df_video["skip_reason"].isna()].copy()

    print("\n" + "=" * 60)
    print("FINAL RESULTS ACTION RECOGNITION")
    print("=" * 60)
    print(f"Total videos in GT:                             {len(df_video)}")
    print(f"Videos actually evaluated (valid subset):   {len(df_valid)}")

    if not df_valid.empty:
        n_frames_total = df_valid["n_gt_frames"].clip(lower=1)

        print(f"Average coverage (macro, per video):               {df_valid['coverage_ratio'].mean():.4f}")
        print(f"Frame accuracy on covered frames (macro):        {df_valid['frame_accuracy_covered'].mean():.4f}")
        print(f"Frame accuracy on all GT frames (macro):      {df_valid['frame_accuracy_all'].mean():.4f}")

        micro_acc_all = float((df_valid["frame_accuracy_all"] * n_frames_total).sum() / n_frames_total.sum())
        print(f"Frame accuracy on all GT frames (micro):      {micro_acc_all:.4f}")

        print(f"Video-level accuracy (top-1):  {df_valid['video_match'].mean():.4f}")

    print("=" * 60)
    print(f"\nResults per-video saved in: {OUTPUT_CSV_PER_VIDEO}")

    if all_y_true:
        df_prf1 = compute_per_class_prf1(all_y_true, all_y_pred)
        df_prf1.to_csv(OUTPUT_CSV_PER_CLASS, index=False)
        print(f"Precision/Recall/F1 per class saved in: {OUTPUT_CSV_PER_CLASS}")
        print("\nPrecision / Recall / F1 per class (on all valid GT frames):")
        print(df_prf1.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "n/a"
        ))
    else:
        print("\nNo evaluable frames for Precision/Recall/F1 calculation.")


if __name__ == "__main__":
    main()