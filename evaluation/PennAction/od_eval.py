import os
import json
import glob
import numpy as np
import pandas as pd
from PIL import Image

# PATH CONFIG
DATA_ROOT = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT"
DET_TRACK_DIR = os.path.join(DATA_ROOT, "video_general_obj_det_finished")
GT_DIR = os.path.join(DATA_ROOT, "groundtruth/PennAction")
DECODE_DIR = os.path.join(DATA_ROOT, "Videos_crop_decode")

OUTPUT_DIR = "/home/ludovico/workspace/AA-STAL/data_pipeline/evaluation/PennAction/od_tracking_validation"
OUTPUT_CSV_PER_VIDEO = os.path.join(OUTPUT_DIR, "od_tracking_validation_per_video.csv")
OUTPUT_CSV_PER_ACTION = os.path.join(OUTPUT_DIR, "od_tracking_validation_per_action.csv")

# Min IoU to consider a detection as "matching" the GT box
IOU_MATCH_THRESHOLD_FOR_ID = 0.1

# Accuracy thresholds (similar to COCO AP50 / AP75)
ACC_THRESHOLDS = [0.5, 0.75]

REQUIRED_GT_COLS = ["frame_id", "x_min", "y_min", "x_max", "y_max"]


# UTILITY FUNCTIONS
def compute_iou_batch(boxes_a, boxes_b):
    """IoU between two arrays Nx4 [xmin, ymin, xmax, ymax]."""
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
    """Boolean mask (N,): True if the box is geometrically valid."""
    return (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])


def box_centers(boxes):
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    return np.stack([cx, cy], axis=1)


def box_diagonal(boxes):
    return np.sqrt((boxes[:, 2] - boxes[:, 0]) ** 2 + (boxes[:, 3] - boxes[:, 1]) ** 2)


def get_frame_dimensions(video_name):
    video_frame_dir = os.path.join(DECODE_DIR, video_name)
    if not os.path.isdir(video_frame_dir):
        return None, None
    frames = sorted(glob.glob(os.path.join(video_frame_dir, "*")))
    if not frames:
        return None, None
    try:
        with Image.open(frames[0]) as img:
            return img.size  # (width, height) - non decodifica i pixel
    except Exception:
        return None, None


def evaluate_track_against_gt(pred_bboxes, gt_frame_idx, gt_boxes, width, height):
    """
    Evaluate a single track against all frames requested by the GT.

    pred_bboxes: list of normalized bbox [x0,y0,x1,y1] or None, one entry per
                 frame of the video. The tracker saves None in frames where the
                 track is lost (occlusion, exit from field of view, etc.).
    gt_frame_idx: array (N,) of 0-based frame indices requested by the GT
    gt_boxes: array (N,4) di bbox GT in pixel

    Returns:
      iou: array (N,) - 0.0 where detection is missing
      detected_mask: array (N,) bool - True if the track has a valid detection for that GT frame
      pred_boxes_px: array (N,4) - predicted bbox in pixel (0 where not present)
    """
    n = len(gt_frame_idx)
    pred_len = len(pred_bboxes)

    detected_mask = np.zeros(n, dtype=bool)
    pred_boxes_px = np.zeros((n, 4), dtype=np.float64)

    within_range = gt_frame_idx < pred_len
    for pos in np.where(within_range)[0]:
        frame_i = gt_frame_idx[pos]
        box = pred_bboxes[frame_i]

        # The tracker saves None when it loses the person in that frame
        if box is None:
            continue
        if len(box) != 4:
            continue

        box_px = np.asarray(box, dtype=np.float64)
        box_px[[0, 2]] *= width
        box_px[[1, 3]] *= height
        pred_boxes_px[pos] = box_px
        detected_mask[pos] = True

    # Discard not valid bbox (xmax<=xmin o ymax<=ymin)
    if detected_mask.any():
        valid_box_mask = is_valid_box_array(pred_boxes_px)
        detected_mask = detected_mask & valid_box_mask

    iou = np.zeros(n, dtype=np.float64)
    if detected_mask.any():
        iou[detected_mask] = compute_iou_batch(gt_boxes[detected_mask], pred_boxes_px[detected_mask])

    return iou, detected_mask, pred_boxes_px


def count_id_switches(per_track_results, n_frames, iou_thr):
    """
    Determines for every frame which track has the
    highest IoU with the GT, then counts how many times this identity changes between
    consecutive frames (ignoring frames without any match above the threshold).
    """
    track_ids = list(per_track_results.keys())
    iou_matrix = np.stack([per_track_results[tid]["iou"] for tid in track_ids], axis=0)

    best_per_frame = np.argmax(iou_matrix, axis=0)
    best_iou_per_frame = np.max(iou_matrix, axis=0)

    active_id_sequence = [
        track_ids[best_per_frame[f]] if best_iou_per_frame[f] >= iou_thr else None
        for f in range(n_frames)
    ]

    switches = 0
    prev_id = None
    for current_id in active_id_sequence:
        if current_id is None:
            continue
        if prev_id is not None and current_id != prev_id:
            switches += 1
        prev_id = current_id

    return switches


def empty_video_result(video_name, n_gt_frames, action):
    result = {
        "video": video_name,
        "action": action,
        "best_track_id": None,
        "n_person_candidates": 0,
        "n_gt_frames": n_gt_frames,
        "coverage_ratio": 0.0,
        "mean_iou": 0.0,
        "mean_iou_when_present": 0.0,
        "mean_center_location_error_norm": np.nan,
        "n_id_switches": 0,
    }
    for thr in ACC_THRESHOLDS:
        result[f"accuracy@{thr}"] = 0.0
    return result


# MAIN
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.csv")))

    video_metrics = []
    skipped = {"no_prediction": 0, "no_frame_dims": 0, "invalid_gt": 0, "no_person_tracks": 0}

    print(f"Found {len(gt_files)} ground truth files. Starting subset validation...")

    for gt_path in gt_files:
        video_name = os.path.splitext(os.path.basename(gt_path))[0]
        json_path = os.path.join(DET_TRACK_DIR, video_name, f"{video_name}.json")

        # Skip if there is no prediction for this video 
        if not os.path.exists(json_path):
            skipped["no_prediction"] += 1
            continue

        width, height = get_frame_dimensions(video_name)
        if width is None or height is None:
            print(f"[{video_name}] Frame dimensions could not be retrieved. Skipping.")
            skipped["no_frame_dims"] += 1
            continue

        df_gt = pd.read_csv(gt_path)
        missing_cols = [c for c in REQUIRED_GT_COLS if c not in df_gt.columns]
        if missing_cols:
            print(f"[{video_name}] CSV GT missing columns {missing_cols}. Skipping.")
            skipped["invalid_gt"] += 1
            continue

        df_gt = df_gt.sort_values("frame_id").reset_index(drop=True)

        # Discard geometrically invalid rows
        gt_boxes_raw = df_gt[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=np.float64)
        valid_gt_mask = is_valid_box_array(gt_boxes_raw)
        n_invalid_gt = int((~valid_gt_mask).sum())
        if n_invalid_gt > 0:
            print(f"[{video_name}] {n_invalid_gt}  GT rows not valid, excluded from calculation.")
        df_gt = df_gt[valid_gt_mask].reset_index(drop=True)

        if df_gt.empty:
            print(f"[{video_name}] No valid GT rows after filtering. Skipping.")
            skipped["invalid_gt"] += 1
            continue

        gt_frame_idx = df_gt["frame_id"].to_numpy(dtype=int) - 1 
        gt_boxes = df_gt[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=np.float64)

        video_action = None
        if "action" in df_gt.columns and not df_gt["action"].empty:
            video_action = df_gt["action"].mode().iat[0]

        with open(json_path, "r") as f:
            pred_data = json.load(f)

        detected_objects = pred_data.get("detected_objects", {})
        person_tracks = {
            tid: info for tid, info in detected_objects.items()
            if info.get("class_name") == "person"
        }

        if not person_tracks:
            print(f"[{video_name}] No person detected by the model.")
            video_metrics.append(empty_video_result(video_name, len(gt_frame_idx), video_action))
            skipped["no_person_tracks"] += 1
            continue

        # Evaluate EVERY track "person" for the entire video
        per_track_results = {}
        for track_id, track_info in person_tracks.items():
            pred_bboxes = track_info.get("bbox", [])
            iou, detected_mask, pred_boxes_px = evaluate_track_against_gt(
                pred_bboxes, gt_frame_idx, gt_boxes, width, height
            )
            per_track_results[track_id] = {
                "iou": iou,
                "detected_mask": detected_mask,
                "pred_boxes_px": pred_boxes_px,
            }

        # Best track selection: choose the track with the highest mean IoU across all GT frames
        best_track_id = max(per_track_results, key=lambda tid: per_track_results[tid]["iou"].mean())
        best = per_track_results[best_track_id]

        n_frames = len(gt_frame_idx)
        coverage_ratio = float(best["detected_mask"].mean())
        mean_iou = float(best["iou"].mean())
        mean_iou_when_present = (
            float(best["iou"][best["detected_mask"]].mean()) if best["detected_mask"].any() else 0.0
        )

        acc_at = {f"accuracy@{thr}": float(np.mean(best["iou"] >= thr)) for thr in ACC_THRESHOLDS}

        # Center Location Error
        mean_cle = np.nan
        if best["detected_mask"].any():
            idx = np.where(best["detected_mask"])[0]
            gt_c = box_centers(gt_boxes[idx])
            pred_c = box_centers(best["pred_boxes_px"][idx])
            diag = box_diagonal(gt_boxes[idx])
            valid_diag = diag > 0
            if valid_diag.any():
                dist = np.linalg.norm(gt_c[valid_diag] - pred_c[valid_diag], axis=1)
                mean_cle = float(np.mean(dist / diag[valid_diag]))

        n_id_switches = count_id_switches(per_track_results, n_frames, IOU_MATCH_THRESHOLD_FOR_ID)

        result = {
            "video": video_name,
            "action": video_action,
            "best_track_id": best_track_id,
            "n_person_candidates": len(person_tracks),
            "n_gt_frames": n_frames,
            "coverage_ratio": coverage_ratio,
            "mean_iou": mean_iou,
            "mean_iou_when_present": mean_iou_when_present,
            "mean_center_location_error_norm": mean_cle,
            "n_id_switches": n_id_switches,
            **acc_at,
        }
        video_metrics.append(result)

        print(
            f"[{video_name}] track={best_track_id} | coverage={coverage_ratio:.2f} | "
            f"mIoU={mean_iou:.3f} | mIoU(presenti)={mean_iou_when_present:.3f} | "
            f"acc@0.5={acc_at['accuracy@0.5']:.3f} | id_switches={n_id_switches}"
        )

    print_and_save_summary(video_metrics, skipped)


def print_and_save_summary(video_metrics, skipped):
    print("\n" + "=" * 60)
    print("SKIP RESUME")
    print("=" * 60)
    for reason, count in skipped.items():
        print(f"  {reason}: {count}")

    if not video_metrics:
        print("\nNo evaluated videos (no match between GT and predictions available).")
        return

    df_metrics = pd.DataFrame(video_metrics)
    df_metrics.to_csv(OUTPUT_CSV_PER_VIDEO, index=False)

    n_frames_total = df_metrics["n_gt_frames"].clip(lower=1)  # evita divisioni per 0

    print("\n" + "=" * 60)
    print("FINAL RESULTS VALIDATION OBJECT DETECTION & TRACKING")
    print("=" * 60)
    print(f"Total videos evaluated (subset): {len(df_metrics)}")
    print(f"Average coverage (presence detection):        {df_metrics['coverage_ratio'].mean():.4f}")
    print(f"Mean IoU - macro avg per video (absences=0): {df_metrics['mean_iou'].mean():.4f}")

    micro_avg_iou = float((df_metrics["mean_iou"] * n_frames_total).sum() / n_frames_total.sum())
    print(f"Mean IoU - micro avg weighted for n. frame:   {micro_avg_iou:.4f}")

    print(f"Mean IoU only on present frames:           {df_metrics['mean_iou_when_present'].mean():.4f}")
    for thr in ACC_THRESHOLDS:
        print(f"Accuracy @IoU={thr}:                          {df_metrics[f'accuracy@{thr}'].mean():.4f}")
    print(f"Mean Center Location Error: {df_metrics['mean_center_location_error_norm'].mean():.4f}")
    print(f"Total ID switches:                          {int(df_metrics['n_id_switches'].sum())}")
    print(f"Average ID switches per video:                  {df_metrics['n_id_switches'].mean():.4f}")
    print("=" * 60)
    print(f"\nResults per-video saved in: {OUTPUT_CSV_PER_VIDEO}")

    # Breakdown for every action
    if "action" in df_metrics.columns and df_metrics["action"].notna().any():
        per_action = (
            df_metrics.dropna(subset=["action"])
            .groupby("action")[["coverage_ratio", "mean_iou", "mean_iou_when_present", "n_id_switches"]]
            .mean()
            .sort_values("mean_iou")
        )
        per_action.to_csv(OUTPUT_CSV_PER_ACTION)
        print(f"Breakdown per action saved in:      {OUTPUT_CSV_PER_ACTION}")
        print("\nMean IoU per action (ordered from worst):")
        print(per_action["mean_iou"].to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()