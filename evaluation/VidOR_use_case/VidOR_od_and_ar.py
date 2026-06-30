import os
import json
import glob
import re
from collections import defaultdict
from scenedetect import detect, ContentDetector
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE PATH ---
PATH_OD = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/video_general_obj_det_partial-owl"
PATH_ACTION = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/action_recognition_finished-example"
PATH_VIDEOS = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/Videos" # Ricerca ricorsiva abilitata
PATH_FRAMES = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/Videos_crop_decode"
PATH_GT = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/groundtruth/VidOR"

AZIONI_CONSENTITE = {
    "lift", "carry", "push", "pull", "pass", "lean_on", "touch", "hit", "play(instrument)", "grab",
    "release", "press", "use", "throw", "clean", "knock", "squeeze", "cut", "open", "close", "watch",
    "hold", "speak_to", "ride", "hug", "hold_hand_of", "bite", "caress", "pat", "wave", "point_to",
    "chase", "feed", "kiss", "kick", "smell", "wave_hand_to", "lick", "drive", "shout_at", "get_on", 
    "get_off", "shake_hand_with"
}

# Cache globale per non rieseguire scenedetect più volte sullo stesso video
SCENE_CACHE = {}

def load_groundtruth() -> dict:
    gt_data = {}
    gt_files = glob.glob(os.path.join(PATH_GT, "*", "*.json"))
    
    for f in gt_files:
        with open(f, 'r') as file:
            data = json.load(file)
            vid = data.get('video_id')
            if not vid:
                continue
                
            valid_persons = set()
            for obj in data.get('subject/objects', []):
                if obj['category'] in ['adult', 'child', 'baby']:
                    valid_persons.add(obj['tid'])
            
            actions = []
            for rel in data.get('relation_instances', []):
                if rel['predicate'] in AZIONI_CONSENTITE:
                    actions.append(rel)
                    
            gt_data[vid] = {
                'width': data.get('width', 1),
                'height': data.get('height', 1),
                'fps': data.get('fps', 30.0),
                'persons': valid_persons,
                'trajectories': data.get('trajectories', []),
                'actions': actions
            }
    return gt_data

def get_clip_offset(clip_name: str, video_id: str, gt_data: dict) -> int:
    match = re.search(r'scene(\d+)(?:_part(\d+))?', clip_name)
    if not match:
        return 0
        
    scene_idx = int(match.group(1))
    part_idx = int(match.group(2)) if match.group(2) else None
    
    if video_id not in SCENE_CACHE:
        video_path = None
        
        # Cerca ricorsivamente in tutte le sottocartelle di PATH_VIDEOS
        search_pattern = os.path.join(PATH_VIDEOS, "**", f"{video_id}.*")
        possible_files = glob.glob(search_pattern, recursive=True)
        
        for file in possible_files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = file
                break
                
        if video_path is None:
            print(f"ATTENZIONE: Video originale {video_id} non trovato in {PATH_VIDEOS} o nelle sue sottocartelle.")
            SCENE_CACHE[video_id] = []
        else:
            try:
                raw_scenes = detect(video_path, ContentDetector())
                if not raw_scenes:
                    SCENE_CACHE[video_id] = [(0.0, 0.0)]
                else:
                    SCENE_CACHE[video_id] = [(s[0].seconds, s[1].seconds) for s in raw_scenes]
            except Exception as e:
                print(f"Errore scenedetect su {video_id}: {e}")
                SCENE_CACHE[video_id] = [(0.0, 0.0)]

    scenes_sec = SCENE_CACHE.get(video_id, [])
    scene_index_zero_based = scene_idx - 1
    
    if scene_index_zero_based < len(scenes_sec):
        start_sec = scenes_sec[scene_index_zero_based][0]
    else:
        start_sec = 0.0
        
    if part_idx is not None:
        start_sec += (part_idx - 1) * 10.0
        
    fps = gt_data[video_id]["fps"]
    offset_frames = int(round(start_sec * fps))
    
    return offset_frames

def calculate_iou(boxA: list, boxB: list) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def main():
    gt_data = load_groundtruth()
    od_folders = glob.glob(os.path.join(PATH_OD, "*"))
    
    metrics = {
        "od_total_iou": 0.0,
        "od_matched_tracks": 0,
        "matched_ious": [], 
        "action_tp": 0,
        "action_fp": 0,
        "action_fn": 0
    }
    
    for folder in od_folders:
        clip_name = os.path.basename(folder)
        od_file = os.path.join(folder, f"{clip_name}.json")
        
        if not os.path.exists(od_file):
            continue
            
        video_id = clip_name.split("_")[0]
        if video_id not in gt_data:
            continue
            
        clip_offset = get_clip_offset(clip_name, video_id, gt_data)
        gt_vid = gt_data[video_id]
        
        with open(od_file, 'r') as f:
            od_res = json.load(f)
            
        detected_objects = od_res.get("detected_objects", {})
        
        for obj_key, obj_data in detected_objects.items():
            if obj_data.get("class_name") != "person":
                continue
                
            pred_bboxes = obj_data.get("bbox", [])
            
            best_gt_tid = None
            best_iou = 0.0
            
            for gt_tid in gt_vid["persons"]:
                total_iou = 0.0
                valid_frames = 0
                
                for frame_idx, pred_box in enumerate(pred_bboxes):
                    if not pred_box:
                        continue

                    gt_frame_idx = frame_idx + clip_offset
                    if gt_frame_idx >= len(gt_vid["trajectories"]):
                        continue
                    
                    gt_frame_data = gt_vid["trajectories"][gt_frame_idx]
                    gt_box_data = next((item["bbox"] for item in gt_frame_data if item["tid"] == gt_tid), None)
                    
                    if gt_box_data:
                        p_xmin = pred_box[0] * gt_vid["width"]
                        p_ymin = pred_box[1] * gt_vid["height"]
                        p_xmax = pred_box[2] * gt_vid["width"]
                        p_ymax = pred_box[3] * gt_vid["height"]
                        
                        g_box = [gt_box_data["xmin"], gt_box_data["ymin"], gt_box_data["xmax"], gt_box_data["ymax"]]
                        total_iou += calculate_iou([p_xmin, p_ymin, p_xmax, p_ymax], g_box)
                        valid_frames += 1
                
                avg_iou = total_iou / valid_frames if valid_frames > 0 else 0
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    best_gt_tid = gt_tid
            
            if best_gt_tid is not None and best_iou > 0.2:
                metrics["od_total_iou"] += best_iou
                metrics["od_matched_tracks"] += 1
                metrics["matched_ious"].append(best_iou)
                
                action_file = os.path.join(PATH_ACTION, f"{clip_name}_{obj_key}_actions.json")
                
                if os.path.exists(action_file):
                    with open(action_file, 'r') as af:
                        action_res = json.load(af)
                        
                    for frame_window, pred_actions in action_res.items():
                        m = re.match(r"frames_(\d+)_to_(\d+)", frame_window)
                        if not m:
                            continue
                            
                        start_f = int(m.group(1)) + clip_offset
                        end_f = int(m.group(2)) + clip_offset
                        
                        gt_actions_in_window = set()
                        for rel in gt_vid["actions"]:
                            if rel["subject_tid"] == best_gt_tid:
                                if max(start_f, rel["begin_fid"]) < min(end_f, rel["end_fid"]):
                                    gt_actions_in_window.add(rel["predicate"])
                                    
                        pred_actions_set = set(pred_actions)
                        metrics["action_tp"] += len(pred_actions_set.intersection(gt_actions_in_window))
                        metrics["action_fp"] += len(pred_actions_set - gt_actions_in_window)
                        metrics["action_fn"] += len(gt_actions_in_window - pred_actions_set)

    print("\n" + "="*30)
    print(" RISULTATI VALIDAZIONE PIPELINE")
    print("="*30)
    
    if metrics["od_matched_tracks"] > 0:
        mean_iou = metrics['od_total_iou'] / metrics['od_matched_tracks']
        print(f"OD/Tracking Mean IoU: {mean_iou:.4f} (over {metrics['od_matched_tracks']} tracks)")
    else:
        print("Nessun match di Tracking trovato o risultati OD assenti.")
        mean_iou = 0.0
        
    tp = metrics["action_tp"]
    fp = metrics["action_fp"]
    fn = metrics["action_fn"]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("-" * 30)
    print(f"Action Labeling Precision: {precision:.4f}")
    print(f"Action Labeling Recall:    {recall:.4f}")
    print(f"Action Labeling F1-Score:  {f1:.4f}")
    print("="*30)

    if metrics["matched_ious"]:
        plt.figure(figsize=(10, 6))
        plt.hist(metrics["matched_ious"], bins=20, color='#4CAF50', edgecolor='black', range=(0.2, 1.0))
        plt.title('Mean IoU distribution for matched tracks (OD & Tracking)', fontsize=14, fontweight='bold')
        plt.xlabel('Mean IoU', fontsize=12)
        plt.ylabel('Frequency (Number of tracks)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.axvline(mean_iou, color='#F44336', linestyle='dashed', linewidth=2.5, 
                    label=f'Media: {mean_iou:.3f}')
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plot_filename = 'od_tracking_iou_histogram.png'
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        
        print(f"\n=> Grafico dell'Object Detection generato e salvato in: {os.path.abspath(plot_filename)}")
    else:
        print("\n=> Dati insufficienti per generare il grafico dell'Object Detection.")

if __name__ == "__main__":
    main()