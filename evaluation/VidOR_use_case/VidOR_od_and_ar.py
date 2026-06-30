import os
import cv2
import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAZIONE PERCORSI
# ==========================================
# AGGIORNATO CON IL NUOVO PERCORSO LINUX
GT_BASE_PATH = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/groundtruth/VidOR"
VIDEO_ORIGINAL_PATH = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/Videos_crop"
OD_RESULTS_PATH = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/video_general_obj_det_partial-owl"
AR_RESULTS_PATH = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/action_recognition_finished-example"

AZIONI_CONSENTITE = [
    "lift", "carry", "push", "pull", "pass", "lean_on", "touch", "hit", "play(instrument)", "grab",
    "release", "press", "use", "throw", "clean", "knock", "squeeze", "cut", "open", "close", "watch",
    "hold", "speak_to", "ride", "hug", "hold_hand_of", "hit", "bite", "caress", "pat", "wave", "point_to",
    "chase", "feed", "kiss", "kick", "smell", "wave_hand_to", "lick", "drive", "shout_at", "get_on", "get_off",
    "shake_hand_with"
]


IOU_MATCH_THRESHOLD = 0.3

# ==========================================
# 2. UTILITY E MATEMATICA
# ==========================================
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def norm_act(act_string):
    """Normalizza le label per evitare mismatch testuali (es. 'lean_on' -> 'lean on')."""
    return str(act_string).lower().replace('_', ' ').strip()

VALID_ACTIONS = {norm_act(a) for a in AZIONI_CONSENTITE}

def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection_area / float(box1_area + box2_area - intersection_area)

def compute_center_error(box1, box2):
    center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
    center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
    return np.linalg.norm(center1 - center2)

# ==========================================
# 3. LETTURA DATI
# ==========================================
def load_vidor_gt(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frame_count = data.get("frame_count", 0)
    if frame_count == 0 and "trajectories" in data:
        frame_count = len(data["trajectories"])
        
    gt_frames = [{} for _ in range(frame_count)]

    # 1. Estrazione Bounding Box
    if "trajectories" in data:
        for f_idx, frame_objs in enumerate(data.get("trajectories", [])):
            if f_idx >= frame_count:
                break
            for obj in frame_objs:
                tid = obj.get("tid")
                bbox = obj.get("bbox")
                
                if tid is not None:
                    if isinstance(bbox, dict):
                        xmin, ymin = bbox.get("xmin"), bbox.get("ymin")
                        xmax, ymax = bbox.get("xmax"), bbox.get("ymax")
                        if None not in (xmin, ymin, xmax, ymax):
                            gt_frames[f_idx][tid] = {
                                "bbox": [xmin, ymin, xmax, ymax],
                                "actions": set()
                            }
                    elif isinstance(bbox, list) and len(bbox) == 4:
                        x, y, w, h = bbox
                        gt_frames[f_idx][tid] = {
                            "bbox": [x, y, x + w, y + h],
                            "actions": set()
                        }

    # 2. Estrazione Azioni (con FILTRO AZIONI_CONSENTITE)
    relations = data.get("relation_instances", [])
    actions = data.get("action_instances", [])
    
    for action_inst in (relations + actions):
        if isinstance(action_inst, dict):
            act_class = action_inst.get("predicate") or action_inst.get("action_class") or action_inst.get("category")
            subj_tid = action_inst.get("subject_tid")
            if subj_tid is None:
                subj_tid = action_inst.get("tid")
                
            start_f = action_inst.get("begin_fid")
            end_f = action_inst.get("end_fid")
            
            if act_class is not None and subj_tid is not None and start_f is not None and end_f is not None:
                act_class_norm = norm_act(act_class)
                # FILTRO: Aggiungi solo se l'azione appartiene alla lista consentita
                if act_class_norm in VALID_ACTIONS:
                    for f_idx in range(start_f, end_f):
                        if f_idx < frame_count and subj_tid in gt_frames[f_idx]:
                            gt_frames[f_idx][subj_tid]["actions"].add(act_class_norm)
                            
        elif isinstance(action_inst, list) and len(action_inst) >= 4:
            act_class_norm = norm_act(action_inst[0])
            # FILTRO: Aggiungi solo se l'azione appartiene alla lista consentita
            if act_class_norm in VALID_ACTIONS:
                subj_tid = action_inst[1]
                
                if len(action_inst) == 5:
                    start_f, end_f = action_inst[3], action_inst[4]
                else:
                    start_f, end_f = action_inst[-2], action_inst[-1]
                
                for f_idx in range(start_f, end_f):
                    if f_idx < frame_count and subj_tid in gt_frames[f_idx]:
                        gt_frames[f_idx][subj_tid]["actions"].add(act_class_norm)

    return gt_frames


# ==========================================
# 3. LETTURA DATI
# ==========================================
def load_vidor_gt(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frame_count = data.get("frame_count", 0)
    if frame_count == 0 and "trajectories" in data:
        frame_count = len(data["trajectories"])
        
    gt_frames = [{} for _ in range(frame_count)]

    # 1. Estrazione Bounding Box (NESSUN FILTRO: carica tutti gli oggetti, umani e non)
    if "trajectories" in data:
        for f_idx, frame_objs in enumerate(data.get("trajectories", [])):
            if f_idx >= frame_count:
                break
            for obj in frame_objs:
                tid = obj.get("tid")
                bbox = obj.get("bbox")
                
                if tid is not None:
                    if isinstance(bbox, dict):
                        xmin, ymin = bbox.get("xmin"), bbox.get("ymin")
                        xmax, ymax = bbox.get("xmax"), bbox.get("ymax")
                        if None not in (xmin, ymin, xmax, ymax):
                            gt_frames[f_idx][tid] = {
                                "bbox": [xmin, ymin, xmax, ymax],
                                "actions": set()
                            }
                    elif isinstance(bbox, list) and len(bbox) == 4:
                        x, y, w, h = bbox
                        gt_frames[f_idx][tid] = {
                            "bbox": [x, y, x + w, y + h],
                            "actions": set()
                        }

    # 2. Estrazione Azioni (Mantiene il filtro AZIONI_CONSENTITE)
    relations = data.get("relation_instances", [])
    actions = data.get("action_instances", [])
    
    for action_inst in (relations + actions):
        if isinstance(action_inst, dict):
            act_class = action_inst.get("predicate") or action_inst.get("action_class") or action_inst.get("category")
            subj_tid = action_inst.get("subject_tid")
            if subj_tid is None:
                subj_tid = action_inst.get("tid")
                
            start_f = action_inst.get("begin_fid")
            end_f = action_inst.get("end_fid")
            
            if act_class is not None and subj_tid is not None and start_f is not None and end_f is not None:
                act_class_norm = norm_act(act_class)
                if act_class_norm in VALID_ACTIONS:
                    for f_idx in range(start_f, end_f):
                        if f_idx < frame_count and subj_tid in gt_frames[f_idx]:
                            gt_frames[f_idx][subj_tid]["actions"].add(act_class_norm)
                            
        elif isinstance(action_inst, list) and len(action_inst) >= 4:
            act_class_norm = norm_act(action_inst[0])
            if act_class_norm in VALID_ACTIONS:
                subj_tid = action_inst[1]
                
                if len(action_inst) == 5:
                    start_f, end_f = action_inst[3], action_inst[4]
                else:
                    start_f, end_f = action_inst[-2], action_inst[-1]
                
                for f_idx in range(start_f, end_f):
                    if f_idx < frame_count and subj_tid in gt_frames[f_idx]:
                        gt_frames[f_idx][subj_tid]["actions"].add(act_class_norm)

    return gt_frames

def build_scene_predictions(od_json_path, action_jsons, video_w, video_h):
    with open(od_json_path, 'r') as f:
        data = json.load(f)
        
    detected_objects = data.get("detected_objects", {})
    
    num_frames = 0
    for obj_val in detected_objects.values():
        num_frames = max(num_frames, len(obj_val.get("bbox", [])))
        
    has_ar = len(action_jsons) > 0
    scene_preds = [{"has_ar": has_ar, "tracks": {}} for _ in range(num_frames)]
    
    track_actions_map = {}
    for a_json in action_jsons:
        match = re.search(r'(person_\d+)_actions', a_json)
        # Se il tuo AR processa anche altri oggetti, puoi generalizzare la regex qui. 
        # Attualmente legge i JSON che contengono "_actions"
        if not match:
            # Fallback se il nome file non ha "person_X" ma magari "dog_X_actions"
            match = re.search(r'([a-zA-Z0-9_]+)_actions', os.path.basename(a_json))
            
        if match:
            track_id = match.group(1)
            track_actions_map[track_id] = load_pred_actions(a_json)

    for obj_key, obj_val in detected_objects.items():
        # RIMOSSO IL FILTRO "PERSON". Lasciamo passare tutti gli oggetti!
        
        bboxes = obj_val.get("bbox", [])
        track_id = obj_key 
        
        for i, bbox in enumerate(bboxes):
            if bbox is not None:
                x1 = bbox[0] * video_w
                y1 = bbox[1] * video_h
                x2 = bbox[2] * video_w
                y2 = bbox[3] * video_h
                
                actions = None
                if track_id in track_actions_map and i in track_actions_map[track_id]:
                    actions = track_actions_map[track_id][i]
                elif track_id in track_actions_map:
                    actions = set()
                
                scene_preds[i]["tracks"][track_id] = {
                    "bbox": [x1, y1, x2, y2],
                    "actions": actions
                }
                
    return scene_preds

# ==========================================
# 4. VALUTAZIONE E METRICHE
# ==========================================
def evaluate_pipeline(gt_frames, pred_frames):
    ious = []
    center_errors = []
    
    ar_hits = 0
    ar_misses = 0
    
    num_eval_frames = min(len(gt_frames), len(pred_frames))
    
    for i in range(num_eval_frames):
        gt_frame = gt_frames[i]
        pred_frame_wrapper = pred_frames[i]
        
        if pred_frame_wrapper is None: 
            continue 
            
        has_ar_for_scene = pred_frame_wrapper["has_ar"]
        pred_tracks = pred_frame_wrapper["tracks"]
        
        for gt_tid, gt_data in gt_frame.items():
            gt_box = gt_data["bbox"]
            gt_actions = gt_data["actions"]
            
            best_iou = 0.0
            best_ce = float('inf')
            best_pred_tid = None
            
            for pred_tid, pred_data in pred_tracks.items():
                iou = compute_iou(gt_box, pred_data["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_ce = compute_center_error(gt_box, pred_data["bbox"])
                    best_pred_tid = pred_tid
            
            ious.append(best_iou)
            center_errors.append(best_ce)
            
            # VALUTAZIONE ACTION RECOGNITION
            if len(gt_actions) > 0:
                if best_pred_tid is not None and best_iou >= IOU_MATCH_THRESHOLD:
                    pred_actions = pred_tracks[best_pred_tid]["actions"]
                    
                    if pred_actions is not None:
                        if len(gt_actions.intersection(pred_actions)) > 0:
                            ar_hits += 1
                        else:
                            ar_misses += 1
                            print(f"[MISMATCH] Frame {i} | GT voleva: {gt_actions} | Modello ha predetto: {pred_actions}")
                else:
                    if has_ar_for_scene:
                        ar_misses += 1
                
    return ious, center_errors, ar_hits, ar_misses

def calculate_metrics(ious, center_errors, ar_hits, ar_misses):
    ious_arr = np.array(ious)
    ce_arr = np.array(center_errors)
    
    iou_thresholds = np.arange(0, 1.05, 0.05)
    cle_thresholds = np.arange(0, 51, 1)
    
    success_rates = [np.mean(ious_arr >= t) for t in iou_thresholds]
    precision_rates = [np.mean(ce_arr <= t) for t in cle_thresholds]
    
    auc_success = np.trapezoid(success_rates, dx=0.05)
    precision_at_20 = precision_rates[20] if len(precision_rates) > 20 else 0.0
    
    total_evaluated_actions = ar_hits + ar_misses
    action_accuracy = (ar_hits / total_evaluated_actions) if total_evaluated_actions > 0 else 0.0

    print("-" * 45)
    print(f"RISULTATI FINALI (Basati su {len(ious_arr)} target GT validi)")
    print("-" * 45)
    print("OBJECT DETECTION E TRACKING:")
    print(f"  Success Rate (AUC):          {auc_success:.4f}")
    print(f"  Precision Rate (CLE < 20px): {precision_at_20:.4f}")
    print("-" * 45)
    print("ACTION RECOGNITION (Top-N Hit Rate, IoU >= 0.3):")
    print(f"  Frame Valutati: {total_evaluated_actions}")
    print(f"  Hits (Almeno 1 corretta): {ar_hits}")
    print(f"  Accuracy / Hit Rate:      {action_accuracy:.4f} ({(action_accuracy*100):.2f}%)")
    print("-" * 45)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(iou_thresholds, success_rates, marker='o', linewidth=2)
    plt.title("Success Plot")
    plt.xlabel("IoU Threshold")
    plt.ylabel("Success Rate")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(cle_thresholds, precision_rates, marker='o', linewidth=2)
    plt.title("Precision Plot")
    plt.xlabel("Location Error Threshold (pixels)")
    plt.ylabel("Precision Rate")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("evaluation_global_results.png")
    print("Grafico salvato come 'evaluation_global_results.png'.")

# ==========================================
# 5. PIPELINE PRINCIPALE
# ==========================================
def run_pipeline():
    all_ious = []
    all_ces = []
    tot_hits, tot_misses = 0, 0
    
    print("\n--- INIZIO PIPELINE ---")
    jsons = glob.glob(os.path.join(GT_BASE_PATH, "**", "*.json"), recursive=True)
    print(f"Trovati {len(jsons)} file JSON GroundTruth in {GT_BASE_PATH}")
    
    if not jsons:
        print(f"[ERRORE CRITICO] Cartella vuota o path errato: {GT_BASE_PATH}")
        return
        
    jsons.sort(key=lambda p: natural_keys(os.path.basename(p)))
    
    for json_path in jsons:
        video_id = os.path.splitext(os.path.basename(json_path))[0]
        
        scene_pattern = os.path.join(VIDEO_ORIGINAL_PATH, f"{video_id}_scene*.mp4")
        scene_videos = glob.glob(scene_pattern)
        scene_videos.sort(key=natural_keys)
        
        if not scene_videos:
            # Silenzioso qui: significa semplicemente che questo video di VidOR non lo hai ancora scaricato/processato
            continue
            
        # GUARDIA NATIVA OPENCV CON FORZATURA FFMPEG
        if not os.path.exists(scene_videos[0]):
            continue
            
        cap = cv2.VideoCapture(scene_videos[0], cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"  [!] Video originale illeggibile (Codec/Corrotto): {scene_videos[0]}")
            continue
            
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        master_preds = []
        elaborato_qualcosa = False
        
        for video_path in scene_videos:
            s_name = os.path.splitext(os.path.basename(video_path))[0]
            
            if not os.path.exists(video_path):
                continue
                
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"  [!] Scena illeggibile: {s_name}")
                continue
            num_frames_in_scene = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            scene_preds = [None] * num_frames_in_scene
            
            # Ricerca RICORSIVA del file OD (ovunque in OD_RESULTS_PATH)
            od_candidates = glob.glob(os.path.join(OD_RESULTS_PATH, "**", f"{s_name}.json"), recursive=True)
            
            if od_candidates:
                single_json_path = od_candidates[0] # Prende il primo trovato
                action_jsons = glob.glob(os.path.join(AR_RESULTS_PATH, "**", f"{s_name}*_actions.json"), recursive=True)
                has_ar = len(action_jsons) > 0
                
                for k in range(num_frames_in_scene):
                    scene_preds[k] = {"has_ar": has_ar, "tracks": {}}
                    
                frames_data = build_scene_predictions(single_json_path, action_jsons, video_w, video_h)
                limit = min(len(frames_data), num_frames_in_scene)
                for k in range(limit):
                    scene_preds[k]["tracks"] = frames_data[k]["tracks"]
                    
                elaborato_qualcosa = True
                print(f"  [+] Elaborata Scena: {s_name} | AR files: {len(action_jsons)}")
                
            else:
                # Se non c'è il json singolo, cerca le parti (es. _part1, _part2)
                part_pattern = os.path.join(OD_RESULTS_PATH, "**", f"{s_name}_part*.json")
                # Filtra per evitare di prendere per sbaglio i file _actions se le cartelle sono mischiate
                part_jsons = [p for p in glob.glob(part_pattern, recursive=True) if "_actions" not in p]
                
                if part_jsons:
                    elaborato_qualcosa = True
                    print(f"  [+] Elaborate {len(part_jsons)} parti per la scena: {s_name}")
                    
                    for p_json in part_jsons:
                        base = os.path.splitext(os.path.basename(p_json))[0]
                        match = re.search(r'_part(\d+)', base)
                        if match:
                            z = int(match.group(1))
                            offset = (z - 1) * 300 
                            
                            action_jsons = glob.glob(os.path.join(AR_RESULTS_PATH, "**", f"{base}*_actions.json"), recursive=True)
                            has_ar = len(action_jsons) > 0
                            
                            for k in range(offset, min(offset + 300, num_frames_in_scene)):
                                scene_preds[k] = {"has_ar": has_ar, "tracks": {}}
                                
                            frames_data = build_scene_predictions(p_json, action_jsons, video_w, video_h)
                            limit = min(len(frames_data), num_frames_in_scene - offset)
                            for k in range(limit):
                                scene_preds[offset + k]["tracks"] = frames_data[k]["tracks"]
                else:
                    print(f"  [-] Nessun risultato OD trovato per la scena: {s_name}")
                            
            master_preds.extend(scene_preds)
        
        if elaborato_qualcosa:
            gt_frames = load_vidor_gt(json_path)
            ious, ces, hits, misses = evaluate_pipeline(gt_frames, master_preds)
            
            all_ious.extend(ious)
            all_ces.extend(ces)
            tot_hits += hits
            tot_misses += misses
        
    if all_ious:
        calculate_metrics(all_ious, all_ces, tot_hits, tot_misses)
    else:
        print("\nNessun dato valido processato.")

if __name__ == "__main__":
    run_pipeline()