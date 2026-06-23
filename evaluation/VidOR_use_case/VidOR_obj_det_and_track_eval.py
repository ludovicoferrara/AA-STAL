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
VIDOR_BASE_PATH = "/mnt/c/Users/ludov/Projects/creazione-sdrogo-dataset-AA-STAL/VidOR"
VIDEO_ORIGINAL_PATH = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/Videos_crop"
OD_RESULTS_PATH = "/mnt/c/Users/ludov/UNIVERSITA/SECONDO ANNO/TESI/Risultati/OD Multipla/video_general_obj_det_partial-dino"

# ==========================================
# 2. UTILITY DI ORDINAMENTO E MATEMATICA
# ==========================================
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """Permette di ordinare correttamente i file: scene2 viene prima di scene10."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def compute_iou(box1, box2):
    """Calcola l'Intersection over Union (IoU) tra due box [x1, y1, x2, y2]."""
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
    """Calcola la distanza euclidea tra i centri di due box."""
    center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
    center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
    return np.linalg.norm(center1 - center2)

# ==========================================
# 3. LETTURA DATI
# ==========================================
def load_vidor_gt(json_path):
    """Carica e formatta la groundtruth di VidOR dal JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    gt_boxes = []
    if "trajectories" in data:
        for frame_objs in data.get("trajectories", []):
            frame_boxes = []
            for obj in frame_objs:
                bbox = obj.get("bbox")
                if isinstance(bbox, dict):
                    xmin = bbox.get("xmin")
                    ymin = bbox.get("ymin")
                    xmax = bbox.get("xmax")
                    ymax = bbox.get("ymax")
                    if None not in (xmin, ymin, xmax, ymax):
                        frame_boxes.append([xmin, ymin, xmax, ymax])
                elif isinstance(bbox, list) and len(bbox) == 4:
                    x, y, w, h = bbox
                    frame_boxes.append([x, y, x + w, y + h])
            gt_boxes.append(frame_boxes)
    else:
        for obj in data.get("objects", []):
            if "bbox" in obj:
                x, y, w, h = obj["bbox"]
                gt_boxes.append([[x, y, x + w, y + h]])

    num_frames = data.get("frame_count", len(gt_boxes))

    return gt_boxes

def extract_bboxes_from_json(json_path, video_w, video_h):
    """
    Estrae le bounding box dal JSON de-normalizzandole.
    La lunghezza è determinata dinamicamente in base ai dati nel JSON.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    detected_objects = data.get("detected_objects", {})
    num_frames = 0
    
    # Determina la lunghezza dei frame calcolati in questa specifica parte
    for obj_val in detected_objects.values():
        bboxes = obj_val.get("bbox", [])
        num_frames = max(num_frames, len(bboxes))
        
    frames_bboxes = [[] for _ in range(num_frames)]
    
    for obj_val in detected_objects.values():
        bboxes = obj_val.get("bbox", [])
        for i, bbox in enumerate(bboxes):
            if bbox is not None:
                # Coordinate assolute [x1, y1, x2, y2]
                x1 = bbox[0] * video_w
                y1 = bbox[1] * video_h
                x2 = bbox[2] * video_w
                y2 = bbox[3] * video_h
                frames_bboxes[i].append([x1, y1, x2, y2])
                
    return frames_bboxes

# ==========================================
# 4. VALUTAZIONE E METRICHE
# ==========================================
def evaluate_tracking(gt_boxes, frames_preds):
    """Valuta le predizioni saltando i frame per i quali non è stato eseguito l'OD."""
    ious = []
    center_errors = []
    
    num_eval_frames = min(len(gt_boxes), len(frames_preds))
    
    for i in range(num_eval_frames):
        preds = frames_preds[i] 
        
        # Se 'preds' è None, significa che questo frame appartiene a una scena/parte scartata. 
        # Ignoriamo del tutto il frame per il calcolo delle metriche.
        if preds is None:
            continue
            
        gt_box = gt_boxes[i]
        is_visible = (occlusions[i] == 0) and (out_of_views[i] == 0)
        
        if not preds:  # Lista vuota [] -> Il modello non ha rilevato nulla
            ious.append(0.0)
            center_errors.append(float('inf'))
        else:
            best_iou = 0.0
            best_ce = float('inf')
                
            for p_box in preds:
                iou = compute_iou(gt_box, p_box)
                if iou >= best_iou:
                    best_iou = iou
                    best_ce = compute_center_error(gt_box, p_box)
                
            ious.append(best_iou)
            center_errors.append(best_ce)
                
    return ious, center_errors

def calculate_metrics(ious, center_errors):
    """Calcola AUC e Precision."""
    ious_arr = np.array(ious)
    ce_arr = np.array(center_errors)
    
    iou_thresholds = np.arange(0, 1.05, 0.05)
    cle_thresholds = np.arange(0, 51, 1)
    
    success_rates = [np.mean(ious_arr >= t) for t in iou_thresholds]
    precision_rates = [np.mean(ce_arr <= t) for t in cle_thresholds]
    
    # Utilizzo della sintassi numpy aggiornata per l'integrazione
    auc_success = np.trapezoid(success_rates, dx=0.05)
    precision_at_20 = precision_rates[20]
    
    print("-" * 30)
    print(f"RISULTATI FINALI (Calcolati su {len(ious_arr)} frame validi)")
    print("-" * 30)
    print(f"Success Rate (AUC): {auc_success:.4f}")
    print(f"Precision Rate (CLE < 20px): {precision_at_20:.4f}")
    
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
    
    jsons = glob.glob(os.path.join(VIDOR_BASE_PATH, "**", "*.json"), recursive=True)
    jsons.sort(key=lambda p: natural_keys(os.path.basename(p)))
    
    for json_path in jsons:
        print(f"Processando file JSON: {json_path}")
            
        gt_boxes = load_vidor_gt(json_path)
        
        scene_name = os.path.splitext(os.path.basename(json_path))[0]
        scene_pattern = os.path.join(VIDEO_ORIGINAL_PATH, f"{scene_name}_scene*.mp4")
        print(f"{scene_pattern}")
        scene_videos = glob.glob(scene_pattern)
        scene_videos.sort(key=natural_keys)
        
        if not scene_videos:
            continue
            
        cap = cv2.VideoCapture(scene_videos[0])
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        master_preds = []
        
        for video_path in scene_videos:
            scene_name = os.path.splitext(os.path.basename(video_path))[0]
            
            cap = cv2.VideoCapture(video_path)
            num_frames_in_scene = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Inizializza l'intera scena con 'None' (Marker per 'Ignora Frame')
            scene_preds = [None] * num_frames_in_scene
            
            # 1. Cerca il file JSON singolo (per scene < 10s)
            single_json_path = os.path.join(OD_RESULTS_PATH, scene_name, f"{scene_name}.json")
            
            if os.path.exists(single_json_path):
                frames_bboxes = extract_bboxes_from_json(single_json_path, video_w, video_h)
                limit = min(len(frames_bboxes), num_frames_in_scene)
                scene_preds[:limit] = frames_bboxes[:limit]
            else:
                # 2. Cerca le sottoparti (per scene > 10s)
                part_pattern = os.path.join(OD_RESULTS_PATH, f"{scene_name}_part*", f"{scene_name}_part*.json")
                part_jsons = glob.glob(part_pattern)
                
                for p_json in part_jsons:
                    # Estrae il numero 'Z' dal nome file 'robot-X_sceneY_partZ.json'
                    base = os.path.splitext(os.path.basename(p_json))[0]
                    match = re.search(r'_part(\d+)', base)
                    if match:
                        z = int(match.group(1))
                        # Offset calcolato supponendo frame a 30fps tagliati a 10s esatti (300 frame per parte)
                        offset = (z - 1) * 300 
                        
                        frames_bboxes = extract_bboxes_from_json(p_json, video_w, video_h)
                        limit = min(len(frames_bboxes), num_frames_in_scene - offset)
                        
                        if limit > 0:
                            # Inserisce i risultati esattamente nell'intervallo temporale corretto
                            scene_preds[offset : offset + limit] = frames_bboxes[:limit]
                            
            master_preds.extend(scene_preds)
                
        ious, ces = evaluate_tracking(gt_boxes, master_preds)
        
        all_ious.extend(ious)
        all_ces.extend(ces)
        
    if all_ious:
        calculate_metrics(all_ious, all_ces)
    else:
        print("Nessun dato valido processato.")

if __name__ == "__main__":
    run_pipeline()