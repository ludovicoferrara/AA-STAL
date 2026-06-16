import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAZIONE PERCORSI E PARAMETRI
# ==========================================
# Imposta la risoluzione reale dei video per de-normalizzare [0, 1] -> pixel
VIDEO_W = 1920  
VIDEO_H = 1080  

GT_FOLDER = "/mnt/c/Users/ludov/Projects/creazione-sdrogo-dataset-AA-STAL/industrial_robot_arm"
OD_RESULTS_PATH = "/mnt/c/Users/ludov/UNIVERSITA/SECONDO ANNO/TESI/Risultati/OD Multipla/video_general_obj_det_finished-dino-ind_rob_arm-001/video_general_obj_det_finished"

# ==========================================
# 2. UTILITY MATEMATICHE E DI RICERCA
# ==========================================
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

def find_matching_json(gt_filename, all_json_paths):
    """
    Associa il file GT (es. 'video_scene1_part1_obj1.txt') al suo JSON corrispondente
    (es. 'video_scene1_part1.json') trovando il match di substringa più lungo.
    """
    best_match = None
    max_len = 0
    gt_name_no_ext = os.path.splitext(gt_filename)[0]
    
    for json_path in all_json_paths:
        json_name = os.path.splitext(os.path.basename(json_path))[0]
        if json_name in gt_name_no_ext and len(json_name) > max_len:
            best_match = json_path
            max_len = len(json_name)
            
    return best_match

# ==========================================
# 3. LETTURA DATI
# ==========================================
def load_gt_txt(gt_path):
    """
    Carica i txt. Assumo formato x, y, w, h separato da virgola o spazio.
    Supporta valori negativi o zeri come indicatore di assenza/occlusione.
    """
    gt_abs = []
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().replace(',', ' ').split()
            if len(parts) >= 4:
                x, y, w, h = map(float, parts[:4])
                if w <= 0 or h <= 0:
                    gt_abs.append(None) # Frame vuoto o target occluso/uscito
                else:
                    gt_abs.append([x, y, x + w, y + h])
            else:
                gt_abs.append(None)
    return gt_abs

def extract_bboxes_from_json(json_path):
    """Estrae e de-normalizza tutte le bbox dal JSON frame per frame."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    detected_objects = data.get("detected_objects", {})
    num_frames = 0
    
    for obj_val in detected_objects.values():
        bboxes = obj_val.get("bbox", [])
        num_frames = max(num_frames, len(bboxes))
        
    frames_bboxes = [[] for _ in range(num_frames)]
    
    for obj_val in detected_objects.values():
        bboxes = obj_val.get("bbox", [])
        for i, bbox in enumerate(bboxes):
            if bbox is not None:
                x1 = bbox[0] * VIDEO_W
                y1 = bbox[1] * VIDEO_H
                x2 = bbox[2] * VIDEO_W
                y2 = bbox[3] * VIDEO_H
                frames_bboxes[i].append([x1, y1, x2, y2])
                
    return frames_bboxes

# ==========================================
# 4. VALUTAZIONE E METRICHE
# ==========================================
def evaluate_tracking_oracle(gt_boxes, frames_preds):
    """
    Valuta il tracking. Se il JSON contiene multipli oggetti, viene 
    automaticamente processato solo quello che massimizza l'IoU con la GT.
    """
    ious = []
    center_errors = []
    
    num_eval_frames = min(len(gt_boxes), len(frames_preds))
    
    for i in range(num_eval_frames):
        gt_box = gt_boxes[i]
        preds = frames_preds[i] 
        
        if gt_box is not None:  # Il target è visibile nella Ground Truth
            if not preds:       # Il modello non ha rilevato nulla (Falso Negativo)
                ious.append(0.0)
                center_errors.append(float('inf'))
            else:
                # ORACLE MATCHING: ignora i falsi positivi nel frame, cerca il target corretto
                best_iou = 0.0
                best_ce = float('inf')
                
                for p_box in preds:
                    iou = compute_iou(gt_box, p_box)
                    if iou >= best_iou:
                        best_iou = iou
                        best_ce = compute_center_error(gt_box, p_box)
                
                ious.append(best_iou)
                center_errors.append(best_ce)
        else:                   # Il target NON è visibile nella Ground Truth (Occluso/Fuori scena)
            if not preds:       # Il modello non ha rilevato nulla (Vero Negativo)
                ious.append(1.0)
                center_errors.append(0.0)
            else:               # Il modello ha rilevato qualcosa di errato (Falso Positivo)
                # In caso di MOT, la presenza di bboxes di altri oggetti NON è un falso positivo
                # per QUESTO specifico target. Di conseguenza, il frame non penalizza l'IoU.
                pass 
                
    return ious, center_errors

def calculate_metrics(ious, center_errors):
    """Calcola AUC e Precision."""
    ious_arr = np.array(ious)
    ce_arr = np.array(center_errors)
    
    iou_thresholds = np.arange(0, 1.05, 0.05)
    cle_thresholds = np.arange(0, 51, 1)
    
    success_rates = [np.mean(ious_arr >= t) for t in iou_thresholds]
    precision_rates = [np.mean(ce_arr <= t) for t in cle_thresholds]
    
    auc_success = np.trapezoid(success_rates, dx=0.05)
    precision_at_20 = precision_rates[20]
    
    print("-" * 30)
    print(f"RISULTATI FINALI (Calcolati su {len(ious_arr)} frame con target visibile/valutabile)")
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
    plt.savefig("evaluation_global_mot_results.png")
    print("Grafico salvato come 'evaluation_global_mot_results.png'.")

# ==========================================
# 5. PIPELINE PRINCIPALE
# ==========================================
def run_pipeline():
    all_ious = []
    all_ces = []
    
    # Raccoglie tutti i file txt di ground truth
    gt_files = glob.glob(os.path.join(GT_FOLDER, "*.txt"))
    # Raccoglie tutti i file JSON dei risultati in modo ricorsivo
    json_files = glob.glob(os.path.join(OD_RESULTS_PATH, "**", "*.json"), recursive=True)
    
    if not gt_files:
        print(f"[!] Nessun file .txt trovato in {GT_FOLDER}")
        return
        
    for gt_path in gt_files:
        gt_filename = os.path.basename(gt_path)
        
        # 1. Associazione GT -> JSON
        matching_json = find_matching_json(gt_filename, json_files)
        
        if not matching_json:
            print(f"  [-] Nessun JSON trovato per: {gt_filename}. Salto.")
            continue
            
        # 2. Estrazione
        gt_boxes = load_gt_txt(gt_path)
        frames_preds = extract_bboxes_from_json(matching_json)
        
        # 3. Valutazione (Oracle Filter)
        ious, ces = evaluate_tracking_oracle(gt_boxes, frames_preds)
        
        all_ious.extend(ious)
        all_ces.extend(ces)
        print(f"  [+] Elaborato: {gt_filename} con {os.path.basename(matching_json)}")
        
    if all_ious:
        calculate_metrics(all_ious, all_ces)
    else:
        print("Nessun dato valido processato.")

if __name__ == "__main__":
    run_pipeline()