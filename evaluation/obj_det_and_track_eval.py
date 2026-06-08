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
LASOT_BASE_PATH = "/mnt/c/Users/ludov/Projects/creazione-sdrogo-dataset-AA-STAL/LaSot_robot"
VIDEO_ORIGINAL_PATH = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/Videos_crop"
OD_RESULTS_PATH = "/mnt/c/Users/ludov/UNIVERSITA/SECONDO ANNO/TESI/Risultati/OD Multipla/video_general_obj_det_finished-dino/video_general_obj_det_finished"
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
def load_lasot_gt(gt_path, occ_path, oov_path):
    """Carica e formatta la groundtruth di LaSOT."""
    with open(gt_path, 'r') as f:
        gt_boxes = [list(map(float, line.strip().split(','))) for line in f]
    
    # Legge l'intero contenuto, converte le virgole in spazi e divide per estrarre i singoli interi
    with open(occ_path, 'r') as f:
        occlusions = [int(x) for x in f.read().replace(',', ' ').split()]
        
    with open(oov_path, 'r') as f:
        out_of_views = [int(x) for x in f.read().replace(',', ' ').split()]

    gt_abs = []
    for box in gt_boxes:
        x, y, w, h = box
        gt_abs.append([x, y, x + w, y + h])
        
    return gt_abs, occlusions, out_of_views

def extract_bboxes_from_json(json_path, num_frames, video_w, video_h):
    """
    Estrae le bounding box dal JSON strutturato per singola scena.
    Restituisce un array di lunghezza 'num_frames' contenente una lista di bbox per frame.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Inizializza array per tutti i frame della scena
    frames_bboxes = [[] for _ in range(num_frames)]
    detected_objects = data.get("detected_objects", {})
    
    # Itera su tutte le entità rilevate (person_1000, robotic arm_1001, ecc.)
    for obj_key, obj_val in detected_objects.items():
        bboxes = obj_val.get("bbox", [])
        
        # Popola i frame con le bbox de-normalizzate dell'entità corrente
        for i, bbox in enumerate(bboxes):
            if i >= num_frames:
                break # Sicurezza per non superare i frame reali del video
            if bbox is not None:
                # [xmin, ymin, xmax, ymax] -> coordinate assolute
                x1 = bbox[0] * video_w
                y1 = bbox[1] * video_h
                x2 = bbox[2] * video_w
                y2 = bbox[3] * video_h
                frames_bboxes[i].append([x1, y1, x2, y2])
                
    return frames_bboxes

# ==========================================
# 4. VALUTAZIONE E METRICHE
# ==========================================
def evaluate_tracking(gt_boxes, occlusions, out_of_views, frames_preds):
    """Valuta le predizioni master rispetto alla GT tramite Oracolo (Miglior IoU)."""
    ious = []
    center_errors = []
    
    # Per sicurezza ci limitiamo al minimo tra la lunghezza GT e le predizioni ricostruite
    num_eval_frames = min(len(gt_boxes), len(frames_preds))
    
    for i in range(num_eval_frames):
        gt_box = gt_boxes[i]
        preds = frames_preds[i] # Lista di bbox rilevate nel frame i-esimo
        
        is_visible = (occlusions[i] == 0) and (out_of_views[i] == 0)
        
        if is_visible:
            if not preds:
                # Falso Negativo
                ious.append(0.0)
                center_errors.append(float('inf'))
            else:
                # ORACLE: Cerca il target predetto con l'IoU maggiore
                best_iou = 0.0
                best_ce = float('inf')
                
                for p_box in preds:
                    iou = compute_iou(gt_box, p_box)
                    if iou >= best_iou:
                        best_iou = iou
                        best_ce = compute_center_error(gt_box, p_box)
                
                ious.append(best_iou)
                center_errors.append(best_ce)
        else:
            if not preds:
                # Vero Negativo
                ious.append(1.0)
                center_errors.append(0.0)
            else:
                # Falso Positivo
                ious.append(0.0)
                center_errors.append(float('inf'))
                
    return ious, center_errors

def calculate_metrics(ious, center_errors):
    """Calcola AUC per il Success Rate e il Precision Rate."""
    ious_arr = np.array(ious)
    ce_arr = np.array(center_errors)
    
    iou_thresholds = np.arange(0, 1.05, 0.05)
    cle_thresholds = np.arange(0, 51, 1)
    
    success_rates = [np.mean(ious_arr >= t) for t in iou_thresholds]
    precision_rates = [np.mean(ce_arr <= t) for t in cle_thresholds]
    
    auc_success = np.trapezoid(success_rates, dx=0.05)
    precision_at_20 = precision_rates[20]
    
    print("-" * 30)
    print("RISULTATI FINALI GLOBALI")
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
    
    # Estrae tutti i robot presenti nella cartella LaSOT
    robots = [d for d in os.listdir(LASOT_BASE_PATH) if os.path.isdir(os.path.join(LASOT_BASE_PATH, d))]
    
    for robot in robots:
        print(f"Processando robot: {robot}")
        
        # Caricamento GT
        gt_path = os.path.join(LASOT_BASE_PATH, robot, "groundtruth.txt")
        occ_path = os.path.join(LASOT_BASE_PATH, robot, "full_occlusion.txt")
        oov_path = os.path.join(LASOT_BASE_PATH, robot, "out_of_view.txt")
        
        if not os.path.exists(gt_path):
            print(f"  [!] Groundtruth mancante per {robot}, salto.")
            continue
            
        gt_boxes, occlusions, out_of_views = load_lasot_gt(gt_path, occ_path, oov_path)
        
        # Recupera tutte le scene originali associate a questo robot
        scene_pattern = os.path.join(VIDEO_ORIGINAL_PATH, f"{robot}_scene*.mp4")
        scene_videos = glob.glob(scene_pattern)
        scene_videos.sort(key=natural_keys)
        
        if not scene_videos:
            print(f"  [!] Nessuna scena video originale trovata per {robot}, salto.")
            continue
            
        # Determina la risoluzione video analizzando la prima scena
        cap = cv2.VideoCapture(scene_videos[0])
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Costruzione dell'array Master per le predizioni
        master_preds = []
        
        for video_path in scene_videos:
            # Nome base della scena, es: "robot-1_scene1"
            scene_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Calcolo durata esatta in frame tramite cv2
            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Percorso del JSON corrispondente (seguendo la tua struttura)
            json_path = os.path.join(OD_RESULTS_PATH, scene_name, f"{scene_name}.json")
            
            if os.path.exists(json_path):
                # Estrae e de-normalizza le bbox
                frames_bboxes = extract_bboxes_from_json(json_path, num_frames, video_w, video_h)
                master_preds.extend(frames_bboxes)
            else:
                # Scena scartata dall'OD: aggiunge frame vuoti (equivalente dei null)
                master_preds.extend([[] for _ in range(num_frames)])
                
        # Valutazione del video ricostruito contro la GT
        ious, ces = evaluate_tracking(gt_boxes, occlusions, out_of_views, master_preds)
        
        # Aggiunta ai risultati globali
        all_ious.extend(ious)
        all_ces.extend(ces)
        
    # Elaborazione finale delle metriche
    if all_ious:
        calculate_metrics(all_ious, all_ces)
    else:
        print("Nessun dato valido processato.")

if __name__ == "__main__":
    run_pipeline()