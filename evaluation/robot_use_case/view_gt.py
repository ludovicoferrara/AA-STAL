import os
import cv2
import glob

# ==========================================
# 1. CONFIGURAZIONE PERCORSI
# ==========================================
# Cartella dei video raw originali (assicurati dell'estensione, qui cerchiamo .mp4)
VIDEO_DIR = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/Videos"

# Cartella con le cartelle dei robot (robot-1, robot-2, ecc.) e i txt
GT_DIR = "/mnt/c/Users/ludov/Projects/creazione-sdrogo-dataset-AA-STAL/LaSot_robot"

# Cartella di output dove verranno salvati i video di debug
OUTPUT_DIR = "debug_videos_gt"

# ==========================================
# 2. CARICAMENTO GROUNDTRUTH
# ==========================================
def load_lasot_gt(gt_path, occ_path, oov_path):
    """Carica e formatta la groundtruth originale di LaSOT (x, y, w, h)."""
    with open(gt_path, 'r') as f:
        # LaSOT usa x, y, w, h
        gt_boxes = [list(map(float, line.strip().split(','))) for line in f]
    
    with open(occ_path, 'r') as f:
        occlusions = [int(x) for x in f.read().replace(',', ' ').split()]
        
    with open(oov_path, 'r') as f:
        out_of_views = [int(x) for x in f.read().replace(',', ' ').split()]

    return gt_boxes, occlusions, out_of_views

# ==========================================
# 3. PIPELINE DI VISUALIZZAZIONE
# ==========================================
def draw_gt_on_videos():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Trova tutte le cartelle robot in GT_DIR
    robots = [d for d in os.listdir(GT_DIR) if os.path.isdir(os.path.join(GT_DIR, d))]
    
    if not robots:
        print(f"[!] Nessuna cartella robot trovata in {GT_DIR}")
        return

    for robot in robots:
        print(f"Elaborazione video di debug per: {robot}")
        
        # Percorsi file GT
        gt_path = os.path.join(GT_DIR, robot, "groundtruth.txt")
        occ_path = os.path.join(GT_DIR, robot, "full_occlusion.txt")
        oov_path = os.path.join(GT_DIR, robot, "out_of_view.txt")
        
        if not os.path.exists(gt_path):
            print(f"  [-] {robot}: File groundtruth.txt mancante. Salto.")
            continue
            
        gt_boxes, occlusions, out_of_views = load_lasot_gt(gt_path, occ_path, oov_path)
        
        # Cerca il video corrispondente (supporta .mp4 o .avi)
        video_path = os.path.join(VIDEO_DIR, f"{robot}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(VIDEO_DIR, f"{robot}.avi")
            if not os.path.exists(video_path):
                print(f"  [-] {robot}: Video corrispondente non trovato in {VIDEO_DIR}. Salto.")
                continue
                
        # Apertura video originale
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup VideoWriter per l'output
        out_path = os.path.join(OUTPUT_DIR, f"{robot}_debug.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        frame_idx = 0
        total_gt_frames = len(gt_boxes)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Disegna solo se abbiamo i dati GT per questo frame
            if frame_idx < total_gt_frames:
                box = gt_boxes[frame_idx]
                occ = occlusions[frame_idx]
                oov = out_of_views[frame_idx]
                
                # Testo HUD (Heads-Up Display) di default
                status_text = "VISIBLE"
                color = (0, 255, 0) # Verde (BGR)
                
                if oov == 1:
                    status_text = "OUT OF VIEW"
                    color = (0, 0, 255) # Rosso
                elif occ == 1:
                    status_text = "FULL OCCLUSION"
                    color = (0, 165, 255) # Arancione
                
                # Disegna la bounding box se non è fuori inquadratura o ha larghezza/altezza valide
                if oov == 0 and len(box) >= 4:
                    x, y, bw, bh = box
                    if bw > 0 and bh > 0:
                        x1, y1 = int(x), int(y)
                        x2, y2 = int(x + bw), int(y + bh)
                        
                        # Disegna il rettangolo
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Scrivi le informazioni a schermo per il debug
                cv2.putText(frame, f"Frame: {frame_idx} / GT_Length: {total_gt_frames}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                cv2.putText(frame, f"Frame: {frame_idx} / GT_Length: {total_gt_frames}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                            
                cv2.putText(frame, f"Status: {status_text}", (30, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            else:
                # Se il video è più lungo della groundtruth, avvisiamo visivamente
                cv2.putText(frame, "WARNING: NO GT DATA FOR THIS FRAME", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            out.write(frame)
            frame_idx += 1
            
        cap.release()
        out.release()
        
        print(f"  [+] Salvato: {out_path} (Elaborati {frame_idx} frame, Video Raw: {total_frames_video})")

if __name__ == "__main__":
    draw_gt_on_videos()