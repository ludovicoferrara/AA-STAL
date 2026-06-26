import os
import json
import glob
import re
import shutil

# ==========================================
# 1. DEFINIZIONE PERCORSI
# ==========================================
BASE_DIR = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT"

VIDEOS_DIR = os.path.join(BASE_DIR, "Videos_crop")
FRAMES_DIR = os.path.join(BASE_DIR, "Videos_crop_decode")
DETECTION_DIR = os.path.join(BASE_DIR, "video_general_obj_det_finished")
ACTION_DIR = os.path.join(BASE_DIR, "action_recognition_finished")

# Cartelle di output in formato AVA
AVA_ROOT = os.path.join(BASE_DIR, "AVA_Dataset")
OUT_ANNOTATIONS = os.path.join(AVA_ROOT, "annotations")
OUT_FRAMES = os.path.join(AVA_ROOT, "frames")

os.makedirs(OUT_ANNOTATIONS, exist_ok=True)
os.makedirs(OUT_FRAMES, exist_ok=True)

# Variabili globali per la mappatura
action_to_id = {}
current_action_id = 1
csv_rows = []

print("Inizio elaborazione per conversione in formato AVA...")

# ==========================================
# 2. ELABORAZIONE FILE
# ==========================================
# Cerca tutti i file delle azioni, assumendo un naming simile a "videoID_sceneX_person_1002_actions.json"
for act_file in glob.glob(os.path.join(ACTION_DIR, "*_actions.json")):
    basename = os.path.basename(act_file)
    
    # Estrae il nome del video e il track_id usando una regex
    match = re.search(r'(.*)_person_(\d+)_actions\.json', basename)
    if not match:
        continue

    video_id = match.group(1)
    person_id = match.group(2)
    person_key = f"person_{person_id}"

    # Trova il corrispondente JSON di object detection nella relativa cartella
    det_folder = os.path.join(DETECTION_DIR, video_id)
    det_files = glob.glob(os.path.join(det_folder, "*.json"))
    if not det_files:
        print(f"ATTENZIONE: Nessun file di detection trovato per la cartella: {video_id}")
        continue
    det_file = det_files[0]

    # Carica i due JSON
    with open(act_file, 'r') as f:
        actions_data = json.load(f)

    with open(det_file, 'r') as f:
        det_data = json.load(f)

    # Verifica se la persona esiste nei rilevamenti
    objects = det_data.get("detected_objects", {})
    if person_key not in objects:
        continue

    # Estrae l'array delle bounding box frame-by-frame
    bboxes = objects[person_key].get("bbox", [])

    # Itera sui blocchi temporali (es. "frames_0_to_30")
    for frame_range, action_list in actions_data.items():
        if not action_list:
            continue

        rmatch = re.search(r'frames_(\d+)_to_(\d+)', frame_range)
        if not rmatch:
            continue
        
        start_f = int(rmatch.group(1))
        end_f = int(rmatch.group(2))

        # Calcola il frame centrale (keyframe) del blocco
        mid_f = start_f + (end_f - start_f) // 2
        
        # Gestione out-of-bounds se il video finisce prima
        if mid_f >= len(bboxes):
            mid_f = len(bboxes) - 1
        if mid_f < 0: 
            continue

        # Calcola timestamp in secondi (AVA utilizza spesso stringhe float / sec)
        timestamp = f"{(mid_f / 30.0):.4f}"

        # Estrae le coordinate [x1, y1, x2, y2]
        bbox = bboxes[mid_f]
        x1, y1, x2, y2 = [format(coord, '.4f') for coord in bbox]

        # Isola al massimo le prime due label per mantenere compatibilità con la tua regola
        selected_actions = action_list[:2]
        
        for act in selected_actions:
            # Aggiorna il dizionario azioni dinamicamente
            if act not in action_to_id:
                action_to_id[act] = current_action_id
                current_action_id += 1
            
            act_id = action_to_id[act]

            # Creazione riga (senza header, come richiesto da AVA/YOWO)
            # Struttura: video_id, timestamp, x1, y1, x2, y2, action_id, person_id
            row = f"{video_id},{timestamp},{x1},{y1},{x2},{y2},{act_id},{person_id}"
            csv_rows.append(row)

    # ==========================================
    # 3. CREAZIONE ALBERO DELLE DIRECTORY (FRAMES)
    # ==========================================
    src_frames = os.path.join(FRAMES_DIR, video_id)
    dst_frames = os.path.join(OUT_FRAMES, video_id)
    
    if os.path.exists(src_frames) and not os.path.exists(dst_frames):
        try:
            # Usa i collegamenti simbolici per risparmiare spazio su disco
            os.symlink(src_frames, dst_frames)
        except OSError:
            # Fallback alla copia fisica se i symlink falliscono
            print(f"Symlink fallito per {video_id}. Avvio copia fisica dei frame...")
            shutil.copytree(src_frames, dst_frames)

# ==========================================
# 4. SALVATAGGIO DEI FILE DI ANNOTAZIONE
# ==========================================

# Scrive il CSV principale
csv_path = os.path.join(OUT_ANNOTATIONS, "ava_train_v2.2.csv")
with open(csv_path, 'w') as f:
    f.write("\n".join(csv_rows) + "\n")
print(f"Scritto dataset CSV: {csv_path} (Totale righe: {len(csv_rows)})")

# Scrive il file Protobuf (.pbtxt)
pbtxt_path = os.path.join(OUT_ANNOTATIONS, "ava_action_list_v2.2.pbtxt")
with open(pbtxt_path, 'w') as f:
    for act_name, act_id in sorted(action_to_id.items(), key=lambda x: x[1]):
        f.write("label {\n")
        f.write(f'  name: "{act_name}"\n')
        f.write(f'  label_id: {act_id}\n')
        # label_type 2 indica generalmente l'interazione con oggetti/azioni fisiche in AVA
        f.write('  label_type: 2\n') 
        f.write("}\n")
print(f"Scritto dizionario azioni Protobuf: {pbtxt_path} (Totale azioni trovate: {len(action_to_id)})")
print("Processo completato con successo.")