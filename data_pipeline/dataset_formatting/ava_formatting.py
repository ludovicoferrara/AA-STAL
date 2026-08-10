import os
import json
import glob
import re
import shutil
import random
from collections import Counter, defaultdict

# ==========================================
# 1. DEFINIZIONE PERCORSI E PARAMETRI
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
OUT_FRAMES_LISTS = os.path.join(AVA_ROOT, "frame_lists")

os.makedirs(OUT_ANNOTATIONS, exist_ok=True)
os.makedirs(OUT_FRAMES, exist_ok=True)
os.makedirs(OUT_FRAMES_LISTS, exist_ok=True)

# Mappatura e raggruppamento dati
action_to_id = {}
current_action_id = 1
video_to_rows = defaultdict(list) # Raggruppa le righe CSV per video_id

print("Inizio elaborazione: Validazione ed estrazione dati...")

# ==========================================
# 2. ELABORAZIONE FILE CON STRICT MODE
# ==========================================
for act_file in glob.glob(os.path.join(ACTION_DIR, "*_actions.json")):
    basename = os.path.basename(act_file)

    match = re.search(r'(.*)_person_(\d+)_actions\.json', basename)
    if not match:
        continue

    video_id = match.group(1)
    person_id = match.group(2)
    person_key = f"person_{person_id}"

    src_frames = os.path.join(FRAMES_DIR, video_id)
    if not os.path.isdir(src_frames):
        continue

    det_folder = os.path.join(DETECTION_DIR, video_id)
    det_files = glob.glob(os.path.join(det_folder, "*.json"))
    if not det_files:
        continue
    det_file = det_files[0]

    with open(act_file, 'r') as f:
        actions_data = json.load(f)

    with open(det_file, 'r') as f:
        det_data = json.load(f)

    objects = det_data.get("detected_objects", {})
    if person_key not in objects:
        continue

    person_data = objects[person_key]
    if person_data.get("class_name") != "person":
        continue

    bboxes = person_data.get("bbox", [])
    max_frames = len(bboxes)

    # Calcolo frequenza azioni
    action_frequencies = Counter()
    windows = []

    for frame_range, action_info in actions_data.items():
        act = action_info.get("action")
        if not act:
            continue

        rmatch = re.search(r'frames_(\d+)_to_(\d+)', frame_range)
        if not rmatch:
            continue

        start_f = int(rmatch.group(1))
        end_f = int(rmatch.group(2))

        action_frequencies[act] += 1
        windows.append((start_f, end_f, act))

    person_temp_rows = []

    for frame_idx in range(0, max_frames, 30):
        bbox = bboxes[frame_idx]
        if not bbox:
            continue

        candidate_actions = []
        for start_f, end_f, act in windows:
            if start_f <= frame_idx <= end_f:
                candidate_actions.append(act)

        if not candidate_actions:
            continue

        best_action = max(candidate_actions, key=lambda a: action_frequencies[a])

        if best_action not in action_to_id:
            action_to_id[best_action] = current_action_id
            current_action_id += 1

        act_id = action_to_id[best_action]
        x1, y1, x2, y2 = [format(coord, '.4f') for coord in bbox]
        timestamp = str(frame_idx // 30)

        row = f"{video_id},{timestamp},{x1},{y1},{x2},{y2},{act_id},{person_id}"
        person_temp_rows.append(row)

    if person_temp_rows:
        # Bufferizza le righe associandole al video
        video_to_rows[video_id].extend(person_temp_rows)

# ==========================================
# 3. SPLIT STRATIFICATO (SHUFFLING)
# ==========================================
valid_videos = list(video_to_rows.keys())

# Shuffle randomico per disperdere i video ordinati per classe d'azione
random.seed(42) # Fissiamo il seed per riproducibilità
random.shuffle(valid_videos)

num_videos = len(valid_videos)
train_split = int(num_videos * 0.8) # 80%
val_split = int(num_videos * 0.9)   # 10%

splits = {
    "train": valid_videos[:train_split],
    "val": valid_videos[train_split:val_split],
    "test": valid_videos[val_split:]
}

print(f"Video validi totali: {num_videos}")
print(f"Suddivisione: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

# Funzione ausiliaria per estrarre l'intero dal nome del frame (Natural Sorting)
def extract_frame_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

# ==========================================
# 4. SALVATAGGIO DATI (ANNOTATIONS & FRAMES LISTS)
# ==========================================
for split_name, split_vids in splits.items():
    if not split_vids:
        continue

    # 1. Scrittura Annotations (CSV bounding box e classi delimitati da VIRGOLA)
    split_annotations = []
    for vid in split_vids:
        split_annotations.extend(video_to_rows[vid])

    csv_path = os.path.join(OUT_ANNOTATIONS, f"ava_{split_name}_v2.2.csv")
    with open(csv_path, 'w') as f:
        f.write("\n".join(split_annotations) + "\n")

    # 2. Scrittura Frames Lists (File delimitati da SPAZIO)
    list_path = os.path.join(OUT_FRAMES_LISTS, f"{split_name}.csv")
    with open(list_path, 'w') as f_list:

        for vid in split_vids:
            # Creazione collegamento / copia per i frame
            src_frames = os.path.join(FRAMES_DIR, vid)
            dst_frames = os.path.join(OUT_FRAMES, vid)

            if os.path.exists(src_frames) and not os.path.exists(dst_frames):
                try:
                    os.symlink(src_frames, dst_frames)
                except OSError:
                    shutil.copytree(src_frames, dst_frames)

            # Lettura e ordinamento fisico dei frame
            frame_files = [img for img in os.listdir(src_frames) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            frame_files.sort(key=extract_frame_number)

            # Scrittura riga per riga con delimitatore SPAZIO
            for i, frame_name in enumerate(frame_files, 1):
                path = f"{vid}/{frame_name}"
                dummy_label = "0"
                row = f"{vid} {vid} {i} {path} {dummy_label}\n"
                f_list.write(row)

# Salvataggio vocabolario azioni (Protobuf)
pbtxt_path = os.path.join(OUT_ANNOTATIONS, "ava_action_list_v2.2.pbtxt")
with open(pbtxt_path, 'w') as f:
    for act_name, act_id in sorted(action_to_id.items(), key=lambda x: x[1]):
        f.write("label {\n")
        f.write(f'  name: "{act_name}"\n')
        f.write(f'  label_id: {act_id}\n')
        f.write('  label_type: 2\n')
        f.write("}\n")

print("Dataset generato e splittato con successo.")