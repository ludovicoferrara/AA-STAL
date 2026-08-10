import os
import shutil
import random
from collections import defaultdict

# ==========================================
# 1. IMPOSTAZIONI
# ==========================================
ORIGINAL_AVA_ROOT = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/AVA_Dataset"
BASE_OUTPUT_DIR = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT"

TRAIN_PERCENTAGES = [0.0625, 0.125, 0.25, 0.5]  
RANDOM_SEED = 42

IN_ANNOTATIONS = os.path.join(ORIGINAL_AVA_ROOT, "annotations")
IN_FRAMES_LISTS = os.path.join(ORIGINAL_AVA_ROOT, "frame_lists")
IN_FRAMES = os.path.join(ORIGINAL_AVA_ROOT, "frames")

train_anno_in = os.path.join(IN_ANNOTATIONS, "ava_train_v2.2.csv")
train_list_in = os.path.join(IN_FRAMES_LISTS, "train.csv")

random.seed(RANDOM_SEED)

# ==========================================
# 2. MAPPATURA VIDEO -> AZIONE (STRATIFICAZIONE)
# ==========================================
video_to_action = {}
video_to_rows = defaultdict(list)

# Legge il CSV di train per raggruppare i video per classe d'azione
with open(train_anno_in, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 8:
            vid_id = parts[0]
            action_id = parts[6]
            video_to_rows[vid_id].append(line)
            # Dato che c'è una sola azione per video, basta assegnarla alla prima occorrenza
            if vid_id not in video_to_action:
                video_to_action[vid_id] = action_id

# Raggruppa i video per azione
action_to_videos = defaultdict(list)
for vid, act in video_to_action.items():
    action_to_videos[act].append(vid)

# Ordina in modo deterministico e mescola (una sola volta) ogni gruppo
for act in action_to_videos:
    action_to_videos[act].sort()
    random.shuffle(action_to_videos[act])

# Legge le righe del frame list originale
with open(train_list_in, 'r') as f:
    train_list_lines = f.readlines()

# Identifica i video di val e test per il trasferimento dei frame
val_test_videos = set()
for split in ["val", "test"]:
    list_path = os.path.join(IN_FRAMES_LISTS, f"{split}.csv")
    if os.path.exists(list_path):
        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    val_test_videos.add(parts[0])

# ==========================================
# 3. GENERAZIONE DEGLI SCAGLIONI (NESTED SUBSETS)
# ==========================================
for pct in TRAIN_PERCENTAGES:
    pct_str = str(pct * 100).replace('.', '_')
    print(f"\nGenerazione subset al {pct_str}%...")
    
    subset_root = os.path.join(BASE_OUTPUT_DIR, f"AVA_Dataset_Subset_{pct_str}")
    out_anno = os.path.join(subset_root, "annotations")
    out_lists = os.path.join(subset_root, "frame_lists")
    out_frames = os.path.join(subset_root, "frames")
    
    os.makedirs(out_anno, exist_ok=True)
    os.makedirs(out_lists, exist_ok=True)
    os.makedirs(out_frames, exist_ok=True)

    # 3a. Trasferimento file invariabili (Val, Test, Vocabolario)
    shutil.copy2(os.path.join(IN_ANNOTATIONS, "ava_action_list_v2.2.pbtxt"), out_anno)
    for split in ["val", "test"]:
        csv_in = os.path.join(IN_ANNOTATIONS, f"ava_{split}_v2.2.csv")
        if os.path.exists(csv_in):
            shutil.copy2(csv_in, out_anno)
        list_in = os.path.join(IN_FRAMES_LISTS, f"{split}.csv")
        if os.path.exists(list_in):
            shutil.copy2(list_in, out_lists)

    # 3b. Selezione stratificata dei video (Slicing = Inclusione progressiva)
    kept_train_videos = set()
    for act, videos in action_to_videos.items():
        # Usa max(1, ...) per garantire che ogni classe abbia almeno 1 video se pct > 0
        num_keep = max(1, int(len(videos) * pct)) if len(videos) > 0 else 0
        # Lo slicing videos[:num_keep] garantisce l'annidamento degli scaglioni
        kept_train_videos.update(videos[:num_keep])
        
    print(f"  - Video mantenuti nel Train: {len(kept_train_videos)}")

    # 3c. Scrittura Annotations Train
    with open(os.path.join(out_anno, "ava_train_v2.2.csv"), 'w') as f_out:
        for vid in kept_train_videos:
            for line in video_to_rows[vid]:
                f_out.write(line)
                
    # 3d. Scrittura Frames Lists Train (usando spazi)
    with open(os.path.join(out_lists, "train.csv"), 'w') as f_out:
        for line in train_list_lines:
            parts = line.strip().split()
            if parts and parts[0] in kept_train_videos:
                f_out.write(line)

    # 3e. Creazione symlink/copia dei frame
    videos_to_keep = kept_train_videos.union(val_test_videos)
    
    for vid in videos_to_keep:
        src_dir = os.path.join(IN_FRAMES, vid)
        dst_dir = os.path.join(out_frames, vid)
        
        if os.path.exists(src_dir) and not os.path.exists(dst_dir):
            try:
                os.symlink(src_dir, dst_dir)
            except OSError:
                shutil.copytree(src_dir, dst_dir)

print("\nTutti i subset stratificati sono stati generati con successo.")