import os
import shutil
import random

# percentage of the training set to keep
TRAIN_PERCENTAGE = 0.25
RANDOM_SEED = 42

# PATHS
ORIGINAL_AVA_ROOT = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/AVA_Dataset"
NEW_AVA_ROOT = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/AVA_Dataset_Subset" + f"_{int(TRAIN_PERCENTAGE*100)}percent"

IN_ANNOTATIONS = os.path.join(ORIGINAL_AVA_ROOT, "annotations")
IN_FRAMES_LISTS = os.path.join(ORIGINAL_AVA_ROOT, "frames_lists")
IN_FRAMES = os.path.join(ORIGINAL_AVA_ROOT, "frames")

OUT_ANNOTATIONS = os.path.join(NEW_AVA_ROOT, "annotations")
OUT_FRAMES_LISTS = os.path.join(NEW_AVA_ROOT, "frames_lists")
OUT_FRAMES = os.path.join(NEW_AVA_ROOT, "frames")

os.makedirs(OUT_ANNOTATIONS, exist_ok=True)
os.makedirs(OUT_FRAMES_LISTS, exist_ok=True)
os.makedirs(OUT_FRAMES, exist_ok=True)

random.seed(RANDOM_SEED)

print(f"Generazione sottoinsieme al {TRAIN_PERCENTAGE*100}% del training set...")

# ==========================================
# 2. TRASFERIMENTO FILE IDENTICI (VAL, TEST, PBTXT)
# ==========================================
# Copia il vocabolario
shutil.copy2(
    os.path.join(IN_ANNOTATIONS, "ava_action_list_v2.2.pbtxt"),
    os.path.join(OUT_ANNOTATIONS, "ava_action_list_v2.2.pbtxt")
)

# Copia annotazioni val e test
for split in ["val", "test"]:
    csv_in = os.path.join(IN_ANNOTATIONS, f"ava_{split}_v2.2.csv")
    if os.path.exists(csv_in):
        shutil.copy2(csv_in, os.path.join(OUT_ANNOTATIONS, f"ava_{split}_v2.2.csv"))
        
    list_in = os.path.join(IN_FRAMES_LISTS, f"{split}.csv")
    if os.path.exists(list_in):
        shutil.copy2(list_in, os.path.join(OUT_FRAMES_LISTS, f"{split}.csv"))

# ==========================================
# 3. CAMPIONAMENTO DEL TRAINING SET
# ==========================================
train_list_in = os.path.join(IN_FRAMES_LISTS, "train.csv")
train_anno_in = os.path.join(IN_ANNOTATIONS, "ava_train_v2.2.csv")

# Identificazione dei video unici nel training set leggendo il file delimitato da spazi
train_videos = set()
with open(train_list_in, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            train_videos.add(parts[0])

train_videos = sorted(list(train_videos))
num_train_videos = len(train_videos)
num_keep = int(num_train_videos * TRAIN_PERCENTAGE)

# Selezione casuale dei video da mantenere
kept_train_videos = set(random.sample(train_videos, num_keep))

print(f"Video originali nel Train: {num_train_videos}")
print(f"Video mantenuti nel Train: {len(kept_train_videos)}")

# ==========================================
# 4. FILTRAGGIO E SCRITTURA NUOVO TRAINING SET
# ==========================================
# Filtraggio frames_lists/train.csv
with open(train_list_in, 'r') as f_in, open(os.path.join(OUT_FRAMES_LISTS, "train.csv"), 'w') as f_out:
    for line in f_in:
        parts = line.strip().split()
        if parts and parts[0] in kept_train_videos:
            f_out.write(line)

# Filtraggio annotations/ava_train_v2.2.csv
with open(train_anno_in, 'r') as f_in, open(os.path.join(OUT_ANNOTATIONS, "ava_train_v2.2.csv"), 'w') as f_out:
    for line in f_in:
        parts = line.strip().split(',')
        if parts and parts[0] in kept_train_videos:
            f_out.write(line)

# ==========================================
# 5. TRASFERIMENTO FRAME
# ==========================================
# Identifica tutti i video necessari per il nuovo dataset (train filtrato + val + test completi)
videos_to_keep = set(kept_train_videos)

for split in ["val", "test"]:
    list_in = os.path.join(IN_FRAMES_LISTS, f"{split}.csv")
    if os.path.exists(list_in):
        with open(list_in, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    videos_to_keep.add(parts[0])

# Crea symlink o copia i frame solo per i video utili
for vid in videos_to_keep:
    src_dir = os.path.join(IN_FRAMES, vid)
    dst_dir = os.path.join(OUT_FRAMES, vid)
    
    if os.path.exists(src_dir) and not os.path.exists(dst_dir):
        try:
            os.symlink(src_dir, dst_dir)
        except OSError:
            shutil.copytree(src_dir, dst_dir)

print("Sottoinsieme generato con successo.")