import os
import shutil
import random
from collections import defaultdict

# SETTINGS
ORIGINAL_AVA_ROOT = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/AVA_Dataset"
BASE_OUTPUT_DIR = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT"

TRAIN_PERCENTAGES = [0.0625, 0.125, 0.25, 0.5, 0.75]  
RANDOM_SEED = 42

IN_ANNOTATIONS = os.path.join(ORIGINAL_AVA_ROOT, "annotations")
IN_FRAMES_LISTS = os.path.join(ORIGINAL_AVA_ROOT, "frame_lists")
IN_FRAMES = os.path.join(ORIGINAL_AVA_ROOT, "frames")

train_anno_in = os.path.join(IN_ANNOTATIONS, "ava_train_v2.2.csv")
train_list_in = os.path.join(IN_FRAMES_LISTS, "train.csv")

random.seed(RANDOM_SEED)

# MAP VIDEO -> ACTION 
video_to_action = {}
video_to_rows = defaultdict(list)

with open(train_anno_in, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 8:
            vid_id = parts[0]
            action_id = parts[6]
            video_to_rows[vid_id].append(line)
            if vid_id not in video_to_action:
                video_to_action[vid_id] = action_id

action_to_videos = defaultdict(list)
for vid, act in video_to_action.items():
    action_to_videos[act].append(vid)

for act in action_to_videos:
    action_to_videos[act].sort()
    random.shuffle(action_to_videos[act])

with open(train_list_in, 'r') as f:
    train_list_lines = f.readlines()

val_test_videos = set()
for split in ["val", "test"]:
    list_path = os.path.join(IN_FRAMES_LISTS, f"{split}.csv")
    if os.path.exists(list_path):
        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    val_test_videos.add(parts[0])

# NESTED SUBSETS
for pct in TRAIN_PERCENTAGES:
    pct_str = str(pct * 100).replace('.', '_')
    print(f"\Subset at {pct_str}%...")
    
    subset_root = os.path.join(BASE_OUTPUT_DIR, f"AVA_Dataset_Subset_{pct_str}")
    out_anno = os.path.join(subset_root, "annotations")
    out_lists = os.path.join(subset_root, "frame_lists")
    out_frames = os.path.join(subset_root, "frames")
    
    os.makedirs(out_anno, exist_ok=True)
    os.makedirs(out_lists, exist_ok=True)
    os.makedirs(out_frames, exist_ok=True)

    shutil.copy2(os.path.join(IN_ANNOTATIONS, "ava_action_list_v2.2.pbtxt"), out_anno)
    for split in ["val", "test"]:
        csv_in = os.path.join(IN_ANNOTATIONS, f"ava_{split}_v2.2.csv")
        if os.path.exists(csv_in):
            shutil.copy2(csv_in, out_anno)
        list_in = os.path.join(IN_FRAMES_LISTS, f"{split}.csv")
        if os.path.exists(list_in):
            shutil.copy2(list_in, out_lists)

    kept_train_videos = set()
    for act, videos in action_to_videos.items():
        num_keep = max(1, int(len(videos) * pct)) if len(videos) > 0 else 0
        kept_train_videos.update(videos[:num_keep])
        
    print(f"  - Video kept in Train: {len(kept_train_videos)}")

    with open(os.path.join(out_anno, "ava_train_v2.2.csv"), 'w') as f_out:
        for vid in kept_train_videos:
            for line in video_to_rows[vid]:
                f_out.write(line)
                
    with open(os.path.join(out_lists, "train.csv"), 'w') as f_out:
        for line in train_list_lines:
            parts = line.strip().split()
            if parts and parts[0] in kept_train_videos:
                f_out.write(line)

    videos_to_keep = kept_train_videos.union(val_test_videos)
    
    for vid in videos_to_keep:
        src_dir = os.path.join(IN_FRAMES, vid)
        dst_dir = os.path.join(out_frames, vid)
        
        if os.path.exists(src_dir) and not os.path.exists(dst_dir):
            try:
                os.symlink(src_dir, dst_dir)
            except OSError:
                shutil.copytree(src_dir, dst_dir)

print("\nSubset generation completed.")