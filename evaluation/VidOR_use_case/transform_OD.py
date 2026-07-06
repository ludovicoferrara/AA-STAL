import os
import json
import re
from pathlib import Path
from collections import defaultdict

# Configurazioni dei percorsi
OD_DIR = Path("/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/video_general_obj_det_partial-dino")
GT_DIR = Path("/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT/groundtruth/VidOR")
TRACKEVAL_OUT = Path("./TrackEval_Workspace/data")

OD_FPS = 30.0
TRACKER_NAME = "DINO_Tracker"
DATASET_NAME = "VidOR-eval"

# Mappatura dalle categorie della Ground Truth agli ID usati dall'Object Detection
# NOTA: Assicurati di aggiungere qui tutte le classi rilevanti. 
# Le classi non presenti in questo dizionario verranno ignorate nella valutazione.
CATEGORY_TO_ID = {
    "adult": 0,   # Raggruppati sotto la classe "person" dell'OD (ID 0)
    "child": 0,   # Raggruppati sotto la classe "person" dell'OD (ID 0)
    "person": 0,
    "table": 1,
    "sofa": 9,
    "cake": 11
}

def find_gt_files(gt_dir):
    gt_map = {}
    for filepath in gt_dir.rglob("*.json"):
        video_name = filepath.stem
        gt_map[video_name] = filepath
    return gt_map

def group_and_sort_od_files(od_dir):
    regex = re.compile(r'^(.*?)_scene(\d+)(?:_part(\d+))?\.json$')
    video_parts = defaultdict(list)
    
    for filepath in od_dir.rglob("*.json"):
        match = regex.match(filepath.name)
        if not match:
            continue
            
        video_name = match.group(1)
        scene_id = int(match.group(2))
        part_id = int(match.group(3) if match.group(3) else 0)
        
        video_parts[video_name].append({
            'path': filepath,
            'scene': scene_id,
            'part': part_id
        })
        
    for name in video_parts:
        video_parts[name].sort(key=lambda x: (x['scene'], x['part']))
        
    return video_parts

def process_dataset():
    gt_map = find_gt_files(GT_DIR)
    od_map = group_and_sort_od_files(OD_DIR)
    
    gt_base_dir = TRACKEVAL_OUT / "gt" / "mot_challenge" / DATASET_NAME
    trk_base_dir = TRACKEVAL_OUT / "trackers" / "mot_challenge" / DATASET_NAME / TRACKER_NAME / "data"
    
    for video_name, od_files in od_map.items():
        if video_name not in gt_map:
            print(f"[-] GT mancante per {video_name}. Salto.")
            continue
            
        # --- 1. LETTURA GROUND TRUTH E SETUP ---
        with open(gt_map[video_name], 'r') as f:
            gt_data = json.load(f)
            
        gt_fps = gt_data["fps"]
        width = gt_data["width"]
        height = gt_data["height"]
        gt_frame_count = gt_data["frame_count"]
        
        # Mappatura interna per associare i TID (Track ID) alla loro classe nella GT
        tid_to_class = {}
        for subj_obj in gt_data.get("subject/objects", []):
            category = subj_obj["category"]
            tid_to_class[subj_obj["tid"]] = CATEGORY_TO_ID.get(category, -1)
            
        vid_gt_dir = gt_base_dir / video_name / "gt"
        vid_gt_dir.mkdir(parents=True, exist_ok=True)
        trk_base_dir.mkdir(parents=True, exist_ok=True)
        
        # --- 2. ESTRAZIONE GROUND TRUTH (MULTI-CLASSE) ---
        gt_mot_lines = []
        for frame_idx, frame_objs in enumerate(gt_data["trajectories"]):
            frame_mot = frame_idx + 1 
            for obj in frame_objs:
                tid = obj["tid"]
                class_id = tid_to_class.get(tid, -1)
                
                # Ignora le classi non mappate
                if class_id == -1:
                    continue
                    
                b = obj["bbox"]
                w = b["xmax"] - b["xmin"]
                h = b["ymax"] - b["ymin"]
                
                # Formato: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, vis
                gt_mot_lines.append(f"{frame_mot},{tid},{b['xmin']},{b['ymin']},{w},{h},1,{class_id},1.0")
                
        with open(vid_gt_dir / "gt.txt", "w") as f:
            f.write("\n".join(gt_mot_lines) + "\n")
            
        # Creazione seqinfo.ini
        seqinfo_content = f"""[Sequence]\nname={video_name}\nimDir=img1\nframeRate={gt_fps}\nseqLength={gt_frame_count}\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n"""
        with open(gt_base_dir / video_name / "seqinfo.ini", "w") as f:
            f.write(seqinfo_content)
            
        # --- 3. ESTRAZIONE OBJECT DETECTION (MULTI-CLASSE) ---
        od_mot_dict = {} 
        cumulative_od_frames = 0
        
        for part_info in od_files:
            with open(part_info['path'], 'r') as f:
                od_data = json.load(f)
                
            objects = od_data.get("detected_objects", {})
            part_frame_count = 0
            
            for obj_key, obj_data in objects.items():
                track_id = obj_data["track_id"]
                class_id = obj_data["class_id"]
                bboxes = obj_data["bbox"]
                
                if len(bboxes) > part_frame_count:
                    part_frame_count = len(bboxes)
                    
                for local_frame_idx, bbox in enumerate(bboxes):
                    if bbox is None:
                        continue
                        
                    global_od_idx = cumulative_od_frames + local_frame_idx
                    timestamp_sec = global_od_idx / OD_FPS
                    gt_frame_idx = int(round(timestamp_sec * gt_fps))
                    
                    frame_mot = gt_frame_idx + 1 
                    
                    if frame_mot > gt_frame_count:
                        continue
                        
                    x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox
                    x_min = x_min_norm * width
                    y_min = y_min_norm * height
                    b_w = (x_max_norm - x_min_norm) * width
                    b_h = (y_max_norm - y_min_norm) * height
                    
                    # Formato: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, vis
                    od_mot_dict[(frame_mot, track_id)] = f"{frame_mot},{track_id},{x_min:.2f},{y_min:.2f},{b_w:.2f},{b_h:.2f},1.0,{class_id},1.0"
            
            cumulative_od_frames += part_frame_count
            
        # --- 4. SCRITTURA TRACKER (OD) ---
        od_mot_lines = [od_mot_dict[k] for k in sorted(od_mot_dict.keys())]
        with open(trk_base_dir / f"{video_name}.txt", "w") as f:
            f.write("\n".join(od_mot_lines) + "\n")
            
        print(f"[+] Elaborato: {video_name} (GT: {len(gt_mot_lines)} obj | OD: {len(od_mot_lines)} obj)")

if __name__ == "__main__":
    process_dataset()
    print("\nElaborazione multi-classe terminata.")