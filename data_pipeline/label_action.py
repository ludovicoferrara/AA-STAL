import os
import glob
import json
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse

# Qwen imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ==========================================
# ACTION VOCABOLARY (CLOSED)
# ==========================================
HUMAN_ACTIONS = ["idle", "walking", "running", "talking", "interacting with object", "standing"]
ROBOT_ACTIONS = ["idle", "moving", "interacting with object"]

def get_action_list(class_name):
    """Returns vocabulary according to entity type """
    if class_name == 'person' or class_name == 'humanoid robot':
        return HUMAN_ACTIONS
    return ROBOT_ACTIONS

def init_qwen_model(device='cuda'):
    """Initializes Qwen2.5-VL in 4-bit to maximize efficiency"""
    print("Initializing Qwen2.5-VL-3B-Instruct (INT4)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

FPS_STIMATI = 30
WINDOW_SIZE_SEC = 2.0  # Durata della finestra temporale da analizzare (in secondi)
WINDOW_FRAMES = int(FPS_STIMATI * WINDOW_SIZE_SEC)
FRAMES_TO_SAMPLE_PER_WINDOW = 6 # Quanti frame passare a Qwen per ogni finestra

def extract_window_frames(video_path, bboxes, start_idx, end_idx, num_samples):
    """
    Estrae frame uniformemente solo all'interno della finestra [start_idx, end_idx].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    # Prendi solo i bounding box validi in QUESTA finestra
    window_bboxes = bboxes[start_idx:end_idx]
    valid_local_indices = [i for i, box in enumerate(window_bboxes) if box is not None]
    
    if not valid_local_indices:
        cap.release()
        return []
        
    step = max(1, len(valid_local_indices) // num_samples)
    sampled_local_indices = valid_local_indices[::step][:num_samples]
    
    frames_pil = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    for local_idx in sampled_local_indices:
        global_idx = start_idx + local_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, global_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        box_norm = bboxes[global_idx]
        x1, y1, x2, y2 = [
            int(box_norm[0] * width), int(box_norm[1] * height),
            int(box_norm[2] * width), int(box_norm[3] * height)
        ]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(frame_rgb))
        
    cap.release()
    return frames_pil


def classify_action(model, processor, frames_pil, class_name):
    """Interroga Qwen2.5-VL per classificare l'azione"""
    if not frames_pil:
        return "unknown"
        
    vocab = get_action_list(class_name)
    vocab_str = ", ".join(vocab)
    
    prompt = (
        f"Analyze this sequence of frames. Focus EXCLUSIVELY on the {class_name} "
        f"enclosed in the thick RED bounding box. "
        f"What action is this specific {class_name} performing? "
        f"Choose EXACTLY ONE label from this list: [{vocab_str}]. "
        f"Reply with ONLY the exact word/phrase from the list, with no punctuation or extra text."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                # Qwen supporta il caricamento di video nativo, ma passandogli le PIL 
                # modificate lo trattiamo come una sequenza di immagini
                {"type": "video", "video": frames_pil},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip().lower()
    
    torch.cuda.empty_cache()
    
    for action in vocab:
        if action in output_text:
            return action
            
    return "unknown"

def process_dataset(video_dir, json_dir):
    model, processor = init_qwen_model()
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    for json_path in tqdm(json_files):
        video_name = os.path.basename(json_path).replace('.json', '.mp4')
        video_path = os.path.join(video_dir, video_name)
        
        if not os.path.exists(video_path):
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        if 'detected_objects' not in data:
            continue
            
        changes_made = False
        
        for obj_key, obj_data in data['detected_objects'].items():
            # Salta se ha già la nuova struttura ad array
            if 'action_labels' in obj_data and len(obj_data['action_labels']) > 0:
                continue
                
            class_name = obj_data['class_name']
            bboxes = obj_data['bbox']
            total_frames = len(bboxes)
            
            obj_data['action_labels'] = []
            
            # Scorriamo il video in finestre di WINDOW_FRAMES
            for start_frame in range(0, total_frames, WINDOW_FRAMES):
                end_frame = min(start_frame + WINDOW_FRAMES, total_frames)
                
                # Assicuriamoci che la finestra abbia una lunghezza sensata
                # (evitiamo di analizzare "code" di 3 frame a fine video)
                if (end_frame - start_frame) < (WINDOW_FRAMES / 2):
                    break
                
                # Estrai clip per QUESTA finestra
                frames = extract_window_frames(
                    video_path, bboxes, start_frame, end_frame, FRAMES_TO_SAMPLE_PER_WINDOW
                )
                
                if not frames:
                    action = "unknown"
                else:
                    action = classify_action(model, processor, frames, class_name)
                
                # Salviamo il risultato per il segmento temporale
                obj_data['action_labels'].append({
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "action": action
                })
                
                print(f"  [{video_name}] {obj_key} (frames {start_frame}-{end_frame}) -> {action}")
                changes_made = True
            
        if changes_made:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assign action labels to detected objects in videos')

    parser.add_argument('--cropped_video_dir', type=str, default='/path/to/Videos', help='Directory cropped video')
    parser.add_argument('--obj_det_dir', type=str, default='/path/to/Videos_crop', help='Directory detection video')
    args = parser.parse_args()
    
    process_dataset(args.cropped_video_dir, args.obj_det_dir)
    print("Action Labeling completed!")