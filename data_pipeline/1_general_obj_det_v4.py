from pathlib import Path
import torch, random
import argparse
import os, sys, glob, imageio

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, Optional
import supervision as sv
from supervision.draw.color import Color, ColorPalette
import time
import warnings

warnings.filterwarnings("ignore")

# from accelerate import Accelerator
# ACCELERATE_AVAILABLE = True

from utils_detectron2 import DefaultPredictor_Lazy

# sam2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor

from kalman_filter import *

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from transformers import AutoModelForCausalLM

# VGGT for camera motion detection
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import torch.nn.functional as F
VGGT_AVAILABLE = True
print("VGGT imported for camera motion detection")



if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.tensor([0.0], device="cuda")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def get_iou(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]


    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width * height

def scale_bbox_within_image(bbox, img_width, img_height, scale=1.5):
    x1, y1, x2, y2 = bbox

    # Calculate the center
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Current width and height
    width = x2 - x1
    height = y2 - y1

    # Scaled width and height
    new_width = width * scale
    new_height = height * scale

    # New bounding box coordinates (before clamping)
    new_x1 = cx - new_width / 2.0
    new_x2 = cx + new_width / 2.0
    new_y1 = cy - new_height / 2.0
    new_y2 = cy + new_height / 2.0

    # Clamp to the image boundaries
    new_x1 = max(0, min(new_x1, img_width))
    new_x2 = max(0, min(new_x2, img_width))
    new_y1 = max(0, min(new_y1, img_height))
    new_y2 = max(0, min(new_y2, img_height))

    if new_x1 < new_x2 and new_y1 < new_y2:
        return (new_x1, new_y1, new_x2, new_y2)
    else:
        return None


def get_mask4bbox(sam2_image_predictor, image_path, bboxes):

    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    sam2_image_predictor.set_image(image)

    input_boxes = np.array(bboxes)
    masks, scores, _ = sam2_image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # print(masks.shape)
    masks_clean = []

    if len(masks.shape) == 3:
        masks = masks[None, :, :, :]
    for m_idx in range(masks.shape[0]):
        mask = masks[m_idx][0]
        masks_clean.append({'segmentation': mask.astype(int).astype(bool)})
    return masks_clean



def isLocked(filename):
    if os.path.exists(filename):
        return True
    try:
        os.mkdir(filename)
        return False
    except:
        return True

def unLock(filename):
    os.rmdir(filename)


def chunk_into_n(lst, n):
    """
    Split `lst` into `n` chunks as evenly as possible.
    Some chunks may be one element larger if len(lst) % n != 0.
    """
    k, m = divmod(len(lst), n)
    chunks = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < m else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks




def init_qwen_model(device='cuda'):
    """Inizializza Qwen2.5-VL-7B-Instruct con quantizzazione a 4-bit per risparmiare VRAM"""
    print("Inizializzazione di Qwen2.5-VL-7B in modalità 4-bit (INT4)...")

    # Configurazione per comprimere il modello a 4-bit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # Formato ottimizzato per i pesi normalizzati
        bnb_4bit_compute_dtype=torch.float16, # I calcoli avvengono comunque in FP16 per stabilità
        bnb_4bit_use_double_quant=True       # Risparmia ulteriore memoria quantizzando le costanti
    )

    # Passiamo alla versione da 7 Miliardi di parametri
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"


    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    print("Qwen2.5-VL-7B caricato con successo in 4-bit!")
    return model, processor





def detect_objects_with_qwen2_5_vl(model, processor, image, text_prompt):
    """Visual Grounding Zero-Shot usando Qwen2.5-VL (con parsing JSON rinforzato)"""
    from qwen_vl_utils import process_vision_info
    import re
    import json

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    img_w, img_h = image_pil.size

    # Prompt draconiano per forzare SOLO l'output JSON
    strict_prompt = (
        f"Locate the following objects in the image: {text_prompt}. "
        "You MUST output ONLY a valid JSON array. Do not include any conversational text, explanations, or markdown formatting. "
        "If none of the objects are present in the image, you must output exactly this empty array: []\n"
        "Format example: [{\"label\": \"robotic arm\", \"bbox_2d\": [ymin, xmin, ymax, xmax]}]"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": strict_prompt}
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
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    del inputs
    del image_inputs
    del video_inputs
    del generated_ids
    del generated_ids_trimmed

    print(f"\n[DEBUG QWEN RAW OUTPUT]: {output_text.strip()}")

    boxes_xyxy = []
    phrases = []
    scores = []
    
    # Estrazione sicura del JSON tramite Regex (cerca tutto ciò che è tra [ e ])
    json_match = re.search(r'\[.*\]', output_text, re.DOTALL)
    
    if not json_match:
        print("Nessun blocco JSON trovato nell'output di Qwen. Restituisco lista vuota.")
        return boxes_xyxy, phrases, scores

    clean_text = json_match.group(0)
    
    try:
        qwen_objects = json.loads(clean_text)
        for obj in qwen_objects:
            label = obj.get("label", "robot_part").strip().lower()
            ymin, xmin, ymax, xmax = obj["bbox_2d"]
            
            # Le coordinate di Qwen sono scalate da 0 a 1000.
            x1 = (xmin / 1000.0) * img_w
            y1 = (ymin / 1000.0) * img_h
            x2 = (xmax / 1000.0) * img_w
            y2 = (ymax / 1000.0) * img_h
            
            boxes_xyxy.append([x1, y1, x2, y2])
            phrases.append(label)
            scores.append(1.0)
    except json.JSONDecodeError as e:
        print(f"Errore di parsing JSON da Qwen: {e}. Testo parsato: {clean_text}")
    except KeyError as e:
        print(f"Chiave mancante nel JSON di Qwen: {e}")

    return boxes_xyxy, phrases, scores


def apply_nms(boxes, phrases, scores, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove overlapping detection boxes"""
    if len(boxes) == 0:
        return [], [], []

    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Sort by confidence
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        # Select box with highest confidence
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Calculate IoU with other boxes
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]

        ious = []
        for other_box in other_boxes:
            iou = get_iou(current_box, other_box)
            ious.append(iou)

        # Keep boxes with IoU below threshold
        ious = np.array(ious)
        indices = indices[1:][ious < iou_threshold]

    # Return kept detections
    filtered_boxes = [boxes[i].tolist() for i in keep]
    filtered_phrases = [phrases[i] for i in keep]
    filtered_scores = [scores[i] for i in keep]

    return filtered_boxes, filtered_phrases, filtered_scores


def filter_and_deduplicate_objects(objects, confidence_threshold=0.45, iou_threshold=0.3):
    """Filter and deduplicate detected objects"""
    if not objects:
        return []

    # Sort by confidence
    objects = sorted(objects, key=lambda x: x.get('confidence', 0), reverse=True)

    filtered_objects = []
    for obj in objects:
        # Confidence filtering
        if obj.get('confidence', 0) < confidence_threshold:
            continue

        # Check for overlap with existing objects
        is_duplicate = False
        for existing_obj in filtered_objects:
            iou = get_iou(obj['bbox'], existing_obj['bbox'])
            if iou > iou_threshold:
                # If high overlap, keep only higher confidence one
                if obj.get('confidence', 0) > existing_obj.get('confidence', 0):
                    filtered_objects.remove(existing_obj)
                else:
                    is_duplicate = True
                break

        if not is_duplicate:
            # Check for duplicate object names (could be different detections of same object)
            class_name = obj['class_name']
            name_duplicate = False
            for existing_obj in filtered_objects:
                if existing_obj['class_name'] == class_name:
                    # For same class objects, check spatial distance
                    bbox1 = obj['bbox']
                    bbox2 = existing_obj['bbox']
                    center1 = [(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2]
                    center2 = [(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2]
                    distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

                    # If distance too close, consider as duplicate detection
                    if distance < 50:  # Reduced to 50 pixel distance threshold
                        if obj.get('confidence', 0) > existing_obj.get('confidence', 0):
                            filtered_objects.remove(existing_obj)
                        else:
                            name_duplicate = True
                        break

            if not name_duplicate:
                filtered_objects.append(obj)

    return filtered_objects


# COCO dataset's 80 classes
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

OBJECTS_CLASS_ID = {
    'person': 0,
    'left_hand': 1,
    'object_in_left_hand': 2,
    'left_hand_2nd_obj': 3,
    'right_hand': 4,
    'object_in_right_hand': 5,
    'right_hand_2nd_obj': 6,
}

# Assign IDs for COCO object classes (starting from 100 to avoid conflict with hand objects)
for i, class_name in enumerate(COCO_CLASSES[1:], start=100):  # Skip person class
    OBJECTS_CLASS_ID[class_name] = i

PERSON            = (255, 0, 255)
RIGHT_HAND_COLOR  = (255, 0, 0)
LEFT_HAND_COLOR   = (0, 0, 255) # rgb
FIRST_COLOR  = (255, 176, 0)
SECOND_COLOR = (0, 170, 100)

COLOR_PALETTE = [
    (255, 0, 255), # purple
    (255, 0, 0),   # red
    (0, 0, 255),   # blue
    (255, 176, 0), # yellow
    (0, 170, 100)  # green
]

# Generate color palette for all object classes
import colorsys
import subprocess

def get_original_video_fps(video_name, video_base_dir):
    """
    Get the original video's FPS by checking multiple possible locations
    """
    possible_paths = [
        os.path.join(video_base_dir, 'Videos_crop', f'{video_name}.mp4'),
    ]

    for video_path in possible_paths:
        if os.path.exists(video_path):
            try:
                # Use ffprobe to get video fps
                cmd = [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate",
                    "-of", "csv=p=0",
                    video_path
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    fps_str = result.stdout.strip()
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        fps = float(num) / float(den)
                    else:
                        fps = float(fps_str)
                    print(f"Found original video {video_path} with FPS: {fps}")
                    return fps
            except Exception as e:
                print(f"Failed to get FPS from {video_path}: {e}")
                continue

    # Fallback: calculate FPS based on frame count and estimated duration
    print(f"Could not find original video for {video_name}, using fallback calculation")
    return None

def calculate_fps_from_frames(frame_count, estimated_duration=5.0):
    """
    Calculate FPS based on frame count and estimated duration
    """
    return frame_count / estimated_duration

def generate_colors(n):
    """Generate n different colors"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1  # 0.7, 0.8, 0.9
        value = 0.8 + (i % 2) * 0.2       # 0.8, 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

# Base colors (hand-related)
BASE_HEX_COLORS = ['#FF00FF', '#FF0000', '#FFB000', '#00AA64', '#0000FF', '#FFB000', '#00AA64']
# Generate colors for all COCO object classes
OBJECT_HEX_COLORS = generate_colors(len(COCO_CLASSES))
# Merge all colors
HEX_COLOR_PALETTE = BASE_HEX_COLORS + OBJECT_HEX_COLORS


def update_color_palette_for_video(video_objects):
    """Update color palette for detected objects in video"""
    global HEX_COLOR_PALETTE, OBJECTS_CLASS_ID

    for i, obj_name in enumerate(video_objects):
        OBJECTS_CLASS_ID[obj_name] = 200 + i

    base_colors = ['#FF00FF', '#FF0000', '#FFB000', '#00AA64', '#0000FF', '#FFB000', '#00AA64']
    object_colors = generate_colors(len(video_objects))
    HEX_COLOR_PALETTE = base_colors + object_colors

    return HEX_COLOR_PALETTE


def init_vggt_model(device='cuda'):
    """Initialize VGGT model for camera motion detection"""
    if not VGGT_AVAILABLE:
        return None

    # Load VGGT model - using the official checkpoint from Hugging Face
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"VGGT model initialized successfully on {device}")
    return model


def detect_camera_motion(vggt_model, image_paths, motion_threshold=0.1, sample_frames=8, device='cuda'):
    """
    Use VGGT to detect camera motion in video

    Args:
        vggt_model: Initialized VGGT model
        image_paths: List of video frame paths
        motion_threshold: Camera motion threshold (threshold for translation and rotation changes)
        sample_frames: Number of frames to sample (for efficiency, not processing all frames)
        device: Computation device

    Returns:
        tuple: (is_moving, motion_score, camera_poses)
            - is_moving: bool, whether camera is moving
            - motion_score: float, motion score
            - camera_poses: list, list of camera poses
    """
    if vggt_model is None or len(image_paths) < 2:
        return False, 0.0, []

    # Sample frames for efficiency
    total_frames = len(image_paths)
    if total_frames > sample_frames:
        indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
        sampled_paths = [image_paths[i] for i in indices]
    else:
        sampled_paths = image_paths

    torch.cuda.empty_cache()

    images_tensor_list = []
    for img_path in sampled_paths:
        image = Image.open(img_path).convert('RGB')
        image = image.resize((518, 518))
        image_array = np.array(image, dtype=np.float16)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        images_tensor_list.append(image_tensor)

    images_tensor = torch.stack(images_tensor_list).to(device)
    del images_tensor_list

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            images_batch = images_tensor[None]  # add batch dimension [1, N, 3, H, W]
            # Aggregate features
            aggregated_tokens_list, ps_idx = vggt_model.aggregator(images_batch)

            # Predict camera poses
            pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]

            # Convert to extrinsic and intrinsic matrices
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])

        del images_tensor, images_batch, aggregated_tokens_list, pose_enc
        # torch.cuda.empty_cache()  # Reduced frequency for better performance

    camera_poses = []
    extrinsic = extrinsic.squeeze(0).cpu().numpy()  # Remove batch dimension

    for i in range(len(extrinsic)):
        camera_matrix = extrinsic[i]  # 4x4 matrix
        translation = camera_matrix[:3, 3]  # Camera position
        rotation = camera_matrix[:3, :3]   # Camera orientation

        camera_poses.append({
            'translation': translation,
            'rotation': rotation,
            'matrix': camera_matrix
        })

    translation_changes = []
    rotation_changes = []

    for i in range(1, len(camera_poses)):
        pos_change = np.linalg.norm(
            camera_poses[i]['translation'] - camera_poses[i-1]['translation']
        )
        translation_changes.append(pos_change)

        rot_change = np.linalg.norm(
            camera_poses[i]['rotation'] - camera_poses[i-1]['rotation'],
            ord='fro'
        )
        rotation_changes.append(rot_change)

    avg_translation_change = np.mean(translation_changes) if translation_changes else 0
    avg_rotation_change = np.mean(rotation_changes) if rotation_changes else 0
    max_translation_change = np.max(translation_changes) if translation_changes else 0
    max_rotation_change = np.max(rotation_changes) if rotation_changes else 0

    motion_score = (avg_translation_change + avg_rotation_change +
                   max_translation_change * 0.5 + max_rotation_change * 0.5)

    is_moving = motion_score > motion_threshold

    print(f"Camera motion: score={motion_score:.4f}, threshold={motion_threshold}, is_moving={is_moving}")

    return is_moving, motion_score, camera_poses



def get_cached_image(image_path, cache, cache_limit=100):
    """
    Get image from cache or load and cache it
    Args:
        image_path: Path to image
        cache: Image cache dict
        cache_limit: Maximum number of images to cache
    Returns:
        np.ndarray: Image array
    """
    if image_path in cache:
        return cache[image_path]

    # Load image
    img = cv2.imread(str(image_path))

    # Manage cache size
    if len(cache) >= cache_limit:
        # Remove oldest entry (FIFO)
        oldest_key = next(iter(cache))
        del cache[oldest_key]

    # Cache image
    cache[image_path] = img
    return img

def preload_video_images(img_paths, max_preload=50):
    """
    Preload video images for faster access
    Args:
        img_paths: List of image paths
        max_preload: Maximum number of images to preload
    Returns:
        dict: Preloaded images
    """
    preloaded = {}
    load_count = min(len(img_paths), max_preload)


    for i in range(load_count):
        img_path = img_paths[i]
        img = cv2.imread(str(img_path))
        if img is not None:
            preloaded[img_path] = img

    return preloaded


def should_skip_video_due_to_camera_motion(vggt_model, video_dir, motion_threshold=0.1, sample_frames=8):

    if vggt_model is None:
        return False, 0.0, "VGGT model not available"

    img_ls = glob.glob(f'{video_dir}/*.jpg')
    img_ls.sort()

    if len(img_ls) < 2:
        return False, 0.0, "Insufficient frames for motion detection"

    is_moving, motion_score, _ = detect_camera_motion(
        vggt_model, img_ls, motion_threshold=motion_threshold, sample_frames=sample_frames
    )

    if is_moving:
        reason = f"Camera is moving (score: {motion_score:.4f} > threshold: {motion_threshold})"
        return True, motion_score, reason
    else:
        reason = f"Camera is stable (score: {motion_score:.4f} <= threshold: {motion_threshold})"
        return False, motion_score, reason


def print_step_header(step_name, step_number=None):
    """Print a simple step header"""
    if step_number:
        print(f"Step {step_number}: {step_name}", flush=True)
    else:
        print(f"{step_name}", flush=True)


def check_and_merge_existing_markers(finished_dir, video_name):
    """
    Check for existing failure markers and merge them to avoid duplicates.
    Returns the path to use for the failure marker.

    Priority order (keep the most informative one):
    1. camera_motion_skipped (has detailed reason)
    2. other specific failure markers
    3. generic failure marker
    """
    base_path = os.path.join(finished_dir, video_name)
    camera_motion_path = os.path.join(finished_dir, f'{video_name}_camera_motion_skipped')

    # Check what already exists
    existing_markers = []
    if os.path.exists(base_path):
        existing_markers.append(('generic', base_path))
    if os.path.exists(camera_motion_path):
        existing_markers.append(('camera_motion', camera_motion_path))

    # Check for other specific markers (future extensibility)
    for item in os.listdir(finished_dir):
        if item.startswith(f'{video_name}_') and item != f'{video_name}_camera_motion_skipped':
            full_path = os.path.join(finished_dir, item)
            if os.path.isdir(full_path):
                existing_markers.append(('other', full_path))

    if len(existing_markers) <= 1:
        return base_path  # No conflict, use generic path

    # Multiple markers exist - merge them
    print(f"Found multiple failure markers for {video_name}: {[m[0] for m in existing_markers]}")

    # Keep camera_motion_skipped if it exists (most informative)
    if any(marker[0] == 'camera_motion' for marker in existing_markers):
        keep_path = camera_motion_path
        print(f"Keeping camera_motion_skipped marker (most informative)")
    else:
        # Keep the first one found
        keep_path = existing_markers[0][1]
        print(f"Keeping {existing_markers[0][0]} marker")

    # Remove other markers
    for marker_type, marker_path in existing_markers:
        if marker_path != keep_path:
            try:
                if os.path.isdir(marker_path):
                    if os.listdir(marker_path):  # Directory not empty
                        print(f"Removing duplicate marker: {marker_path}")
                        import shutil
                        shutil.rmtree(marker_path)
                    else:  # Empty directory
                        os.rmdir(marker_path)
                        print(f"Removed empty duplicate marker: {marker_path}")
            except Exception as e:
                print(f"Warning: Could not remove duplicate marker {marker_path}: {e}")

    return keep_path





if __name__ == '__main__':

    # Initialize timing statistics
    timing_stats = {
        'model_loading': 0,
        'camera_motion_detection': 0,
        'object_analysis_detection': 0,
        'initial_sam2_tracking': 0,
        'human_hand_association': 0,
        'second_sam2_tracking': 0,
        'sam2_propagation': 0,
        'reformatting_visualization': 0,
        'saving_results': 0,
        'total_video_processing': 0
    }

    script_start_time = time.time()

    # cur_parser = argparse.ArgumentParser(description='demo code')
    # cur_parser.add_argument('--video_dir', type=str, default='', help='video directionary', required=True)
    # cur_parser.add_argument('--single_video_name', type=str, default='', help='process only this specific video (video name without extension)')
    # cur_parser.add_argument('--video_names', type=str, nargs='+', default=[], help='process only these specific videos (video names without extension)')

    # cur_parser.add_argument('--chunk_num', type=int, default=10, help='split videos into chunks')
    # cur_parser.add_argument('--chunk_idx', type=int, default=0, help='which chunk to process (use -1 to process all videos)')
    # cur_parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')

    # # VGGT Camera Motion Detection Arguments
    # cur_parser.add_argument('--enable_camera_motion_detection', action='store_true',
    #                        help='Enable VGGT-based camera motion detection to skip videos with moving cameras')
    # cur_parser.add_argument('--camera_motion_threshold', type=float, default=0.3,
    #                        help='Threshold for camera motion detection (default: 0.3). Lower values are more sensitive.')
    # cur_parser.add_argument('--camera_motion_sample_frames', type=int, default=4,
    #                        help='Number of frames to sample for camera motion detection (default: 4, reduced for memory)')

    # # Memory optimization arguments
    # # cur_parser.add_argument('--disable_qwen', action='store_true',
    # #                        help='Disable Qwen2-VL-72B model to save GPU memory (will use fallback objects)')
    # cur_parser.add_argument('--max_video_frames', type=int, default=300,
    #                        help='Skip videos with more than this many frames (default: 300, reduced from 500)')

    # cur_args = cur_parser.parse_args()

    video_dir='/content/drive/MyDrive/AA-STAL/data_pipeline/DATA_ROOT'
    single_video_name=''
    video_names=[]
    chunk_num=10
    chunk_idx=-1
    body_detector='vitdet'
    enable_camera_motion_detection=False #da cambiare
    camera_motion_threshold=0.3
    camera_motion_sample_frames=4
    disable_qwen=False
    max_video_frames=300


    # Auto-detect accelerate environment
    accelerator = None
    # if ACCELERATE_AVAILABLE:
    #     # Try to initialize accelerator - this will work if launched with 'accelerate launch'
    #     accelerator = Accelerator()
    #     device = accelerator.device
    #     total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    #     print(f"ACCELERATE DETECTED:")
    #     print(f"   • Total GPUs available: {total_gpus}")
    #     print(f"   • Accelerate processes: {accelerator.num_processes}")
    #     print(f"   • Current process index: {accelerator.process_index}")
    #     print(f"   • Current device: {device}")

    #     if accelerator.num_processes == 1 and total_gpus > 1:
    #         print(f"Using single process mode")
    #         print(f"   Use: accelerate launch --num_processes {total_gpus} script.py")
    #     elif accelerator.num_processes > 1:
    #         print(f"Multi-GPU setup: using {accelerator.num_processes} GPUs simultaneously")
    # else:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     print(f"Single GPU mode: {device} (accelerate not installed)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Single GPU mode: {device} (accelerate not installed)")

    print("Loading models...", flush=True)
    model_loading_start = time.time()

    # Memory optimization
    if torch.cuda.is_available():
        # Set PyTorch memory management for better fragmentation handling
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

        # Print GPU memory info
        if accelerator is not None:
            device_idx = accelerator.local_process_index
            total_mem = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
            reserved_mem = torch.cuda.memory_reserved(device_idx) / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(device_idx) / (1024**3)
            print(f"GPU {device_idx} Memory: {total_mem:.2f}GB total, {reserved_mem:.2f}GB reserved, {allocated_mem:.2f}GB allocated")

    # Load detector
    # if cur_args.body_detector == 'vitdet':
    if body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        cfg_path = f'configs/cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    # elif cur_args.body_detector == 'regnety':
    elif body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # sam2
    sam2_checkpoint = "./saved_models/sam2_models/sam2.1_hiera_small.pt"
    model_cfg       = "configs/sam2.1/sam2.1_hiera_s.yaml"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch.autocast(device_type='cuda', dtype=torch.float16).__enter__()

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
    # sam2 image model
    image_predictor = SAM2ImagePredictor(sam2)
    # sam video model
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    predictor.to(dtype=torch.float16)

    # Image cache for faster access
    image_cache = {}


    # Models
    print("Loading models...")

    qwen_model, qwen_processor = init_qwen_model(device)

    # VGGT model for camera motion detection
    vggt_model = None
    # if cur_args.enable_camera_motion_detection:
    if enable_camera_motion_detection:
        print("Loading VGGT model for camera motion detection...")
        vggt_model = init_vggt_model(device)
        if vggt_model is not None:
            print("VGGT model loaded - camera motion detection enabled")
        else:
            print("Camera motion detection disabled")
    else:
        print("Camera motion detection disabled")

    model_loading_time = time.time() - model_loading_start
    timing_stats['model_loading'] = model_loading_time
    print(f"Models loaded in {model_loading_time:.1f}s", flush=True)

    # video_dir  = cur_args.video_dir
    save_dir            = os.path.join(video_dir, 'video_general_obj_det_finished')
    finished_dir        = os.path.join(video_dir, 'ignore')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(finished_dir, exist_ok=True)

    # Load video list
    decode_dir = os.path.join(video_dir, 'Videos_crop_decode')
    print(f'Input directory: {decode_dir}')
    video_ls = glob.glob(f'{decode_dir}/*')

    random.seed(0)
    random.shuffle(video_ls)

    # Check if processing specific videos
    # if cur_args.video_names:
    if video_names:
        # Filter to only the specified videos
        # target_video_names = set(cur_args.video_names)
        target_video_names = set(video_names)
        videos_to_process = []
        found_videos = set()

        for video_path in video_ls:
            video_name = os.path.basename(video_path)
            if video_name in target_video_names:
                videos_to_process.append(video_path)
                found_videos.add(video_name)

        missing_videos = target_video_names - found_videos
        if missing_videos:
            print(f'WARNING: Videos not found in {decode_dir}: {list(missing_videos)}')

        if videos_to_process:
            print(f'Processing SPECIFIC videos: {len(videos_to_process)} videos from provided list')
            print(f'Videos to process: {[os.path.basename(v) for v in videos_to_process]}')
        else:
            print(f'ERROR: None of the specified videos found in {decode_dir}')
            print(f'Available videos: {[os.path.basename(v) for v in video_ls[:5]]}...')
            exit(1)
    # elif cur_args.single_video_name:
    elif single_video_name:
        # Filter to only the specified video
        target_video_path = None
        for video_path in video_ls:
            video_name = os.path.basename(video_path)
            # if video_name == cur_args.single_video_name:
            if video_name == single_video_name:
                target_video_path = video_path
                break

        if target_video_path:
            videos_to_process = [target_video_path]
            # print(f'Processing SINGLE video: {cur_args.single_video_name}')
            print(f'Processing SINGLE video: {single_video_name}')
        else:
            # print(f'ERROR: Video {cur_args.single_video_name} not found in {decode_dir}')
            print(f'ERROR: Video {single_video_name} not found in {decode_dir}')
            print(f'Available videos: {[os.path.basename(v) for v in video_ls[:5]]}...')
            exit(1)
    # elif cur_args.chunk_idx == -1:
    elif chunk_idx == -1:
        videos_to_process = video_ls
        print(f'Processing ALL {len(videos_to_process)} videos')
    else:
        # video_chunk_ls = chunk_into_n(video_ls, cur_args.chunk_num)
        # videos_to_process = video_chunk_ls[cur_args.chunk_idx]
        # print(f'Processing chunk {cur_args.chunk_idx}/{cur_args.chunk_num-1} ({len(videos_to_process)} videos)')
        video_chunk_ls = chunk_into_n(video_ls, chunk_num)
        videos_to_process = video_chunk_ls[chunk_idx]
        print(f'Processing chunk {chunk_idx}/{chunk_num-1} ({len(videos_to_process)} videos)')

    # Split videos across accelerator processes if using accelerate
    # if accelerator is not None and accelerator.num_processes > 1:
    #     total_videos = len(videos_to_process)
    #     videos_to_process = [videos_to_process[i] for i in range(accelerator.process_index, len(videos_to_process), accelerator.num_processes)]

    #     print(f'VIDEO DISTRIBUTION:')
    #     print(f'   • Total: {total_videos}, Per GPU: ~{total_videos // accelerator.num_processes}, GPU {accelerator.process_index}: {len(videos_to_process)}')

    #     # Show first few video names for verification
    #     if videos_to_process:
    #         print(f'   • Processing {len(videos_to_process)} videos on GPU {accelerator.process_index}')
    # else:
    #     print(f'Processing {len(videos_to_process)} videos on single GPU')

    skipped_videos = []
    skipped_reasons = {}

    total_video_processing_start = time.time()

    for video_idx, video_dir in enumerate(tqdm(videos_to_process)):
        video_start_time = time.time()
        video_name = video_dir.split('/')[-1].split('.')[0]

        if accelerator is not None and accelerator.num_processes > 1:
            print(f'GPU {accelerator.process_index}: Processing video {video_idx+1}/{len(videos_to_process)}: {video_name}', flush=True)
        else:
            print(f'Processing video {video_idx+1}/{len(videos_to_process)}: {video_name}', flush=True)

        # Optimized: reduce GPU memory clearing frequency
        # torch.cuda.empty_cache()



        out_path = os.path.join(save_dir, video_name+'.mp4')
        pkl_path = os.path.join(save_dir, video_name+'.json')

        # Check for any existing failure markers and merge if necessary
        finished_path = check_and_merge_existing_markers(finished_dir, video_name)

        # Skip if already processed successfully or marked as failed
        if os.path.exists(out_path):
            continue  # Successfully processed
        if os.path.exists(finished_path):
            continue  # Already marked as failed

        # Also check for camera motion marker specifically
        camera_motion_marker = os.path.join(finished_dir, f'{video_name}_camera_motion_skipped')
        if os.path.exists(camera_motion_marker):
            continue  # Already marked as camera motion skipped


        img_ls = glob.glob(f'{video_dir}/*.jpg')
        img_ls.sort()

        # if len(img_ls) > cur_args.max_video_frames:
        if len(img_ls) > max_video_frames:
            print(f'Skipping video: Too many frames ({len(img_ls)})')
            os.makedirs(finished_path, exist_ok=True)
            continue

        # Preload images for faster access (optimization)
        video_image_cache = preload_video_images(img_ls, max_preload=min(10, len(img_ls)))
        first_img = get_cached_image(img_ls[0], video_image_cache)
        img_height, img_width = first_img.shape[:2]

        # Camera Motion Detection
        camera_motion_start = time.time()
        # if cur_args.enable_camera_motion_detection and vggt_model is not None:
        if enable_camera_motion_detection and vggt_model is not None:
            should_skip, motion_score, skip_reason = should_skip_video_due_to_camera_motion(
                vggt_model, video_dir,
                # motion_threshold=cur_args.camera_motion_threshold,
                # sample_frames=cur_args.camera_motion_sample_frames
                motion_threshold=camera_motion_threshold,
                sample_frames=camera_motion_sample_frames
            )

            if should_skip:
                print(f'Skipping video: Camera motion')
                skipped_videos.append(video_name)
                skipped_reasons[video_name] = {
                    'reason': 'Camera motion detected',
                    'motion_score': motion_score,
                    # 'threshold': cur_args.camera_motion_threshold,
                    'threshold': camera_motion_threshold,
                    'detail': skip_reason
                }
                # Use camera_motion_skipped marker (more informative than generic)
                camera_motion_marker = os.path.join(finished_dir, f'{video_name}_camera_motion_skipped')
                os.makedirs(camera_motion_marker, exist_ok=True)
                with open(os.path.join(camera_motion_marker, 'skip_reason.txt'), 'w') as f:
                    f.write(f'Video skipped due to camera motion\n')
                    f.write(f'Motion score: {motion_score:.4f}\n')

                    # f.write(f'Threshold: {cur_args.camera_motion_threshold}\n')
                    f.write(f'Threshold: {camera_motion_threshold}\n')
                    f.write(f'Detail: {skip_reason}\n')

                # Update finished_path to point to the camera motion marker for consistency
                finished_path = camera_motion_marker
                continue
            else:
                print(f'Processing video')

        camera_motion_time = time.time() - camera_motion_start
        timing_stats['camera_motion_detection'] += camera_motion_time

        # Object Analysis and Detection
        object_analysis_start = time.time()

        # --- NEW ACTOR-CENTRIC ---
        print_step_header("Detecting Actors in first frame", 1)

        # who can act
        actor_classes = ["person", "humanoid_robot", "robotic arm"]
        update_color_palette_for_video(actor_classes)

        first_frame_objects = []
        if len(img_ls) > 0:
            # --- HUMAN-CENTRIC DETECTION WITH DETECTRON2 ---
            with torch.cuda.amp.autocast(enabled=False):
                d2_out = detector(first_img)

            det_instances = d2_out['instances']
            valid_human_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.70)
            human_bboxes = det_instances.pred_boxes.tensor[valid_human_idx].cpu().numpy().astype(np.float32)
            human_scores = det_instances.scores[valid_human_idx].cpu().numpy().astype(np.float32)

            del d2_out
            del det_instances
            torch.cuda.empty_cache()

            for bbox, score in zip(human_bboxes, human_scores):
                if get_bbox_area(bbox) > 500:
                    first_frame_objects.append({
                        'bbox': bbox.tolist(),
                        'class_name': 'person',
                        'class_id': 200,  # ID mappato per person
                        'confidence': float(score)
                    })


            # --- 2. ROBOT DETECTION WITH QWEN2.5-VL ---
            with torch.cuda.amp.autocast(enabled=False):
                r_bboxes, r_phrases, r_scores = detect_objects_with_qwen2_5_vl(
                    qwen_model, qwen_processor, first_img,
                    "humanoid robot, robotic arm and autonomous mobile robot"
                )

            torch.cuda.empty_cache()

            for bbox, phrase, score in zip(r_bboxes, r_phrases, r_scores):
                name = phrase.strip().lower()
                if get_bbox_area(bbox) > 500 and ("robot" in name or "arm" in name):
                    class_id = 201 if "humanoid" in name or "robot" in name else 202
                    first_frame_objects.append({
                        'bbox': bbox,
                        'class_name': name,
                        'class_id': class_id,
                        'confidence': float(score)
                    })

            first_frame_objects = filter_and_deduplicate_objects(
                first_frame_objects, confidence_threshold=0.40, iou_threshold=0.45
            )

            for act in first_frame_objects:
                print(f"  -> {act['class_name']} (Conf: {act['confidence']:.2f}) at {act['bbox']}")

        if not first_frame_objects:
            os.makedirs(finished_path, exist_ok=True)
            continue

        object_analysis_time = time.time() - object_analysis_start
        timing_stats['object_analysis_detection'] += object_analysis_time

        # FINISH NEW ACTOR-CENTRIC

        out_dir = os.path.join(save_dir, video_name)
        os.makedirs(out_dir, exist_ok=True)

        # SAM2 propagation
        sam2_propagation_start = time.time()

        # Check if we have any objects to track
        has_objects_to_track = len(first_frame_objects) > 0

        torch.cuda.empty_cache()

        if not has_objects_to_track:
            print("No humans or objects to track, skipping SAM2 propagation")
            video_segments = {}  # Empty dictionary for consistency
            global_object_id_to_class = {} # Inizializza vuoto per evitare NameError successivi
        else:
            print_step_header("SAM2 video propagation", 5)

            global_object_id_to_class = {} # Inizializzazione dizionario

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                video_segments = {}
                inference_state = predictor.init_state(video_path=video_dir,
                                                 offload_video_to_cpu=True,
                                                 offload_state_to_cpu=True,
                                                 async_loading_frames=False)

                for obj_idx, obj_data in enumerate(first_frame_objects):
                    obj_id = 1000 + obj_idx
                    global_object_id_to_class[obj_id] = obj_data['class_name']
                    bbox_tensor = torch.tensor(obj_data['bbox'], dtype=torch.float32, device='cpu')
                    
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        box=bbox_tensor
                    )

                # PROPAGAZIONE OTTIMIZZATA PER LA RAM
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    frame_boxes = {}
                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                        # Convertiamo SUBITO in bounding box e scartiamo la maschera gigante
                        boxes = sv.mask_to_xyxy(mask)
                        if len(boxes) > 0:
                            frame_boxes[out_obj_id] = list(boxes[0])
                            
                    video_segments[out_frame_idx] = frame_boxes
                
                # LIBERIAMO LA RAM DI SISTEMA CANCELLANDO LO STATO
                predictor.reset_state(inference_state)
                del inference_state
                import gc
                gc.collect()

        sam2_propagation_time = time.time() - sam2_propagation_start
        timing_stats['sam2_propagation'] += sam2_propagation_time

        # Reformatting and Visualization
        reformatting_start = time.time()
        print_step_header("Reformatting and visualization", 6)

        # Calcolo FPS con fallback obbligatorio prima dell'inizializzazione del writer
        original_fps = get_original_video_fps(video_name, video_dir.replace('/Videos_crop_decode/' + video_name, ''))
        if original_fps is None:
            frame_count = len(img_ls)
            original_fps = calculate_fps_from_frames(frame_count, estimated_duration=5.0)
            print(f"Using calculated FPS: {original_fps:.2f}")
        else:
            print(f"Using original video FPS: {original_fps:.2f}")

        writer = imageio.get_writer(out_path, fps=original_fps, codec='libx264')

        motion = {'detected_objects': {}}

        for i, obj_data in enumerate(first_frame_objects):
            obj_id = 1000 + i
            obj_key = f"{obj_data['class_name']}_{obj_id}"

            first_bbox = obj_data['bbox']

            x1, y1, x2, y2 = first_bbox
            bbox_norm_first = [
                round(x1/img_width, 4), round(y1/img_height, 4),
                round(x2/img_width, 4), round(y2/img_height, 4)
            ]

            motion['detected_objects'][obj_key] = {
                'class_name': obj_data['class_name'],
                'class_id': obj_data['class_id'],
                'track_id': obj_id,
                'bbox': [bbox_norm_first]
            }

        for i_idx, img_path in tqdm(enumerate(img_ls)):
            img_name = img_path.split('/')[-1]

            img_cv2 = get_cached_image(img_path, video_image_cache)

            if i_idx not in video_segments:
                if i_idx > 0:
                    for obj_key in list(motion['detected_objects'].keys()):
                        motion['detected_objects'][obj_key]['bbox'].append(None)
            else:
                segments = video_segments[i_idx]
                if i_idx > 0:
                    for obj_key in list(motion['detected_objects'].keys()):
                        motion['detected_objects'][obj_key]['bbox'].append(None)

                for obj_id in segments:
                    if obj_id >= 1000:
                        bbox = segments[obj_id]
                        bbox_area = get_bbox_area(bbox)
                        if bbox_area > 50:
                            class_name = global_object_id_to_class.get(obj_id)
                            if class_name is None:
                                continue

                            x1, y1, x2, y2 = bbox
                            bbox_norm = [round(x1/img_width, 4), round(y1/img_height, 4), round(x2/img_width, 4), round(y2/img_height, 4)]

                            obj_key = f"{class_name}_{obj_id}"
                            if obj_key in motion['detected_objects']:
                                if i_idx != 0:
                                    motion['detected_objects'][obj_key]['bbox'][-1] = bbox_norm
                            else:
                                motion['detected_objects'][obj_key] = {
                                    'class_name': class_name,
                                    'class_id': OBJECTS_CLASS_ID.get(class_name, 999),
                                    'track_id': obj_id,
                                    'bbox': [None] * i_idx + [bbox_norm]
                                }

            annotated_frame = img_cv2.copy()
            drawing_objects = []

            for obj_key, obj_data in motion['detected_objects'].items():
                current_bbox_norm = obj_data['bbox'][i_idx] if i_idx < len(obj_data['bbox']) else None

                if current_bbox_norm is not None:
                    x1_norm, y1_norm, x2_norm, y2_norm = current_bbox_norm
                    x1 = x1_norm * img_width
                    y1 = y1_norm * img_height
                    x2 = x2_norm * img_width
                    y2 = y2_norm * img_height

                    bbox_abs = [x1, y1, x2, y2]
                    class_name = obj_data['class_name']
                    drawing_objects.append((class_name, bbox_abs))

            for (name, bbox) in drawing_objects:
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                    continue

                class_id = OBJECTS_CLASS_ID.get(name, 999)
                detections = sv.Detections(
                    xyxy=np.array(bbox)[None, :],
                    class_id = np.array([class_id]),
                )

                box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(HEX_COLOR_PALETTE))
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(HEX_COLOR_PALETTE), smart_position=True)
                annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[name])

            # Scrivi lo streaming dei dati direttamente su disco per preservare la RAM di sistema
            writer.append_data(annotated_frame[:, :, ::-1])
            cv2.imwrite(os.path.join(out_dir, img_name), annotated_frame)

            del annotated_frame
            del img_cv2

        writer.close()
        reformatting_time = time.time() - reformatting_start
        timing_stats['reformatting_visualization'] += reformatting_time

        # Save results (Step 7)
        saving_start = time.time()
        print_step_header("Saving results", 7)

        with open(pkl_path, 'w') as f:
            json.dump(motion, f, indent=4, cls=NpEncoder)

        saving_time = time.time() - saving_start
        timing_stats['saving_results'] += saving_time
        # Update timing stats
        video_total_time = time.time() - video_start_time
        timing_stats['total_video_processing'] += video_total_time

        if accelerator is not None and accelerator.num_processes > 1:
            print(f"GPU {accelerator.process_index}: Video {video_name} completed in {video_total_time:.1f}s", flush=True)
        else:
            print(f"Video {video_name} completed in {video_total_time:.1f}s", flush=True)

        # Aggressive memory cleanup for Qwen-72B multi-GPU compatibility
        if 'video_image_cache' in locals():
            video_image_cache.clear()
            del video_image_cache
            
        if 'motion' in locals():
            motion.clear()
            del motion
            
        if 'video_segments' in locals():
            video_segments.clear()
            del video_segments
            
        if 'inference_state' in locals():
            try:
                predictor.reset_state(inference_state)
            except:
                pass
            del inference_state
            
        if 'first_img' in locals():
            del first_img

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import ctypes
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


    # Final summary
    total_script_time = time.time() - script_start_time
    total_videos = len(videos_to_process)
    processed_videos = total_videos - len(skipped_videos)

    if accelerator is not None and accelerator.num_processes > 1:
        print(f'\nGPU {accelerator.process_index} FINAL SUMMARY:')
        print(f'   • Time: {total_script_time/60:.1f}m, Processed: {processed_videos}/{total_videos}')
        if processed_videos > 0:
            avg_time = timing_stats["total_video_processing"]/processed_videos
            print(f'   • Avg time: {avg_time:.1f}s')
        if skipped_videos:
            print(f'   • Skipped: {len(skipped_videos)} videos')

        # Wait for all processes to complete and show total stats on main process
        if accelerator.process_index == 0:
            print(f'\nALL GPU PROCESSING COMPLETED!')
            print(f'   • Total GPUs used: {accelerator.num_processes}')
    else:
        print(f'\nPROCESSING COMPLETED:')
        print(f'   • Time: {total_script_time/60:.1f}m, Processed: {processed_videos}/{total_videos}')
        if processed_videos > 0:
            avg_time = timing_stats["total_video_processing"]/processed_videos
            print(f'   • Avg time: {avg_time:.1f}s')
        if skipped_videos:
            print(f'   • Skipped: {len(skipped_videos)} videos')
