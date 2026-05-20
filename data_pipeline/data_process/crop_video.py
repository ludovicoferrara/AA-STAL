import os
import glob
import subprocess
import argparse
from tqdm import tqdm
from scenedetect import detect, ContentDetector

def crop_video(video_path, output_path, start_time, duration):
    """Estrae un segmento video ricodificandolo per garantire keyframe precisi"""
    cmd = [
        "ffmpeg",
        "-y",  # Sovrascrive automaticamente i file esistenti
        "-loglevel", "error",
        "-ss", str(start_time),  # Posizionato prima dell'input per fast-seeking
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd)

def chunk_into_n(lst, n):
    """Suddivide la lista in n chunk"""
    chunk_size = len(lst) // n
    remainder = len(lst) % n
    
    chunks = []
    start = 0
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    
    return chunks

def process_videos(input_dir, output_dir, chunk_idx=None, chunk_num=None):
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    video_files.extend(glob.glob(os.path.join(input_dir, "*.avi")))
    video_files.extend(glob.glob(os.path.join(input_dir, "*.mov")))
    video_files.sort()
    
    if chunk_idx is not None and chunk_num is not None:
        video_chunks = chunk_into_n(video_files, chunk_num)
        if chunk_idx >= len(video_chunks):
            print(f"Errore: Chunk {chunk_idx} fuori range (totale: {len(video_chunks)})")
            return
        video_files = video_chunks[chunk_idx]
        print(f"Elaborazione chunk {chunk_idx}/{chunk_num-1}: {len(video_files)} video")
    else:
        print(f"Trovati {len(video_files)} video da elaborare")
    
    for video_path in tqdm(video_files):
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        
        # 1. Rilevamento dei cambi di inquadratura (Cut detection)
        # ContentDetector(threshold=27.0) è il valore di default ottimale
        scene_list = detect(video_path, ContentDetector())
        
        if not scene_list:
            print(f"Nessuna scena rilevata in {video_name} o file illeggibile.")
            continue
            
        print(f"Video {video_name}: trovate {len(scene_list)} scene.")
        
        # 2. Filtraggio temporale ed estrazione
        valid_scenes = 0
        for i, scene in enumerate(scene_list):
            start_sec = scene[0].seconds
            end_sec = scene[1].seconds
            duration = end_sec - start_sec
            
            # Regola 1: Scarta se minore di 3 secondi
            if duration < 3.0:
                continue
                
            # Regola 2: Conserva integralmente se tra 3 e 10 secondi
            elif duration <= 10.0:
                final_start = start_sec
                final_duration = duration
                
            # Regola 3: Estrai i 6 secondi centrali se maggiore di 10 secondi
            else:
                center_time = start_sec + (duration / 2.0)
                final_start = center_time - 3.0
                final_duration = 6.0
                
            valid_scenes += 1
            output_name = f"{base_name}_scene{i+1}.mp4"
            output_path = os.path.join(output_dir, output_name)
            
            crop_video(video_path, output_path, final_start, final_duration)
            
        print(f"  -> Salvate {valid_scenes} clip valide su {len(scene_list)} totali.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop videos based on scene detection and temporal thresholds')
    parser.add_argument('--in_dir', type=str, default='/path/to/Videos', help='Directory input video')
    parser.add_argument('--out_dir', type=str, default='/path/to/Videos_crop', help='Directory output video')
    parser.add_argument('--chunk_idx', type=int, default=None, help='Indice del chunk')
    parser.add_argument('--chunk_num', type=int, default=None, help='Numero totale di chunk')
    args = parser.parse_args()
    
    process_videos(args.in_dir, args.out_dir, args.chunk_idx, args.chunk_num)
    print(f"Elaborazione completata. Video salvati in: {args.out_dir}")