import os
import glob
import subprocess
import argparse
from tqdm import tqdm
from scenedetect import detect, ContentDetector

def get_video_duration(video_path):
    """Ottiene la durata totale del video in secondi tramite ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def crop_video(video_path, output_path, start_time, duration):
    """Estrae un segmento video ricodificandolo per garantire keyframe precisi."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]
    subprocess.run(cmd)

def chunk_into_n(lst, n):
    """Suddivide la lista in n chunk."""
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
        
        # 1. Rilevamento delle scene
        try:
            raw_scenes = detect(video_path, ContentDetector())
        except Exception as e:
            print(f"Errore di lettura scenedetect per {video_name}: {e}")
            raw_scenes = []

        scenes_sec = []
        
        # 2. Gestione caso video continuo vs video con tagli
        if not raw_scenes:
            # Nessun taglio rilevato: tratta il video intero come un'unica scena
            total_duration = get_video_duration(video_path)
            if total_duration > 0:
                scenes_sec = [(0.0, total_duration)]
                print(f"Video {video_name}: Nessun taglio. Trattato come scena singola.")
            else:
                print(f"Video {video_name}: File corrotto o illeggibile.")
                continue
        else:
            # Converti i frame timecode di scenedetect in secondi
            scenes_sec = [(s[0].seconds, s[1].seconds) for s in raw_scenes]
            print(f"Video {video_name}: trovate {len(scenes_sec)} scene.")

        # 3. Estrazione per ogni scena identificata
        valid_scenes = 0
        for i, (start_sec, end_sec) in enumerate(scenes_sec):
            duration = end_sec - start_sec
            
            final_start = start_sec
            final_duration = duration
                    
            valid_scenes += 1
            output_name = f"{base_name}_scene{i+1}.mp4"
            output_path = os.path.join(output_dir, output_name)
            
            crop_video(video_path, output_path, final_start, final_duration)
            
        print(f"  -> Salvate {valid_scenes} clip valide su {len(scenes_sec)} totali.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop videos based on scene detection and temporal thresholds')
    parser.add_argument('--in_dir', type=str, default='/path/to/Videos', help='Directory input video')
    parser.add_argument('--out_dir', type=str, default='/path/to/Videos_crop', help='Directory output video')
    parser.add_argument('--chunk_idx', type=int, default=None, help='Indice del chunk')
    parser.add_argument('--chunk_num', type=int, default=None, help='Numero totale di chunk')
    args = parser.parse_args()
    
    process_videos(args.in_dir, args.out_dir, args.chunk_idx, args.chunk_num)
    print(f"Elaborazione completata. Video salvati in: {args.out_dir}")