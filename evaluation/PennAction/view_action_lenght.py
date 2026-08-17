import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = "/home/ludovico/workspace/AA-STAL/data_pipeline/DATA_ROOT"
GT_DIR = os.path.join(DATA_ROOT, "groundtruth/PennAction")

def main():
    gt_files = glob.glob(os.path.join(GT_DIR, "*.csv"))
    
    if not gt_files:
        print("Nessun file CSV trovato nella directory specificata.")
        return

    print(f"Analisi di {len(gt_files)} video in corso...")

    action_lengths = {}

    for gt_path in gt_files:
        try:
            df = pd.read_csv(gt_path)
            if df.empty:
                continue
            
            action = df['action'].iloc[0].strip().lower()
            
            num_frames = len(df)
            
            if action not in action_lengths:
                action_lengths[action] = []
                
            action_lengths[action].append(num_frames)
            
        except Exception as e:
            print(f"Errore nella lettura di {gt_path}: {e}")

    actions = []
    avg_frames = []
    
    print("\nRisultati numerici:")
    print("-" * 40)
    for action, lengths in action_lengths.items():
        avg = np.mean(lengths)
        actions.append(action)
        avg_frames.append(avg)
        print(f"{action.ljust(20)}: {avg:.1f} frame medi (su {len(lengths)} video)")
    print("-" * 40)

    # PLOTTING
    sorted_indices = np.argsort(avg_frames)
    actions_sorted = [actions[i] for i in sorted_indices]
    avg_frames_sorted = [avg_frames[i] for i in sorted_indices]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(actions_sorted, avg_frames_sorted, color='steelblue')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, 
                 bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}', 
                 ha='left', 
                 va='center')

    plt.xlabel('Avg number of frames', fontsize=12)
    plt.ylabel('Action', fontsize=12)
    plt.title('Average Length of Videos per Action (PennAction)', fontsize=14, pad=15)
    
    global_mean = np.mean([l for lengths in action_lengths.values() for l in lengths])
    plt.axvline(global_mean, color='red', linestyle='--', alpha=0.7, label=f'Global Average ({global_mean:.1f})')
    plt.legend()

    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    output_img = "frame_medi_per_azione.png"
    plt.savefig(output_img, dpi=300)
    print(f"\nGrafico salvato con successo in: {os.path.abspath(output_img)}")

if __name__ == "__main__":
    main()