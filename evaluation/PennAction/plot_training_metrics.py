import re
import matplotlib.pyplot as plt
import os

def parse_training_logs(file_path):
    """
    Parses the YOWOv2 training log file to extract mAP per epoch for each dataset subset.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Log file '{file_path}' not found.")
        return None

    training_data = {}
    
    current_subset = None
    current_map = None

    subset_pattern = re.compile(r"checkpoints_(\d+(?:_\d+)?)/?'")
    map_pattern = re.compile(r"mAP:\s+([0-9.]+)")
    epoch_pattern = re.compile(r"Saving state,\s+epoch:\s+(\d+)")

    print(f"[INFO] Parsing log file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            
            subset_match = subset_pattern.search(line)
            if subset_match:
                subset_str = subset_match.group(1).replace('_', '.')
                current_subset = float(subset_str)
                
                if current_subset not in training_data:
                    training_data[current_subset] = {}
                    print(f"[INFO] Found new training run for dataset subset: {current_subset}%")
                continue

            map_match = map_pattern.search(line)
            if map_match:
                current_map = float(map_match.group(1))
                continue

            epoch_match = epoch_pattern.search(line)
            if epoch_match and current_map is not None and current_subset is not None:
                epoch = int(epoch_match.group(1))
                
                training_data[current_subset][epoch] = current_map
                
                current_map = None

    print("[INFO] Parsing completed successfully.\n")
    return training_data

def plot_map_vs_epochs(data):
    """
    Generates Graph 1: Epochs on X-axis, mAP on Y-axis. 
    One line for each dataset subset.
    """
    plt.figure(figsize=(10, 6))

    # Sort subsets to ensure the legend is ordered
    for subset in sorted(data.keys()):
        epochs_dict = data[subset]
        
        epochs = sorted(epochs_dict.keys())
        map_values = [epochs_dict[e] for e in epochs]
        
        # Plot the curve
        plt.plot(epochs, map_values, marker='o', linewidth=2, label=f"{subset}% Dataset")

    plt.title("Model Accuracy (mAP@0.5) vs. Training Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("mAP @ 0.5 IOU", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Dataset Split", fontsize=10)
    plt.tight_layout()
    
    output_filename = "plot_map_vs_epochs.png"
    plt.savefig(output_filename, dpi=300)
    print(f"[INFO] Saved Graph 1 as '{output_filename}'")
    plt.show()

def plot_final_map_vs_dataset(data):
    """
    Generates Graph 2: Dataset Percentage on X-axis, Final Epoch mAP on Y-axis.
    """
    plt.figure(figsize=(8, 5))

    percentages = sorted(data.keys())
    final_maps = []

    for p in percentages:
        final_epoch = max(data[p].keys())
        final_map = data[p][final_epoch]
        final_maps.append(final_map)
        print(f"[DATA] Subset {p:5.2f}% -> Final Epoch: {final_epoch} -> mAP: {final_map:.4f}")

    # Plot the curve
    plt.plot(percentages, final_maps, marker='s', color='darkblue', linestyle='-', linewidth=2, markersize=8)
    
    plt.title("Final Accuracy (mAP@0.5) vs. Dataset Growth", fontsize=14, fontweight='bold')
    plt.xlabel("Dataset Subset (%)", fontsize=12)
    plt.ylabel("Final Epoch mAP @ 0.5 IOU", fontsize=12)
    
    plt.xticks(percentages)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_filename = "plot_final_map_vs_dataset.png"
    plt.savefig(output_filename, dpi=300)
    print(f"[INFO] Saved Graph 2 as '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    log_file = "/home/ludovico/workspace/AA-STAL/evaluation/PennAction/whole_training - Copia.txt"
    
    parsed_data = parse_training_logs(log_file)
    
    if parsed_data:
        plot_map_vs_epochs(parsed_data)
        plot_final_map_vs_dataset(parsed_data)
        print("[INFO] All tasks finished.")