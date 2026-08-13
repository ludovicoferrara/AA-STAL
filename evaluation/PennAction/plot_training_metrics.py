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

    # Dictionary to store the data: { subset_percentage: { epoch: mAP_value, ... }, ... }
    # Example: { 6.25: { 1: 0.082, 2: 0.159, ... }, 12.5: { ... } }
    training_data = {}
    
    current_subset = None
    current_map = None

    # Regex patterns based on your specific YOWOv2 output format
    # Matches: save_folder='/.../checkpoints_6_25/' -> Extracts '6_25'
    subset_pattern = re.compile(r"checkpoints_(\d+(?:_\d+)?)/?'")
    # Matches: mAP: 0.08269542123732236
    map_pattern = re.compile(r"mAP:\s+([0-9.]+)")
    # Matches: Saving state, epoch: 1
    epoch_pattern = re.compile(r"Saving state,\s+epoch:\s+(\d+)")

    print(f"[INFO] Parsing log file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            
            # 1. Detect dataset subset percentage
            subset_match = subset_pattern.search(line)
            if subset_match:
                # Convert '6_25' to 6.25, '25' to 25.0
                subset_str = subset_match.group(1).replace('_', '.')
                current_subset = float(subset_str)
                
                if current_subset not in training_data:
                    training_data[current_subset] = {}
                    print(f"[INFO] Found new training run for dataset subset: {current_subset}%")
                continue

            # 2. Extract mAP evaluation value
            map_match = map_pattern.search(line)
            if map_match:
                current_map = float(map_match.group(1))
                continue

            # 3. Extract epoch number and link it to the previously found mAP
            epoch_match = epoch_pattern.search(line)
            if epoch_match and current_map is not None and current_subset is not None:
                epoch = int(epoch_match.group(1))
                
                # Assign mAP to the specific epoch. 
                # If a run crashed and restarted (like your 100% run), it simply overwrites the old keys.
                training_data[current_subset][epoch] = current_map
                
                # Reset current_map to avoid double-logging
                current_map = None

    print("[INFO] Parsing completed successfully.\n")
    return training_data

def plot_map_vs_epochs(data):
    """
    Generates Graph 1: Epochs on X-axis, mAP on Y-axis. 
    One line for each dataset subset.
    """
    plt.figure(figsize=(10, 6))

    # Sort subsets to ensure the legend is ordered (6.25 -> 100)
    for subset in sorted(data.keys()):
        epochs_dict = data[subset]
        
        # Sort epochs just in case they were parsed out of order
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
        # Get the maximum epoch number for the current subset (the final epoch)
        final_epoch = max(data[p].keys())
        final_map = data[p][final_epoch]
        final_maps.append(final_map)
        print(f"[DATA] Subset {p:5.2f}% -> Final Epoch: {final_epoch} -> mAP: {final_map:.4f}")

    # Plot the curve
    plt.plot(percentages, final_maps, marker='s', color='darkblue', linestyle='-', linewidth=2, markersize=8)
    
    plt.title("Final Accuracy (mAP@0.5) vs. Dataset Growth", fontsize=14, fontweight='bold')
    plt.xlabel("Dataset Subset (%)", fontsize=12)
    plt.ylabel("Final Epoch mAP @ 0.5 IOU", fontsize=12)
    
    # Ensure X-axis ticks exactly match the dataset percentages
    plt.xticks(percentages)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_filename = "plot_final_map_vs_dataset.png"
    plt.savefig(output_filename, dpi=300)
    print(f"[INFO] Saved Graph 2 as '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    # Define the input log file
    log_file = "/home/ludovico/workspace/AA-STAL/evaluation/PennAction/whole_training_log.txt"
    
    # Parse the data
    parsed_data = parse_training_logs(log_file)
    
    if parsed_data:
        # Generate the two requested plots
        plot_map_vs_epochs(parsed_data)
        plot_final_map_vs_dataset(parsed_data)
        print("[INFO] All tasks finished.")