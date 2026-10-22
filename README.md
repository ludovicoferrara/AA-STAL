# An Auto-Annotation Pipeline for Action Detection

This repository contains the code and resources for the master's thesis work "An Auto-Annotation Pipeline for Action Detection". It implements an offline, zero-shot auto-annotation pipeline that processes raw videos to produce datasets with bounding boxes and labels for human actions and object interactions. The primary goal is to automatically generate datasets in the AVA format to train or fine-tune lightweight, real-time Action Detection (AD) models, such as YOWOv2.

## Pipeline Architecture

The pipeline consists of the following sequential stages:
1. **Data Pre-processing**: Raw videos are segmented based on shot changes, and 30 Frames Per Second (FPS) are extracted. Unsuitable or blurry clips are discarded using the VGGT model.
2. **Object Detection and Tracking**: The pipeline uses Detectron2 to detect human actors and GroundingDINO to detect objects. SAM2 is then utilized to track these detections across temporal windows.
3. **Person–Object Association**: A standalone geometric stage associates tracked objects with people using bounding-box containment, proximity, and temporal consistency. The output preserves frame-level evidence and window-level associations without modifying the detection results.
4. **Action Recognition**: The Qwen3-VL-8B-Instruct model (loaded in 4-bit precision) assigns action labels to the tracked humans and uses only the objects associated with the target actor.
5. **Output Production**: The processed data is formatted into an AVA-like dataset structure. Additionally, the pipeline supports generating event logs suitable for process mining applications.

## Repository Structure

The repository is organized into three main directories:

*   **`data_pipeline/`**: Contains utility files and model weights intended to be loaded onto Google Drive. 
    *   Must contain a manually created **`DATA_ROOT/`** folder where the raw input videos will be placed.
    *   **`jupiter_notebooks/`**: Contains the execution flow of the pipeline:
        *   `pre_processing.ipynb`: Handles video cropping and frame extraction.
        *   `obj_det_w_dino.ipynb`: Executes human/object detection and SAM2 tracking.
        *   `person_object_association.ipynb`: Associates tracked objects with people from the detection JSON files; it does not rerun detection or tracking.
        *   `action_labeling.ipynb`: Runs the Qwen3-VL action recognition inference.
        *   `ava_formatting_and_training.ipynb`: Formats the final annotations into the AVA structure and contains the commands to launch the YOWOv2 training and evaluation.
        *   `process_mining_formatting.ipynb`: Adapts the pipeline output into event logs for process mining tasks.
*   **`evaluation/`**: Contains scripts to evaluate the direct results and metrics (e.g., IoU, F1-score) of the auto-annotation pipeline against a ground-truth dataset.
*   **`YOWOv2_finetuning/YOWOv2/`**: Contains the YOWOv2 repository, modified specifically to support this fine-tuning use case and single-actor dataset constraints. Functions from this directory are called via the `ava_formatting_and_training.ipynb` notebook.

## Environment and Execution

The entire workflow has been designed and tested on **Google Colab**, utilizing Google Drive for storage and file management. 

To replicate the experiment on the ([Penn Action Dataset](https://dreamdragon.github.io/PennAction/)):
1. Clone this repository into your Google Drive environment.
2. Create the `DATA_ROOT` directory inside `data_pipeline/` and upload the dataset.
3. Open the files in `data_pipeline/jupiter_notebooks/` using Google Colab.
4. Execute the notebooks sequentially following the pipeline architecture (Pre-processing -> Object Detection -> Person–Object Association -> Action Labeling -> Formatting & Training).

The association stage can also be run without Jupyter:

```bash
python data_pipeline/person_object_association.py \
  data_pipeline/DATA_ROOT/video_general_obj_det_finished \
  data_pipeline/DATA_ROOT/person_object_associations
```

The scoring method and JSON contracts are documented in [`docs/person_object_association_spec.md`](docs/person_object_association_spec.md).

The final cell of `person_object_association.ipynb` generates sampled diagnostic frames under `DATA_ROOT/person_object_association_visualizations/`. People are drawn in red, temporally accepted objects in green, and each accepted association is represented by a yellow line and its frame-level score. The visualization step never modifies the original frames.

## Reference
This work is part of a Master's Degree in Computer Science at the University of Camerino (UNICAM) by Ludovico Ferrara.

## Main External References
```
@article{yang2023yowov2,
  title={YOWOv2: A Stronger yet Efficient Multi-level Detection Framework for Real-time Spatio-temporal Action Detection},
  author={Yang, Jianhua and Kun, Dai},
  journal={arXiv preprint arXiv:2302.06848},
  year={2023}
}
```

```
@INPROCEEDINGS{pennaction,
  author={Zhang, Weiyu and Zhu, Menglong and Derpanis, Konstantinos G.},
  booktitle={2013 IEEE International Conference on Computer Vision}, 
  title={From Actemes to Action: A Strongly-Supervised Representation for Detailed Action Understanding}, 
  year={2013},
  volume={},
  number={},
  pages={2248-2255},
  keywords={Spatiotemporal phenomena;Training;Trajectory;Visualization;Semantics;Cameras;Context;action classification;action detection},
  doi={10.1109/ICCV.2013.280}}
  ```
