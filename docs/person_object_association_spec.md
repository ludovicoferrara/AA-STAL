# Person–Object Association Module

## 1. Purpose

This specification defines a standalone pipeline stage that associates tracked objects with tracked people using only two-dimensional bounding-box geometry and temporal consistency.

The module is intended to provide a simple, inexpensive, and measurable baseline. It must remain independent from object detection, object tracking, action recognition, and depth estimation so that the association method can be evaluated and replaced without rerunning the upstream models.

The module answers the following question:

> Given the tracked people and objects visible in a video, which object is spatially associated with each person during each frame and temporal window?

An association is a geometric hypothesis. It does not by itself prove physical contact or manipulation.

## 2. Scope

### 2.1 In scope

- Read the tracking JSON produced by the existing object-detection and SAM2 tracking stage.
- Identify person tracks and non-person object tracks.
- Compute frame-level person–object association scores from bounding boxes.
- Resolve competition when the same object is a candidate for multiple people.
- Aggregate frame-level evidence into temporal windows.
- Save frame-level and window-level associations in a separate JSON file.
- Preserve score components and configuration metadata for evaluation and reproducibility.

### 2.2 Out of scope

- Object detection or re-identification.
- Modification of SAM2 tracks.
- Action recognition.
- Vision-language model inference.
- Depth estimation.
- Hand or body keypoint estimation.
- Definitive physical-contact classification.

These capabilities may be introduced later without changing the upstream detection output.

## 3. Pipeline position

The module must be implemented as an independent stage:

```text
pre_processing.ipynb
        ↓
obj_det_w_dino.ipynb
        ↓
person_object_association.ipynb
        ↓
action_labeling.ipynb
        ↓
process_mining_formatting.ipynb
```

The object-detection notebook remains responsible only for identifying and tracking entities. The new module reads its output and does not modify it.

## 4. Input contract

### 4.1 Source

One tracking JSON file per video, as currently generated under `video_general_obj_det_finished/<video_id>/`.

### 4.2 Required structure

```json
{
  "detected_objects": {
    "person_1000": {
      "class_name": "person",
      "class_id": 0,
      "track_id": 1000,
      "bbox": [
        [0.10, 0.08, 0.55, 0.96],
        [0.11, 0.08, 0.56, 0.96],
        null
      ]
    },
    "cup_1004": {
      "class_name": "cup",
      "class_id": 4,
      "track_id": 1004,
      "bbox": [
        [0.42, 0.38, 0.48, 0.49],
        [0.43, 0.38, 0.49, 0.49],
        null
      ]
    }
  }
}
```

Bounding boxes use normalized coordinates in `[x1, y1, x2, y2]` format. A `null` entry means that the track is not available in that frame.

### 4.3 Validation

The module must:

- reject malformed bounding boxes where `x1 >= x2` or `y1 >= y2`;
- clamp small numerical deviations to the `[0, 1]` interval;
- treat missing or invalid boxes as absent observations;
- verify that all tracks use the same frame-axis length, or safely pad shorter tracks with `null`;
- identify people by `class_name == "person"`, not only by the dictionary-key prefix.

## 5. Frame-level association

### 5.1 Candidate generation

For each frame:

1. Collect all valid person boxes.
2. Collect all valid non-person object boxes.
3. Expand each person box by a configurable fraction on all sides.
4. Evaluate every geometrically plausible person–object pair.

The default person-box expansion is `10%` of its width and height, clipped to the image boundaries.

Pairs may be discarded before scoring if the shortest distance between the two boxes exceeds the configured maximum distance.

### 5.2 Intersection over Object Area

Standard IoU is unsuitable as the primary measure because manipulated objects are generally much smaller than people. The primary overlap metric is Intersection over Object Area (`IoO`):

```text
IoO(person, object) = intersection_area(expanded_person_box, object_box)
                      / area(object_box)
```

Properties:

- `IoO = 1.0` when the object is fully contained in the expanded person box.
- `IoO = 0.0` when the boxes do not overlap.
- A small object can receive a high score even when standard IoU is low.

The implementation must save this component independently as `overlap_score`.

### 5.3 Proximity score

To support objects immediately outside the person box, such as a racket held by an extended arm, calculate the shortest Euclidean distance between the two rectangles.

Let `d` be the shortest box-to-box distance and `diag_p` the diagonal of the unexpanded person box. Define:

```text
normalized_distance = d / max(diag_p, epsilon)
proximity_score = max(0, 1 - normalized_distance / max_distance_ratio)
```

The default `max_distance_ratio` is `0.25`. Therefore, the proximity score becomes zero when the boxes are separated by more than one quarter of the person's bounding-box diagonal.

Intersecting boxes have distance zero and therefore `proximity_score = 1.0`.

### 5.4 Combined score

The default frame-level score is:

```text
association_score = 0.75 × overlap_score + 0.25 × proximity_score
```

Default eligibility rules:

```text
association_score >= 0.30
and
(overlap_score > 0 or proximity_score > 0)
```

All weights and thresholds must be configurable and written to the output metadata.

### 5.5 Multi-person conflict resolution

An object may be associated with at most one person in a frame in the baseline implementation.

If multiple people produce eligible scores for the same object:

1. Assign the object to the person with the highest `association_score`.
2. If scores are equal within a configurable tolerance, choose the person with the higher `overlap_score`.
3. If the tie remains, choose the person with the smallest track ID to guarantee deterministic output.

A person may be associated with multiple objects in the same frame.

Objects for which no candidate reaches the threshold must remain explicitly unassociated rather than being forced onto the closest person.

## 6. Temporal aggregation

### 6.1 Window definition

For compatibility with the current action-recognition stage, the default windows are:

```text
window_size = 60 frames
stride = 30 frames
```

If a video contains no more than 60 frames, create one window covering the entire video. Window end indices must be represented consistently as exclusive bounds: `[start_frame, end_frame)`.

### 6.2 Pair aggregation

For each person–object pair in a window, calculate:

- `valid_pair_frames`: frames in which both tracks have valid boxes;
- `associated_frames`: frames in which the pair wins frame-level assignment;
- `support_ratio = associated_frames / valid_pair_frames`;
- `mean_score`: mean association score across associated frames;
- `median_score`: median association score across associated frames;
- `max_consecutive_frames`: longest uninterrupted run of winning assignments.

A pair is accepted at window level when either:

```text
support_ratio >= 0.50
```

or:

```text
max_consecutive_frames >= 5
```

The two criteria and their default values must be configurable.

If multiple objects pass for the same person, all passing objects may be retained. The downstream consumer may select the highest-scoring object if a single object is required.

## 7. Output contract

### 7.1 Location

```text
DATA_ROOT/person_object_associations/<video_id>_associations.json
```

### 7.2 Structure

```json
{
  "schema_version": "1.0",
  "video_id": "video_001",
  "frame_count": 180,
  "metadata": {
    "method": "bbox_overlap_proximity",
    "person_bbox_expansion": 0.10,
    "overlap_weight": 0.75,
    "proximity_weight": 0.25,
    "max_distance_ratio": 0.25,
    "minimum_frame_score": 0.30,
    "window_size": 60,
    "stride": 30,
    "minimum_support_ratio": 0.50,
    "minimum_consecutive_frames": 5
  },
  "frame_associations": {
    "120": [
      {
        "person_id": "person_1000",
        "person_track_id": 1000,
        "object_id": "cup_1004",
        "object_track_id": 1004,
        "object_class": "cup",
        "overlap_score": 0.72,
        "proximity_score": 1.0,
        "association_score": 0.79
      }
    ]
  },
  "window_associations": [
    {
      "start_frame": 120,
      "end_frame": 180,
      "person_id": "person_1000",
      "person_track_id": 1000,
      "object_id": "cup_1004",
      "object_track_id": 1004,
      "object_class": "cup",
      "valid_pair_frames": 60,
      "associated_frames": 50,
      "support_ratio": 0.8333,
      "mean_score": 0.74,
      "median_score": 0.76,
      "max_consecutive_frames": 22,
      "accepted": true
    }
  ],
  "unassociated_objects": {
    "120": ["chair_1007"]
  }
}
```

Scores should be rounded only during serialization. Internal calculations should use full floating-point precision.

## 8. Downstream integration

### 8.1 Action recognition

When action recognition is enabled, `action_labeling.ipynb` should read window-level associations and display only objects accepted for the target person and window. It must no longer display every non-person object visible in the video window.

If no accepted object exists, the person must still be processed without object boxes.

### 8.2 Process-mining formatting

`process_mining_formatting.ipynb` must obtain the object field from the association output instead of listing every visible object.

Recommended additional event-log fields are:

```text
ObjectTrack
AssociationScore
AssociationMethod
AssociationSupport
```

An event with no accepted association should use an explicit missing value such as `Nothing`, consistently with the current output convention.

## 9. Configuration

The implementation should expose the following parameters in one configuration cell or object:

| Parameter | Default |
|---|---:|
| `PERSON_BBOX_EXPANSION` | `0.10` |
| `OVERLAP_WEIGHT` | `0.75` |
| `PROXIMITY_WEIGHT` | `0.25` |
| `MAX_DISTANCE_RATIO` | `0.25` |
| `MINIMUM_FRAME_SCORE` | `0.30` |
| `WINDOW_SIZE` | `60` |
| `STRIDE` | `30` |
| `MINIMUM_SUPPORT_RATIO` | `0.50` |
| `MINIMUM_CONSECUTIVE_FRAMES` | `5` |

Weights must sum to `1.0`. Invalid configurations must fail early with a clear error.

## 10. Evaluation requirements

The baseline should be evaluated separately from detection quality. A small manually annotated validation set should include:

- a manipulated object fully inside the person box;
- an object held outside the original person box;
- a background object overlapping the person in 2D;
- a foreground object overlapping the person in 2D;
- two people close to the same object;
- an object passing briefly near a person;
- an object temporarily occluded;
- a frame containing no manipulated object.

Recommended metrics are:

- pair-level precision, recall, and F1;
- percentage of objects left unassociated;
- object-to-person assignment accuracy in multi-person frames;
- window-level association accuracy;
- error distribution by object class.

The background and foreground overlap cases are expected limitations of this baseline and should be retained as explicit test cases for a future depth-aware version.

## 11. Acceptance criteria

The implementation is complete when:

1. It processes an existing tracking JSON without rerunning detection or SAM2.
2. It produces deterministic frame-level and window-level associations.
3. It never assigns a single object to multiple people in the same frame.
4. It permits multiple objects to be associated with one person.
5. It preserves unassociated objects rather than forcing an assignment.
6. It handles missing tracks and malformed boxes without terminating the entire dataset run.
7. It records all thresholds and weights in output metadata.
8. Its windows align with the current `60`-frame, `30`-frame-stride action windows.
9. The action-labeling and process-mining stages can consume the output without modifying the original detection JSON.
10. Runtime is negligible relative to object detection and tracking for the same video.

## 12. Future extension: depth-aware association

A future schema version may add a `depth_score` while preserving the same module boundary and output identifiers:

```text
association_score =
    spatial_score × depth_compatibility_score
```

The depth-aware implementation should reuse SAM2 masks or stored segmentation maps, but must not require changes to the detection semantics defined by this specification.

