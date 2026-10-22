"""Associate tracked objects with people using 2D bounding-box geometry.

This module consumes the JSON produced by ``obj_det_w_dino.ipynb`` and writes
frame-level and window-level person-object associations without changing the
upstream tracking data.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class AssociationConfig:
    person_bbox_expansion: float = 0.10
    overlap_weight: float = 0.75
    proximity_weight: float = 0.25
    max_distance_ratio: float = 0.25
    minimum_frame_score: float = 0.30
    window_size: int = 60
    stride: int = 30
    minimum_support_ratio: float = 0.50
    minimum_consecutive_frames: int = 5
    tie_tolerance: float = 1e-9

    def validate(self) -> None:
        if not math.isclose(
            self.overlap_weight + self.proximity_weight, 1.0, abs_tol=1e-9
        ):
            raise ValueError("overlap_weight and proximity_weight must sum to 1.0")
        if self.person_bbox_expansion < 0:
            raise ValueError("person_bbox_expansion must be non-negative")
        if self.max_distance_ratio <= 0:
            raise ValueError("max_distance_ratio must be greater than zero")
        if not 0 <= self.minimum_frame_score <= 1:
            raise ValueError("minimum_frame_score must be between 0 and 1")
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError("window_size and stride must be greater than zero")
        if not 0 <= self.minimum_support_ratio <= 1:
            raise ValueError("minimum_support_ratio must be between 0 and 1")
        if self.minimum_consecutive_frames <= 0:
            raise ValueError("minimum_consecutive_frames must be greater than zero")


def _valid_bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2))
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def _area(box: BBox) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _expanded(box: BBox, fraction: float) -> BBox:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (
        max(0.0, box[0] - width * fraction),
        max(0.0, box[1] - height * fraction),
        min(1.0, box[2] + width * fraction),
        min(1.0, box[3] + height * fraction),
    )


def intersection_over_object(person_box: BBox, object_box: BBox) -> float:
    intersection_width = max(
        0.0, min(person_box[2], object_box[2]) - max(person_box[0], object_box[0])
    )
    intersection_height = max(
        0.0, min(person_box[3], object_box[3]) - max(person_box[1], object_box[1])
    )
    return (intersection_width * intersection_height) / max(_area(object_box), 1e-12)


def _box_distance(first: BBox, second: BBox) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(dx, dy)


def proximity(person_box: BBox, object_box: BBox, max_distance_ratio: float) -> float:
    diagonal = math.hypot(person_box[2] - person_box[0], person_box[3] - person_box[1])
    normalized_distance = _box_distance(person_box, object_box) / max(diagonal, 1e-12)
    return max(0.0, 1.0 - normalized_distance / max_distance_ratio)


def _track_id(key: str, track: dict[str, Any]) -> int | str:
    if "track_id" in track:
        return track["track_id"]
    suffix = key.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else suffix


def _sortable_track_id(value: int | str) -> tuple[int, int | str]:
    return (0, value) if isinstance(value, int) else (1, str(value))


def _normalise_tracks(data: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], int]:
    raw_tracks = data.get("detected_objects")
    if not isinstance(raw_tracks, dict):
        raise ValueError("Input JSON must contain a 'detected_objects' object")

    frame_count = max(
        (len(track.get("bbox", [])) for track in raw_tracks.values() if isinstance(track, dict)),
        default=0,
    )
    tracks: dict[str, dict[str, Any]] = {}
    for key, raw_track in raw_tracks.items():
        if not isinstance(raw_track, dict):
            continue
        boxes = list(raw_track.get("bbox", []))
        boxes.extend([None] * (frame_count - len(boxes)))
        tracks[key] = {
            **raw_track,
            "track_id": _track_id(key, raw_track),
            "bbox": [_valid_bbox(box) for box in boxes],
        }
    return tracks, frame_count


def _candidate(
    person_key: str,
    person: dict[str, Any],
    person_box: BBox,
    object_key: str,
    obj: dict[str, Any],
    object_box: BBox,
    config: AssociationConfig,
) -> dict[str, Any] | None:
    overlap_score = intersection_over_object(
        _expanded(person_box, config.person_bbox_expansion), object_box
    )
    proximity_score = proximity(person_box, object_box, config.max_distance_ratio)
    association_score = (
        config.overlap_weight * overlap_score
        + config.proximity_weight * proximity_score
    )
    if association_score < config.minimum_frame_score:
        return None
    if overlap_score <= 0 and proximity_score <= 0:
        return None
    return {
        "person_id": person_key,
        "person_track_id": person["track_id"],
        "object_id": object_key,
        "object_track_id": obj["track_id"],
        "object_class": obj.get("class_name", "unknown"),
        "overlap_score": overlap_score,
        "proximity_score": proximity_score,
        "association_score": association_score,
    }


def _choose_candidate(
    candidates: list[dict[str, Any]], config: AssociationConfig
) -> dict[str, Any]:
    best = candidates[0]
    for candidate in candidates[1:]:
        score_delta = candidate["association_score"] - best["association_score"]
        if score_delta > config.tie_tolerance:
            best = candidate
            continue
        if abs(score_delta) <= config.tie_tolerance:
            overlap_delta = candidate["overlap_score"] - best["overlap_score"]
            if overlap_delta > config.tie_tolerance:
                best = candidate
                continue
            if abs(overlap_delta) <= config.tie_tolerance and _sortable_track_id(
                candidate["person_track_id"]
            ) < _sortable_track_id(best["person_track_id"]):
                best = candidate
    return best


def _windows(frame_count: int, size: int, stride: int) -> list[tuple[int, int]]:
    if frame_count == 0:
        return []
    if frame_count <= size:
        return [(0, frame_count)]
    return [(start, start + size) for start in range(0, frame_count - size + 1, stride)]


def _longest_consecutive(frames: Iterable[int]) -> int:
    longest = current = 0
    previous: int | None = None
    for frame in sorted(frames):
        current = current + 1 if previous is not None and frame == previous + 1 else 1
        longest = max(longest, current)
        previous = frame
    return longest


def associate_tracking_data(
    data: dict[str, Any],
    video_id: str,
    config: AssociationConfig | None = None,
) -> dict[str, Any]:
    config = config or AssociationConfig()
    config.validate()
    tracks, frame_count = _normalise_tracks(data)
    people = {
        key: track for key, track in tracks.items() if track.get("class_name") == "person"
    }
    objects = {
        key: track for key, track in tracks.items() if track.get("class_name") != "person"
    }

    raw_frame_associations: dict[int, list[dict[str, Any]]] = {}
    raw_unassociated: dict[int, list[str]] = {}
    for frame_index in range(frame_count):
        by_object: dict[str, list[dict[str, Any]]] = {}
        visible_objects: list[str] = []
        for object_key, obj in objects.items():
            object_box = obj["bbox"][frame_index]
            if object_box is None:
                continue
            visible_objects.append(object_key)
            for person_key, person in people.items():
                person_box = person["bbox"][frame_index]
                if person_box is None:
                    continue
                candidate = _candidate(
                    person_key,
                    person,
                    person_box,
                    object_key,
                    obj,
                    object_box,
                    config,
                )
                if candidate is not None:
                    by_object.setdefault(object_key, []).append(candidate)

        winners = [_choose_candidate(candidates, config) for candidates in by_object.values()]
        winners.sort(
            key=lambda item: (
                _sortable_track_id(item["person_track_id"]),
                _sortable_track_id(item["object_track_id"]),
            )
        )
        if winners:
            raw_frame_associations[frame_index] = winners
        assigned_objects = {item["object_id"] for item in winners}
        missing = sorted(set(visible_objects) - assigned_objects)
        if missing:
            raw_unassociated[frame_index] = missing

    window_associations: list[dict[str, Any]] = []
    for start, end in _windows(frame_count, config.window_size, config.stride):
        for person_key, person in people.items():
            for object_key, obj in objects.items():
                valid_frames = [
                    frame
                    for frame in range(start, end)
                    if person["bbox"][frame] is not None and obj["bbox"][frame] is not None
                ]
                if not valid_frames:
                    continue
                observations = [
                    association
                    for frame in valid_frames
                    for association in raw_frame_associations.get(frame, [])
                    if association["person_id"] == person_key
                    and association["object_id"] == object_key
                ]
                if not observations:
                    continue
                associated_frame_ids = [
                    frame
                    for frame in valid_frames
                    if any(
                        item["person_id"] == person_key and item["object_id"] == object_key
                        for item in raw_frame_associations.get(frame, [])
                    )
                ]
                scores = [item["association_score"] for item in observations]
                support_ratio = len(associated_frame_ids) / len(valid_frames)
                consecutive = _longest_consecutive(associated_frame_ids)
                window_associations.append(
                    {
                        "start_frame": start,
                        "end_frame": end,
                        "person_id": person_key,
                        "person_track_id": person["track_id"],
                        "object_id": object_key,
                        "object_track_id": obj["track_id"],
                        "object_class": obj.get("class_name", "unknown"),
                        "valid_pair_frames": len(valid_frames),
                        "associated_frames": len(associated_frame_ids),
                        "support_ratio": support_ratio,
                        "mean_score": mean(scores),
                        "median_score": median(scores),
                        "max_consecutive_frames": consecutive,
                        "accepted": support_ratio >= config.minimum_support_ratio
                        or consecutive >= config.minimum_consecutive_frames,
                    }
                )

    metadata = asdict(config)
    metadata.pop("tie_tolerance")
    metadata["method"] = "bbox_overlap_proximity"
    return {
        "schema_version": "1.0",
        "video_id": video_id,
        "frame_count": frame_count,
        "metadata": metadata,
        "frame_associations": {
            str(frame): [_rounded(item) for item in associations]
            for frame, associations in raw_frame_associations.items()
        },
        "window_associations": [_rounded(item) for item in window_associations],
        "unassociated_objects": {
            str(frame): object_ids for frame, object_ids in raw_unassociated.items()
        },
    }


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def process_tracking_file(
    input_path: str | Path,
    output_path: str | Path,
    config: AssociationConfig | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open("r", encoding="utf-8") as source:
        tracking_data = json.load(source)
    result = associate_tracking_data(tracking_data, input_path.stem, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False)
    return result


def process_directory(
    input_directory: str | Path,
    output_directory: str | Path,
    config: AssociationConfig | None = None,
) -> tuple[int, int]:
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    processed = skipped = 0
    for video_directory in sorted(path for path in input_directory.iterdir() if path.is_dir()):
        json_files = sorted(video_directory.glob("*.json"))
        if not json_files:
            skipped += 1
            continue
        output_path = output_directory / f"{video_directory.name}_associations.json"
        process_tracking_file(json_files[0], output_path, config)
        processed += 1
    return processed, skipped


def _pixel_bbox(box: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(box[0] * width),
        round(box[1] * height),
        round(box[2] * width),
        round(box[3] * height),
    )


def _box_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = position
    text_box = draw.textbbox((x, y), text, font=font, stroke_width=1)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    y = max(0, y - text_height - 6)
    draw.rectangle((x, y, x + text_width + 6, y + text_height + 6), fill=color)
    draw.text(
        (x + 3, y + 2),
        text,
        fill=(255, 255, 255),
        font=font,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )


def _accepted_pair_keys(
    association_data: dict[str, Any], frame_index: int
) -> set[tuple[str, str]]:
    return {
        (item["person_id"], item["object_id"])
        for item in association_data.get("window_associations", [])
        if item.get("accepted", False)
        and item["start_frame"] <= frame_index < item["end_frame"]
    }


def render_association_frames(
    tracking_data: dict[str, Any],
    association_data: dict[str, Any],
    frames_directory: str | Path,
    output_directory: str | Path,
    frame_step: int = 30,
    include_frame_only_associations: bool = False,
) -> int:
    """Render diagnostic frames for accepted person-object associations.

    People are red, temporally accepted objects are green, and optional
    frame-level associations that failed temporal aggregation are orange.
    Original frames are never modified.
    """
    if frame_step <= 0:
        raise ValueError("frame_step must be greater than zero")
    tracks, frame_count = _normalise_tracks(tracking_data)
    frames_directory = Path(frames_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frames_directory.glob("*.jpg"))
    if not frame_paths:
        frame_paths = sorted(frames_directory.glob("*.png"))
    usable_frames = min(frame_count, len(frame_paths))
    font = ImageFont.load_default()
    rendered = 0

    for frame_index in range(0, usable_frames, frame_step):
        frame_items = association_data.get("frame_associations", {}).get(
            str(frame_index), []
        )
        accepted_pairs = _accepted_pair_keys(association_data, frame_index)
        selected_items = [
            item
            for item in frame_items
            if include_frame_only_associations
            or (item["person_id"], item["object_id"]) in accepted_pairs
        ]

        with Image.open(frame_paths[frame_index]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size

        person_ids = sorted(
            key
            for key, track in tracks.items()
            if track.get("class_name") == "person"
            and track["bbox"][frame_index] is not None
        )
        for person_id in person_ids:
            person = tracks.get(person_id)
            if person is None or person["bbox"][frame_index] is None:
                continue
            person_box = _pixel_bbox(person["bbox"][frame_index], width, height)
            draw.rectangle(person_box, outline=(220, 30, 30), width=4)
            _draw_label(draw, (person_box[0], person_box[1]), person_id, (180, 0, 0), font)

        for item in selected_items:
            obj = tracks.get(item["object_id"])
            person = tracks.get(item["person_id"])
            if obj is None or person is None:
                continue
            object_bbox = obj["bbox"][frame_index]
            person_bbox = person["bbox"][frame_index]
            if object_bbox is None or person_bbox is None:
                continue
            accepted = (item["person_id"], item["object_id"]) in accepted_pairs
            color = (20, 170, 60) if accepted else (230, 135, 20)
            object_box = _pixel_bbox(object_bbox, width, height)
            person_box = _pixel_bbox(person_bbox, width, height)
            draw.rectangle(object_box, outline=color, width=4)
            draw.line(
                (_box_center(person_box), _box_center(object_box)),
                fill=(255, 210, 20),
                width=3,
            )
            label = (
                f"{item['object_id']}  score={item['association_score']:.2f}"
                + ("" if accepted else "  frame-only")
            )
            _draw_label(draw, (object_box[0], object_box[1]), label, color, font)

        legend = "RED: person | GREEN: accepted object | YELLOW: association"
        draw.rectangle((5, 5, 5 + len(legend) * 7, 25), fill=(0, 0, 0))
        draw.text((9, 8), legend, fill=(255, 255, 255), font=font)
        image.save(output_directory / f"frame_{frame_index:06d}.jpg", quality=92)
        rendered += 1
    return rendered


def render_directory(
    tracking_directory: str | Path,
    association_directory: str | Path,
    frames_root: str | Path,
    output_root: str | Path,
    frame_step: int = 30,
    include_frame_only_associations: bool = False,
) -> tuple[int, int]:
    """Render diagnostic images for every video with all required inputs."""
    tracking_directory = Path(tracking_directory)
    association_directory = Path(association_directory)
    frames_root = Path(frames_root)
    output_root = Path(output_root)
    rendered = skipped = 0
    for association_path in sorted(association_directory.glob("*_associations.json")):
        video_id = association_path.name.removesuffix("_associations.json")
        tracking_files = sorted((tracking_directory / video_id).glob("*.json"))
        frames_directory = frames_root / video_id
        if not tracking_files or not frames_directory.is_dir():
            skipped += 1
            continue
        with tracking_files[0].open("r", encoding="utf-8") as source:
            tracking_data = json.load(source)
        with association_path.open("r", encoding="utf-8") as source:
            association_data = json.load(source)
        rendered += render_association_frames(
            tracking_data,
            association_data,
            frames_directory,
            output_root / video_id,
            frame_step,
            include_frame_only_associations,
        )
    return rendered, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Tracking JSON or directory containing video folders")
    parser.add_argument("output", help="Output JSON or output directory")
    args = parser.parse_args()
    source = Path(args.input)
    if source.is_file():
        process_tracking_file(source, args.output)
    elif source.is_dir():
        processed, skipped = process_directory(source, args.output)
        print(f"Processed {processed} videos; skipped {skipped} folders without JSON files.")
    else:
        parser.error(f"Input does not exist: {source}")


if __name__ == "__main__":
    main()
