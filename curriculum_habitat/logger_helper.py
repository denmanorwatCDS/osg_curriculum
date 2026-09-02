"""Helpers for converting completed episode data into loggable videos."""

import math
from collections.abc import Mapping

import cv2
import numpy as np


def _as_rgb(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(
            "Video frames must have shape (height, width, channels) "
            f"with at least three channels; got {frame.shape}"
        )

    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        finite_values = frame[np.isfinite(frame)]
        if finite_values.size and finite_values.max() <= 1.0:
            frame = frame * 255.0
        frame = np.nan_to_num(
            frame, nan=0.0, posinf=255.0, neginf=0.0
        )
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _item_at(values, step, number_of_frames):
    if values is None:
        return None
    if isinstance(values, np.ndarray) and values.ndim == 0:
        return values.item()
    if isinstance(values, (list, tuple, np.ndarray)):
        if len(values) == 0:
            return None
        if len(values) == number_of_frames:
            return values[step]
        if len(values) == 1:
            return values[0]
        if step < len(values):
            return values[step]
        return values[-1]
    return values


def _scalar_text(value, precision=3):
    if value is None:
        return "N/A"
    array = np.asarray(value)
    if array.size != 1:
        return "N/A"
    try:
        return f"{float(array.reshape(-1)[0]):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _boolean_text(value):
    if value is None:
        return "N/A"
    return "yes" if bool(value) else "no"


def _goal_position_at(values, step, number_of_frames):
    if values is None:
        return None
    positions = np.asarray(values)
    if positions.size == 0:
        return None
    if positions.ndim == 1:
        return positions
    if positions.shape[0] == number_of_frames:
        return positions[step]
    if positions.shape[0] == 1:
        return positions[0]
    if step < positions.shape[0]:
        return positions[step]
    return positions[-1]


def _draw_line(canvas, text, y, color):
    margin = max(8, canvas.shape[1] // 50)
    available_width = max(1, canvas.shape[1] - 2 * margin)
    base_scale = min(0.65, max(0.38, canvas.shape[1] / 900.0))
    (text_width, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, base_scale, 1
    )
    scale = base_scale
    if text_width > available_width:
        scale *= available_width / text_width
    cv2.putText(
        canvas,
        text,
        (margin, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _episode_and_environment_index(description, fallback_index):
    if description is None:
        return None, fallback_index
    if not isinstance(description, Mapping):
        raise TypeError(
            "Each episode description must be a mapping or None; got "
            f"{type(description).__name__}"
        )

    environment_index = description.get(
        "environment_index", fallback_index
    )
    if "previous_episode" in description:
        return description["previous_episode"], environment_index
    return description, environment_index


def fetch_videos(episode_descriptions):
    """Render completed episode descriptions as RGB videos.

    Each description must contain ``frames`` and may contain per-frame
    ``reward``, ``distance_to_goal``, ``angle_to_goal``, and
    ``goal_position`` values. Static fields are ``controlled_by_expert``,
    ``is_success``, ``return``, and ``goal_object``. An optional
    ``environment_index`` is displayed in the header; otherwise the item's
    position in ``episode_descriptions`` is used.

    A description may also be an environment-slot mapping containing a
    ``previous_episode`` entry. In that form, only ``previous_episode`` is
    rendered; ``current_episode`` is ignored.

    Returns:
        A list of RGB ``uint8`` arrays with shape ``(T, H, W, C)``.
    """
    if episode_descriptions is None:
        raise TypeError("episode_descriptions must be an iterable, not None")

    rendered_videos = []
    for fallback_index, description in enumerate(episode_descriptions):
        episode, environment_index = _episode_and_environment_index(
            description, fallback_index
        )
        if episode is None:
            continue
        if not isinstance(episode, Mapping):
            raise TypeError(
                "previous_episode must be a mapping or None; got "
                f"{type(episode).__name__}"
            )

        stored_frames = episode.get("frames")
        if stored_frames is None or len(stored_frames) == 0:
            continue

        number_of_frames = len(stored_frames)
        first_frame = _as_rgb(stored_frames[0])
        frame_height, frame_width = first_frame.shape[:2]
        panel_height = max(96, frame_height // 3)

        expert_text = _boolean_text(episode.get("controlled_by_expert"))
        success = episode.get("is_success")
        success_text = _boolean_text(success)
        return_text = _scalar_text(episode.get("return"), precision=3)
        goal_object = episode.get("goal_object")
        goal_object_text = "N/A" if goal_object is None else str(goal_object)

        video_frames = []
        for step, stored_frame in enumerate(stored_frames):
            camera_frame = _as_rgb(stored_frame)
            if camera_frame.shape[:2] != (frame_height, frame_width):
                camera_frame = cv2.resize(
                    camera_frame,
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_AREA,
                )

            canvas = np.zeros(
                (frame_height + 2 * panel_height, frame_width, 3),
                dtype=np.uint8,
            )
            canvas[:panel_height] = (24, 31, 43)
            canvas[panel_height:panel_height + frame_height] = camera_frame
            canvas[panel_height + frame_height:] = (17, 24, 34)

            header_spacing = panel_height // 4
            _draw_line(
                canvas,
                (
                    f"Environment index: {environment_index} | "
                    f"Expert-controlled: {expert_text}"
                ),
                header_spacing,
                (235, 245, 255),
            )
            _draw_line(
                canvas,
                f"Success: {success_text} | Episode return: {return_text}",
                2 * header_spacing,
                (130, 230, 170) if bool(success) else (255, 190, 120),
            )
            _draw_line(
                canvas,
                f"Goal object: {goal_object_text}",
                3 * header_spacing,
                (180, 205, 255),
            )

            reward = _item_at(
                episode.get("reward"), step, number_of_frames
            )
            distance = _item_at(
                episode.get("distance_to_goal"), step, number_of_frames
            )
            angle = _item_at(
                episode.get("angle_to_goal"), step, number_of_frames
            )
            goal_position = _goal_position_at(
                episode.get("goal_position"), step, number_of_frames
            )

            angle_text = _scalar_text(angle, precision=3)
            angle_degrees_text = "N/A"
            if angle is not None and np.asarray(angle).size == 1:
                try:
                    angle_degrees_text = (
                        f"{math.degrees(float(np.asarray(angle).item())):.1f}"
                    )
                except (TypeError, ValueError):
                    pass

            goal_position_text = "N/A"
            if goal_position is not None:
                goal_position = np.asarray(goal_position).reshape(-1)
                if goal_position.size >= 2:
                    goal_position_text = (
                        f"x={_scalar_text(goal_position[0], 2)}, "
                        f"z={_scalar_text(goal_position[1], 2)}"
                    )

            footer_start = panel_height + frame_height
            footer_spacing = panel_height // 4
            _draw_line(
                canvas,
                (
                    f"Step: {step + 1}/{number_of_frames} | "
                    f"Reward: {_scalar_text(reward, 4)}"
                ),
                footer_start + footer_spacing,
                (235, 245, 255),
            )
            _draw_line(
                canvas,
                (
                    f"Distance to goal: {_scalar_text(distance, 3)} m | "
                    f"Angle to goal: {angle_text} rad "
                    f"({angle_degrees_text} deg)"
                ),
                footer_start + 2 * footer_spacing,
                (255, 220, 145),
            )
            _draw_line(
                canvas,
                f"Goal position: {goal_position_text}",
                footer_start + 3 * footer_spacing,
                (180, 205, 255),
            )
            video_frames.append(canvas)

        rendered_videos.append(np.stack(video_frames, axis=0))

    return rendered_videos
