"""
video_processing.py
--------------------
Cluster 04: Video Processing

Performs basic video processing operations using OpenCV, following the
same pattern as image_processing.py (Cluster 02) and audio_processing.py
(Cluster 03).

Operations performed:
    1. Grayscale conversion
    2. Resize to 50% of original dimensions
    3. Extract first frame as a thumbnail image
    4. Trim to the first half of the video's duration
    5. Reverse frame order

Supported formats: MP4, AVI, MKV, MOV

Usage:
    python video_processing.py <path-to-video>
"""

import os
import sys
import cv2

SUPPORTED_FORMATS = {".mp4", ".avi", ".mkv", ".mov"}


def _open_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file. It may be corrupt or unsupported.")
    return cap


def _get_properties(cap):
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def _read_all_frames(cap):
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def convert_to_grayscale(frames):
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]


def resize_frames(frames, width, height):
    new_w, new_h = max(1, width // 2), max(1, height // 2)
    return [cv2.resize(f, (new_w, new_h)) for f in frames], new_w, new_h


def extract_thumbnail(frames, output_dir):
    if not frames:
        return
    cv2.imwrite(os.path.join(output_dir, "thumbnail.jpg"), frames[0])


def trim_frames(frames, fps, seconds=None):
    """Keep only the first `seconds` seconds of frames. Defaults to half the clip."""
    if seconds is None:
        seconds = (len(frames) / fps) / 2 if fps else 0
    n_keep = max(1, int(seconds * fps))
    return frames[:n_keep]


def reverse_frames(frames):
    return frames[::-1]


def _write_video(frames, output_path, fps, width, height, is_gray=False):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=not is_gray)
    for frame in frames:
        writer.write(frame)
    writer.release()


def process_video(input_path, output_dir):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported video format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    cap = _open_video(input_path)
    props = _get_properties(cap)
    frames = _read_all_frames(cap)
    cap.release()

    if not frames:
        raise ValueError("No frames could be read from this video.")

    os.makedirs(output_dir, exist_ok=True)
    fps, width, height = props["fps"], props["width"], props["height"]

    # 1. Grayscale
    gray_frames = convert_to_grayscale(frames)
    _write_video(gray_frames, os.path.join(output_dir, "grayscale.mp4"), fps, width, height, is_gray=True)

    # 2. Resize (50%)
    resized_frames, rw, rh = resize_frames(frames, width, height)
    _write_video(resized_frames, os.path.join(output_dir, "resized.mp4"), fps, rw, rh)

    # 3. Thumbnail (first frame)
    extract_thumbnail(frames, output_dir)

    # 4. Trim (first half of the clip)
    trimmed = trim_frames(frames, fps)
    _write_video(trimmed, os.path.join(output_dir, "trimmed.mp4"), fps, width, height)

    # 5. Reverse
    reversed_clip = reverse_frames(frames)
    _write_video(reversed_clip, os.path.join(output_dir, "reversed.mp4"), fps, width, height)

    print("\nVideo processing completed successfully.")
    print("\nGenerated outputs:")
    for name in ("grayscale.mp4", "resized.mp4", "thumbnail.jpg", "trimmed.mp4", "reversed.mp4"):
        print(f" - {name}")


def main():
    if len(sys.argv) != 2:
        print("\nUsage:")
        print("python video_processing.py <path-to-video>")
        return

    input_path = sys.argv[1]
    output_dir = os.path.join(os.path.dirname(os.path.dirname(input_path)), "output")

    try:
        process_video(input_path, output_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nError during video reading or writing: {e}")


if __name__ == "__main__":
    main()