"""
Performs three basic operations on 16-bit PCM WAV audio:
    1. Stereo to Mono conversion
    2. Volume normalization
    3. Audio reversal

Currently supports: WAV, 16-bit PCM audio only.

"""

import os
import sys
import wave

import numpy as np

SUPPORTED_EXTENSIONS = {".wav"}
SUPPORTED_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


def _read_wav(file_path):
    """Read a WAV file and return (samples, params). samples shape: (n_frames, n_channels)."""
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width != SUPPORTED_SAMPLE_WIDTH_BYTES:
        raise ValueError(
            f"Unsupported WAV bit depth: {sample_width * 8}-bit. "
            f"Only 16-bit PCM WAV is supported."
        )

    samples = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
    else:
        samples = samples.reshape(-1, 1)

    params = {
        "n_channels": n_channels,
        "sample_width": sample_width,
        "framerate": framerate,
    }
    return samples, params


def _write_wav(file_path, samples, params, n_channels=None):
    """Write int16 samples (n_frames, n_channels) back out as a WAV file."""
    out_channels = n_channels if n_channels is not None else params["n_channels"]
    samples = np.clip(samples, -32768, 32767).astype(np.int16)

    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(out_channels)
        wf.setsampwidth(params["sample_width"])
        wf.setframerate(params["framerate"])
        wf.writeframes(samples.tobytes())


def stereo_to_mono(samples):
    """Average all channels down to a single mono channel."""
    mono = samples.mean(axis=1).astype(np.int16)
    return mono.reshape(-1, 1)


def normalize_audio(samples, target_peak=32767):
    """Scale samples so the loudest peak uses the full 16-bit amplitude range."""
    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples.copy()  # silent audio, nothing to normalize
    scale = target_peak / float(peak)
    normalized = samples.astype(np.float32) * scale
    return normalized.astype(np.int16)


def reverse_audio(samples):
    """Reverse the order of audio frames (time axis)."""
    return samples[::-1].copy()


def process_audio(input_path, output_dir):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{ext}'. Audio processing currently supports: WAV (16-bit PCM)."
        )

    try:
        samples, params = _read_wav(input_path)
    except wave.Error as e:
        raise ValueError(f"Invalid or corrupt WAV file: {e}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Stereo to Mono
    mono = stereo_to_mono(samples)
    _write_wav(os.path.join(output_dir, "mono.wav"), mono, params, n_channels=1)

    # 2. Normalize (on the original, unmodified channel layout)
    normalized = normalize_audio(samples)
    _write_wav(os.path.join(output_dir, "normalized.wav"), normalized, params)

    # 3. Reverse
    reversed_samples = reverse_audio(samples)
    _write_wav(os.path.join(output_dir, "reversed.wav"), reversed_samples, params)

    print("\nAudio processing completed successfully.")
    print("\nGenerated outputs:")
    for name in ("mono.wav", "normalized.wav", "reversed.wav"):
        print(f" - {name}")


def main():
    if len(sys.argv) != 2:
        print("\nUsage:")
        print("python audio_processing.py <path-to-wav>")
        return

    input_path = sys.argv[1]
    # e.g. Cluster03-Audio-Processing/.../output   (sibling of input/)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(input_path)), "output")

    try:
        process_audio(input_path, output_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nError during audio reading or writing: {e}")


if __name__ == "__main__":
    main()