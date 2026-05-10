#!/usr/bin/env python3
"""
Video transcription tool using faster-whisper.
Supports video files up to 2+ hours long.

Requirements:
    pip install faster-whisper
    brew install ffmpeg  (macOS) or apt install ffmpeg (Linux)

Usage:
    python transcribe.py VIDEO_FILE [options]
    python transcribe.py lecture.mp4 --format srt --model large-v3
    python transcribe.py meeting.mkv -f txt -t
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv"}
SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
SUPPORTED_FORMATS = ["txt", "srt", "vtt", "json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe video files up to 2+ hours long using Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="Input video file path")
    parser.add_argument(
        "-m", "--model",
        default="medium",
        choices=SUPPORTED_MODELS,
        help="Whisper model size (default: medium). Larger = slower but more accurate.",
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Language code e.g. 'en', 'es', 'ja' (default: auto-detect)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path (default: same name as input with new extension)",
    )
    parser.add_argument(
        "-f", "--format",
        default="txt",
        choices=SUPPORTED_FORMATS,
        help="Output format: txt, srt, vtt, json (default: txt)",
    )
    parser.add_argument(
        "-t", "--timestamps",
        action="store_true",
        help="Include timestamps in txt output",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="'transcribe' to keep original language, 'translate' to translate to English (default: transcribe)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "float32", "float16", "int8"],
        help="Compute precision (default: auto — float16 on Apple Silicon/GPU, int8 on CPU)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    if args.input.suffix.lower() not in SUPPORTED_EXTENSIONS:
        parser.error(
            f"Unsupported file type '{args.input.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if args.output is None:
        ext = "." + args.format
        args.output = args.input.with_suffix(ext)

    return args


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print(
            "Error: ffmpeg not found. Install it with:\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html",
            file=sys.stderr,
        )
        sys.exit(1)


def extract_audio(video_path: Path, verbose: bool = False) -> Path:
    """Extract audio from video to a temporary 16kHz mono WAV file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)

    if verbose:
        print(f"Extracting audio from {video_path.name}...", file=sys.stderr)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",               # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",      # 16kHz sample rate (Whisper's native rate)
        "-ac", "1",          # mono
        "-f", "wav",
        str(tmp_path),
        "-y",                # overwrite without asking
        "-loglevel", "error",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        tmp_path.unlink(missing_ok=True)
        print(f"Error: ffmpeg failed to extract audio.\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
        print(f"Audio extracted ({size_mb:.1f} MB temp WAV).", file=sys.stderr)

    return tmp_path


def resolve_device_and_compute(device: str, compute_type: str) -> tuple[str, str]:
    """Resolve 'auto' device and compute type to concrete values."""
    if device == "auto":
        device = "cuda" if shutil.which("nvidia-smi") else "cpu"

    if compute_type == "auto":
        # float16 is only supported on CUDA; use int8 for all CPU targets
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def transcribe_audio(
    audio_path: Path,
    model_size: str,
    language: str | None,
    task: str,
    device: str,
    compute_type: str,
    verbose: bool,
) -> tuple[list, dict]:
    """Load Whisper model and transcribe audio. Returns (segments, info)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "Error: faster-whisper not installed. Run:\n"
            "  pip install faster-whisper",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        print(
            f"Loading '{model_size}' model (device={device}, compute={compute_type})...",
            file=sys.stderr,
        )
        print(
            "(First run will download the model, which may take a few minutes.)",
            file=sys.stderr,
        )

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    if verbose:
        print("Transcribing...", file=sys.stderr)

    t0 = time.time()

    segments_gen, info = model.transcribe(
        str(audio_path),
        language=language,
        task=task,
        beam_size=5,
        vad_filter=True,       # Skip silent regions; helps with long recordings
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    if verbose and language is None:
        print(
            f"Detected language: {info.language} "
            f"(probability {info.language_probability:.0%})",
            file=sys.stderr,
        )

    segments = []
    for seg in segments_gen:
        segments.append(seg)
        if verbose:
            elapsed = time.time() - t0
            print(
                f"  [{_fmt_time(seg.start)} --> {_fmt_time(seg.end)}] {seg.text.strip()}",
                file=sys.stderr,
            )

    elapsed = time.time() - t0
    if verbose:
        print(
            f"Done. {len(segments)} segments in {elapsed:.1f}s.",
            file=sys.stderr,
        )

    return segments, info


def _fmt_time(seconds: float, separator: str = ",") -> str:
    """Format seconds as HH:MM:SS,mmm (SRT style) or HH:MM:SS.mmm (VTT style)."""
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}{separator}{ms:03d}"


def write_txt(segments, output_path: Path, timestamps: bool) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            if timestamps:
                f.write(f"[{_fmt_time(seg.start, ':')} --> {_fmt_time(seg.end, ':')}] ")
            f.write(seg.text.strip() + "\n")


def write_srt(segments, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{_fmt_time(seg.start)} --> {_fmt_time(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")


def write_vtt(segments, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(
                f"{_fmt_time(seg.start, '.')} --> {_fmt_time(seg.end, '.')}\n"
            )
            f.write(seg.text.strip() + "\n\n")


def write_json(segments, output_path: Path) -> None:
    data = [
        {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip()}
        for seg in segments
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    check_ffmpeg()

    device, compute_type = resolve_device_and_compute(args.device, args.compute_type)

    audio_path = None
    try:
        audio_path = extract_audio(args.input, verbose=args.verbose)

        segments, info = transcribe_audio(
            audio_path=audio_path,
            model_size=args.model,
            language=args.language,
            task=args.task,
            device=device,
            compute_type=compute_type,
            verbose=args.verbose,
        )

        writers = {
            "txt": lambda: write_txt(segments, args.output, args.timestamps),
            "srt": lambda: write_srt(segments, args.output),
            "vtt": lambda: write_vtt(segments, args.output),
            "json": lambda: write_json(segments, args.output),
        }
        writers[args.format]()

        duration_min = info.duration / 60 if hasattr(info, "duration") and info.duration else 0
        print(
            f"Transcript saved to: {args.output}\n"
            f"  {len(segments)} segments"
            + (f", {duration_min:.1f} min audio" if duration_min else ""),
            file=sys.stderr,
        )

    finally:
        if audio_path and audio_path.exists():
            audio_path.unlink()


if __name__ == "__main__":
    main()
