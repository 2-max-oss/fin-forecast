"""Streamlit UI for the video transcription tool."""

import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transcribe import (
    check_ffmpeg,
    extract_audio,
    resolve_device_and_compute,
    transcribe_audio,
    write_json,
    write_srt,
    write_txt,
    write_vtt,
    SUPPORTED_MODELS,
)

st.set_page_config(page_title="Video Transcriber", page_icon="🎙️", layout="centered")
st.title("🎙️ Video Transcriber")
st.caption("Transcribe video files up to 2+ hours using Whisper AI — runs locally, no API key needed.")

# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        SUPPORTED_MODELS,
        index=SUPPORTED_MODELS.index("medium"),
        help="Larger models are more accurate but slower. First run downloads the model.",
    )
    output_format = st.selectbox("Output format", ["txt", "srt", "vtt", "json"])
    timestamps = st.checkbox(
        "Include timestamps (txt only)",
        value=False,
        disabled=output_format != "txt",
    )
    language = st.text_input(
        "Language (optional)",
        placeholder="e.g. en, es, ja — leave blank to auto-detect",
    )
    task = st.radio("Task", ["transcribe", "translate"], horizontal=True,
                    help="'translate' converts any language to English.")

    st.divider()
    st.markdown("**Model sizes (approx. speed on Apple Silicon)**")
    st.markdown("""
| Model | Speed |
|-------|-------|
| tiny | ~30x real-time |
| base | ~20x real-time |
| small | ~15x real-time |
| medium | ~10x real-time |
| large-v3 | ~5x real-time |
""")

# ── File input ────────────────────────────────────────────────────────────────
tab_upload, tab_local = st.tabs(["Upload file (< 200 MB)", "Local file path (any size)"])

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mkv", "mov", "avi", "webm", "m4v", "flv", "wmv"],
    )
    if uploaded:
        st.video(uploaded)

with tab_local:
    local_path_str = st.text_input(
        "Full path to video file",
        placeholder="/Users/you/Movies/recording.mp4",
    )
    if local_path_str:
        local_path = Path(local_path_str.strip())
        if not local_path.exists():
            st.error(f"File not found: `{local_path}`")
            local_path = None
        else:
            size_gb = local_path.stat().st_size / (1024 ** 3)
            st.success(f"Found: **{local_path.name}** ({size_gb:.2f} GB)")
    else:
        local_path = None

# Determine active input
using_upload = uploaded is not None
using_local = local_path is not None
has_input = using_upload or using_local

if has_input and st.button("Transcribe", type="primary", use_container_width=True):
        # Check ffmpeg
        import shutil
        if shutil.which("ffmpeg") is None:
            st.error(
                "**ffmpeg not found.** Install it with:\n\n"
                "```\nbrew install ffmpeg\n```\n\n"
                "Then restart the app."
            )
            st.stop()

        # Resolve video path — either save upload to temp file, or use local path directly
        owns_tmp_video = False
        if using_local:
            tmp_video_path = local_path
            video_name = local_path.name
        else:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_video:
                tmp_video.write(uploaded.read())
                tmp_video_path = Path(tmp_video.name)
            video_name = uploaded.name
            owns_tmp_video = True

        audio_path = None
        try:
            with st.status("Working...", expanded=True) as status:
                st.write(f"Extracting audio from **{video_name}**...")
                audio_path = extract_audio(tmp_video_path, verbose=False)

                st.write(f"Loading **{model}** model (downloads on first use)...")
                device, compute_type = resolve_device_and_compute("auto", "auto")

                st.write("Transcribing — this may take a few minutes for long videos...")
                t0 = time.time()
                lang = language.strip() or None
                segments, info = transcribe_audio(
                    audio_path=audio_path,
                    model_size=model,
                    language=lang,
                    task=task,
                    device=device,
                    compute_type=compute_type,
                    verbose=False,
                )
                elapsed = time.time() - t0

                status.update(label="Done!", state="complete")

            # ── Results ───────────────────────────────────────────────────────
            detected_lang = getattr(info, "language", "unknown")
            duration_min = getattr(info, "duration", 0) / 60
            col1, col2, col3 = st.columns(3)
            col1.metric("Segments", len(segments))
            col2.metric("Audio length", f"{duration_min:.1f} min")
            col3.metric("Time taken", f"{elapsed:.0f}s")

            # Write to a temp output file then read back for download
            ext = "." + output_format
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="w", encoding="utf-8") as tmp_out:
                tmp_out_path = Path(tmp_out.name)

            writers = {
                "txt": lambda: write_txt(segments, tmp_out_path, timestamps),
                "srt": lambda: write_srt(segments, tmp_out_path),
                "vtt": lambda: write_vtt(segments, tmp_out_path),
                "json": lambda: write_json(segments, tmp_out_path),
            }
            writers[output_format]()

            transcript_text = tmp_out_path.read_text(encoding="utf-8")
            tmp_out_path.unlink(missing_ok=True)

            # Preview (txt and srt)
            if output_format in ("txt", "srt", "vtt"):
                with st.expander("Preview transcript", expanded=True):
                    st.text_area("", transcript_text, height=300, label_visibility="collapsed")
            else:
                st.json(transcript_text)

            # Download button
            stem = Path(video_name).stem
            st.download_button(
                label=f"Download .{output_format}",
                data=transcript_text,
                file_name=f"{stem}.{output_format}",
                mime="text/plain",
                use_container_width=True,
            )

        finally:
            if owns_tmp_video:
                tmp_video_path.unlink(missing_ok=True)
            if audio_path:
                audio_path.unlink(missing_ok=True)
