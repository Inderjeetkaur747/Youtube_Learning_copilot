import threading

_model = None
_load_lock = threading.Lock()
_infer_lock = threading.Lock()  # Whisper is CPU-bound; run one transcription at a time.


def get_model():
    """Load Faster-Whisper once. Safe to call from several threads."""
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from faster_whisper import WhisperModel

                from .config import settings

                _model = WhisperModel(
                    settings.whisper_model,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                    cpu_threads=0,
                    num_workers=1,
                )
    return _model


def preload_model_async() -> None:
    """Start loading Whisper in the background so it is ready when audio arrives."""
    threading.Thread(target=_safe_preload, name="whisper-preload", daemon=True).start()


def _safe_preload() -> None:
    try:
        get_model()
    except Exception:  # the real error will surface when transcription is attempted
        pass


def transcribe_audio(path: str, progress_callback=None) -> list[dict]:
    """Transcribe a complete audio file (fallback when streaming is unavailable)."""
    model = get_model()
    out = []
    with _infer_lock:
        segments, _ = model.transcribe(
            path,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        for i, segment in enumerate(segments):
            text = segment.text.strip()
            if not text:
                continue
            out.append({"text": text, "start": float(segment.start), "end": float(segment.end)})
            if progress_callback and i % 10 == 0:
                # Best-effort progress; exact percentage depends on transcript length.
                progress_callback(min(70, 30 + i))
    if progress_callback:
        progress_callback(70)
    return out


def transcribe_array(audio, offset: float = 0.0, language: str | None = None):
    """Transcribe a 16 kHz mono float32 numpy window.

    Returns ``(segments, language)``. Timestamps are shifted by ``offset`` so they are
    absolute positions in the video. Pass the returned language back in for the next
    window so Whisper does not re-detect (and possibly flip) the language every time.
    """
    model = get_model()
    out = []
    with _infer_lock:
        segments, info = model.transcribe(
            audio,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            language=language,
        )
        for segment in segments:
            text = segment.text.strip()
            if text:
                out.append(
                    {
                        "text": text,
                        "start": offset + float(segment.start),
                        "end": offset + float(segment.end),
                    }
                )
    detected = language
    if detected is None and getattr(info, "language_probability", 1.0) >= 0.5:
        detected = getattr(info, "language", None)
    return out, detected
