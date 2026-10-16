import os
import threading
import time
from contextlib import closing

from .chunking import ChunkStream
from .config import settings
from .embeddings import embed_texts, warmup_async
from .logger import get_logger, timed
from .rag import stream_answer
from .transcribe import transcribe_audio
from .vector_store import (
    delete_video,
    get_video_chunks,
    get_video_meta,
    mark_complete,
    upsert_chunks,
)
from .youtube_utils import (
    download_audio,
    extract_target,
    fetch_captions,
    format_timestamp,
    get_playlist_video_ids,
    get_video_info,
)

log = get_logger(__name__)

_PUBLIC_META_KEYS = ("video_id", "title", "channel", "duration", "webpage_url")


class JobCancelled(RuntimeError):
    """Raised inside ingestion when the user cancels the job."""


def _progress(callback, value, text=None):
    if callback:
        callback(min(100, max(0, int(value))), text)


def _check(cancel):
    if cancel is not None and cancel.is_set():
        raise JobCancelled("Cancelled")


def _public_meta(meta: dict) -> dict:
    """Keep only what the UI/vector store need (drops the large caption-track tables)."""
    return {key: meta.get(key) for key in _PUBLIC_META_KEYS}


def _slice_by_time(segments: list[dict], seconds: float):
    """Yield consecutive groups of segments, each covering about ``seconds`` of video."""
    group: list[dict] = []
    boundary = None
    for seg in segments:
        if boundary is None:
            boundary = float(seg["start"]) + seconds
        if group and float(seg["start"]) >= boundary:
            yield group
            group = []
            boundary = float(seg["start"]) + seconds
        group.append(seg)
    if group:
        yield group


class _Indexer:
    """Chunk, embed and store transcript segments as they arrive.

    Chunks are saved with ``complete=False`` so questions can already be answered from
    the part indexed so far; ``finish`` flips the whole video to complete.
    """

    def __init__(self, meta: dict, on_partial=None):
        self.meta = meta
        self.on_partial = on_partial
        self.stream = ChunkStream(settings.chunk_seconds)
        self.chunk_count = 0
        self.indexed_until = 0.0

    def add(self, segments: list[dict]):
        self._index(self.stream.add(segments))

    def finish(self):
        self._index(self.stream.flush())
        if self.chunk_count == 0:
            raise ValueError("No transcript could be created for this video.")
        mark_complete(self.meta["video_id"])

    def _index(self, chunks: list[dict]):
        if not chunks:
            return
        vectors = embed_texts([c["text"] for c in chunks])
        upsert_chunks(self.meta, chunks, vectors, complete=False)
        self.chunk_count += len(chunks)
        self.indexed_until = chunks[-1]["end_time"]
        if self.on_partial:
            self.on_partial(self.indexed_until, self.chunk_count)


def _index_captions(segments, indexer, meta, progress_callback, cancel):
    total = float(meta.get("duration") or 0) or float(segments[-1]["end"]) or 1.0
    for part in _slice_by_time(segments, settings.caption_slice_seconds):
        _check(cancel)
        indexer.add(part)
        done = min(1.0, float(part[-1]["end"]) / total)
        _progress(
            progress_callback,
            35 + 60 * done,
            f"Indexed up to {format_timestamp(indexer.indexed_until)} — "
            "you can already ask questions",
        )
    indexer.finish()


def _transcribe_streaming(meta, indexer, progress_callback, cancel):
    """Transcribe while audio is still downloading. Raises AudioStreamUnavailable early."""
    from .audio_stream import SAMPLE_RATE, stream_audio_windows
    from .transcribe import preload_model_async, transcribe_array

    duration = float(meta.get("duration") or 0)
    # Load Whisper while yt-dlp is still connecting; the two overlap instead of queueing.
    preload_model_async()
    language = None
    stream = stream_audio_windows(meta["video_id"], cancel=cancel, duration=duration)
    with closing(stream):
        for offset, audio in stream:
            _check(cancel)
            segments, language = transcribe_array(audio, offset, language)
            indexer.add(segments)
            covered = offset + audio.size / SAMPLE_RATE
            if duration:
                pct = 25 + 70 * min(1.0, covered / duration)
                label = f"{format_timestamp(covered)} of {format_timestamp(duration)}"
            else:
                pct = min(90, 25 + covered / 60)
                label = format_timestamp(covered)
            _progress(
                progress_callback,
                pct,
                f"Transcribed {label} — you can already ask questions",
            )
            _check(cancel)  # stop before decoding another window, not after
    _check(cancel)
    indexer.finish()


def _transcribe_download(meta, indexer, progress_callback, cancel, timings):
    """Fallback: download the whole audio file first, then transcribe it."""
    path = None
    try:
        _progress(progress_callback, 25, "Downloading small audio track...")
        with timed(timings, "download"):
            path = download_audio(meta["video_id"])
        _check(cancel)
        _progress(progress_callback, 40, "Transcribing with Whisper...")
        segments = transcribe_audio(path, progress_callback=progress_callback)
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    if not segments:
        raise ValueError("No transcript could be created for this video.")
    _progress(progress_callback, 85, "Saving your learning space...")
    indexer.add(segments)
    indexer.finish()


def process_video(
    value: str,
    progress_callback=None,
    on_meta=None,
    on_partial=None,
    cancel: threading.Event | None = None,
):
    """Prepare one video with cache-first, caption-first, streaming ingestion.

    ``on_meta(meta)`` fires as soon as the title/duration are known.
    ``on_partial(indexed_until_seconds, chunk_count)`` fires each time more of the video
    becomes searchable, so the UI can unlock questions before ingestion finishes.
    ``cancel`` is a ``threading.Event``; set it to stop work at the next safe point.
    """
    timings = {}
    started = time.perf_counter()
    kind, video_id = extract_target(value)
    if kind != "video":
        raise ValueError("This is a playlist URL. Use process_playlist() for playlist links.")

    warmup_async()  # the embedding model loads while we talk to YouTube

    # Cache check happens before ANY YouTube network call.
    _progress(progress_callback, 5, "Checking your learning library...")
    cached_meta = get_video_meta(video_id)
    if cached_meta:
        if cached_meta.get("complete", True):
            timings["total"] = round(time.perf_counter() - started, 3)
            log.info("cache hit video=%s total=%ss", video_id, timings["total"])
            _progress(progress_callback, 100, "Already processed — ready to learn ✨")
            if on_meta:
                on_meta(cached_meta)
            return {
                "meta": cached_meta,
                "timings": timings,
                "cached": True,
                "source": "cache",
            }
        # An earlier run was interrupted part-way. Start clean rather than serve half a video.
        log.info("discarding incomplete index video=%s", video_id)
        delete_video(video_id)

    _check(cancel)
    _progress(progress_callback, 10, "Getting video details...")
    with timed(timings, "info"):
        info = get_video_info(video_id)
    meta = _public_meta(info)
    if on_meta:
        on_meta(meta)

    has_captions = bool(info.get("subtitles") or info.get("automatic_captions"))
    too_long = bool(info["duration"] and info["duration"] > settings.max_video_minutes * 60)
    if too_long and not has_captions:
        # Long videos are fine when captions exist; only Whisper needs the limit.
        raise ValueError(
            f"This video is {info['duration'] // 60} minutes long and has no usable captions. "
            f"For Whisper processing, the limit is {settings.max_video_minutes} minutes."
        )

    indexer = _Indexer(meta, on_partial)
    segments = None
    source = ""
    if settings.use_yt_captions:
        _check(cancel)
        _progress(progress_callback, 20, "Looking for YouTube captions...")
        with timed(timings, "captions"):
            segments = fetch_captions(info)
        if segments:
            source = "YouTube captions"

    if segments:
        _progress(progress_callback, 35, "Captions found — indexing...")
        with timed(timings, "index"):
            _index_captions(segments, indexer, meta, progress_callback, cancel)
    else:
        source = "Whisper"
        if too_long:
            raise ValueError(
                f"This video is longer than {settings.max_video_minutes} minutes and its captions "
                "could not be downloaded. I stopped before starting the slower Whisper fallback."
            )
        _check(cancel)
        with timed(timings, "transcribe"):
            streamed = False
            if settings.stream_whisper:
                from .audio_stream import AudioStreamUnavailable

                try:
                    _progress(progress_callback, 25, "Streaming audio — transcribing as it arrives...")
                    _transcribe_streaming(meta, indexer, progress_callback, cancel)
                    streamed = True
                except AudioStreamUnavailable as exc:
                    log.warning("audio streaming unavailable (%s); using full download", exc)
            if not streamed:
                _transcribe_download(meta, indexer, progress_callback, cancel, timings)

    timings["total"] = round(time.perf_counter() - started, 3)
    log.info("ingestion source=%s timings=%s", source, timings)
    _progress(progress_callback, 100, "Ready to learn ✨")
    return {
        "meta": meta,
        "timings": timings,
        "cached": False,
        "source": source,
        "chunk_count": indexer.chunk_count,
    }


def process_playlist(value: str, progress_callback=None):
    """Process a playlist item-by-item; one failed video does not abort the playlist."""
    kind, _ = extract_target(value)
    if kind != "playlist":
        raise ValueError("Expected a YouTube playlist URL.")

    video_ids = get_playlist_video_ids(value, settings.max_playlist_videos)
    if not video_ids:
        raise ValueError("I could not find any videos in that playlist.")

    results = []
    failures = []
    total = len(video_ids)

    for index, video_id in enumerate(video_ids, 1):
        base = ((index - 1) / total) * 100
        span = 100 / total

        def item_progress(value, text=None, base=base, span=span, index=index):
            overall = base + (max(0, min(100, value)) / 100) * span
            if progress_callback:
                progress_callback(
                    overall,
                    f"Playlist: video {index}/{total} — {text or 'processing...'}",
                )

        try:
            result = process_video(video_id, progress_callback=item_progress)
            results.append(result)
        except Exception as exc:
            log.warning("playlist item failed video=%s error=%s", video_id, exc)
            failures.append({"video_id": video_id, "error": str(exc)})

    if not results:
        details = "; ".join(f["error"] for f in failures[:3])
        raise RuntimeError(f"No playlist videos could be prepared. {details}")

    return {
        "results": results,
        "failures": failures,
        "processed": len(results),
        "requested": len(video_ids),
        "playlist_limited": len(video_ids) >= settings.max_playlist_videos,
    }


def discard_incomplete(video_id: str) -> None:
    """Remove a half-indexed video (used after a cancelled or failed run)."""
    meta = get_video_meta(video_id)
    if meta and not meta.get("complete", True):
        delete_video(video_id)


def prepare_answer(question, video_ids, history):
    return stream_answer(question, video_ids, history, settings.top_k)


def all_chunks(video_id):
    return get_video_chunks(video_id)
