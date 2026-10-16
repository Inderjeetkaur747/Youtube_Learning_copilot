"""Stream a video's audio straight into the transcriber.

Instead of downloading the whole file and only then starting Whisper, yt-dlp writes the
audio to a pipe, PyAV decodes it on the fly, and we hand Whisper one window at a time.
The first window is ready after seconds, not after the whole download.
"""

import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator

import numpy as np

SAMPLE_RATE = 16000
# Lowest-bitrate audio is plenty for speech. Prefer webm/opus: it decodes sequentially
# from a pipe without needing to seek.
AUDIO_FORMAT = "worstaudio[ext=webm]/worstaudio[ext=m4a]/worstaudio/bestaudio"


class AudioStreamUnavailable(RuntimeError):
    """Streaming could not start. Nothing was produced, so the caller can fall back."""


def iter_windows(
    blocks: Iterable[np.ndarray],
    first_seconds: float,
    window_seconds: float,
    sample_rate: int = SAMPLE_RATE,
) -> Iterator[tuple[float, np.ndarray]]:
    """Regroup decoded audio blocks into ``(offset_seconds, samples)`` windows.

    The first window is shorter so the first transcript appears quickly.
    """
    target = max(sample_rate, int(first_seconds * sample_rate))
    later = max(sample_rate, int(window_seconds * sample_rate))
    pending: list[np.ndarray] = []
    pending_len = 0
    done_samples = 0
    for block in blocks:
        if block.size == 0:
            continue
        pending.append(block)
        pending_len += block.size
        while pending_len >= target:
            data = np.concatenate(pending)
            yield done_samples / sample_rate, data[:target]
            rest = data[target:]
            done_samples += target
            pending = [rest] if rest.size else []
            pending_len = rest.size
            target = later
    if pending_len >= int(0.3 * sample_rate):  # ignore a sliver of trailing silence
        yield done_samples / sample_rate, np.concatenate(pending)


class _PipeReader:
    """Expose only ``read`` so PyAV treats the pipe as a non-seekable stream."""

    def __init__(self, raw):
        self._raw = raw

    def read(self, size=-1):
        return self._raw.read(size)


def _resample(resampler, frame) -> list[np.ndarray]:
    out = resampler.resample(frame)
    if out is None:
        return []
    frames = out if isinstance(out, list | tuple) else [out]
    return [f.to_ndarray().reshape(-1).astype(np.float32, copy=False) for f in frames]


def _decode_blocks(reader) -> Iterator[np.ndarray]:
    import av

    container = av.open(reader, mode="r")
    try:
        if not container.streams.audio:
            raise AudioStreamUnavailable("The downloaded stream has no audio track.")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            yield from _resample(resampler, frame)
        yield from _resample(resampler, None)
    finally:
        container.close()


def build_ytdlp_command(video_id: str) -> list[str]:
    from .config import settings

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--no-cache-dir",
        "--socket-timeout",
        str(settings.yt_socket_timeout),
        "--retries",
        str(settings.yt_retries),
        "--fragment-retries",
        str(settings.yt_fragment_retries),
        "-f",
        AUDIO_FORMAT,
        "-o",
        "-",
    ]
    if settings.yt_cookies_file:
        cmd += ["--cookies", settings.yt_cookies_file]
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")
    return cmd


def _stderr_tail(handle) -> str:
    try:
        handle.seek(0)
        text = handle.read().decode("utf-8", "replace").strip()
    except Exception:
        return ""
    return text[-300:]


def _stop(proc) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def stream_audio_windows(
    video_id: str,
    cancel=None,
    duration: float = 0,
    first_seconds: float | None = None,
    window_seconds: float | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(offset_seconds, float32_16k_mono_samples)`` while audio is downloading.

    Raises ``AudioStreamUnavailable`` if nothing could be produced (so the caller may
    fall back to a normal download); any later failure is raised as a regular error.
    """
    from .config import settings

    first = first_seconds if first_seconds is not None else settings.stream_first_window_seconds
    window = window_seconds if window_seconds is not None else settings.stream_window_seconds

    try:
        import av  # noqa: F401
    except ImportError as exc:
        raise AudioStreamUnavailable("PyAV is not installed.") from exc

    errors = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            build_ytdlp_command(video_id), stdout=subprocess.PIPE, stderr=errors
        )
    except OSError as exc:
        errors.close()
        raise AudioStreamUnavailable(f"Could not start yt-dlp: {exc}") from exc

    produced = False
    covered = 0.0
    try:
        try:
            blocks = _decode_blocks(_PipeReader(proc.stdout))
            for offset, samples in iter_windows(blocks, first, window):
                if cancel is not None and cancel.is_set():
                    return
                produced = True
                covered = offset + samples.size / SAMPLE_RATE
                yield offset, samples
        except AudioStreamUnavailable:
            raise
        except Exception as exc:
            if produced:
                raise
            detail = _stderr_tail(errors)
            raise AudioStreamUnavailable(
                f"Audio stream could not be decoded: {exc}. {detail}".strip()
            ) from exc

        if not produced:
            raise AudioStreamUnavailable(f"yt-dlp produced no audio. {_stderr_tail(errors)}".strip())
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        if proc.returncode not in (0, None) and duration and covered < 0.9 * duration:
            raise RuntimeError(
                "The audio download stopped early "
                f"(got {int(covered)}s of {int(duration)}s). {_stderr_tail(errors)}".strip()
            )
    finally:
        _stop(proc)
        errors.close()
