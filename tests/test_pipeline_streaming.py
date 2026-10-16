import threading
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from src import pipeline
from src.config import settings

INFO = {
    "video_id": "dQw4w9WgXcQ",
    "title": "Demo",
    "channel": "Chan",
    "duration": 600,
    "language": "en",
    "subtitles": {},
    "automatic_captions": {},
    "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
}


def seg(start, end, text="word"):
    return {"text": f"{text}{int(start)}", "start": float(start), "end": float(end)}


class Env:
    """Patches every external dependency of process_video and records what happens."""

    def __init__(self, info=None, captions=None, cached=None, **setting_overrides):
        self.events = []
        self.info = {**INFO, **(info or {})}
        self.captions = captions
        self.cached = cached
        self.settings = replace(settings, **setting_overrides)
        self.stack = ExitStack()

    def __enter__(self):
        ev = self.events

        def upsert(meta, chunks, vectors, complete=True):
            ev.append(("upsert", [c["chunk_index"] for c in chunks], complete))

        def fake_embed(texts):
            return [[0.0, 0.0, 0.0] for _ in texts]

        patches = {
            "settings": self.settings,
            "warmup_async": MagicMock(),
            "get_video_meta": MagicMock(return_value=self.cached),
            "get_video_info": MagicMock(return_value=self.info),
            "fetch_captions": MagicMock(return_value=self.captions),
            "embed_texts": fake_embed,
            "upsert_chunks": upsert,
            "mark_complete": MagicMock(side_effect=lambda vid: ev.append(("complete", vid))),
            "delete_video": MagicMock(side_effect=lambda vid: ev.append(("delete", vid))),
        }
        self.mocks = {}
        for name, value in patches.items():
            self.mocks[name] = self.stack.enter_context(patch(f"src.pipeline.{name}", value))
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def test_complete_cache_hit_skips_youtube_entirely():
    cached = {"video_id": "dQw4w9WgXcQ", "title": "Demo", "complete": True}
    with Env(cached=cached) as env:
        result = pipeline.process_video("dQw4w9WgXcQ")
    assert result["cached"] is True and result["source"] == "cache"
    env.mocks["get_video_info"].assert_not_called()


def test_interrupted_index_is_discarded_and_redone():
    cached = {"video_id": "dQw4w9WgXcQ", "title": "Demo", "complete": False}
    captions = [seg(0, 30), seg(30, 60)]
    with Env(cached=cached, captions=captions) as env:
        result = pipeline.process_video("dQw4w9WgXcQ")
    assert env.events[0] == ("delete", "dQw4w9WgXcQ")
    assert result["cached"] is False


def test_captions_are_indexed_in_slices_and_only_marked_complete_at_the_end():
    captions = [seg(t, t + 30) for t in range(0, 600, 30)]  # 10 minutes
    partials, metas = [], []
    with Env(captions=captions, caption_slice_seconds=180, chunk_seconds=75) as env:
        result = pipeline.process_video(
            "dQw4w9WgXcQ",
            on_meta=metas.append,
            on_partial=lambda until, n: partials.append((until, n)),
        )
    upserts = [e for e in env.events if e[0] == "upsert"]
    assert len(upserts) >= 3, "captions should be indexed slice by slice"
    assert all(e[2] is False for e in upserts), "chunks stay incomplete until the end"
    assert env.events[-1] == ("complete", "dQw4w9WgXcQ")
    assert [e for e in env.events if e[0] == "complete"] == [env.events[-1]]
    assert partials == sorted(partials), "indexed range only moves forward"
    assert result["source"] == "YouTube captions"
    assert result["chunk_count"] == partials[-1][1]
    assert metas and metas[0]["title"] == "Demo"
    assert "subtitles" not in result["meta"], "heavy caption tables are not leaked to the UI"


def test_chunk_indexes_are_continuous_across_slices():
    captions = [seg(t, t + 30) for t in range(0, 900, 30)]
    with Env(captions=captions, caption_slice_seconds=120) as env:
        pipeline.process_video("dQw4w9WgXcQ")
    indexes = [i for e in env.events if e[0] == "upsert" for i in e[1]]
    assert indexes == list(range(len(indexes)))


def fake_stream_factory(env, windows, consumed):
    """A stand-in for audio_stream.stream_audio_windows yielding fake audio windows."""
    import numpy as np

    def stream(video_id, cancel=None, duration=0, **kw):
        for offset in windows:
            consumed.append(offset)
            env.events.append(("window", offset))
            yield float(offset), np.zeros(16000 * 100, dtype=np.float32)

    return stream


def test_whisper_streams_and_indexes_before_the_audio_has_finished():
    pytest.importorskip("numpy")
    consumed = []
    partial_seen_at = []

    def fake_transcribe(audio, offset, language):
        base = int(offset)
        return [seg(base + 0, base + 40), seg(base + 40, base + 80), seg(base + 80, base + 100)], "en"

    with Env(stream_whisper=True, chunk_seconds=75) as env:
        stream = fake_stream_factory(env, [0, 100, 200], consumed)
        with patch("src.audio_stream.stream_audio_windows", stream), patch(
            "src.transcribe.transcribe_array", fake_transcribe
        ), patch("src.transcribe.preload_model_async"):
            result = pipeline.process_video(
                "dQw4w9WgXcQ",
                on_partial=lambda until, n: partial_seen_at.append(len(consumed)),
            )
    assert result["source"] == "Whisper"
    # The first chunks became searchable while later windows had not been read yet.
    assert partial_seen_at[0] < 3
    assert env.events[-1][0] == "complete"
    windows = [e for e in env.events if e[0] == "window"]
    first_upsert = next(i for i, e in enumerate(env.events) if e[0] == "upsert")
    assert first_upsert < env.events.index(windows[-1]), "indexing starts before the last window"


def test_whisper_falls_back_to_full_download_when_streaming_is_unavailable():
    pytest.importorskip("numpy")
    from src.audio_stream import AudioStreamUnavailable

    def broken_stream(*args, **kwargs):
        raise AudioStreamUnavailable("PyAV missing")
        yield  # pragma: no cover

    with Env(stream_whisper=True) as env, patch(
        "src.audio_stream.stream_audio_windows", broken_stream
    ), patch("src.transcribe.preload_model_async"), patch(
        "src.pipeline.download_audio", return_value="/nonexistent/audio.m4a"
    ) as download, patch(
        "src.pipeline.transcribe_audio", return_value=[seg(0, 30), seg(30, 60)]
    ) as transcribe:
        result = pipeline.process_video("dQw4w9WgXcQ")
    download.assert_called_once()
    transcribe.assert_called_once()
    assert result["source"] == "Whisper"
    assert env.events[-1][0] == "complete"


def test_streaming_can_be_switched_off():
    with Env(stream_whisper=False) as env, patch(
        "src.pipeline.download_audio", return_value="/nonexistent/a.m4a"
    ), patch("src.pipeline.transcribe_audio", return_value=[seg(0, 30)]), patch(
        "src.pipeline._transcribe_streaming"
    ) as streaming:
        pipeline.process_video("dQw4w9WgXcQ")
    streaming.assert_not_called()
    assert env.events[-1][0] == "complete"


def test_cancel_stops_streaming_and_never_marks_complete():
    pytest.importorskip("numpy")
    cancel = threading.Event()
    consumed = []

    def fake_transcribe(audio, offset, language):
        cancel.set()  # the user hits cancel while the first window is being processed
        return [seg(int(offset), int(offset) + 40)], "en"

    with Env(stream_whisper=True) as env:
        stream = fake_stream_factory(env, [0, 100, 200], consumed)
        with patch("src.audio_stream.stream_audio_windows", stream), patch(
            "src.transcribe.transcribe_array", fake_transcribe
        ), patch("src.transcribe.preload_model_async"):
            with pytest.raises(pipeline.JobCancelled):
                pipeline.process_video("dQw4w9WgXcQ", cancel=cancel)
    assert not any(e[0] == "complete" for e in env.events)
    assert consumed == [0], "no further audio is pulled after cancel"


def test_long_video_without_captions_is_refused_before_any_work():
    with Env(info={"duration": 3 * 3600}, captions=None) as env, pytest.raises(ValueError) as err:
        pipeline.process_video("dQw4w9WgXcQ")
    assert "no usable captions" in str(err.value)
    assert not env.events


def test_long_video_with_captions_is_allowed():
    info = {"duration": 3 * 3600, "automatic_captions": {"en": [{"ext": "json3", "url": "u"}]}}
    captions = [seg(t, t + 30) for t in range(0, 600, 30)]
    with Env(info=info, captions=captions) as env:
        result = pipeline.process_video("dQw4w9WgXcQ")
    assert result["source"] == "YouTube captions"
    assert env.events[-1][0] == "complete"


def test_empty_transcript_raises_a_clear_error():
    with Env(captions=[]) as env, patch(
        "src.pipeline.download_audio", return_value="/nonexistent/a.m4a"
    ), patch("src.pipeline.transcribe_audio", return_value=[]), pytest.raises(ValueError) as err:
        pipeline.process_video("dQw4w9WgXcQ")
    assert "No transcript" in str(err.value)
    assert not any(e[0] == "complete" for e in env.events)
