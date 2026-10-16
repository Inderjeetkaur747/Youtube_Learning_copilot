import pytest

np = pytest.importorskip("numpy")

from src.audio_stream import iter_windows  # noqa: E402


def blocks(total, size):
    left = total
    while left > 0:
        n = min(size, left)
        yield np.ones(n, dtype=np.float32)
        left -= n


def test_first_window_is_short_then_regular_windows():
    out = list(iter_windows(blocks(16000 * 50, 4000), first_seconds=10, window_seconds=20, sample_rate=16000))
    assert [round(offset) for offset, _ in out] == [0, 10, 30]
    assert [len(w) // 16000 for _, w in out] == [10, 20, 20]


def test_no_audio_is_lost_or_duplicated():
    total = 16000 * 47 + 123
    out = list(iter_windows(blocks(total, 999), first_seconds=5, window_seconds=11, sample_rate=16000))
    assert sum(len(w) for _, w in out) == total
    # Each window starts exactly where the previous one ended.
    position = 0
    for offset, window in out:
        assert round(offset * 16000) == position
        position += len(window)


def test_tiny_trailing_sliver_is_dropped_but_real_tail_is_kept():
    kept = list(iter_windows(blocks(16000 * 10 + 8000, 5000), 10, 10, 16000))
    assert len(kept) == 2  # 0.5 s of trailing audio is worth transcribing
    dropped = list(iter_windows(blocks(16000 * 10 + 1000, 5000), 10, 10, 16000))
    assert len(dropped) == 1  # 1/16 s of trailing audio is not


def test_short_clip_yields_single_window():
    out = list(iter_windows(blocks(16000 * 3, 1000), 90, 180, 16000))
    assert len(out) == 1 and out[0][0] == 0


def test_empty_stream_yields_nothing():
    assert list(iter_windows(iter(()), 90, 180)) == []


# --- process / error handling, with a fake PyAV and a fake yt-dlp process -------------

import io  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
from unittest.mock import patch  # noqa: E402

from src import audio_stream  # noqa: E402


class FakeFrame:
    def __init__(self, samples):
        self.samples = samples

    def to_ndarray(self):
        return self.samples.reshape(1, -1)


def make_fake_av(frames, fail_at=None):
    class Container:
        streams = types.SimpleNamespace(audio=[object()])
        closed = False

        def decode(self, stream):
            for i, frame in enumerate(frames):
                if fail_at == i:
                    raise OSError("invalid data found")
                yield frame

        def close(self):
            Container.closed = True

    class Resampler:
        def __init__(self, **kwargs):
            pass

        def resample(self, frame):
            return [] if frame is None else [frame]

    return types.SimpleNamespace(open=lambda reader, mode="r": Container(), AudioResampler=Resampler)


class FakeProc:
    def __init__(self, returncode=0):
        self.stdout = io.BytesIO(b"")
        self._rc = returncode
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self._rc
        return self._rc

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def seconds_of_audio(n):
    return FakeFrame(np.ones(16000 * n, dtype=np.float32))


def run_stream(frames, proc=None, fail_at=None, duration=0, consume=None):
    proc = proc or FakeProc()
    fake_av = make_fake_av(frames, fail_at)
    with patch.dict(sys.modules, {"av": fake_av}), patch(
        "src.audio_stream.subprocess.Popen", return_value=proc
    ):
        gen = audio_stream.stream_audio_windows(
            "dQw4w9WgXcQ", duration=duration, first_seconds=5, window_seconds=5
        )
        if consume is not None:
            out = []
            for item in gen:
                out.append(item)
                if len(out) >= consume:
                    gen.close()
                    break
            return out, proc
        return list(gen), proc


def test_stream_yields_windows_from_decoded_frames():
    out, proc = run_stream([seconds_of_audio(4)] * 3)  # 12 s of audio -> 5 + 5 + 2
    assert [round(offset) for offset, _ in out] == [0, 5, 10]
    assert proc.returncode is not None


def test_failure_before_any_audio_allows_fallback():
    with pytest.raises(audio_stream.AudioStreamUnavailable):
        run_stream([seconds_of_audio(4)] * 3, fail_at=0)


def test_empty_stream_allows_fallback():
    with pytest.raises(audio_stream.AudioStreamUnavailable):
        run_stream([])


def test_missing_ytdlp_allows_fallback():
    with patch.dict(sys.modules, {"av": make_fake_av([])}), patch(
        "src.audio_stream.subprocess.Popen", side_effect=OSError("no such file")
    ), pytest.raises(audio_stream.AudioStreamUnavailable):
        list(audio_stream.stream_audio_windows("dQw4w9WgXcQ"))


def test_failure_after_audio_started_is_a_real_error_not_a_silent_fallback():
    with pytest.raises(OSError):
        run_stream([seconds_of_audio(6), seconds_of_audio(6)], fail_at=1)


def test_download_that_dies_early_is_reported():
    proc = FakeProc(returncode=1)
    with pytest.raises(RuntimeError) as err:
        run_stream([seconds_of_audio(6)], proc=proc, duration=600)
    assert "stopped early" in str(err.value)


def test_download_exit_code_is_ignored_when_audio_is_complete():
    out, _ = run_stream([seconds_of_audio(6)], proc=FakeProc(returncode=1), duration=6)
    assert len(out) == 2  # 5 s + 1 s; whole video covered, so no error


def test_closing_the_stream_stops_the_download():
    out, proc = run_stream([seconds_of_audio(6)] * 5, consume=1)
    assert len(out) == 1
    assert proc.terminated, "yt-dlp must be killed when the consumer stops reading"
