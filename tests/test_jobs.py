import threading
import time

from src.jobs import DONE, ERROR, JobManager


def wait_for(manager, video_id, states, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        snap = manager.snapshot(video_id)
        if snap and snap["status"] in states:
            return snap
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {states}: {manager.snapshot(video_id)}")


def test_job_reports_progress_meta_and_result():
    def runner(video_id, progress_callback, on_meta, on_partial, cancel):
        on_meta({"video_id": video_id, "title": "T"})
        progress_callback(40, "working")
        on_partial(90.0, 2)
        return {"meta": {"video_id": video_id, "title": "T"}, "source": "YouTube captions"}

    manager = JobManager(runner=runner, workers=1)
    manager.submit("vid")
    snap = wait_for(manager, "vid", {DONE})
    assert snap["meta"]["title"] == "T"
    assert snap["source"] == "YouTube captions"
    assert snap["percent"] == 100
    assert snap["indexed_until"] == 90.0 and snap["chunk_count"] == 2


def test_same_video_is_never_processed_twice_at_once():
    release = threading.Event()
    calls = []

    def runner(video_id, **kwargs):
        calls.append(video_id)
        release.wait(2)
        return {"meta": {"video_id": video_id}, "source": "cache"}

    manager = JobManager(runner=runner, workers=2)
    first = manager.submit("same")
    second = manager.submit("same")
    assert first is second
    release.set()
    wait_for(manager, "same", {DONE})
    assert calls == ["same"]


def test_failure_is_reported_not_raised():
    def runner(video_id, **kwargs):
        raise ValueError("no captions and too long")

    manager = JobManager(runner=runner, workers=1)
    manager.submit("bad")
    snap = wait_for(manager, "bad", {ERROR})
    assert "too long" in snap["error"]
    manager.forget("bad")
    assert manager.snapshot("bad") is None


def test_cancel_sets_event_and_cleans_up(monkeypatch=None):
    from unittest.mock import patch

    started = threading.Event()

    def runner(video_id, cancel, **kwargs):
        started.set()
        cancel.wait(2)
        raise RuntimeError("stopped")

    manager = JobManager(runner=runner, workers=1)
    with patch("src.jobs.pipeline.discard_incomplete") as discard:
        manager.submit("c")
        assert started.wait(2)
        assert manager.cancel("c") is True
        snap = wait_for(manager, "c", {"cancelled"})
        discard.assert_called_once_with("c")
    assert snap["status"] == "cancelled"


def test_finished_job_can_be_resubmitted():
    manager = JobManager(runner=lambda video_id, **kw: {"meta": {"video_id": video_id}}, workers=1)
    manager.submit("again")
    wait_for(manager, "again", {DONE})
    manager.submit("again")
    wait_for(manager, "again", {DONE})
