"""Background ingestion jobs.

The Streamlit script must never block on YouTube or Whisper: it starts a job here, shows
the player immediately, and polls ``JobManager.snapshot`` to draw progress. One manager
lives for the whole server process, so two visitors asking for the same video share one job.
"""

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from . import pipeline
from .config import settings
from .logger import get_logger

log = get_logger(__name__)

QUEUED, RUNNING, DONE, ERROR, CANCELLED = "queued", "running", "done", "error", "cancelled"
ACTIVE_STATES = {QUEUED, RUNNING}


class Job:
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.status = QUEUED
        self.percent = 0
        self.message = "Waiting to start..."
        self.meta: dict | None = None
        self.source = ""
        self.error = ""
        self.indexed_until = 0.0
        self.chunk_count = 0
        self.created = time.time()
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()

    # -- callbacks handed to the pipeline (called from the worker thread) --------------
    def on_progress(self, value, text=None):
        with self._lock:
            self.percent = int(value)
            if text:
                self.message = text

    def on_meta(self, meta):
        with self._lock:
            self.meta = dict(meta)

    def on_partial(self, indexed_until, chunk_count):
        with self._lock:
            self.indexed_until = float(indexed_until)
            self.chunk_count = int(chunk_count)

    def _finish(self, status, **fields):
        with self._lock:
            self.status = status
            for key, value in fields.items():
                setattr(self, key, value)

    def snapshot(self) -> dict:
        """A plain-dict copy that is safe to read from the UI thread."""
        with self._lock:
            return {
                "video_id": self.video_id,
                "status": self.status,
                "percent": self.percent,
                "message": self.message,
                "meta": dict(self.meta) if self.meta else None,
                "source": self.source,
                "error": self.error,
                "indexed_until": self.indexed_until,
                "chunk_count": self.chunk_count,
                "created": self.created,
            }


class JobManager:
    def __init__(self, runner: Callable | None = None, workers: int | None = None):
        self._runner = runner
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, workers or settings.job_workers), thread_name_prefix="ingest"
        )
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _get_runner(self):
        # Looked up at call time so tests (and hot reloads) can swap pipeline.process_video.
        return self._runner if self._runner is not None else pipeline.process_video

    def submit(self, video_id: str) -> Job:
        """Start ingesting a video, or return the job already working on it."""
        with self._lock:
            existing = self._jobs.get(video_id)
            if existing and existing.status in ACTIVE_STATES:
                return existing
            job = Job(video_id)
            self._jobs[video_id] = job
        self._pool.submit(self._run, job)
        return job

    def _run(self, job: Job):
        if job.cancel_event.is_set():
            job._finish(CANCELLED, message="Cancelled")
            return
        job._finish(RUNNING, message="Starting...")
        try:
            result = self._get_runner()(
                job.video_id,
                progress_callback=job.on_progress,
                on_meta=job.on_meta,
                on_partial=job.on_partial,
                cancel=job.cancel_event,
            )
        except pipeline.JobCancelled:
            self._discard(job.video_id)
            job._finish(CANCELLED, message="Cancelled")
            return
        except Exception as exc:
            if job.cancel_event.is_set():
                self._discard(job.video_id)
                job._finish(CANCELLED, message="Cancelled")
            else:
                log.warning("job failed video=%s error=%s", job.video_id, exc)
                job._finish(ERROR, error=str(exc), message="Could not prepare this video")
            return
        job.on_meta(result["meta"])
        job._finish(
            DONE,
            percent=100,
            message="Ready to learn ✨",
            source=result.get("source", ""),
        )

    @staticmethod
    def _discard(video_id: str):
        try:
            pipeline.discard_incomplete(video_id)
        except Exception as exc:  # cleanup is best-effort
            log.warning("could not discard partial video=%s error=%s", video_id, exc)

    def snapshot(self, video_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(video_id)
        return job.snapshot() if job else None

    def cancel(self, video_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(video_id)
        if job and job.status in ACTIVE_STATES:
            job.cancel_event.set()
            return True
        return False

    def forget(self, video_id: str):
        """Drop a finished job (e.g. after the user dismisses an error)."""
        with self._lock:
            job = self._jobs.get(video_id)
            if job and job.status not in ACTIVE_STATES:
                del self._jobs[video_id]

    def is_active(self, video_id: str) -> bool:
        snap = self.snapshot(video_id)
        return bool(snap and snap["status"] in ACTIVE_STATES)


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    """The process-wide manager (created on first use)."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager
