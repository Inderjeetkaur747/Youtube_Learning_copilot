from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    use_yt_captions: bool = _bool("USE_YT_CAPTIONS", True)
    max_video_minutes: int = int(os.getenv("MAX_VIDEO_MINUTES", "60"))

    # Network resilience. Keep these modest so a bad YouTube request fails fast.
    yt_socket_timeout: int = int(os.getenv("YT_SOCKET_TIMEOUT", "12"))
    yt_retries: int = int(os.getenv("YT_RETRIES", "2"))
    yt_fragment_retries: int = int(os.getenv("YT_FRAGMENT_RETRIES", "2"))
    yt_extractor_retries: int = int(os.getenv("YT_EXTRACTOR_RETRIES", "2"))
    caption_timeout: float = float(os.getenv("CAPTION_TIMEOUT", "10"))

    # Playlist protection. A playlist is processed video-by-video so one bad item
    # does not kill the whole job.
    max_playlist_videos: int = int(os.getenv("MAX_PLAYLIST_VIDEOS", "10"))
    playlist_preview_limit: int = int(os.getenv("PLAYLIST_PREVIEW_LIMIT", "25"))
    playlist_search_results: int = int(os.getenv("PLAYLIST_SEARCH_RESULTS", "6"))

    # Streaming ingestion. Whisper transcribes audio while it is still arriving, and
    # every finished window is indexed immediately so questions work before the end.
    stream_whisper: bool = _bool("STREAM_WHISPER", True)
    stream_first_window_seconds: int = int(os.getenv("STREAM_FIRST_WINDOW_SECONDS", "90"))
    stream_window_seconds: int = int(os.getenv("STREAM_WINDOW_SECONDS", "180"))
    caption_slice_seconds: int = int(os.getenv("CAPTION_SLICE_SECONDS", "900"))
    job_workers: int = int(os.getenv("JOB_WORKERS", "2"))
    library_cache_seconds: float = float(os.getenv("LIBRARY_CACHE_SECONDS", "10"))

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    qdrant_upsert_batch_size: int = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "64"))
    normalize_hinglish: bool = _bool("NORMALIZE_HINGLISH", True)

    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_path: str = os.getenv("QDRANT_PATH", "data/qdrant")
    chunk_seconds: int = int(os.getenv("CHUNK_SECONDS", "75"))
    top_k: int = int(os.getenv("TOP_K", "3"))
    chapter_minutes: int = int(os.getenv("CHAPTER_MINUTES", "6"))
    quiz_questions: int = int(os.getenv("QUIZ_QUESTIONS", "5"))
    shared_library: bool = _bool("SHARED_LIBRARY", True)
    yt_cookies: str = os.getenv("YT_COOKIES", "")
    yt_cookies_file: str = os.getenv("YT_COOKIES_FILE", "")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
