import re
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, urlparse

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
PLAYLIST_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PLAYLIST_WORD_RE = re.compile(r"\bplay\s?-?lists?\b", re.IGNORECASE)
_PLAYLIST_PREFIXES = ("PL", "OL", "FL", "RD", "UU", "LL")
_HIDDEN_TITLES = {"[private video]", "[deleted video]"}
# YouTube's "Type: Playlist" search filter.
_PLAYLIST_SEARCH_FILTER = "EgIQAw%3D%3D"


class YouTubeNetworkError(RuntimeError):
    """A user-friendly wrapper around transient YouTube/network failures."""


def _normalise_url(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Please paste a YouTube video or playlist URL.")
    return value if "://" in value else "https://" + value


def _parse_url(value: str):
    return urlparse(_normalise_url(value))


def extract_video_id(value: str) -> str:
    value = value.strip()
    if VIDEO_ID_RE.fullmatch(value):
        return value
    try:
        u = _parse_url(value)
        host = u.netloc.lower().replace("www.", "")
        vid = None
        if host == "youtu.be":
            vid = u.path.strip("/").split("/")[0]
        elif host in {"youtube.com", "m.youtube.com"}:
            if u.path == "/watch":
                vid = parse_qs(u.query).get("v", [None])[0]
            else:
                parts = [p for p in u.path.split("/") if p]
                if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
                    vid = parts[1]
        if vid and VIDEO_ID_RE.fullmatch(vid):
            return vid
    except Exception:
        pass
    raise ValueError("Invalid YouTube video URL or video ID")


def extract_playlist_id(value: str) -> str:
    """Extract a playlist ID from a pure playlist URL."""
    try:
        u = _parse_url(value)
        host = u.netloc.lower().replace("www.", "")
        if host not in {"youtube.com", "m.youtube.com"}:
            raise ValueError
        playlist_id = parse_qs(u.query).get("list", [None])[0]
        if playlist_id and PLAYLIST_ID_RE.fullmatch(playlist_id):
            return playlist_id
    except Exception:
        pass
    raise ValueError("Invalid YouTube playlist URL")


def is_playlist_url(value: str) -> bool:
    """True only when the input is a playlist URL without a usable video ID."""
    try:
        u = _parse_url(value)
        host = u.netloc.lower().replace("www.", "")
        has_list = bool(parse_qs(u.query).get("list", [None])[0])
        has_video = bool(parse_qs(u.query).get("v", [None])[0])
        return host in {"youtube.com", "m.youtube.com"} and has_list and not has_video
    except Exception:
        return False


def extract_target(value: str) -> tuple[str, str]:
    """Return ('video', id) or ('playlist', id)."""
    if is_playlist_url(value):
        return "playlist", extract_playlist_id(value)
    return "video", extract_video_id(value)


def format_timestamp(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def parse_json3(data: dict) -> list[dict]:
    out = []
    for e in data.get("events", []):
        parts = e.get("segs") or []
        text = "".join(p.get("utf8", "") for p in parts).replace("\n", " ").strip()
        if not text:
            continue
        start = e.get("tStartMs", 0) / 1000
        dur = e.get("dDurationMs", 0) / 1000
        out.append({"text": text, "start": start, "end": start + dur})
    return out


def _ydl_opts(*, noplaylist: bool = True, extract_flat: bool = False):
    from .config import settings

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": noplaylist,
        "socket_timeout": settings.yt_socket_timeout,
        "retries": settings.yt_retries,
        "fragment_retries": settings.yt_fragment_retries,
        "extractor_retries": settings.yt_extractor_retries,
        "retry_sleep_functions": {"http": lambda n: min(2 * n, 4)},
        "extract_flat": extract_flat,
    }
    if settings.yt_cookies_file:
        opts["cookiefile"] = settings.yt_cookies_file
    return opts


def _friendly_network_error(exc: Exception) -> YouTubeNetworkError:
    message = str(exc).lower()
    if "getaddrinfo" in message or "name or service not known" in message or "nodename" in message:
        return YouTubeNetworkError(
            "YouTube could not be reached because DNS/network lookup failed. "
            "Please check your internet connection and try again."
        )
    if "timed out" in message or "timeout" in message:
        return YouTubeNetworkError(
            "YouTube took too long to respond. Please try again; long videos and playlists "
            "are processed in smaller steps now."
        )
    return YouTubeNetworkError(
        "YouTube could not be reached right now. Please try again in a moment."
    )


def get_video_info(value: str) -> dict:
    import yt_dlp

    vid = extract_video_id(value)
    url = f"https://www.youtube.com/watch?v={vid}"
    try:
        with yt_dlp.YoutubeDL(_ydl_opts()) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise _friendly_network_error(exc) from exc

    return {
        "video_id": vid,
        "title": info.get("title") or vid,
        "channel": info.get("channel") or info.get("uploader") or "Unknown",
        "duration": int(info.get("duration") or 0),
        "language": info.get("language") or info.get("original_language") or "",
        "subtitles": info.get("subtitles") or {},
        "automatic_captions": info.get("automatic_captions") or {},
        "webpage_url": url,
    }


def _entry_duration(entry: dict) -> int:
    try:
        return int(entry.get("duration") or 0)
    except (TypeError, ValueError):
        return 0


def parse_playlist_data(data: dict, limit: int | None = None) -> dict:
    """Turn yt-dlp's flat playlist info into a small, UI-friendly dict (no network)."""
    entries = []
    seen = set()
    for entry in data.get("entries") or []:
        if not entry:
            continue
        vid = entry.get("id")
        title = entry.get("title") or ""
        if not vid or not VIDEO_ID_RE.fullmatch(vid) or vid in seen:
            continue
        if title.strip().lower() in _HIDDEN_TITLES:
            continue
        seen.add(vid)
        entries.append(
            {
                "video_id": vid,
                "title": title or "Untitled",
                "channel": entry.get("channel") or entry.get("uploader") or "",
                "duration": _entry_duration(entry),
            }
        )
        if limit and len(entries) >= limit:
            break
    count = data.get("playlist_count") or data.get("n_entries")
    return {
        "playlist_id": data.get("id") or "",
        "title": data.get("title") or "Playlist",
        "channel": data.get("channel") or data.get("uploader") or "",
        "video_count": int(count) if isinstance(count, int | float) and count else None,
        "entries": entries,
    }


def get_playlist_entries(value: str, limit: int | None = None) -> dict:
    """List a playlist's videos (title, channel, duration) without downloading media."""
    import yt_dlp

    from .config import settings

    playlist_id = extract_playlist_id(value) if "list=" in value else value
    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    limit = limit or settings.playlist_preview_limit
    opts = _ydl_opts(noplaylist=False, extract_flat=True)
    opts["playlistend"] = limit  # do not page through hundreds of items we will not show
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            data = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise _friendly_network_error(exc) from exc

    parsed = parse_playlist_data(data or {}, limit)
    parsed["playlist_id"] = parsed["playlist_id"] or playlist_id
    return parsed


def get_playlist_video_ids(value: str, limit: int | None = None) -> list[str]:
    """Extract playlist entries without downloading any media."""
    from .config import settings

    limit = limit or settings.max_playlist_videos
    return [e["video_id"] for e in get_playlist_entries(value, limit)["entries"]]


def wants_playlists(query: str) -> bool:
    """True when the user is asking for playlists ("python playlist", "dsa playlists")."""
    return bool(_PLAYLIST_WORD_RE.search(query or ""))


def _playlist_id_from_entry(entry: dict) -> str | None:
    url = entry.get("url") or entry.get("webpage_url") or ""
    list_param = parse_qs(urlparse(url).query).get("list", [None])[0] if url else None
    candidate = list_param or entry.get("id") or ""
    if not candidate or not PLAYLIST_ID_RE.fullmatch(candidate) or VIDEO_ID_RE.fullmatch(candidate):
        return None
    if list_param or candidate.startswith(_PLAYLIST_PREFIXES):
        return candidate
    return None  # channels (UC...) and other result types are not playlists


def parse_playlist_search_entries(entries: list, limit: int) -> list[dict]:
    """Keep only playlist results from a YouTube search and normalise them (no network)."""
    out = []
    seen = set()
    for entry in entries or []:
        if not entry:
            continue
        playlist_id = _playlist_id_from_entry(entry)
        if not playlist_id or playlist_id in seen:
            continue
        seen.add(playlist_id)
        thumbs = entry.get("thumbnails") or []
        thumbnail = (thumbs[-1].get("url") if thumbs else None) or entry.get("thumbnail")
        count = entry.get("playlist_count") or entry.get("n_entries")
        out.append(
            {
                "playlist_id": playlist_id,
                "title": entry.get("title") or "Untitled playlist",
                "channel": entry.get("channel") or entry.get("uploader") or "",
                "video_count": int(count) if isinstance(count, int | float) and count else None,
                "thumbnail": thumbnail,
                "url": f"https://www.youtube.com/playlist?list={playlist_id}",
            }
        )
        if len(out) >= limit:
            break
    return out


def search_playlists(query: str, limit: int | None = None) -> list[dict]:
    """Search YouTube for playlists (not videos)."""
    import yt_dlp

    from .config import settings

    limit = limit or settings.playlist_search_results
    base = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    # First use YouTube's playlist filter; if that yields nothing, scan normal results.
    attempts = ((f"{base}&sp={_PLAYLIST_SEARCH_FILTER}", limit * 2), (base, 30))
    last_error = None
    for url, fetch in attempts:
        opts = _ydl_opts(extract_flat=True)
        opts["playlistend"] = fetch
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                data = ydl.extract_info(url, download=False)
        except Exception as exc:
            last_error = exc
            continue
        found = parse_playlist_search_entries((data or {}).get("entries") or [], limit)
        if found:
            return found
    if last_error is not None:
        raise _friendly_network_error(last_error) from last_error
    return []


def _pick_track(info: dict):
    """Prefer human captions, then auto captions, and prefer the spoken language when known."""
    preferred = (info.get("language") or info.get("original_language") or "").lower()
    preferred_base = preferred.split("-")[0].split("_")[0]

    def rank(lang: str) -> tuple[int, int, str]:
        normalized = lang.lower()
        base = normalized.split("-")[0].split("_")[0]
        if preferred and normalized == preferred:
            return (0, 0, normalized)
        if preferred_base and base == preferred_base:
            return (1, 0, normalized)
        if base == "en":
            return (2, 0, normalized)
        return (3, 0, normalized)

    for source in (info.get("subtitles", {}), info.get("automatic_captions", {})):
        candidates = []
        for lang, tracks in source.items():
            for track in tracks:
                if track.get("ext") == "json3" and track.get("url"):
                    candidates.append((rank(lang), lang, track["url"]))
        if candidates:
            _, lang, url = sorted(candidates)[0]
            return lang, url
    return None


def fetch_captions(info: dict):
    import httpx
    from .config import settings

    chosen = _pick_track(info)
    if not chosen:
        return None
    try:
        with httpx.Client(
            timeout=httpx.Timeout(settings.caption_timeout, connect=settings.caption_timeout),
            follow_redirects=True,
        ) as client:
            response = client.get(chosen[1])
            response.raise_for_status()
            return parse_json3(response.json())
    except Exception as exc:
        # Caption failure should not kill ingestion; Whisper can be the fallback.
        from .logger import get_logger
        get_logger(__name__).warning("caption fetch failed; falling back if allowed: %s", exc)
        return None


def search_youtube(query: str, limit: int = 5) -> list[dict]:
    import yt_dlp

    opts = _ydl_opts(extract_flat=True)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            data = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    except Exception as exc:
        raise _friendly_network_error(exc) from exc

    out = []
    for e in data.get("entries", [])[:limit]:
        if not e:
            continue
        out.append(
            {
                "video_id": e.get("id"),
                "title": e.get("title") or "Untitled",
                "channel": e.get("channel") or e.get("uploader") or "Unknown",
                "duration": int(e.get("duration") or 0),
            }
        )
    return out


def download_audio(video_id: str, workdir: str = "data/tmp") -> str:
    import yt_dlp
    from .config import settings

    Path(workdir).mkdir(parents=True, exist_ok=True)
    template = str(Path(workdir) / f"{video_id}.%(ext)s")
    opts = {
        **_ydl_opts(noplaylist=True),
        "skip_download": False,
        "format": "worstaudio[ext=m4a]/worstaudio/bestaudio",
        "outtmpl": template,
        "postprocessors": [],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
            return ydl.prepare_filename(info)
    except Exception as exc:
        raise _friendly_network_error(exc) from exc
