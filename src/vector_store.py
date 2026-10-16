import threading
import time
from uuid import NAMESPACE_URL, uuid5

from .config import settings

COLLECTION = "youtube_chunks"
_client = None
_ensured = False
# Background ingestion and the chat UI share one client; local Qdrant is not safe for
# concurrent use from several threads, so every client call happens under this lock.
_LOCK = threading.RLock()
_library_cache: dict = {"at": 0.0, "data": None}


def get_client():
    global _client, _ensured
    with _LOCK:
        if _client is None:
            from qdrant_client import QdrantClient

            _client = (
                QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                    timeout=settings.yt_socket_timeout,
                )
                if settings.qdrant_url
                else QdrantClient(path=settings.qdrant_path)
            )
        if not _ensured:  # checking the collection on every call was pure overhead
            ensure_collection(_client)
            _ensured = True
        return _client


def ensure_collection(client=None):
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

    client = client or _client
    if not client:
        return
    names = {c.name for c in client.get_collections().collections}
    if COLLECTION not in names:
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    if settings.qdrant_url:
        for field, schema in (
            ("video_id", PayloadSchemaType.KEYWORD),
            ("chunk_index", PayloadSchemaType.INTEGER),
        ):
            try:
                client.create_payload_index(
                    COLLECTION,
                    field_name=field,
                    field_schema=schema,
                    wait=True,
                )
            except Exception:
                pass


def invalidate_library() -> None:
    with _LOCK:
        _library_cache["data"] = None


def _video_filter(video_id: str):
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))])


def get_video_meta(video_id: str):
    """Return cached metadata without contacting YouTube.

    ``complete`` is False while a video is still being indexed (or if indexing was
    interrupted). Videos stored by older versions have no flag and count as complete.
    """
    with _LOCK:
        points, _ = get_client().scroll(
            COLLECTION,
            scroll_filter=_video_filter(video_id),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
    if not points:
        return None
    payload = points[0].payload or {}
    return {
        "video_id": video_id,
        "title": payload.get("video_title", video_id),
        "channel": payload.get("channel", "Unknown"),
        "duration": int(payload.get("duration") or 0),
        "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
        "cached": True,
        "complete": bool(payload.get("complete", True)),
    }


def video_exists(video_id: str) -> bool:
    with _LOCK:
        pts, _ = get_client().scroll(
            COLLECTION,
            scroll_filter=_video_filter(video_id),
            limit=1,
            with_payload=False,
        )
    return bool(pts)


def upsert_chunks(
    meta: dict,
    chunks: list[dict],
    vectors: list[list[float]],
    complete: bool = True,
):
    """Upload in batches so a long transcript never becomes one huge write.

    Pass ``complete=False`` while indexing is still in progress, then call
    ``mark_complete`` when the last chunk is stored.
    """
    from qdrant_client.models import PointStruct

    if len(chunks) != len(vectors):
        raise ValueError("Chunk/vector count mismatch during indexing")

    batch_size = max(1, settings.qdrant_upsert_batch_size)

    with _LOCK:
        client = get_client()
        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start : start + batch_size]
            vector_batch = vectors[start : start + batch_size]
            points = []
            for c, v in zip(chunk_batch, vector_batch, strict=True):
                payload = {
                    **c,
                    "video_id": meta["video_id"],
                    "video_title": meta["title"],
                    "channel": meta["channel"],
                    "duration": meta["duration"],
                    "complete": complete,
                }
                points.append(
                    PointStruct(
                        id=str(uuid5(NAMESPACE_URL, f'{meta["video_id"]}:{c["chunk_index"]}')),
                        vector=v,
                        payload=payload,
                    )
                )
            client.upsert(COLLECTION, points=points, wait=True)
    invalidate_library()


def mark_complete(video_id: str) -> None:
    """Flag every chunk of a video as fully indexed."""
    with _LOCK:
        get_client().set_payload(
            COLLECTION,
            payload={"complete": True},
            points=_video_filter(video_id),
            wait=True,
        )
    invalidate_library()


def search(query_vector: list[float], video_ids: list[str] | None, limit: int):
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    flt = None
    if video_ids:
        match = (
            MatchValue(value=video_ids[0])
            if len(video_ids) == 1
            else MatchAny(any=video_ids)
        )
        flt = Filter(must=[FieldCondition(key="video_id", match=match)])
    with _LOCK:
        res = get_client().query_points(
            COLLECTION,
            query=query_vector,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        ).points
    return [{**(p.payload or {}), "score": float(p.score)} for p in res]


def get_video_chunks(video_id: str):
    pts = []
    offset = None
    while True:
        with _LOCK:
            batch, offset = get_client().scroll(
                COLLECTION,
                scroll_filter=_video_filter(video_id),
                limit=256,
                offset=offset,
                with_payload=True,
            )
        pts.extend(p.payload for p in batch)
        if offset is None:
            break
    return sorted(pts, key=lambda x: x["chunk_index"])


_LIBRARY_FIELDS = ["video_id", "video_title", "channel", "duration", "complete"]


def _scan_videos() -> list[dict]:
    seen: dict[str, dict] = {}
    offset = None
    while True:
        with _LOCK:
            batch, offset = get_client().scroll(
                COLLECTION,
                limit=256,
                offset=offset,
                with_payload=_LIBRARY_FIELDS,  # skip the heavy transcript text
                with_vectors=False,
            )
        for p in batch:
            x = p.payload or {}
            vid = x.get("video_id")
            if not vid:
                continue
            entry = seen.setdefault(
                vid,
                {
                    "video_id": vid,
                    "title": x.get("video_title", vid),
                    "channel": x.get("channel", "Unknown"),
                    "duration": x.get("duration", 0),
                    "complete": True,
                },
            )
            if not x.get("complete", True):
                entry["complete"] = False
        if offset is None:
            break
    return list(seen.values())


def list_videos(force: bool = False):
    """List stored videos. Cached briefly: the UI asks on every interaction."""
    with _LOCK:
        cached = _library_cache["data"]
        fresh = time.monotonic() - _library_cache["at"] < settings.library_cache_seconds
        if not force and cached is not None and fresh:
            return [dict(v) for v in cached]
        data = _scan_videos()
        _library_cache.update(at=time.monotonic(), data=data)
        return [dict(v) for v in data]


def delete_video(video_id: str):
    from qdrant_client.models import FilterSelector

    with _LOCK:
        get_client().delete(
            COLLECTION,
            points_selector=FilterSelector(filter=_video_filter(video_id)),
            wait=True,
        )
    invalidate_library()
