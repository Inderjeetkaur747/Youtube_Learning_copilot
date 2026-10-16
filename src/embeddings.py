import threading

_model = None
_load_lock = threading.Lock()
_encode_lock = threading.Lock()
_warm_started = False


def _get_model():
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                from .config import settings

                _model = SentenceTransformer(settings.embedding_model, device="cpu")
    return _model


def warmup() -> None:
    """Load the embedding model now so the first real request does not pay for it."""
    try:
        _get_model()
    except Exception:  # surfaced later, when embeddings are actually needed
        pass


def warmup_async() -> None:
    """Start warming the model in the background. Only the first call does anything."""
    global _warm_started
    if _warm_started:
        return
    _warm_started = True
    threading.Thread(target=warmup, name="embedding-warmup", daemon=True).start()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    from .config import settings

    model = _get_model()
    with _encode_lock:  # background indexing and chat queries may overlap
        arr = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    return arr.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
