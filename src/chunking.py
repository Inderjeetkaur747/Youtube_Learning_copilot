def chunk_segments(segments: list[dict], chunk_seconds: int = 75) -> list[dict]:
    if not segments: return []
    chunks=[]; i=0; idx=0
    while i < len(segments):
        start=i; start_time=float(segments[i]["start"]); j=i
        while j+1 < len(segments) and float(segments[j+1]["end"]) - start_time <= chunk_seconds:
            j += 1
        text=" ".join(s["text"] for s in segments[start:j+1]).strip()
        chunks.append({"chunk_index":idx,"text":text,"start_time":start_time,"end_time":float(segments[j]["end"])})
        idx += 1
        if j >= len(segments)-1: break
        i = max(j, i+1)  # one-segment overlap
    return chunks

def group_chapter_windows(chunks: list[dict], minutes: int = 6) -> list[list[dict]]:
    if not chunks: return []
    width=minutes*60; groups=[]; current=[]; anchor=chunks[0]["start_time"]
    for c in chunks:
        if current and c["start_time"] - anchor >= width:
            groups.append(current); current=[]; anchor=c["start_time"]
        current.append(c)
    if current: groups.append(current)
    return groups


class ChunkStream:
    """Incremental version of ``chunk_segments``.

    Feed transcript segments as they arrive. ``add`` returns only chunks that are
    final (a later segment proved they cannot grow); ``flush`` returns the rest.
    ``add(all) + flush()`` produces exactly what ``chunk_segments(all)`` produces.
    """

    def __init__(self, chunk_seconds: int = 75):
        self.chunk_seconds = chunk_seconds
        self._segments: list[dict] = []
        self._start = 0  # index of the first segment of the chunk being built
        self._next_index = 0

    def add(self, segments: list[dict]) -> list[dict]:
        self._segments.extend(segments)
        return self._drain(final=False)

    def flush(self) -> list[dict]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[dict]:
        segs = self._segments
        out: list[dict] = []
        while self._start < len(segs):
            i = self._start
            start_time = float(segs[i]["start"])
            j = i
            while j + 1 < len(segs) and float(segs[j + 1]["end"]) - start_time <= self.chunk_seconds:
                j += 1
            closed = j + 1 < len(segs)  # the next segment did not fit, so this chunk is final
            if not closed and not final:
                break
            out.append(
                {
                    "chunk_index": self._next_index,
                    "text": " ".join(s["text"] for s in segs[i : j + 1]).strip(),
                    "start_time": start_time,
                    "end_time": float(segs[j]["end"]),
                }
            )
            self._next_index += 1
            if not closed:
                self._start = len(segs)
                break
            self._start = max(j, i + 1)  # one-segment overlap, same as chunk_segments
        if self._start:
            del segs[: self._start]  # drop consumed segments so memory stays flat
            self._start = 0
        return out
