from src.chunking import chunk_segments, group_chapter_windows

def seg(i,start,end): return {"text":f"s{i}","start":start,"end":end}

def test_empty(): assert chunk_segments([])==[]
def test_short_one_chunk():
    out=chunk_segments([seg(1,0,10),seg(2,10,20)],75); assert len(out)==1 and out[0]["text"]=="s1 s2"
def test_long_split_and_overlap():
    items=[seg(i,i*30,(i+1)*30) for i in range(5)]
    out=chunk_segments(items,75); assert len(out)>=2; assert "s1" in out[0]["text"] and "s1" in out[1]["text"]
def test_chapter_groups():
    chunks=[{"start_time":0},{"start_time":100},{"start_time":400},{"start_time":500}]
    groups=group_chapter_windows(chunks,6); assert len(groups)==2


def test_chunk_stream_matches_batch_chunking():
    import random

    from src.chunking import ChunkStream

    rng = random.Random(7)
    for _ in range(300):
        t, items = 0.0, []
        for i in range(rng.randint(0, 40)):
            dur = rng.choice([0.5, 2, 5, 12, 40, 90])
            gap = rng.choice([0, 0, 1, 3])
            items.append(seg(i, t + gap, t + gap + dur))
            t += gap + dur
        size = rng.choice([10, 30, 75, 120])
        stream, got, pos = ChunkStream(size), [], 0
        while pos < len(items):
            step = rng.randint(1, 6)
            got += stream.add(items[pos : pos + step])
            pos += step
        got += stream.flush()
        assert got == chunk_segments(items, size)


def test_chunk_stream_only_emits_final_chunks_early():
    from src.chunking import ChunkStream

    stream = ChunkStream(75)
    # 60s of speech: the chunk could still grow, so nothing may be emitted yet.
    assert stream.add([seg(1, 0, 30), seg(2, 30, 60)]) == []
    # This segment does not fit in 75s, so the first chunk is now final.
    early = stream.add([seg(3, 60, 100)])
    assert [c["text"] for c in early] == ["s1 s2"]
