from src.rag import build_prompt

def test_prompt_has_numbered_excerpt_and_timestamp():
    s=[{"video_title":"Demo","start_time":155,"end_time":230,"text":"Race conditions happen when timing changes shared state."}]
    messages=build_prompt("What is a race condition?",s,[])
    text=messages[-1]["content"]
    assert '[1] "Demo" (02:35-03:50)' in text
    assert "What is a race condition?" in text


def test_hinglish_query_is_detected():
    from src.rag import _looks_hinglish
    assert _looks_hinglish("ye concept simple language mein samjhao") is True
    assert _looks_hinglish("What is a race condition?") is False
