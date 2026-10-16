import random
from .chunking import group_chapter_windows
from .config import settings
from .llm import chat, parse_json
from .youtube_utils import format_timestamp

def generate_chapters(chunks):
    out = []
    for group in group_chapter_windows(chunks, settings.chapter_minutes):
        text = " ".join(c["text"] for c in group)
        prompt = [
            {"role": "system", "content": 'Return JSON only: {"title": "3-7 words", "summary": "one sentence"}.'},
            {"role": "user", "content": text},
        ]
        data = parse_json(chat(prompt, json_mode=True))
        out.append({"title": data["title"], "summary": data["summary"], "start_time": group[0]["start_time"]})
    return out

def generate_summary(chapters):
    outline = "\n".join(f'- {c["title"]}: {c["summary"]}' for c in chapters)
    prompt = [
        {"role": "system", "content": "Return JSON only with tldr string and takeaways array (max 6)."},
        {"role": "user", "content": outline},
    ]
    return parse_json(chat(prompt, json_mode=True))

def validate_quiz_item(item):
    return (
        isinstance(item, dict)
        and isinstance(item.get("question"), str)
        and isinstance(item.get("options"), list)
        and len(item["options"]) == 4
        and isinstance(item.get("answer_index"), int)
        and 0 <= item["answer_index"] < 4
        and isinstance(item.get("explanation"), str)
    )

def shuffle_quiz_item(item, rng=None):
    rng = rng or random
    correct = item["options"][item["answer_index"]]
    options = item["options"][:]
    rng.shuffle(options)
    return {**item, "options": options, "answer_index": options.index(correct)}

def generate_quiz(chunks):
    if not chunks:
        return []
    n = min(settings.quiz_questions, len(chunks))
    idxs = [round(i * (len(chunks) - 1) / max(1, n - 1)) for i in range(n)]
    questions = []
    for idx in idxs:
        chunk = chunks[idx]
        prompt = [
            {"role": "system", "content": "Create one multiple-choice question from the excerpt. JSON only with question, options (4 strings), answer_index (0-3), explanation."},
            {"role": "user", "content": chunk["text"]},
        ]
        try:
            item = parse_json(chat(prompt, json_mode=True))
            if validate_quiz_item(item):
                questions.append({**shuffle_quiz_item(item), "start_time": chunk["start_time"], "review_time": format_timestamp(chunk["start_time"])})
        except Exception:
            pass
    return questions
