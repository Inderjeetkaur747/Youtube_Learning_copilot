from .config import settings
from .embeddings import embed_query
from .llm import chat, chat_stream
from .vector_store import search
from .youtube_utils import format_timestamp

SYSTEM = """You are YouTube Learning Copilot, a friendly AI tutor for students.

Grounding rules:
- Use ONLY the numbered video excerpts for factual claims about the video.
- Cite relevant claims with [n]. Never invent a citation number.
- If the excerpts do not contain the answer, say that the video does not seem to cover it.
- Explain concepts step-by-step when useful, but do not add facts that are not supported by the excerpts.
- Keep answers concise first, then add detail if the student asks for more.

Language rules:
- Answer in the same language/style as the student's latest question.
- If the student uses English, answer in clear simple English.
- If the student uses Hinglish/Roman Hindi, answer naturally in Hinglish/Roman Hindi while keeping technical terms in English when that is clearer.
- Do not force Hindi script unless the student uses it.
- Preserve the student's technical vocabulary and level.
"""

HINGLISH_MARKERS = {
    "kya", "kaise", "kyun", "kyu", "kyunki", "batao", "samjhao", "samjha", "samajh",
    "matlab", "mujhe", "mera", "meri", "mere", "hum", "hume", "hai", "hain", "ho", "hoga",
    "hoti", "hota", "ka", "ke", "ki", "me", "mein", "se", "ko", "ye", "woh", "isko", "usko",
    "acha", "accha", "phir", "bas", "thoda", "simple", "kyon", "konsa", "kaunsa", "difference",
}


def _looks_hinglish(text: str) -> bool:
    words = {w.strip(".,!?;:()[]{}\"").lower() for w in text.split() if w.strip()}
    return len(words & HINGLISH_MARKERS) >= 1


def rewrite_question(question: str, history: list[dict]) -> str:
    """Create a retrieval-friendly standalone English query only when needed."""
    needs_rewrite = bool(history) or (settings.normalize_hinglish and _looks_hinglish(question))
    if not needs_rewrite:
        return question

    try:
        recent = history[-4:]
        prompt = [
            {
                "role": "system",
                "content": (
                    "Rewrite the student's newest question into one short, standalone, retrieval-friendly "
                    "English search question. If the question is Hinglish/Roman Hindi, translate its meaning "
                    "faithfully into English. Preserve technical terms exactly. Do not answer the question. "
                    "Return only the rewritten question."
                ),
            },
            {"role": "user", "content": f"History: {recent}\nQuestion: {question}"},
        ]
        return chat(prompt, temperature=0).strip() or question
    except Exception:
        return question


def build_prompt(question: str, sources: list[dict], history: list[dict]):
    excerpts = []
    for i, source in enumerate(sources, 1):
        excerpts.append(
            f'[{i}] "{source.get("video_title", "Video")}" '
            f'({format_timestamp(source["start_time"])}-{format_timestamp(source["end_time"])}) '
            f'{source["text"]}'
        )
    messages = [{"role": "system", "content": SYSTEM}]
    for message in history[-4:]:
        messages.append({"role": message["role"], "content": message["content"]})
    content = (
        "Retrieved video excerpts:\n"
        + "\n\n".join(excerpts)
        + "\n\nStudent's original question: "
        + question
        + "\n\nAnswer the student directly in the same language/style as the original question."
    )
    messages.append({"role": "user", "content": content})
    return messages


def prepare(question: str, video_ids: list[str], history: list[dict], top_k: int):
    standalone = rewrite_question(question, history)
    sources = search(embed_query(standalone), video_ids, top_k)
    return standalone, sources, build_prompt(question, sources, history)


def stream_answer(question: str, video_ids: list[str], history: list[dict], top_k: int):
    standalone, sources, messages = prepare(question, video_ids, history, top_k)
    return chat_stream(messages, temperature=0.2), sources, standalone
