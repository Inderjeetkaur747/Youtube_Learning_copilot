import html
import time

import streamlit as st

from src.config import settings
from src.features import generate_chapters, generate_quiz, generate_summary
from src.jobs import ACTIVE_STATES, CANCELLED, DONE, ERROR, get_manager
from src.pipeline import all_chunks, prepare_answer
from src.vector_store import delete_video, list_videos
from src.youtube_utils import (
    extract_target,
    format_timestamp,
    get_playlist_entries,
    search_playlists,
    search_youtube,
    wants_playlists,
)

st.set_page_config(
    page_title="YouTube Learning Copilot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULTS = {
    "active": None,
    "player_video": None,
    "start": 0,
    "messages": [],
    "summaries": {},
    "quizzes": {},
    "my_videos": [],
    "library": [],
    "quiz_gen": 0,
    "transcript_source": None,
    "results": [],
    "playlist_results": [],
    "result_kind": "video",
    "playlist_view": None,
    "tracked": [],
    "titles": {},
    "welcome_dismissed": False,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

st.markdown(
    """
<style>
:root { --bg: #07111f; --panel: #0d1a2b; --panel2: #101f33; --border: #20344e; --text: #eef6ff; --muted: #91a4bc; --accent: #54a8ff; }
.stApp { background: radial-gradient(circle at 80% 0%, #112b48 0, #07111f 42%, #050b14 100%); color: var(--text); }
.block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 2rem; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(13,26,43,.72); border: 1px solid var(--border); border-radius: 18px; }
[data-testid="stChatMessage"] { border: 1px solid rgba(73,108,145,.35); border-radius: 16px; background: rgba(11,27,45,.75); margin-bottom: .65rem; }
[data-testid="stChatMessage"] p { line-height: 1.65; }
.stButton > button { border-radius: 10px; border: 1px solid #27405e; background: #0f2136; color: #eaf4ff; transition: all .15s ease; }
.stButton > button:hover { border-color: #54a8ff; transform: translateY(-1px); }
button[kind="primary"] { background: linear-gradient(135deg,#2e83d5,#55aafc) !important; border: 0 !important; color: white !important; }
[data-testid="stChatInput"] { border-color: #2b4b6d !important; background: #0b1726 !important; }
[data-testid="stTabs"] button { color: #9db2c9; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #66b3ff; }
.copilot-title { font-size: 2.35rem; font-weight: 800; letter-spacing: -.04em; margin-bottom: 0; }
.copilot-subtitle { color: var(--muted); margin-top: .2rem; margin-bottom: 1.2rem; }
.badge { display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .65rem; border-radius:999px; background:#0c2138; border:1px solid #244565; color:#a9d5ff; font-size:.78rem; }
.hero { padding: 1.25rem; border:1px solid #203b59; border-radius:20px; background:linear-gradient(135deg,rgba(15,35,58,.96),rgba(9,22,38,.92)); margin-bottom:1rem; }
.hero h3 { margin:.15rem 0 .35rem; }
.hero p { color:#a8b9cb; margin:0; }
.section-label { color:#a9bfd7; font-size:.76rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; margin-bottom:.45rem; }
.source-pill { display:inline-block; padding:.18rem .5rem; border-radius:999px; border:1px solid #254462; color:#8fbde4; font-size:.72rem; background:#0a192a; }
.video-card { padding:.75rem; border-radius:14px; background:#0a1727; border:1px solid #1d334d; }
.tip { color:#90a6bd; font-size:.85rem; padding:.5rem .1rem; }
.pl-meta { color:#8ea5bd; font-size:.78rem; margin-top:-.2rem; }
@media (max-width: 1100px) { .copilot-title {font-size:1.8rem;} }
</style>
""",
    unsafe_allow_html=True,
)


jobs = get_manager()


@st.cache_resource
def _warm_models():
    """Load the embedding model once per server, in the background, before it is needed."""
    from src.embeddings import warmup_async

    warmup_async()
    return True


_warm_models()


def get_library():
    if not settings.shared_library:
        return st.session_state.my_videos
    try:
        return list_videos()
    except Exception:
        return []


def start_learning(video_id, title=None, focus=True):
    """Start ingesting a video in the background and (optionally) show it right away.

    The player appears immediately and streams from YouTube; preparation continues in a
    worker thread, so nothing here waits for a download or a transcript.
    """
    if title:
        st.session_state.titles[video_id] = title
    jobs.submit(video_id)
    if video_id not in st.session_state.tracked:
        st.session_state.tracked.append(video_id)
    if focus:
        st.session_state.update(
            active={
                "video_id": video_id,
                "title": title or st.session_state.titles.get(video_id) or "Loading your video…",
                "pending": True,
            },
            player_video=video_id,
            start=0,
            messages=[],
        )


def open_video(meta):
    video_id = meta["video_id"]
    if jobs.is_active(video_id) or meta.get("complete", True) is False:
        # Still being prepared, or an earlier run was interrupted: (re)start the job.
        start_learning(video_id, meta.get("title"))
        return
    st.session_state.update(
        active={**meta, "pending": False}, player_video=video_id, start=0, messages=[]
    )


def learn(video_id, title=None, focus=True):
    """Open a video that is already in the library, otherwise start preparing it."""
    known = {v["video_id"]: v for v in get_library()}.get(video_id)
    if known and known.get("complete", True) and not jobs.is_active(video_id):
        if focus:
            open_video(known)
        return
    start_learning(video_id, title, focus)


def learn_many(entries):
    for n, entry in enumerate(entries):
        learn(entry["video_id"], entry["title"], focus=n == 0)


def jump_to(video_id, seconds):
    st.session_state.update(player_video=video_id, start=int(seconds))


def open_playlist(source, limit=None):
    """Load a playlist's videos for browsing. Nothing is downloaded or processed yet."""
    limit = limit or settings.playlist_preview_limit
    data = get_playlist_entries(source, limit)
    if not data["entries"]:
        raise ValueError("I could not find any videos in that playlist.")
    st.session_state.playlist_view = {**data, "limit": limit, "source": source}


def close_playlist():
    st.session_state.playlist_view = None


def cancel_job(video_id):
    jobs.cancel(video_id)


def dismiss_job(video_id):
    jobs.forget(video_id)
    if video_id in st.session_state.tracked:
        st.session_state.tracked.remove(video_id)
    active = st.session_state.active
    if active and active["video_id"] == video_id and active.get("pending"):
        st.session_state.active = None


def retry_job(video_id):
    jobs.forget(video_id)
    start_learning(video_id, st.session_state.titles.get(video_id))


def remove_video(video_id):
    # A running job deletes its own half-indexed video when it stops; otherwise delete now.
    if not jobs.cancel(video_id):
        delete_video(video_id)
    st.session_state.my_videos = [v for v in st.session_state.my_videos if v["video_id"] != video_id]
    if video_id in st.session_state.tracked:
        st.session_state.tracked.remove(video_id)
    if st.session_state.active and st.session_state.active["video_id"] == video_id:
        st.session_state.update(active=None, player_video=None, start=0, messages=[])


def reap_jobs():
    """Fold finished background jobs into the session (library, active lesson, toast)."""
    for video_id in list(st.session_state.tracked):
        snap = jobs.snapshot(video_id)
        if snap is None:
            st.session_state.tracked.remove(video_id)
            continue
        active = st.session_state.active
        is_active = bool(active and active["video_id"] == video_id)
        if snap["status"] == DONE:
            meta = snap["meta"] or {"video_id": video_id, "title": video_id}
            if video_id not in [v["video_id"] for v in st.session_state.my_videos]:
                st.session_state.my_videos.append(meta)
            if is_active:
                st.session_state.active = {**meta, "pending": False}
                st.session_state.transcript_source = snap["source"] or "cache"
                st.toast("Video ready — ask me anything!")
            st.session_state.tracked.remove(video_id)
        elif snap["status"] == CANCELLED:
            if is_active and active.get("pending"):
                st.session_state.active = None
            st.session_state.tracked.remove(video_id)


def lesson_state(active):
    """Return (state, snapshot): ready | partial | preparing | failed."""
    if not active.get("pending"):
        return "ready", None
    snap = jobs.snapshot(active["video_id"])
    if snap is None or snap["status"] in (ERROR, CANCELLED):
        return "failed", snap
    if snap["chunk_count"] > 0:
        return "partial", snap
    return "preparing", snap


def _stage(snap):
    """Coarse progress milestones; when one changes the whole page is redrawn."""
    return (bool(snap["meta"]), snap["chunk_count"] > 0, snap["status"])


job_slots: dict = {}
job_stage: dict = {}


def render_jobs_panel():
    """Progress for lessons being prepared. Live bars are filled in by watch_jobs()."""
    for video_id in st.session_state.tracked:
        snap = jobs.snapshot(video_id)
        if snap is None:
            continue
        title = (snap["meta"] or {}).get("title") or st.session_state.titles.get(video_id) or video_id
        with st.container(border=True):
            st.markdown(f"**{html.escape(title)}**")
            if snap["status"] in ACTIVE_STATES:
                job_slots[video_id] = st.empty()
                job_stage[video_id] = _stage(snap)
                job_slots[video_id].progress(
                    snap["percent"] / 100, text=snap["message"] or "Working..."
                )
                st.button("Cancel", key=f"cancel_{video_id}", on_click=cancel_job, args=(video_id,))
            elif snap["status"] == ERROR:
                st.error(snap["error"] or "Could not prepare this video.")
                a, b = st.columns(2)
                a.button("Try again", key=f"retry_{video_id}", on_click=retry_job, args=(video_id,))
                b.button("Dismiss", key=f"dismiss_{video_id}", on_click=dismiss_job, args=(video_id,))


def watch_jobs():
    """Keep progress bars moving until something worth redrawing happens.

    This deliberately runs as the last thing in the script instead of in a timed
    fragment: any click or chat message interrupts it cleanly, so a streaming answer is
    never cut off by a background timer.
    """
    if not job_slots:
        return
    while True:
        for video_id, slot in job_slots.items():
            snap = jobs.snapshot(video_id)
            if snap is None or _stage(snap) != job_stage[video_id]:
                st.rerun()
            slot.progress(snap["percent"] / 100, text=snap["message"] or "Working...")
        time.sleep(0.5)


def render_playlist_browser(library):
    view = st.session_state.playlist_view
    if not view:
        return
    in_library = {v["video_id"]: v for v in library}
    entries = view["entries"]
    with st.container(border=True):
        head, close = st.columns([6, 1])
        head.markdown(f"#### {html.escape(view['title'])}")
        close.button("✕", key="close_playlist", on_click=close_playlist, help="Close playlist")
        count = view.get("video_count") or len(entries)
        st.caption(" · ".join(x for x in (view.get("channel"), f"{count} videos") if x))
        take = min(settings.max_playlist_videos, len(entries))
        st.button(
            f" Learn the first {take} videos",
            key="pl_learn_all",
            type="primary",
            on_click=learn_many,
            args=(entries[:take],),
            help="Starts right away; the first video opens while the rest are prepared in the background.",
        )
        for i, entry in enumerate(entries):
            thumb, info = st.columns([1.1, 2.4])
            thumb.image(f"https://i.ytimg.com/vi/{entry['video_id']}/mqdefault.jpg")
            info.markdown(f"**{i + 1}. {html.escape(entry['title'])}**")
            if entry["duration"]:
                info.markdown(
                    f'<div class="pl-meta">{format_timestamp(entry["duration"])}</div>',
                    unsafe_allow_html=True,
                )
            done = in_library.get(entry["video_id"])
            info.button(
                "✓ Open" if done and done.get("complete", True) else "Learn",
                key=f"pl_{entry['video_id']}",
                on_click=learn,
                args=(entry["video_id"], entry["title"]),
            )
        if len(entries) >= view["limit"] and (view.get("video_count") or 10**9) > len(entries):
            if st.button("Show more videos", key="pl_more"):
                try:
                    with st.spinner("Loading more videos..."):
                        open_playlist(view["source"], view["limit"] + settings.playlist_preview_limit)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not load more videos: {exc}")


def render_sources(sources, key_prefix):
    if not sources:
        return
    st.markdown('<div class="section-label">Video references</div>', unsafe_allow_html=True)
    for n, source in enumerate(sources, 1):
        label = f"[{n}] {format_timestamp(source['start_time'])}"
        if len({x.get('video_id') for x in sources}) > 1:
            label += f" · {source.get('video_title', 'Video')}"
        st.button(label, key=f"{key_prefix}_{n}", on_click=jump_to, args=(source["video_id"], source["start_time"]))
    with st.expander("View excerpts"):
        for n, source in enumerate(sources, 1):
            st.markdown(
                f"**[{n}] {format_timestamp(source['start_time'])}–{format_timestamp(source['end_time'])}**  \n"
                f"{source['text']}"
            )


st.markdown('<div class="copilot-title">YouTube Learning Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="copilot-subtitle">Your AI study partner — ask questions naturally in English or Hinglish, get grounded answers, and jump straight to the lesson.</div>',
    unsafe_allow_html=True,
)

left, mid, right = st.columns([1.05, 1.75, 1.25], gap="large")

reap_jobs()
videos = get_library()

with left:
    st.markdown('<div class="section-label">LEARNING SOURCE</div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["I have a link", " Find a video or playlist"])
    with t1:
        url = st.text_input(
            "YouTube video or playlist URL",
            key="url",
            placeholder="Paste a YouTube link...",
        )
        st.caption("A video starts playing instantly. A playlist shows its videos so you can pick.")
        if st.button("Start learning →", type="primary", disabled=not bool(url.strip()), key="process"):
            try:
                kind, ident = extract_target(url)
                if kind == "playlist":
                    with st.spinner("Loading playlist..."):
                        open_playlist(url)
                else:
                    start_learning(ident)
            except Exception as exc:
                st.error(f"Could not open that link: {exc}")
    with t2:
        search_question = st.text_input(
            "What do you want to learn?",
            key="searchq",
            placeholder="e.g. Explain transformers simply, or: python playlist",
        )
        mode = st.radio(
            "Show me",
            ["Auto", "Videos", "Playlists"],
            horizontal=True,
            key="search_mode",
            help="Auto switches to playlists when your search mentions the word 'playlist'.",
        )
        if st.button("Search YouTube", disabled=not bool(search_question.strip()), key="search"):
            as_playlists = mode == "Playlists" or (mode == "Auto" and wants_playlists(search_question))
            try:
                with st.spinner("Searching playlists..." if as_playlists else "Searching..."):
                    if as_playlists:
                        st.session_state.playlist_results = search_playlists(search_question)
                        st.session_state.results = []
                    else:
                        st.session_state.results = search_youtube(search_question, 5)
                        st.session_state.playlist_results = []
                st.session_state.result_kind = "playlist" if as_playlists else "video"
            except Exception as exc:
                st.error(f"Could not search YouTube: {exc}")

        if st.session_state.result_kind == "playlist":
            if not st.session_state.playlist_results and st.session_state.get("searchq"):
                st.info("No playlists found. Try other words, or switch to Videos.")
            for i, found in enumerate(st.session_state.playlist_results):
                with st.container(border=True):
                    if found.get("thumbnail"):
                        st.image(found["thumbnail"])
                    st.markdown(f"** {html.escape(found['title'])}**")
                    bits = [found.get("channel"), f"{found['video_count']} videos" if found.get("video_count") else None]
                    st.caption(" · ".join(b for b in bits if b) or "Playlist")
                    if st.button("Open playlist", key=f"openpl_{i}"):
                        try:
                            with st.spinner("Loading playlist..."):
                                open_playlist(found["url"])
                        except Exception as exc:
                            st.error(f"Could not open that playlist: {exc}")
        else:
            for i, result in enumerate(st.session_state.results):
                with st.container(border=True):
                    st.image(f"https://i.ytimg.com/vi/{result['video_id']}/mqdefault.jpg")
                    st.markdown(f"**{html.escape(result['title'])}**")
                    st.caption(f"{result['channel']} · {format_timestamp(result['duration'])}")
                    st.button(
                        "Learn from this",
                        key=f"learn_{i}",
                        on_click=start_learning,
                        args=(result["video_id"], result["title"]),
                    )

    render_jobs_panel()
    render_playlist_browser(videos)

    if st.session_state.transcript_source:
        st.markdown(
            f'<span class="source-pill">Transcript: {html.escape(st.session_state.transcript_source)}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### My library")
    if not videos:
        st.markdown('<div class="tip">Your processed lessons will appear here.</div>', unsafe_allow_html=True)
    for video in videos:
        a, b = st.columns([5, 1])
        is_open = st.session_state.active and st.session_state.active["video_id"] == video["video_id"]
        unfinished = video.get("complete", True) is False or jobs.is_active(video["video_id"])
        prefix = (" " if is_open else "") + ("⏳ " if unfinished else "")
        a.button(prefix + video["title"], key=f"open_{video['video_id']}", on_click=open_video, args=(video,))
        b.button("🗑", key=f"del_{video['video_id']}", on_click=remove_video, args=(video["video_id"],))

with mid:
    active = st.session_state.active
    if not active:
        st.markdown(
            '<div class="hero"><span class="badge">● AI TUTOR READY</span><h3>What are you learning today?</h3>'
            '<p>Paste a YouTube video or playlist — or search for one — then ask questions exactly like you would ask a teacher. English, Hinglish, follow-ups — all welcome.</p></div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        if c1.button("Explain a concept", key="quick_explain"):
            st.info("Add a video first, then ask: 'Isko simple language mein samjhao.'")
        if c2.button("Make study notes", key="quick_notes"):
            st.info("Add a video first, then open Summary & chapters.")
    else:
        state, snap = lesson_state(active)
        lesson_title = ((snap or {}).get("meta") or {}).get("title") or active["title"]
        st.markdown(f"### {html.escape(lesson_title)}")
        if state == "preparing":
            st.markdown(
                '<div class="hero"><span class="badge">⏳ GETTING READY</span>'
                "<h3>Your lesson is being prepared</h3>"
                "<p>The video is already playing on the right — you do not have to wait. "
                "Questions unlock as soon as the first minutes are indexed.</p></div>",
                unsafe_allow_html=True,
            )
        elif state == "failed":
            reason = (snap or {}).get("error") or "This lesson is no longer being prepared."
            st.error(f"Could not prepare this video: {reason}")
            st.caption("The player on the right still works. You can try again.")
            st.button(
                "Try again",
                key="retry_active",
                on_click=retry_job,
                args=(active["video_id"],),
            )
        else:
            ready = state == "ready"
            if ready:
                st.markdown('<span class="badge">● AI tutor connected</span>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<span class="badge">● Still indexing — ask about the first '
                    f"{format_timestamp(snap['indexed_until'])} now</span>",
                    unsafe_allow_html=True,
                )
            ask, summary, quiz = st.tabs([" Ask AI Tutor", "Study Notes", "Quiz Me"])

            with ask:
                if not ready:
                    st.info(
                        f"Ready for questions up to {format_timestamp(snap['indexed_until'])}. "
                        "The rest is still being processed in the background."
                    )
                whole = (
                    st.toggle("Search across my whole library", value=False, disabled=len(videos) < 2, key="whole")
                    if len(videos) > 1
                    else False
                )
                if not st.session_state.messages:
                    st.markdown(
                        '<div class="tip">Try: <b>"ye concept simple Hinglish mein samjhao"</b> · <b>"what is the main idea?"</b> · <b>"iska real-world example kya hai?"</b></div>',
                        unsafe_allow_html=True,
                    )
                with st.container(height=480):
                    for i, message in enumerate(st.session_state.messages):
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                            if message.get("sources"):
                                render_sources(message["sources"], f"m{i}")

                question = st.chat_input("Ask your AI tutor in English or Hinglish…")
                if question:
                    st.session_state.messages.append({"role": "user", "content": question})
                    try:
                        ids = [v["video_id"] for v in videos] if whole else [active["video_id"]]
                        history = st.session_state.messages[:-1]
                        stream, sources, _ = prepare_answer(question, ids, history)
                        with st.chat_message("assistant"):
                            answer = st.write_stream(stream)
                            render_sources(sources, f"new{len(st.session_state.messages)}")
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                        if sources:
                            jump_to(sources[0]["video_id"], sources[0]["start_time"])
                            st.rerun()
                    except Exception as exc:
                        st.session_state.messages.pop()
                        st.error(f"Could not answer that question: {exc}")

            with summary:
                video_id = active["video_id"]
                if not ready:
                    st.caption("Study notes unlock when the video finishes processing.")
                if st.button("Generate summary & chapters", key="gensum", disabled=not ready):
                    try:
                        with st.spinner("Creating your study notes..."):
                            chapters = generate_chapters(all_chunks(video_id))
                            data = generate_summary(chapters)
                            st.session_state.summaries[video_id] = {"chapters": chapters, **data}
                    except Exception as exc:
                        st.error(f"Could not create the summary: {exc}")
                data = st.session_state.summaries.get(video_id)
                if data:
                    st.markdown("#### TL;DR")
                    st.write(data.get("tldr", ""))
                    st.markdown("#### Key takeaways")
                    for item in data.get("takeaways", []):
                        st.markdown(f"- {item}")
                    st.markdown("#### Chapters")
                    for i, chapter in enumerate(data.get("chapters", [])):
                        st.button(
                            f"▶ {format_timestamp(chapter['start_time'])}  {chapter['title']}",
                            key=f"chap_{i}",
                            on_click=jump_to,
                            args=(video_id, chapter["start_time"]),
                        )
                        st.caption(chapter["summary"])

            with quiz:
                video_id = active["video_id"]
                existing = st.session_state.quizzes.get(video_id)
                if not ready:
                    st.caption("The quiz unlocks when the video finishes processing.")
                if st.button("New quiz" if existing else "Generate quiz", key="genquiz", disabled=not ready):
                    try:
                        with st.spinner("Creating your quiz..."):
                            st.session_state.quiz_gen += 1
                            st.session_state.quizzes[video_id] = {
                                "gen": st.session_state.quiz_gen,
                                "questions": generate_quiz(all_chunks(video_id)),
                            }
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Could not create the quiz: {exc}")
                quiz_data = st.session_state.quizzes.get(video_id)
                if quiz_data:
                    score = 0
                    placeholder = st.empty()
                    for i, item in enumerate(quiz_data["questions"]):
                        choice = st.radio(
                            f"{i + 1}. {item['question']}",
                            item["options"],
                            index=None,
                            key=f"quiz_{quiz_data['gen']}_{i}",
                        )
                        if choice is not None:
                            chosen = item["options"].index(choice)
                            if chosen == item["answer_index"]:
                                score += 1
                                st.success(item["explanation"])
                            else:
                                st.error(item["explanation"])
                        st.button(
                            f"▶ Review at {item['review_time']}",
                            key=f"review_{quiz_data['gen']}_{i}",
                            on_click=jump_to,
                            args=(video_id, item["start_time"]),
                        )
                    placeholder.markdown(f"### Score: {score} / {len(quiz_data['questions'])}")

with right:
    st.markdown('<div class="section-label">LESSON PLAYER</div>', unsafe_allow_html=True)
    if st.session_state.player_video:
        st.video(
            f"https://www.youtube.com/watch?v={st.session_state.player_video}",
            start_time=int(st.session_state.start),
        )
        st.caption(f"▶ Starting at {format_timestamp(st.session_state.start)}")
    else:
        st.markdown(
            '<div class="hero"><h3>Your lesson</h3><p>The player will appear here. Ask a question and I’ll take you to the relevant moment in the video.</p></div>',
            unsafe_allow_html=True,
        )


watch_jobs()
