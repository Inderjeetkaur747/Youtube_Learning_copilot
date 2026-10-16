#  YouTube Learning Copilot

Turn a YouTube video into an interactive tutor. Paste a video link or search for a topic, then ask questions, get answers grounded in the video with numbered citations, jump to the cited moment, generate chapters and a summary, and test yourself with a quiz.

## Why this project exists

Long learning videos are useful, but finding one exact explanation again can be slow. This app prepares the video's transcript once, stores small timestamped chunks in a vector database, and retrieves the most relevant chunks whenever you ask a question. The LLM is told to answer only from those retrieved excerpts.

## Features

| Feature | What it does |
|---|---|
| Paste a YouTube URL | Loads a video directly |
| Search YouTube | Finds up to 5 videos without an API key |
| Instant playback | The video starts playing the moment you submit it; preparation runs in the background |
| Streaming ingestion | Whisper transcribes audio while it downloads; questions unlock as soon as the first minutes are indexed |
| Playlist search | Ask for a "python playlist" (or pick Playlists) and get playlists, not videos |
| Playlist browser | Paste a playlist link to see its videos and choose which to learn |
| Fast captions first | Uses YouTube captions when available |
| Whisper fallback | Transcribes audio only when captions are unavailable |
| RAG chat | Retrieves top-K transcript chunks and asks Groq to answer from them |
| Clickable citations | Jump the player to the source timestamp |
| Follow-up memory | Rewrites questions like “explain that more simply” into standalone questions |
| Summary + chapters | Builds a chapter outline and concise study notes |
| Quiz | Creates 5 MCQs with explanations and review timestamps |
| Video library | Reopen or delete processed videos |
| Multi-video search | Optionally retrieve from the whole library |
| FastAPI | Optional REST endpoints |
| Docker + tests + CI | Reproducible setup and basic quality checks |

## Simple architecture

### Part A — prepare a video

```text
YouTube URL / search result
        │
        ▼
yt-dlp metadata + caption tracks
        │
        ├── captions available ──► parse timestamped captions
        │
        └── no captions ──► stream audio ──► Faster-Whisper (window by window)
                                            │
                                            ▼
                                  timestamped segments
                                            │
                                            ▼
                                   ~75 second chunks
                                            │
                                            ▼
                              MiniLM 384-d embeddings
                                            │
                                            ▼
                                     Qdrant storage
```

### Part B — answer a question

Example question: **“What is a race condition?”**

```text
Question
   │
   ├── if it is a follow-up: rewrite to a standalone question
   │
   ▼
MiniLM query embedding
   │
   ▼
Qdrant cosine search (top 3)
   │
   ▼
[1] 02:35-03:50 excerpt
[2] 08:10-09:18 excerpt
[3] 14:02-15:03 excerpt
   │
   ▼
Groq LLM: answer only from these excerpts
   │
   ▼
Streamed answer with citations such as [1]
   │
   ▼
Click ▶ [1] 02:35 to jump the YouTube player
```

## Tools used and why

- **Streamlit** — quick interactive Python UI with chat, tabs, buttons and video playback.
- **yt-dlp** — reads YouTube metadata, captions, search results and audio without requiring a YouTube API key.
- **Faster-Whisper** — CPU-friendly speech-to-text fallback.
- **sentence-transformers / all-MiniLM-L6-v2** — small embedding model; normalized 384-dimensional vectors work well for semantic search.
- **Qdrant** — vector database used for cosine-similarity retrieval and metadata filtering by video.
- **Groq** — streams LLM answers and returns JSON for study tools.
- **FastAPI** — optional REST interface.
- **pytest + Ruff + GitHub Actions** — tests and code-quality checks.

This project intentionally does **not** use LangChain, LlamaIndex, BM25, hybrid search, reranking, RRF, or a similarity threshold. The RAG pipeline is written directly so it is easy to explain in an interview.

## Project structure

```text
youtube-learning-copilot/
├── app.py
├── api.py
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── ruff.toml
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── src/
│   ├── config.py
│   ├── logger.py
│   ├── youtube_utils.py
│   ├── transcribe.py
│   ├── audio_stream.py
│   ├── jobs.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm.py
│   ├── rag.py
│   ├── features.py
│   └── pipeline.py
├── benchmarks/benchmark.py
├── eval/generate_eval_set.py
├── eval/run_eval.py
├── tests/            # incl. streaming, jobs, playlist tests
├── README.md
└── DEPLOY.md
```

## Setup — Windows

Use Python 3.11 for the smoothest ML dependency setup.

```powershell
cd youtube-learning-copilot
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and replace `your_key_here` with your Groq API key. Then run:

```powershell
streamlit run app.py
```

## Setup — macOS / Linux

```bash
cd youtube-learning-copilot
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`, add `GROQ_API_KEY`, then:

```bash
streamlit run app.py
```

## How to use the app

1. Open **I have a link** and paste a YouTube video or playlist URL, or open **Find a video or playlist** and search.
   - A **video** starts playing immediately while it is prepared in the background.
   - A **playlist** (pasted, or found by searching e.g. `python playlist`, or by setting *Show me* to Playlists) opens a list of its videos. Click **Learn** on one, or **Learn the first N videos**.
2. Watch the progress bar on the left. You do **not** need to wait for it to finish: once the first minutes are indexed, the tutor shows *Still indexing — ask about the first mm:ss now*. Study notes and quiz unlock when processing completes. Use **Cancel** to stop a job.
3. In ** Ask AI Tutor**, ask a question about the video.
4. Read the streamed answer. Use the citation buttons under it to jump to the relevant moment.
5. Ask a follow-up such as **“explain that more simply.”**
6. Open ** Study Notes** to create the summary and chapters (after processing finishes).
7. Open ** Quiz Me** to create MCQs. Use **Review at mm:ss** after answering.
8. Use **My library** to reopen or delete videos. A timer marks a video that is still processing or was interrupted; opening it resumes preparation.

## API

Start the optional API:

```bash
uvicorn api:app --reload
```

Main endpoints: `/health`, `/videos`, `/ask`, `/videos/{id}/summary`, `/videos/{id}/quiz`, `/playlists/search?q=...`, `/playlists/{id}`.

## Tests and code quality

For development dependencies:

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

The unit tests avoid network access and heavy ML model downloads. They cover URL parsing, timestamp formatting, json3 captions, chunking (including incremental chunking), JSON parsing, quiz validation/shuffling, RAG prompt formatting, playlist parsing and detection, audio windowing, background jobs, and the streaming pipeline with its fallback and cancel paths.

## Benchmark

First make sure the video can be processed and your Groq key is configured.

```bash
python benchmarks/benchmark.py "https://www.youtube.com/watch?v=VIDEO_ID" --question "What are the main ideas?"
```

It writes `benchmarks/results.md`.

## Retrieval evaluation

Generate a small evaluation set from an already processed video:

```bash
python eval/generate_eval_set.py VIDEO_ID
python eval/run_eval.py --set eval/eval_set.json --k 3
```

It reports **Hit@k** and **MRR** for plain vector retrieval.

## Docker

```bash
cp .env.example .env
# put your GROQ_API_KEY in .env
docker compose up --build
```

- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- Qdrant: `http://localhost:6333`

## Fast ingestion

The ingestion path is optimized for normal captioned YouTube videos:

1. Extract the video ID locally.
2. Check Qdrant before contacting YouTube, so an already-processed video opens from cache immediately.
3. Use YouTube human captions first, then auto-generated captions in the spoken language when available.
4. Only run Faster-Whisper when no usable captions exist, and stream the audio into it instead of downloading the whole file first.
5. Embed and store chunks as they are produced (captions in slices, Whisper window by window), so questions work before the end.
6. Store deterministic Qdrant points so processing is idempotent.

The UI also shows the transcript source (`YouTube captions`, `Whisper`, or `cache`). Exact timing depends on network speed, CPU, video length, and whether Whisper is required.

## Settings

| Variable | Default | Meaning |
|---|---:|---|
| `GROQ_API_KEY` | required | Groq key |
| `GROQ_MODEL` | llama-3.3-70b-versatile | answer/study LLM |
| `WHISPER_MODEL` | base | fallback transcription model |
| `WHISPER_DEVICE` | cpu | Whisper device |
| `WHISPER_COMPUTE_TYPE` | int8 | CPU-efficient compute |
| `USE_YT_CAPTIONS` | true | use captions before Whisper |
| `MAX_VIDEO_MINUTES` | 60 | longest no-caption video accepted |
| `STREAM_WHISPER` | true | transcribe audio while it downloads (false = download first) |
| `STREAM_FIRST_WINDOW_SECONDS` | 90 | length of the first audio window, kept short so questions unlock early |
| `STREAM_WINDOW_SECONDS` | 180 | length of later audio windows |
| `CAPTION_SLICE_SECONDS` | 900 | captions are indexed in slices of this length |
| `JOB_WORKERS` | 2 | videos prepared at the same time (Whisper still runs one at a time) |
| `LIBRARY_CACHE_SECONDS` | 10 | how long the library list is cached |
| `MAX_PLAYLIST_VIDEOS` | 10 | videos started by "Learn the first N" |
| `PLAYLIST_PREVIEW_LIMIT` | 25 | videos listed when a playlist is opened |
| `PLAYLIST_SEARCH_RESULTS` | 6 | playlists shown per search |
| `EMBEDDING_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | embedding model |
| `QDRANT_URL` | empty | set for server/cloud mode |
| `QDRANT_API_KEY` | empty | hosted Qdrant key |
| `QDRANT_PATH` | data/qdrant | local vector DB folder |
| `CHUNK_SECONDS` | 75 | target transcript chunk duration |
| `TOP_K` | 3 | chunks retrieved per question |
| `CHAPTER_MINUTES` | 6 | approximate chapter window |
| `QUIZ_QUESTIONS` | 5 | target quiz length |
| `SHARED_LIBRARY` | true | library visibility setting for deployments |
| `YT_COOKIES_FILE` | empty | optional yt-dlp cookie file |
| `LOG_LEVEL` | INFO | terminal logging level |

## Common problems

**`GROQ_API_KEY is not set`** — copy `.env.example` to `.env` and add the key.

**YouTube blocks captions or media** — yt-dlp can occasionally be rate-limited. The app tries captions first and falls back to Whisper if caption fetching fails. A cookie file can help in environments where YouTube requires authentication.

**Whisper is slow** — transcription is the fallback only. It runs locally on CPU and is much slower than using captions. Streaming lets you start asking questions early, but total time is still bounded by CPU speed; a smaller `WHISPER_MODEL` is faster.

**Streaming falls back to a full download** — check the log for `audio streaming unavailable`. It usually means PyAV or yt-dlp could not decode the stream; the app then uses the old download-then-transcribe path.

**Qdrant local folder is locked** — do not run multiple Python processes against the same local `data/qdrant` folder. Docker/server mode is better when running Streamlit and FastAPI at the same time.

**First embedding run is slow** — the MiniLM model is downloaded on first use. After that it is loaded in the background when the app starts.

## Honest limitations

- The jump timestamp is the beginning of the retrieved chunk, not the exact sentence.
- Caption quality depends on the creator/YouTube; Whisper quality depends on audio quality.
- MiniLM has a short context window, so chunks are intentionally kept small.
- There is no reranker or hybrid keyword search; this is deliberately plain vector retrieval.
- LLM output can still be imperfect even with grounding rules.
- Public deployments should use hosted Qdrant rather than a local on-disk database.

## Small dictionary

- **RAG** — retrieve relevant source text first, then give it to an LLM before asking for an answer.
- **Embedding** — a vector of numbers representing the meaning of text.
- **Vector database** — database designed to store vectors and find similar ones.
- **Cosine similarity** — compares the direction of two vectors to estimate semantic similarity.
- **Chunk** — a small part of the transcript with start/end timestamps.
- **Top-K** — the K most similar chunks returned by retrieval.
- **Grounding** — making the answer depend on supplied evidence instead of general model knowledge.
- **MRR** — Mean Reciprocal Rank; rewards relevant results appearing near the top.

## What I built

I built an end-to-end YouTube learning assistant from scratch: YouTube ingestion, caption/Whisper transcription, timestamped chunking, MiniLM embeddings, Qdrant vector search, grounded Groq answers with citations, follow-up memory, clickable video jumps, chapter/summary generation, quiz generation, a video library, an optional API, Docker packaging, tests, CI, a benchmark and a basic retrieval evaluation.

## Exact commands

```bash
# app
streamlit run app.py

# optional API
uvicorn api:app --reload

# tests
pytest -q

# lint
ruff check .

# benchmark
python benchmarks/benchmark.py "https://www.youtube.com/watch?v=VIDEO_ID" --question "What are the main ideas?"

# retrieval evaluation
python eval/generate_eval_set.py VIDEO_ID
python eval/run_eval.py --set eval/eval_set.json --k 3
```

## Student-friendly AI tutor

The chat accepts normal English and Hinglish questions. For likely Hinglish queries, the retrieval layer converts the question into a short English search query while the final answer is generated in the student's original language/style. This keeps vector retrieval reliable without forcing students to write formal English.

Examples:
- `ye concept simple language mein samjhao`
- `iska real world example kya hai?`
- `why does this happen?`
- `isko thoda aur deeply explain karo`

The UI is designed as a student workspace: video source/library on the left, AI tutor chat and study tools in the center, and the lesson player on the right.

## Processing reliability improvements

The current version also includes:

- cache-first ingestion before YouTube network calls
- YouTube socket/read timeouts and retries
- friendly handling for DNS errors such as Windows `[Errno 11001] getaddrinfo failed`
- caption requests with their own timeout
- long-video protection before starting an expensive Whisper job
- playlist URL detection and video-by-video playlist processing
- per-video playlist progress and partial-failure handling
- batched Qdrant writes for long transcripts
- configurable playlist/network settings in `.env`

Useful settings include `YT_SOCKET_TIMEOUT`, `YT_RETRIES`, `CAPTION_TIMEOUT`, `MAX_PLAYLIST_VIDEOS`, and `QDRANT_UPSERT_BATCH_SIZE`.

## Streaming and background processing

- **Instant playback.** Clicking *Start learning* shows the YouTube player immediately. Preparation runs in a background worker (`src/jobs.py`), so the page never blocks.
- **Progressive indexing.** Captions are indexed in ~15-minute slices. For videos with no captions, `src/audio_stream.py` pipes yt-dlp audio into PyAV and Faster-Whisper transcribes it in windows (first 90 s, then 180 s) as it arrives, instead of downloading the whole file first. Each window is embedded and stored immediately, so you can ask about the first minutes while the rest is processing. Study notes and quiz unlock when it finishes.
- **Safe states.** Chunks are stored with `complete=false` until the last one is saved. An interrupted run is detected and redone; videos from older versions count as complete.
- **Fallback.** If streaming cannot start (PyAV/yt-dlp problem), the app falls back to the previous download-then-transcribe path. Set `STREAM_WHISPER=false` to force that path.
- **Other speedups.** The embedding model warms up in the background at app start, the vector DB collection is checked once instead of on every call, and the library list is cached and only reads the fields it needs.
- A windowed transcript can clip a word at a window edge; this has little effect on retrieval.

## Playlists

Type something containing the word *playlist* in **Find a video or playlist** (or set *Show me* to Playlists) to search playlists. Open one to see its videos, then learn a single video or the first N (`MAX_PLAYLIST_VIDEOS`). Pasting a playlist URL opens the same browser instead of processing everything blindly. API: `GET /playlists/search?q=...`, `GET /playlists/{id}`.
