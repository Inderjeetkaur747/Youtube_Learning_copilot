# Deployment

## Streamlit Community Cloud
1. Push this repository to GitHub.
2. Create an app with `app.py` as the entry point.
3. Add `GROQ_API_KEY` in Secrets.
4. For a public app, prefer hosted Qdrant and set `QDRANT_URL` and `QDRANT_API_KEY`.
5. Set `SHARED_LIBRARY=false` if you do not want every visitor to see the same library list.

## Docker
Copy `.env.example` to `.env`, add your key, then run `docker compose up --build`.
