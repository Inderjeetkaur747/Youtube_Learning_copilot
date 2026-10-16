from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.features import generate_chapters, generate_quiz, generate_summary
from src.pipeline import all_chunks, prepare_answer, process_playlist, process_video
from src.vector_store import list_videos
from src.youtube_utils import extract_target, get_playlist_entries, search_playlists

app=FastAPI(title="YouTube Learning Copilot API")
class Ask(BaseModel): question:str; video_ids:list[str]; history:list[dict]=[]
@app.get("/health")
def health(): return {"ok":True}
@app.get("/videos")
def videos(): return list_videos()
@app.post("/videos")
def add_video(payload:dict):
    try:
        kind, _ = extract_target(payload["url"])
        return process_playlist(payload["url"]) if kind == "playlist" else process_video(payload["url"])
    except Exception as e:
        raise HTTPException(400, str(e)) from e
@app.get("/playlists/search")
def playlists_search(q:str):
    try: return search_playlists(q)
    except Exception as e: raise HTTPException(502,str(e)) from e
@app.get("/playlists/{playlist_id}")
def playlist(playlist_id:str):
    try: return get_playlist_entries(playlist_id)
    except Exception as e: raise HTTPException(502,str(e)) from e
@app.post("/ask")
def ask(req:Ask):
    try:
        stream,sources,standalone=prepare_answer(req.question,req.video_ids,req.history); return {"answer":"".join(stream),"sources":sources,"standalone_question":standalone}
    except Exception as e: raise HTTPException(500,str(e)) from e
@app.get("/videos/{video_id}/summary")
def summary(video_id:str):
    chapters=generate_chapters(all_chunks(video_id)); return {"chapters":chapters,**generate_summary(chapters)}
@app.get("/videos/{video_id}/quiz")
def quiz(video_id:str): return {"questions":generate_quiz(all_chunks(video_id))}
