import argparse, json
from src.pipeline import all_chunks
p=argparse.ArgumentParser(); p.add_argument("video_id"); p.add_argument("--out",default="eval/eval_set.json"); a=p.parse_args()
chunks=all_chunks(a.video_id); data=[{"question":f"What is discussed around chunk {c['chunk_index']}?","video_id":a.video_id,"relevant_chunk_index":c["chunk_index"]} for c in chunks[::max(1,len(chunks)//10 or 1)][:10]]
open(a.out,"w",encoding="utf-8").write(json.dumps(data,indent=2)); print(f"Wrote {len(data)} items to {a.out}")
