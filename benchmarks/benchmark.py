import argparse, time
from pathlib import Path
from src.pipeline import process_video, prepare_answer

p=argparse.ArgumentParser(); p.add_argument("video"); p.add_argument("--question",default="What are the main ideas?"); a=p.parse_args()
r=process_video(a.video); t=time.perf_counter(); stream,_,_=prepare_answer(a.question,[r["meta"]["video_id"]],[]); answer="".join(stream); latency=time.perf_counter()-t
text=f"# Benchmark Results\n\nIngestion: `{r['timings']}`\n\nQuery latency: `{latency:.3f}s`\n\nAnswer preview: {answer[:300]}\n"
Path("benchmarks/results.md").write_text(text,encoding="utf-8"); print(text)
