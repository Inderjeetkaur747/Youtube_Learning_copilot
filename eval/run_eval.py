import argparse,json
from src.embeddings import embed_query
from src.vector_store import search
p=argparse.ArgumentParser(); p.add_argument("--set",default="eval/eval_set.json"); p.add_argument("--k",type=int,default=3); a=p.parse_args(); items=json.load(open(a.set,encoding="utf-8")); hits=0; rr=0
for x in items:
    res=search(embed_query(x["question"]),[x["video_id"]],a.k); ranks=[i+1 for i,r in enumerate(res) if r["chunk_index"]==x["relevant_chunk_index"]]
    if ranks: hits+=1; rr+=1/ranks[0]
n=max(1,len(items)); print({"Hit@k":hits/n,"MRR":rr/n,"n":len(items)})
