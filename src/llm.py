import json
import re
import time
from .config import settings
_client=None

def _client_get():
    global _client
    if _client is None:
        if not settings.groq_api_key: raise RuntimeError("GROQ_API_KEY is not set")
        from groq import Groq
        _client=Groq(api_key=settings.groq_api_key)
    return _client

def _retry(call):
    for i in range(4):
        try: return call()
        except Exception as e:
            if i==3 or "rate" not in str(e).lower(): raise
            time.sleep(2**i)

def chat(messages, temperature=0.2, json_mode=False):
    kwargs={"model":settings.groq_model,"messages":messages,"temperature":temperature}
    if json_mode: kwargs["response_format"]={"type":"json_object"}
    r=_retry(lambda:_client_get().chat.completions.create(**kwargs))
    return r.choices[0].message.content or ""

def chat_stream(messages, temperature=0.2):
    stream=_retry(lambda:_client_get().chat.completions.create(model=settings.groq_model,messages=messages,temperature=temperature,stream=True))
    for item in stream:
        token=item.choices[0].delta.content
        if token: yield token

def parse_json(text: str):
    cleaned=re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.I|re.M).strip()
    try: return json.loads(cleaned)
    except json.JSONDecodeError:
        start=min([i for i in (cleaned.find("{"),cleaned.find("[")) if i>=0], default=-1)
        for end in range(len(cleaned), start, -1):
            try: return json.loads(cleaned[start:end])
            except Exception: pass
        raise
