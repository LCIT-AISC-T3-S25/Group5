from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import os
import json
import re

# ==================== Load Configuration ====================

try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    raise RuntimeError("Missing config.json in project root")

# ==================== Load Models and Data ====================

faiss_index = faiss.read_index(CONFIG["faiss_index_path"])
embedder = SentenceTransformer(CONFIG["embedder_model_path"])
endpoint = CONFIG.get("rag_endpoint", "/stream-rag")

with open(CONFIG["passage_map_path"], "rb") as f:
    id_to_passage = pickle.load(f)

# ==================== FastAPI Setup ====================

app = FastAPI(title="🧠 RAG Streaming API")

class QueryRequest(BaseModel):
    question: str
    k: int | None = None

# ==================== Helper Functions ====================

def retrieve_top_k(query, k=None):
    if k is None:
        k = CONFIG.get("top_k_default", 3)
    query_embed = embedder.encode([query]).astype("float32")
    D, I = faiss_index.search(query_embed, k)
    return [{
        "id": int(i),
        "score": float(d),
        "passage": id_to_passage.get(i, "UNKNOWN")
    } for i, d in zip(I[0], D[0])]

def rewrite_query_groq(question, return_wait=False):
    url = CONFIG["groq_url"]
    headers = {
        "Authorization": f"Bearer {CONFIG['groq_api_key']}",
        "Content-Type": "application/json"
    }
    prompt = (
        "Rewrite the following biomedical question for optimal passage retrieval. "
        "If the question is vague or not about biomedicine, reply: OUT OF SCOPE.\n"
        f"Question: {question}\nRewritten:"
    )
    data = {
        "model": CONFIG["groq_model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": CONFIG.get("max_tokens_rewrite", 64)
    }
    resp = requests.post(url, headers=headers, json=data)
    try:
        out = resp.json()
        if "choices" in out:
            rewritten = out["choices"][0]["message"]["content"].strip()
            return (rewritten, None) if return_wait else rewritten
        elif "error" in out:
            wait_seconds = 10
            if "message" in out["error"]:
                match = re.search(r'try again in ([\\d\\.]+)s', out["error"]["message"])
                if match:
                    wait_seconds = float(match.group(1))
            return ("API error", wait_seconds) if return_wait else "API error"
        else:
            return ("API call failed", 10) if return_wait else "API call failed"
    except Exception:
        return ("API call failed", 10) if return_wait else "API call failed"

def answer_with_groq_streaming(question, context):
    headers = {
        "Authorization": f"Bearer {CONFIG['groq_api_key']}",
        "Content-Type": "application/json"
    }

    prompt = (
        "You are a biomedical expert. Answer the following question using ONLY the context below. "
        "If the answer is not present, reply: \"I'm sorry, I cannot answer that question based on the provided information.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    data = {
        "model": CONFIG["groq_model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": CONFIG.get("max_tokens_answer", 256),
        "stream": True
    }

    def stream():
        with requests.post(CONFIG["groq_url"], headers=headers, json=data, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line and line.startswith(b"data: "):
                    payload = line[6:]
                    if payload == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
                    except Exception:
                        continue
    return stream()

# ==================== Endpoint ====================

@app.post(endpoint)
def stream_rag(request: QueryRequest):
    original_question = request.question
    rewritten_question = rewrite_query_groq(original_question)
    passages = retrieve_top_k(rewritten_question, request.k)

    for i, p in enumerate(passages):
        print(f"[Passage {i}] ID={p['id']}, Score={p['score']:.4f}")
        print(p["passage"])
        print("—" * 50)

    context = "\n".join([p["passage"] for p in passages])
    return StreamingResponse(answer_with_groq_streaming(original_question, context), media_type="text/plain")
