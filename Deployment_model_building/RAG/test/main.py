# rag_streaming_api/main.py

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

# ==================== Setup ====================

# Load FAISS index
faiss_index = faiss.read_index("index.faiss")

# Load SentenceTransformer model
embedder = SentenceTransformer("embedder_model/")

# Load passage map (should match FAISS row indices)
with open("id_to_passage.pkl", "rb") as f:
    id_to_passage = pickle.load(f)

# ==================== FastAPI App ====================

app = FastAPI(title="🧠 RAG Streaming API")

class QueryRequest(BaseModel):
    question: str
    k: int = 3

# ==================== Helper Functions ====================

def retrieve_top_k(query, k=3):
    query_embed = embedder.encode([query]).astype("float32")
    D, I = faiss_index.search(query_embed, k)
    return [{
        "id": int(i),
        "score": float(d),
        "passage": id_to_passage.get(i, "UNKNOWN")
    } for i, d in zip(I[0], D[0])]

API_ERROR = "API error"
API_FAILED = "API call failed"

def rewrite_query_groq(question, return_wait=False):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }
    prompt = (
        "Rewrite the following biomedical question for optimal passage retrieval. "
        "If the question is vague or not about biomedicine, reply: OUT OF SCOPE.\n"
        f"Question: {question}\nRewritten:"
    )
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64
    }

    try:
        response = requests.post(url, headers=headers, json=data).json()
        return _handle_rewrite_response(response, return_wait)
    except Exception:
        return (API_FAILED, 10) if return_wait else API_FAILED

def _handle_rewrite_response(response, return_wait):
    if "choices" in response:
        content = response["choices"][0]["message"]["content"].strip()
        return (content, None) if return_wait else content

    if "error" in response:
        wait = _extract_retry_time(response["error"].get("message", ""))
        return (API_ERROR, wait) if return_wait else API_ERROR

    return (API_FAILED, 10) if return_wait else API_FAILED

def _extract_retry_time(msg):
    match = re.search(r'try again in ([\d.]+)s', msg)
    return float(match.group(1)) if match else 10


def answer_with_groq_streaming(question, context):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = _generate_answer_prompt(question, context)
    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "stream": True
    }

    return _stream_groq_response(headers, data)

def _generate_answer_prompt(question, context):
    return (
        "You are a biomedical expert. Answer the following question using ONLY the context below. "
        "If the answer is not present, reply: \"I'm sorry, I cannot answer that question based on the provided information.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

def _stream_groq_response(headers, data):
    def stream():
        with requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]
                except Exception:
                    continue
    return stream()


# ==================== Endpoint ====================

@app.post("/stream-rag")
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
