from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

app = FastAPI(title="Grounded QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Chunk(BaseModel):
    chunk_id: str
    text: str

class RequestModel(BaseModel):
    question: str
    chunks: List[Chunk]

def tokenize(text):
    return re.findall(r"\w+", text.lower())

@app.post("/")
def grounded_qa(req: RequestModel):

    if not req.question.strip() or len(req.chunks)==0:
        return {
            "answer":"I don't know",
            "citations":[],
            "confidence":0.0,
            "answerable":False
        }

    q_words=set(tokenize(req.question))

    best_chunk=None
    best_score=0

    for chunk in req.chunks:

        words=set(tokenize(chunk.text))
        score=len(q_words & words)

        if score>best_score:
            best_score=score
            best_chunk=chunk

    if best_chunk is None or best_score==0:
        return {
            "answer":"I don't know",
            "citations":[],
            "confidence":0.2,
            "answerable":False
        }

    confidence=min(0.95,0.4+0.1*best_score)

    return {
        "answer":best_chunk.text,
        "citations":[best_chunk.chunk_id],
        "confidence":round(confidence,2),
        "answerable":True
    }
