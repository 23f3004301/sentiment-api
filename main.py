from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import json, os, sys, re, traceback
from io import StringIO
from typing import List

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

client = OpenAI(api_key=os.environ.get("AIPIPE_TOKEN"), base_url="https://aipipe.org/openai/v1")

# ── SENTIMENT ─────────────────────────────────────────────────────────────────
class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Analyze sentiment. Return JSON with sentiment (positive/negative/neutral) and rating (1-5)."},
            {"role": "user", "content": request.comment}
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "s", "strict": True, "schema": {
            "type": "object",
            "properties": {"sentiment": {"type": "string", "enum": ["positive","negative","neutral"]}, "rating": {"type": "integer"}},
            "required": ["sentiment","rating"], "additionalProperties": False
        }}}
    )
    return SentimentResponse(**json.loads(response.choices[0].message.content))

# ── CODE INTERPRETER ──────────────────────────────────────────────────────────
class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    error: List[int]
    result: str

@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest):
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(request.code, {})
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        return CodeResponse(error=[], result=output)
    except Exception:
        tb = traceback.format_exc()
        sys.stdout = old_stdout
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "Return JSON with error_lines (array of int line numbers)."},
                {"role": "user", "content": f"Code:\n{request.code}\n\nError:\n{tb}"}
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "e", "strict": True, "schema": {
                "type": "object",
                "properties": {"error_lines": {"type": "array", "items": {"type": "integer"}}},
                "required": ["error_lines"], "additionalProperties": False
            }}}
        )
        result = json.loads(response.choices[0].message.content)
        return CodeResponse(error=result["error_lines"], result=tb)

# ── ASK / TIMESTAMP ───────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    video_url: str
    topic: str

class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str

def get_video_id(url: str) -> str:
    m = re.search(r'(?:v=|youtu\.be/|/v/)([^&\n?#]+)', url)
    return m.group(1) if m else None

def secs_to_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"

@app.post("/ask", response_model=AskResponse)
async def find_timestamp(request: AskRequest):
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        video_id = get_video_id(request.video_url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")

        # Fetch transcript
        try:
            api = YouTubeTranscriptApi()
            entries = list(api.fetch(video_id, languages=['en']))
        except Exception:
            try:
                api = YouTubeTranscriptApi()
                listing = api.list(video_id)
                transcript = listing.find_generated_transcript(['en']).fetch()
                entries = list(transcript)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Transcript fetch failed: {e}")

        # Build timestamped transcript (full, no truncation)
        lines = [f"[{secs_to_ts(e.start)}] {e.text}" for e in entries]
        full_transcript = "\n".join(lines)

        # Direct text search first (free, instant)
        topic_lower = request.topic.lower()
        for window in [1, 2, 3, 5]:
            for i in range(len(entries)):
                chunk = entries[i:i+window]
                combined = " ".join(e.text for e in chunk).lower()
                if topic_lower in combined:
                    return AskResponse(timestamp=secs_to_ts(chunk[0].start), video_url=request.video_url, topic=request.topic)

        # Keyword scoring fallback
        words = [w for w in topic_lower.split() if len(w) > 4]
        best_score, best_sec = 0, 0
        for i in range(len(entries)):
            chunk = entries[i:i+5]
            combined = " ".join(e.text for e in chunk).lower()
            score = sum(1 for w in words if w in combined)
            if score > best_score:
                best_score, best_sec = score, chunk[0].start

        if best_score >= max(2, len(words) // 2):
            return AskResponse(timestamp=secs_to_ts(best_sec), video_url=request.video_url, topic=request.topic)

        # GPT on full transcript (fits in 128k context)
        ai_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": 'Return JSON: {"timestamp": "HH:MM:SS"} for the first occurrence of the topic.'},
                {"role": "user", "content": f'Topic: "{request.topic}"\n\nTranscript:\n{full_transcript[:100000]}'}
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "ts", "strict": True, "schema": {
                "type": "object",
                "properties": {"timestamp": {"type": "string"}},
                "required": ["timestamp"], "additionalProperties": False
            }}},
            max_tokens=20
        )
        ts = json.loads(ai_response.choices[0].message.content).get("timestamp", "00:00:00")
        return AskResponse(timestamp=ts, video_url=request.video_url, topic=request.topic)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
