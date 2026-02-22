from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import json
import os
import sys
from io import StringIO
import traceback
from typing import List
import subprocess
import re
import tempfile
import glob

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)

# ─── SENTIMENT ENDPOINT ───────────────────────────────────────────────────────
class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str
    rating: int

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "rating": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        }
    },
    "required": ["sentiment", "rating"],
    "additionalProperties": False
}

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=422, detail="Comment cannot be empty")
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis engine. "
                        "Analyze the comment and return sentiment "
                        "('positive','negative','neutral') and "
                        "rating (1-5 where 5=very positive, 1=very negative)."
                    )
                },
                {"role": "user", "content": request.comment}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_analysis",
                    "strict": True,
                    "schema": SENTIMENT_SCHEMA
                }
            }
        )
        result = json.loads(response.choices[0].message.content)
        return SentimentResponse(**result)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── CODE INTERPRETER ENDPOINT ───────────────────────────────────────────────
class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    error: List[int]
    result: str

def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {})
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout

ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "error_lines": {
            "type": "array",
            "items": {"type": "integer"}
        }
    },
    "required": ["error_lines"],
    "additionalProperties": False
}

def analyze_error_with_ai(code: str, traceback_output: str) -> List[int]:
    prompt = f"""Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{traceback_output}

Return only the line number(s) where the error is located."""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a Python error analysis expert. Identify exact line numbers where errors occur."
            },
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "error_analysis",
                "strict": True,
                "schema": ERROR_SCHEMA
            }
        }
    )
    result = json.loads(response.choices[0].message.content)
    return result["error_lines"]

@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest):
    if not request.code or not request.code.strip():
        raise HTTPException(status_code=422, detail="Code cannot be empty")
    execution_result = execute_python_code(request.code)
    if execution_result["success"]:
        return CodeResponse(error=[], result=execution_result["output"])
    try:
        error_lines = analyze_error_with_ai(request.code, execution_result["output"])
    except Exception:
        error_lines = []
    return CodeResponse(error=error_lines, result=execution_result["output"])

# ─── YOUTUBE TIMESTAMP ENDPOINT ──────────────────────────────────────────────

import glob
import subprocess
import tempfile
import shutil

class AskRequest(BaseModel):
    video_url: str
    topic: str

class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str

def get_video_id(url: str) -> str:
    import re
    m = re.search(r'(?:v=|youtu\.be/|/v/)([^&\n?#]+)', url)
    return m.group(1) if m else None

def seconds_to_hhmmss(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_transcript_entries(video_url: str, tmpdir: str):
    """
    Returns list of (start_seconds, text) tuples.
    Tries: 1) youtube-transcript-api  2) yt-dlp VTT
    """
    import re as re2

    # --- Method 1: youtube-transcript-api ---
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        video_id = get_video_id(video_url)
        if video_id:
            try:
                api = YouTubeTranscriptApi()
                transcript = api.fetch(video_id, languages=['en'])
                entries = [(e.start, e.text) for e in transcript]
                if entries:
                    return entries
            except Exception:
                pass
            # try without language filter
            try:
                api = YouTubeTranscriptApi()
                listing = api.list(video_id)
                transcript = listing.find_generated_transcript(['en']).fetch()
                entries = [(e.start, e.text) for e in transcript]
                if entries:
                    return entries
            except Exception:
                pass
    except ImportError:
        pass

    # --- Method 2: yt-dlp VTT ---
    output_template = os.path.join(tmpdir, "sub")
    for flag in ["--write-auto-subs", "--write-subs"]:
        cmd = [
            "yt-dlp", flag,
            "--skip-download",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--output", output_template,
            "--no-warnings", "--quiet",
            video_url
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
        except Exception:
            pass
        vtt_files = glob.glob(os.path.join(tmpdir, "*.vtt"))
        if vtt_files:
            with open(vtt_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
            entries = []
            lines = content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                ts = re2.match(r'(\d{2}):(\d{2}):(\d{2})\.\d+', line)
                if ts and '-->' in line:
                    h, m, s = int(ts.group(1)), int(ts.group(2)), int(ts.group(3))
                    total_sec = h * 3600 + m * 60 + s
                    i += 1
                    text_parts = []
                    while i < len(lines) and lines[i].strip():
                        clean = re2.sub(r'<[^>]+>', '', lines[i].strip())
                        if clean:
                            text_parts.append(clean)
                        i += 1
                    if text_parts:
                        entries.append((total_sec, ' '.join(text_parts)))
                else:
                    i += 1
            if entries:
                return entries

    return []

def find_timestamp_with_ai(entries: list, topic: str) -> str:
    """Send FULL transcript to AI in one call. 128k context fits any video."""
    # Build timestamped transcript
    lines = []
    for sec, text in entries:
        ts = seconds_to_hhmmss(sec)
        lines.append(f"[{ts}] {text}")
    full_transcript = '\n'.join(lines)

    # First try direct substring match (no AI needed, instant)
    topic_lower = topic.lower().strip()
    # sliding window of 1-5 consecutive entries
    for window in [1, 2, 3, 5]:
        for i in range(len(entries)):
            chunk = entries[i:i+window]
            combined = ' '.join(t for _, t in chunk).lower()
            if topic_lower in combined:
                return seconds_to_hhmmss(chunk[0][0])
    # partial phrase match (first 30 chars of topic)
    partial = topic_lower[:40]
    for i in range(len(entries)):
        chunk = entries[i:i+3]
        combined = ' '.join(t for _, t in chunk).lower()
        if partial in combined:
            return seconds_to_hhmmss(chunk[0][0])

    # Keyword match: find most keyword-dense segment
    topic_words = [w for w in topic_lower.split() if len(w) > 4]
    if topic_words:
        best_score, best_sec = 0, None
        for i in range(len(entries)):
            chunk = entries[i:i+5]
            combined = ' '.join(t for _, t in chunk).lower()
            score = sum(1 for w in topic_words if w in combined)
            if score > best_score:
                best_score = score
                best_sec = chunk[0][0]
        if best_sec is not None and best_score >= max(2, len(topic_words) // 2):
            return seconds_to_hhmmss(best_sec)

    # AI fallback: send full transcript (fits in 128k context)
    # Truncate to 100k chars max (~75k tokens) if very long
    if len(full_transcript) > 100000:
        # Send in two halves if needed
        half = len(full_transcript) // 2
        segments = [full_transcript[:half + 5000], full_transcript[half - 5000:]]
    else:
        segments = [full_transcript]

    for segment in segments:
        prompt = f"""You are analyzing a YouTube video transcript with timestamps.

Find the FIRST moment this topic is spoken:
"{topic}"

TRANSCRIPT:
{segment}

Return the HH:MM:SS timestamp from the transcript line where this topic FIRST appears.
If not in this segment, return exactly: NOT_FOUND"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": 'Return ONLY a JSON object: {"timestamp": "HH:MM:SS"} or {"timestamp": "NOT_FOUND"}'},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ts",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"timestamp": {"type": "string"}},
                        "required": ["timestamp"],
                        "additionalProperties": False
                    }
                }
            },
            max_tokens=50
        )
        result = json.loads(response.choices[0].message.content)
        ts = result.get("timestamp", "NOT_FOUND")
        if ts != "NOT_FOUND" and re.match(r'^\d{2}:\d{2}:\d{2}$', ts):
            return ts

    return "00:00:00"


@app.post("/ask", response_model=AskResponse)
async def find_timestamp(request: AskRequest):
    tmpdir = tempfile.mkdtemp()
    try:
        entries = get_transcript_entries(request.video_url, tmpdir)
        
        if not entries:
            # No transcript available at all
            raise HTTPException(status_code=400, detail="Could not download transcript for this video")
        
        timestamp = find_timestamp_with_ai(entries, request.topic)
        
        return AskResponse(
            timestamp=timestamp,
            video_url=request.video_url,
            topic=request.topic
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
