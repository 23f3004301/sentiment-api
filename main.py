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

class AskRequest(BaseModel):
    video_url: str
    topic: str

class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str

TIMESTAMP_SCHEMA = {
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "string",
            "description": "Timestamp in HH:MM:SS format e.g. 00:05:47"
        }
    },
    "required": ["timestamp"],
    "additionalProperties": False
}

def parse_vtt_to_entries(vtt_content: str) -> list:
    """Parse VTT into list of (seconds, text) tuples."""
    entries = []
    lines = vtt_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        ts_match = re.match(r'(\d{2}):(\d{2}):(\d{2})\.\d+', line)
        if ts_match and '-->' in line:
            h, m, s = int(ts_match.group(1)), int(ts_match.group(2)), int(ts_match.group(3))
            total_seconds = h * 3600 + m * 60 + s
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() != '':
                clean = re.sub(r'<[^>]+>', '', lines[i].strip())
                if clean:
                    text_lines.append(clean)
                i += 1
            if text_lines:
                entries.append((total_seconds, ' '.join(text_lines)))
        else:
            i += 1
    return entries

def seconds_to_hhmmss(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def fuzzy_search_transcript(entries: list, topic: str) -> int:
    """
    Search transcript for topic. Returns best matching timestamp in seconds.
    Strategy:
    1. Direct substring match (case-insensitive)
    2. Sliding window of 3-5 entries combined (for phrases split across lines)
    3. Word overlap scoring
    """
    topic_lower = topic.lower().strip()
    topic_words = set(topic_lower.split())

    best_score = 0
    best_sec = None

    # Strategy 1 & 2: sliding window over 1, 3, 5 combined entries
    for window in [1, 3, 5]:
        for i in range(len(entries)):
            chunk_entries = entries[i:i+window]
            chunk_text = ' '.join(t for _, t in chunk_entries).lower()
            start_sec = chunk_entries[0][0]

            # Direct substring match - highest priority
            if topic_lower in chunk_text:
                return start_sec

            # Word overlap score
            chunk_words = set(chunk_text.split())
            overlap = len(topic_words & chunk_words)
            score = overlap / len(topic_words) if topic_words else 0

            if score > best_score:
                best_score = score
                best_sec = start_sec

    return best_sec

def download_subtitles(video_url: str, tmpdir: str) -> str:
    """Download subtitles. Returns path to VTT file or None."""
    output_template = os.path.join(tmpdir, "subtitle")

    for sub_flag in ["--write-auto-subs", "--write-subs"]:
        cmd = [
            "yt-dlp",
            sub_flag,
            "--skip-download",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--output", output_template,
            "--no-warnings",
            "--quiet",
            video_url
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        vtt_files = glob.glob(os.path.join(tmpdir, "*.vtt"))
        if vtt_files:
            return vtt_files[0]

    return None

def ai_search_transcript(entries: list, topic: str) -> str:
    """Fallback: send transcript chunks to AI to find topic."""
    # Build full transcript text
    transcript_chunks = []
    for sec, text in entries:
        ts = seconds_to_hhmmss(sec)
        transcript_chunks.append(f"[{ts}] {text}")

    full_transcript = '\n'.join(transcript_chunks)

    # Send in chunks of 15000 chars, ask AI which chunk has the topic
    # First try first 30000 chars, then next 30000, etc.
    chunk_size = 30000
    for start in range(0, len(full_transcript), chunk_size):
        chunk = full_transcript[start:start + chunk_size]
        if topic.lower()[:20] in chunk.lower() or any(
            w in chunk.lower() for w in topic.lower().split() if len(w) > 5
        ):
            # This chunk likely has the answer
            prompt = f"""Find the EXACT timestamp when this topic/phrase is first spoken:
"{topic}"

TRANSCRIPT SECTION:
{chunk}

Return the HH:MM:SS timestamp from the transcript line that best matches the topic.
If not found in this section, return "00:00:00"."""

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "Return only a JSON with timestamp in HH:MM:SS format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "timestamp_finder",
                        "strict": True,
                        "schema": TIMESTAMP_SCHEMA
                    }
                }
            )
            result = json.loads(response.choices[0].message.content)
            ts = result.get("timestamp", "00:00:00")
            if ts != "00:00:00":
                return ts

    # Final fallback: send last section of transcript to AI
    last_chunk = full_transcript[-30000:] if len(full_transcript) > 30000 else full_transcript
    prompt = f"""In this YouTube video transcript, find when this topic is FIRST spoken:
"{topic}"

TRANSCRIPT:
{last_chunk[:15000]}

Return the HH:MM:SS timestamp."""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Return only JSON with timestamp in HH:MM:SS format."},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "timestamp_finder",
                "strict": True,
                "schema": TIMESTAMP_SCHEMA
            }
        }
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("timestamp", "00:00:00")


@app.post("/ask", response_model=AskResponse)
async def find_timestamp(request: AskRequest):
    tmpdir = tempfile.mkdtemp()
    try:
        # Step 1: Download subtitles
        vtt_path = download_subtitles(request.video_url, tmpdir)

        if vtt_path:
            with open(vtt_path, 'r', encoding='utf-8') as f:
                vtt_content = f.read()

            entries = parse_vtt_to_entries(vtt_content)

            if entries:
                # Step 2: Direct fuzzy search first (fast, no AI needed)
                best_sec = fuzzy_search_transcript(entries, request.topic)

                if best_sec is not None:
                    timestamp = seconds_to_hhmmss(best_sec)
                    # Validate format
                    if re.match(r'^\d{2}:\d{2}:\d{2}$', timestamp):
                        return AskResponse(
                            timestamp=timestamp,
                            video_url=request.video_url,
                            topic=request.topic
                        )

                # Step 3: AI-assisted search on full transcript
                timestamp = ai_search_transcript(entries, request.topic)
                if re.match(r'^\d{2}:\d{2}:\d{2}$', timestamp):
                    return AskResponse(
                        timestamp=timestamp,
                        video_url=request.video_url,
                        topic=request.topic
                    )

        # Final fallback
        return AskResponse(
            timestamp="00:00:00",
            video_url=request.video_url,
            topic=request.topic
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Subtitle download timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
