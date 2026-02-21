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
            "description": "Timestamp in HH:MM:SS format"
        }
    },
    "required": ["timestamp"],
    "additionalProperties": False
}

def parse_vtt_to_text(vtt_content: str) -> list:
    """
    Parse VTT subtitle file into list of
    (timestamp_seconds, text) tuples.
    """
    entries = []
    lines = vtt_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Match timestamp lines like: 00:01:23.456 --> 00:01:25.789
        ts_match = re.match(
            r'(\d{2}):(\d{2}):(\d{2})\.\d+ --> ', line
        )
        if ts_match:
            h, m, s = int(ts_match.group(1)), int(ts_match.group(2)), int(ts_match.group(3))
            total_seconds = h * 3600 + m * 60 + s
            # Collect text lines after timestamp
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() != '':
                # Remove VTT tags like <c>, </c>, <00:00:00.000>
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

def download_subtitles(video_url: str, tmpdir: str) -> str:
    """
    Download auto-generated subtitles using yt-dlp.
    Returns the path to the downloaded .vtt file.
    """
    output_template = os.path.join(tmpdir, "subtitle")

    # Try auto-generated subtitles first
    cmd = [
        "yt-dlp",
        "--write-auto-subs",       # auto-generated subtitles
        "--skip-download",          # don't download video/audio
        "--sub-lang", "en",         # English only
        "--sub-format", "vtt",      # VTT format
        "--output", output_template,
        "--no-warnings",
        video_url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    # Find downloaded .vtt file
    vtt_files = glob.glob(os.path.join(tmpdir, "*.vtt"))
    if vtt_files:
        return vtt_files[0]

    # Fallback: try manual subtitles
    cmd[1] = "--write-subs"
    subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    vtt_files = glob.glob(os.path.join(tmpdir, "*.vtt"))
    if vtt_files:
        return vtt_files[0]

    return None

@app.post("/ask", response_model=AskResponse)
async def find_timestamp(request: AskRequest):
    tmpdir = tempfile.mkdtemp()

    try:
        # Step 1: Download subtitles
        vtt_path = download_subtitles(request.video_url, tmpdir)

        transcript_text = ""

        if vtt_path:
            # Step 2: Parse VTT into timestamped entries
            with open(vtt_path, 'r', encoding='utf-8') as f:
                vtt_content = f.read()

            entries = parse_vtt_to_text(vtt_content)

            # Build readable transcript with timestamps
            transcript_chunks = []
            for sec, text in entries:
                ts = seconds_to_hhmmss(sec)
                transcript_chunks.append(f"[{ts}] {text}")

            transcript_text = '\n'.join(transcript_chunks)

        # Step 3: Ask AI to find the timestamp
        if transcript_text:
            prompt = f"""You have a YouTube video transcript with timestamps.
Find when this topic/phrase is first spoken: "{request.topic}"

TRANSCRIPT:
{transcript_text[:12000]}

Return the exact timestamp (HH:MM:SS) when this topic first appears."""
        else:
            # Fallback: ask AI based on general knowledge of the video
            prompt = f"""For this YouTube video: {request.video_url}
Find the timestamp (HH:MM:SS) when this topic is first spoken: "{request.topic}"
Use your knowledge of the video content to estimate the timestamp."""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a video timestamp finder. "
                        "Return timestamps in HH:MM:SS format only. "
                        "Example: 00:05:47 or 01:23:45"
                    )
                },
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
        timestamp = result["timestamp"]

        # Ensure HH:MM:SS format
        if re.match(r'^\d{2}:\d{2}:\d{2}$', timestamp):
            pass
        elif re.match(r'^\d{1,2}:\d{2}$', timestamp):
            timestamp = "00:" + timestamp.zfill(5)
        else:
            timestamp = "00:00:00"

        return AskResponse(
            timestamp=timestamp,
            video_url=request.video_url,
            topic=request.topic
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Video download timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
