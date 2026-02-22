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

import glob, subprocess, tempfile, shutil, time
import google.generativeai as genai

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class AskRequest(BaseModel):
    video_url: str
    topic: str

class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str

def download_audio(video_url: str, tmpdir: str) -> str:
    """Download audio using yt-dlp. Returns path to audio file."""
    output_template = os.path.join(tmpdir, "audio.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "5",       # medium quality, smaller file
        "--output", output_template,
        "--no-warnings",
        "--quiet",
        video_url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    # Find downloaded file
    for ext in ["mp3", "m4a", "ogg", "wav", "opus", "webm"]:
        path = os.path.join(tmpdir, f"audio.{ext}")
        if os.path.exists(path):
            return path
    
    # Glob fallback
    files = glob.glob(os.path.join(tmpdir, "audio.*"))
    if files:
        return files[0]
    
    raise FileNotFoundError(f"Audio download failed. stderr: {result.stderr[:500]}")

def ask_gemini_for_timestamp(audio_path: str, topic: str) -> str:
    """Upload audio to Gemini and ask for timestamp."""
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Upload audio file
    print(f"Uploading audio file: {audio_path} ({os.path.getsize(audio_path)//1024}KB)")
    audio_file = genai.upload_file(
        path=audio_path,
        mime_type="audio/mpeg"
    )
    
    # Wait for file to be processed
    while audio_file.state.name == "PROCESSING":
        time.sleep(2)
        audio_file = genai.get_file(audio_file.name)
    
    if audio_file.state.name == "FAILED":
        raise ValueError("Gemini file processing failed")
    
    # Ask Gemini for timestamp
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""Listen to this audio carefully.

Find the EXACT moment when the speaker FIRST mentions or discusses this topic:
"{topic}"

Return ONLY the timestamp in HH:MM:SS format (e.g., 00:58:49).
Do not include any explanation, just the timestamp."""

    response = model.generate_content([audio_file, prompt])
    raw = response.text.strip()
    print(f"Gemini raw response: {raw}")
    
    # Extract HH:MM:SS from response
    import re as re2
    match = re2.search(r'\d{1,2}:\d{2}:\d{2}', raw)
    if match:
        ts = match.group(0)
        # Ensure HH:MM:SS format
        parts = ts.split(':')
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{h:02d}:{m:02d}:{s:02d}"
    
    # Try MM:SS format
    match2 = re2.search(r'\d{1,2}:\d{2}', raw)
    if match2:
        parts = match2.group(0).split(':')
        m, s = int(parts[0]), int(parts[1])
        return f"00:{m:02d}:{s:02d}"
    
    return "00:00:00"


@app.post("/ask", response_model=AskResponse)
async def find_timestamp(request: AskRequest):
    tmpdir = tempfile.mkdtemp()
    try:
        # Step 1: Download audio
        audio_path = download_audio(request.video_url, tmpdir)
        
        # Step 2: Ask Gemini with audio
        timestamp = ask_gemini_for_timestamp(audio_path, request.topic)
        
        return AskResponse(
            timestamp=timestamp,
            video_url=request.video_url,
            topic=request.topic
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
