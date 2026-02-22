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
from google import genai as google_genai
from google.genai import types as genai_types

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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
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
# ── ASK / TIMESTAMP ENDPOINT ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    video_url: str
    topic: str

class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


def find_timestamp_gemini(video_url: str, topic: str) -> str:
    """
    Pass YouTube URL directly to Gemini — no audio download needed.
    Gemini 2.0 Flash supports YouTube URLs natively.
    """
    gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""Watch this YouTube video carefully.

Find the EXACT timestamp (HH:MM:SS) when this topic is FIRST spoken or discussed:
"{topic}"

Rules:
- Give me the moment the speaker FIRST says this, not the chapter heading time.
- Return ONLY the timestamp in HH:MM:SS format, nothing else.
- Example valid response: 00:58:49
- If you cannot find it, return: 00:00:00
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=genai_types.Content(
            parts=[
                genai_types.Part(
                    file_data=genai_types.FileData(
                        file_uri=video_url,
                        mime_type="video/*"
                    )
                ),
                genai_types.Part(text=prompt)
            ]
        )
    )

    text = response.text.strip()
    # Extract HH:MM:SS from response
    import re as re2
    match = re2.search(r'\d{1,2}:\d{2}:\d{2}', text)
    if match:
        raw = match.group(0)
        parts = raw.split(":")
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{h:02d}:{m:02d}:{s:02d}"
    return "00:00:00"


@app.post("/ask", response_model=AskResponse)
async def find_timestamp(request: AskRequest):
    try:
        timestamp = find_timestamp_gemini(request.video_url, request.topic)
        return AskResponse(
            timestamp=timestamp,
            video_url=request.video_url,
            topic=request.topic
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
