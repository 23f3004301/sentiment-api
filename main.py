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

# ─── SENTIMENT ENDPOINT (already existing) ───────────────────────────────────

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


# ─── CODE INTERPRETER ENDPOINT (NEW) ─────────────────────────────────────────

class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    error: List[int]
    result: str

# ── Part 1: Tool Function ─────────────────────────────────────────────────────
def execute_python_code(code: str) -> dict:
    """Execute Python code, capture exact stdout or traceback."""
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

# ── Part 2: AI Error Analysis ─────────────────────────────────────────────────
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
    """Use LLM structured output to identify error line numbers."""
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

# ── Main Endpoint ─────────────────────────────────────────────────────────────
@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest):
    if not request.code or not request.code.strip():
        raise HTTPException(status_code=422, detail="Code cannot be empty")

    # Step 1: Execute the code
    execution_result = execute_python_code(request.code)

    # Step 2: If success → return immediately, no AI needed
    if execution_result["success"]:
        return CodeResponse(
            error=[],
            result=execution_result["output"]
        )

    # Step 3: If error → use AI to find line numbers
    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution_result["output"]
        )
    except Exception as e:
        # If AI fails, still return the traceback
        error_lines = []

    return CodeResponse(
        error=error_lines,
        result=execution_result["output"]
    )