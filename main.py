from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import json

app = FastAPI()

# ── AI Pipe config ──────────────────────────────────────────────
client = OpenAI(
    api_key="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDQzMDFAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.c78hKkboZXsPTMZGeGxjkBLNteWDtZ9Eq_0W0AejpYE",          # paste your token
    base_url="https://aipipe.org/openai/v1"    # AI Pipe proxy URL
)

# ── Request & Response schemas ──────────────────────────────────
class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: str   # "positive", "negative", "neutral"
    rating: int      # 1-5

# ── The JSON schema we enforce on the model ─────────────────────
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

# ── Endpoint ────────────────────────────────────────────────────
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
                        "Analyze the given comment and return:\n"
                        "- sentiment: 'positive', 'negative', or 'neutral'\n"
                        "- rating: integer 1-5 "
                        "(5=very positive, 3=neutral, 1=very negative)"
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            # ✅ This is the KEY part — enforcing structured output
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