import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your portfolio domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    system: str = "You are a helpful assistant."

@app.get("/")
async def root():
    return {"message": "Abhi's Portfolio API is live! ✨"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic API key not configured")

    # Claude specifically needs messages to start with 'user'
    # and alternate between 'user' and 'assistant'
    formatted_messages = []
    for msg in request.messages:
        formatted_messages.append({"role": msg.role, "content": msg.content})

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                ANTHROPIC_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-5-sonnet-20240620",
                    "max_tokens": 1024,
                    "system": request.system,
                    "messages": formatted_messages
                },
                timeout=30.0
            )

            if response.status_code != 200:
                error_detail = response.text
                print(f"Anthropic API Error: {error_detail}")
                raise HTTPException(status_code=response.status_code, detail=error_detail)

            return response.json()

        except Exception as e:
            print(f"Backend Crash: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
