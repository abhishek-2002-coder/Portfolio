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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    system: str = "You are a helpful assistant."

@app.get("/")
async def root():
    return {"message": "Abhi's Portfolio API (Gemini Edition) is live! ✨"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    # Format messages for Gemini API
    # Gemini uses 'user' and 'model' (Anthropic used 'assistant')
    contents = []
    for msg in request.messages:
        role = "user" if msg.role == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg.content}]
        })

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                GEMINI_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "system_instruction": {
                        "parts": [{"text": request.system}]
                    },
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": 1024,
                    }
                },
                timeout=30.0
            )

            if response.status_code != 200:
                error_detail = response.text
                print(f"Gemini API Error: {error_detail}")
                raise HTTPException(status_code=response.status_code, detail=error_detail)

            # Format Gemini response to match what the frontend expects
            # (Anthropic format: data.content[0].text)
            gemini_data = response.json()
            try:
                bot_text = gemini_data['candidates'][0]['content']['parts'][0]['text']
                return {
                    "content": [{"text": bot_text}]
                }
            except (KeyError, IndexError) as e:
                print(f"Parsing Error: {str(e)} - Data: {gemini_data}")
                raise HTTPException(status_code=500, detail="Error parsing response from Gemini")

        except Exception as e:
            print(f"Backend Crash: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
