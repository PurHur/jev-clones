"""
OpenAI-compatible shim for mailguard-jev-style-1.5b.
Exposes /v1/chat/completions so any OpenAI client can use the model locally.

Usage:
    pip install fastapi uvicorn transformers torch
    python serve_openai_shim.py
"""

import json
import re
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "JackKozmo29/mailguard-jev-style-1.5b"

print(f"Loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
print("Model ready.")

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    max_tokens: int = 120
    temperature: float = 0.0


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    messages = [m.model_dump() for m in req.messages]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=req.max_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    content = m.group(0) if m else text.strip()

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": ids["input_ids"].shape[1], "completion_tokens": len(out[0]) - ids["input_ids"].shape[1], "total_tokens": len(out[0])},
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
