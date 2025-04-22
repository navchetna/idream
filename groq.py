from fastapi import FastAPI, Request
from pydantic import BaseModel
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("Llama-3.2-3B-Instruct", "CPU")

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(data: PromptRequest):
    result = pipe.generate(data.prompt, max_new_tokens=100, do_sample=False)
    return {"answer": result}
