from fastapi import FastAPI
from input_schema import GenerateRequest, GenerateResponse
from handler import infer

app = FastAPI()

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    return infer(request)

@app.get("/")
async def root():
    return {"message": "COCO LLM API is running"} 