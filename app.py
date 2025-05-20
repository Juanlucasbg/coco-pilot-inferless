from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import time
from typing import Optional, List, Dict, Any
from input_schema import GenerateRequest, GenerateResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="COCO LLM API",
    description="API for interacting with the COCO LLM model",
    version="1.0.0"
)

# Initialize model and tokenizer
MODEL_NAME = "coco-llm"  # Update this with your actual model name
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    logger.info(f"Model loaded successfully on {device}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.get("/")
async def root():
    return {"message": "Welcome to COCO LLM API"}

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        start_time = time.time()
        
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                num_return_sequences=request.num_return_sequences,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=request.repetition_penalty,
                do_sample=True
            )
        
        # Decode and return the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate metadata
        generation_time = time.time() - start_time
        tokens_generated = len(tokenizer.encode(generated_text))
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            metadata={
                "generation_time": round(generation_time, 2),
                "tokens_generated": tokens_generated,
                "model_name": MODEL_NAME,
                "device": device
            }
        )
    
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 