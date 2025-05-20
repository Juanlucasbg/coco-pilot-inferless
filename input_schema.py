from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class GenerateRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    num_return_sequences: Optional[int] = 1
    stop_sequences: Optional[List[str]] = None
    repetition_penalty: Optional[float] = 1.0

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Write a story about a magical forest",
                "max_length": 256,
                "temperature": 0.8,
                "top_p": 0.9,
                "num_return_sequences": 1,
                "stop_sequences": ["###", "END"],
                "repetition_penalty": 1.2
            }
        }

class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "generated_text": "In the heart of the ancient forest, where sunlight danced through emerald leaves...",
                "prompt": "Write a short story about a magical forest",
                "metadata": {
                    "generation_time": 1.23,
                    "tokens_generated": 45
                }
            }
        } 