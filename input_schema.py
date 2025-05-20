from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1
    stop_sequences: list = None
    repetition_penalty: float = 1.0

class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    metadata: dict = None 