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

# Define the input schema for Inferless
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "max_length": {"type": "integer", "default": 512},
        "temperature": {"type": "number", "default": 0.7},
        "top_p": {"type": "number", "default": 0.9},
        "num_return_sequences": {"type": "integer", "default": 1},
        "stop_sequences": {"type": "array", "items": {"type": "string"}, "default": None},
        "repetition_penalty": {"type": "number", "default": 1.0}
    },
    "required": ["prompt"]
}

# Define the output schema for Inferless
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "generated_text": {"type": "string"},
        "prompt": {"type": "string"},
        "metadata": {
            "type": "object",
            "properties": {
                "generation_time": {"type": "number"},
                "tokens_generated": {"type": "integer"},
                "model_name": {"type": "string"},
                "device": {"type": "string"}
            }
        }
    },
    "required": ["generated_text", "prompt"]
} 