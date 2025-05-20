from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any

class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    prompt: str = Field(
        default="",
        description="The input prompt for text generation",
        example="Write a story about a magical forest"
    )
    max_length: Optional[int] = Field(
        default=512,
        description="Maximum length of the generated text",
        ge=1,
        le=2048
    )
    temperature: Optional[float] = Field(
        default=0.7,
        description="Sampling temperature for text generation",
        ge=0.0,
        le=2.0
    )
    top_p: Optional[float] = Field(
        default=0.9,
        description="Nucleus sampling parameter",
        ge=0.0,
        le=1.0
    )
    num_return_sequences: Optional[int] = Field(
        default=1,
        description="Number of sequences to generate",
        ge=1,
        le=5
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="List of sequences that will stop the generation"
    )
    repetition_penalty: Optional[float] = Field(
        default=1.0,
        description="Penalty for repeating tokens",
        ge=1.0,
        le=2.0
    )

class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    generated_text: str = Field(
        default="",
        description="The generated text response"
    )
    prompt: str = Field(
        default="",
        description="The original input prompt"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the generation"
    )

# Example input JSON
EXAMPLE_INPUT = {
    "prompt": "Write a short story about a magical forest",
    "max_length": 256,
    "temperature": 0.8,
    "top_p": 0.9,
    "num_return_sequences": 1,
    "stop_sequences": ["###", "END"],
    "repetition_penalty": 1.2
}

# Example output JSON
EXAMPLE_OUTPUT = {
    "generated_text": "In the heart of the ancient forest, where sunlight danced through emerald leaves...",
    "prompt": "Write a short story about a magical forest",
    "metadata": {
        "generation_time": 1.23,
        "tokens_generated": 45
    }
} 