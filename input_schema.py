from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict

class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    prompt: str = Field(..., description="The input prompt for text generation")
    max_length: Optional[int] = Field(default=512, description="Maximum length of the generated text")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature for text generation")
    top_p: Optional[float] = Field(default=0.9, description="Nucleus sampling parameter")
    num_return_sequences: Optional[int] = Field(default=1, description="Number of sequences to generate")
    stop_sequences: Optional[List[str]] = Field(default=None, description="List of sequences that will stop the generation")
    repetition_penalty: Optional[float] = Field(default=1.0, description="Penalty for repeating tokens")

class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    generated_text: str = Field(..., description="The generated text response")
    prompt: str = Field(..., description="The original input prompt")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata about the generation")

# Define the input schema for Inferless
INPUT_SCHEMA = {
    "type": "object",
    "title": "GenerateRequest",
    "description": "Schema for text generation request",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "The input prompt for text generation"
        },
        "max_length": {
            "type": "integer",
            "description": "Maximum length of the generated text",
            "default": 512
        },
        "temperature": {
            "type": "number",
            "description": "Sampling temperature for text generation",
            "default": 0.7
        },
        "top_p": {
            "type": "number",
            "description": "Nucleus sampling parameter",
            "default": 0.9
        },
        "num_return_sequences": {
            "type": "integer",
            "description": "Number of sequences to generate",
            "default": 1
        },
        "stop_sequences": {
            "type": "array",
            "description": "List of sequences that will stop the generation",
            "items": {
                "type": "string"
            },
            "default": []
        },
        "repetition_penalty": {
            "type": "number",
            "description": "Penalty for repeating tokens",
            "default": 1.0
        }
    },
    "required": ["prompt"]
}

# Define the output schema for Inferless
OUTPUT_SCHEMA = {
    "type": "object",
    "title": "GenerateResponse",
    "description": "Schema for text generation response",
    "properties": {
        "generated_text": {
            "type": "string",
            "description": "The generated text response"
        },
        "prompt": {
            "type": "string",
            "description": "The original input prompt"
        },
        "metadata": {
            "type": "object",
            "description": "Additional metadata about the generation",
            "properties": {
                "generation_time": {
                    "type": "number",
                    "description": "Time taken for generation in seconds"
                },
                "tokens_generated": {
                    "type": "integer",
                    "description": "Number of tokens generated"
                },
                "model_name": {
                    "type": "string",
                    "description": "Name of the model used"
                },
                "device": {
                    "type": "string",
                    "description": "Device used for generation"
                }
            }
        }
    },
    "required": ["generated_text", "prompt"]
} 