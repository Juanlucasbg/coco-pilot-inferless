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
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "title": "GenerateRequest",
    "description": "Schema for text generation request",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "The input prompt for text generation",
            "minLength": 1
        },
        "max_length": {
            "type": "integer",
            "description": "Maximum length of the generated text",
            "default": 512,
            "minimum": 1,
            "maximum": 2048
        },
        "temperature": {
            "type": "number",
            "description": "Sampling temperature for text generation",
            "default": 0.7,
            "minimum": 0.0,
            "maximum": 2.0
        },
        "top_p": {
            "type": "number",
            "description": "Nucleus sampling parameter",
            "default": 0.9,
            "minimum": 0.0,
            "maximum": 1.0
        },
        "num_return_sequences": {
            "type": "integer",
            "description": "Number of sequences to generate",
            "default": 1,
            "minimum": 1,
            "maximum": 5
        },
        "stop_sequences": {
            "type": ["array", "null"],
            "description": "List of sequences that will stop the generation",
            "items": {
                "type": "string"
            },
            "default": None
        },
        "repetition_penalty": {
            "type": "number",
            "description": "Penalty for repeating tokens",
            "default": 1.0,
            "minimum": 1.0,
            "maximum": 2.0
        }
    },
    "required": ["prompt"],
    "additionalProperties": False
}

# Define the output schema for Inferless
OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
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
            "type": ["object", "null"],
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
            },
            "required": ["generation_time", "tokens_generated", "model_name", "device"]
        }
    },
    "required": ["generated_text", "prompt"],
    "additionalProperties": False
} 