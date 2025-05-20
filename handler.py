from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time
import logging
from input_schema import GenerateRequest, GenerateResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Model:
    def __init__(self):
        try:
            self.model_name = os.getenv("MODEL_NAME", "coco-llm")
            self.device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
            
            logger.info(f"Initializing model {self.model_name} on {self.device}")
            
            # Initialize model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def predict(self, request: GenerateRequest) -> GenerateResponse:
        try:
            start_time = time.time()
            
            # Validate input
            if not request.prompt:
                raise ValueError("Prompt cannot be empty")
            
            # Tokenize input
            inputs = self.tokenizer(request.prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    num_return_sequences=request.num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=True
                )
            
            # Decode and return the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate metadata
            generation_time = time.time() - start_time
            tokens_generated = len(self.tokenizer.encode(generated_text))
            
            return GenerateResponse(
                generated_text=generated_text,
                prompt=request.prompt,
                metadata={
                    "generation_time": round(generation_time, 2),
                    "tokens_generated": tokens_generated,
                    "model_name": self.model_name,
                    "device": self.device
                }
            )
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise

# Initialize the model
try:
    model = Model()
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

def infer(request: GenerateRequest) -> GenerateResponse:
    return model.predict(request) 