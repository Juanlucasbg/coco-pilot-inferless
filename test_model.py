import requests
import json
from input_schema import EXAMPLE_INPUT
import time

def test_model(prompt, max_length=256, temperature=0.8):
    """Test the model with a given prompt and parameters"""
    url = "http://localhost:8000/generate"
    
    payload = {
        "prompt": prompt,
        "max_length": max_length,
        "temperature": temperature,
        "top_p": 0.9,
        "num_return_sequences": 1,
        "repetition_penalty": 1.2
    }
    
    try:
        print(f"\nTesting with prompt: {prompt}")
        print("-" * 50)
        
        start_time = time.time()
        response = requests.post(url, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"Generated text: {result['generated_text']}")
            print(f"\nMetadata:")
            print(f"- Generation time: {result['metadata']['generation_time']} seconds")
            print(f"- Tokens generated: {result['metadata']['tokens_generated']}")
            print(f"- Model: {result['metadata']['model_name']}")
            print(f"- Device: {result['metadata']['device']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def main():
    # Test cases
    test_cases = [
        {
            "prompt": "Write a short story about a magical forest",
            "max_length": 256,
            "temperature": 0.8
        },
        {
            "prompt": "Explain quantum computing in simple terms",
            "max_length": 512,
            "temperature": 0.7
        },
        {
            "prompt": "Write a poem about artificial intelligence",
            "max_length": 128,
            "temperature": 0.9
        }
    ]
    
    print("Starting model tests...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}")
        test_model(**test_case)
        time.sleep(1)  # Add a small delay between tests

if __name__ == "__main__":
    main() 