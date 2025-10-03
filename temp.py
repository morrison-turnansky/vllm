import requests

url = "http://localhost:8000/v1/completions"

data = {
    "prompt": [
        "Explain block quantized FP8 models in simple terms.",
        "Summarize the benefits of FP8 quantization."
    ],
    "max_tokens": 100
}

response = requests.post(url, json=data)
result = response.json()

for i, item in enumerate(result["results"]):
    print(f"Prompt {i+1} output:\n{item['text']}\n")
