# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# # import requests

# # url = "http://localhost:8000/v1/completions"

# # data = {
# #     "prompt": [
# #         "Explain block quantized FP8 models in simple terms.",
# #         "Summarize the benefits of FP8 quantization."
# #     ],
# #     "max_tokens": 100
# # }

# # response = requests.post(url, json=data)
# # result = response.json()

# # for i, item in enumerate(result["results"]):
# #     print(f"Prompt {i+1} output:\n{item['text']}\n")

# import requests
# import time

# url = "http://localhost:8000/v1/completions"

# prompts = [
#     "Write a short story about a cat in space.",
#     "Explain quantum computing in simple terms.",
#     "Generate Python code for a Fibonacci function."
# ]

# start_time = time.time()
# for _ in range(10000):
#     for prompt in prompts:
#         r = requests.post(url, json={"prompt": prompt, "max_tokens": 50})
#         result = r.json()
#         print("Output:", result["choices"][0]["text"])

# end_time = time.time()

# total_time = end_time - start_time
# print(f"Total time for {len(prompts)} prompts: {total_time:.2f} s")
# print(f"Average latency per prompt: {total_time/len(prompts):.2f} s")
# print(f"Throughput: {len(prompts)/total_time:.2f} prompts/sec")

import time

import requests

url = "http://localhost:8000/v1/completions"

prompts = ["Hello world", "Explain quantum computing", "Write a poem"]
for j in range(100):
    for prompt in prompts:
        start = time.time()
        r = requests.post(url, json={"prompt": prompt, "max_tokens": 50})
        end = time.time()

        resp = r.json()
        # vLLM includes "request_stats" if you enable it
        stats = resp.get("request_stats", {})

        print(f"Prompt: {prompt}")
        print(f"Output tokens: {len(resp['choices'][0]['text'].split())}")
        print(f"Latency: {end-start:.3f}s")
        print(f"Stats: {stats}")
