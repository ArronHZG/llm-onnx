"""Deep debug: check SGLang service configuration and available models"""
import requests
import json

base = "http://33.29.18.248:44400"

# Check model info in detail
print("=== Model Info ===")
r = requests.get(f"{base}/v1/models", timeout=10)
data = r.json()
for m in data.get("data", []):
    print(json.dumps(m, indent=2))

print("\n=== SGLang Info (if available) ===")
for endpoint in ["/info", "/health", "/get_model_info", "/get_logit_processor_info"]:
    try:
        r = requests.get(f"{base}{endpoint}", timeout=5)
        if r.status_code == 200:
            text = r.text[:500]
            print(f"  {endpoint}: {text}")
    except Exception as e:
        print(f"  {endpoint}: {e}")

# Test with a very simple prompt and very few tokens
print("\n=== Minimal test ===")
r = requests.post(f"{base}/v1/chat/completions", json={
    "model": "/home/hadoop-djst-algoplat/model/Qwen/Qwen3.6-27B",
    "messages": [{"role": "user", "content": "Say OK"}],
    "max_tokens": 5,
    "stream": False,
}, timeout=30)
d = r.json()
msg = d["choices"][0]["message"]
print(f"content: {repr(msg.get('content'))}")
print(f"reasoning: {repr(msg.get('reasoning_content', ''))}")
print(f"finish_reason: {d['choices'][0].get('finish_reason')}")

# Try with detailed logprobs to see actual tokens
print("\n=== With logprobs ===")
r = requests.post(f"{base}/v1/chat/completions", json={
    "model": "/home/hadoop-djst-algoplat/model/Qwen/Qwen3.6-27B",
    "messages": [{"role": "user", "content": "Say OK"}],
    "max_tokens": 5,
    "stream": False,
    "logprobs": 1,
    "top_logprobs": 3,
}, timeout=30)
d = r.json()
choice = d["choices"][0]
msg = choice.get("message", {})
logprobs = choice.get("logprobs")
print(f"content: {repr(msg.get('content'))}")
reasoning_val = repr(msg.get('reasoning_content', ''))[:200]
print(f"reasoning: {reasoning_val}")
if logprobs:
    print(f"logprobs keys: {list(logprobs.keys()) if isinstance(logprobs, dict) else type(logprobs)}")
    # Try to see token texts
    if isinstance(logprobs, dict) and 'content' in logprobs:
        tokens = logprobs['content']
        if tokens:
            for t in tokens[:5]:
                print(f"  token: {t}")

