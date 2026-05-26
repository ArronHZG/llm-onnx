"""Raw API test - dump full response structure"""
import requests
import json

url = "http://33.29.18.248:44400/v1/chat/completions"
model = "/home/hadoop-djst-algoplat/model/Qwen/Qwen3.6-27B"

payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 32,
    "stream": False,
}
resp = requests.post(url, json=payload, timeout=60)
data = resp.json()

# Print full response (truncated)
print("=== Full response keys ===")
print(f"Top keys: {list(data.keys())}")
print()

choice = data.get("choices", [{}])[0]
print(f"=== Choice ===")
print(json.dumps(choice, indent=2, ensure_ascii=False)[:2000])
print()

msg = choice.get("message", {})
print(f"=== Message keys: {list(msg.keys()) if msg else 'None'} ===")
if msg:
    for k, v in msg.items():
        val_repr = repr(v)
        if len(val_repr) > 200:
            val_repr = val_repr[:200] + f"...(total {len(str(v))} chars)"
        print(f"  {k}: {val_repr}")

