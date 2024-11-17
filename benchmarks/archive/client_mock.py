import requests
import json

# 请求的URL和请求体数据
url = "http://localhost:8000/generate_stream"
headers = {
    "Content-Type": "application/json",
    "User-Agent": "Benchmark Client"
}
data = {
    "model_dir": "huggyllama/llama-13b",
    "lora_dir": "dummy-lora-13b-rank-64-49",
    "inputs": "你好，世界，今天你想听什么？",
    "parameters": {
        "do_sample": True,
        "ignore_eos": False,
        "max_new_tokens": 2048
    }
}

# 发起 POST 请求并设置流式读取
response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

# 初始化文本拼接容器
generated_text = ""

# 逐行处理流式响应
for line in response.iter_lines():
    if line:
        try:
            # 解析 JSON 数据
            line_data = json.loads(line.decode("utf-8").replace("data:", ""))
            token = line_data.get("token", {})
            text = token.get("text", "")
            generated_text += text  # 拼接文本
            print(text, end="", flush=True)  # 流式输出拼接的文本
        except json.JSONDecodeError:
            # 跳过无法解析的行
            continue

# 打印最终生成的完整文本
# print("\n\n完整生成的文本：")
print(generated_text)
