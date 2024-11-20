import requests
import json

# 请求的URL和请求体数据
url = "http://localhost:8000/generate_stream"
headers = {
    "Content-Type": "application/json",
    "User-Agent": "Benchmark Client"
}
data = {
    "model_dir": "huggyllama/llama-7b",
    "lora_dir": "tloen/alpaca-lora-7b-1",
    # "lora_dir": "MBZUAI/bactrian-x-llama-7b-lora-1",
    "inputs": "你好llama，我想请教一下long-tail长度的requests分布对transformer推理有什么影响？",
    # "inputs": "hello hello hello",
    "parameters": {
        "do_sample": False,
        "ignore_eos": False,
        "max_new_tokens": 2048,
        "temperature": 0.4,
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
print()
# print(generated_text)
