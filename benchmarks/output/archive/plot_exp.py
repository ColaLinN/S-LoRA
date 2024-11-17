import json
import matplotlib.pyplot as plt

# 初始化存储数据的列表
num_adapters = []
throughputs = []
latencies = []

# 读取 JSONL 文件
with open("data.jsonl", "r") as file:
    for line in file:
        # 解析每一行 JSON 数据
        data = json.loads(line)
        
        # 提取 num_adapters 和 throughput, avg_latency
        num_adapters.append(data["config"]["num_adapters"])
        throughputs.append(data["result"]["throughput"])
        latencies.append(data["result"]["avg_latency"])

# 创建图表
plt.figure(figsize=(12, 5))

# 绘制吞吐量图表
plt.subplot(1, 2, 1)
plt.plot(num_adapters, throughputs, marker='o', color='blue', label="Throughput (req/s)")
plt.xlabel("Number of Adapters")
plt.ylabel("Throughput (req/s)")
plt.title("Throughput vs Number of Adapters")
plt.legend()

# 绘制延迟图表
plt.subplot(1, 2, 2)
plt.plot(num_adapters, latencies, marker='o', color='orange', label="Average Latency (s)")
plt.xlabel("Number of Adapters")
plt.ylabel("Average Latency (s)")
plt.title("Average Latency vs Number of Adapters")
plt.legend()

# 调整布局并显示图表
plt.tight_layout()
plt.show()
