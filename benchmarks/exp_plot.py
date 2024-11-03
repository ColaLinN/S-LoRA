import json
import matplotlib.pyplot as plt

def getData(filePath):
    num_adapters = []
    throughputs = []
    latencies = []
    with open(filePath, "r") as file:
        for line in file:
            # 解析每一行 JSON 数据
            data = json.loads(line)

            # 提取 num_adapters 和 throughput, avg_latency
            num_adapters.append(data["config"]["num_adapters"])
            throughputs.append(data["result"]["throughput"])
            latencies.append(data["result"]["avg_latency"])
    return num_adapters, throughputs, latencies

# slora = "20241031_synthetic_num_adapters_a100_40GB_S4_short_duration_full_slora.jsonl"
# slora_no_mem = "20241031_synthetic_num_adapters_a100_40GB_S4_short_duration_slora_no_mem.jsonl"
slora = "20241031_synthetic_num_adapters_a100_40GB_S4_full_slora.jsonl"
slora_no_mem = "20241031_synthetic_num_adapters_a100_40GB_S4_slora_no_mem.jsonl"
slora_bmm = "20241031_synthetic_num_adapters_a100_40GB_S4_slora_bmm.jsonl"

# 创建图表
plt.figure(figsize=(12, 5))
# 绘制吞吐量图表
plt.subplot(2, 1, 1)
num_adapters, throughputs, latencies = getData(slora)
plt.plot(num_adapters, throughputs, marker='o', color='blue', label="slora")
plt.ylim(0, max(throughputs) * 1.3)  # 设置 y 轴从 0 开始，给出 10% 的上限缓冲
num_adapters, throughputs, latencies = getData(slora_no_mem)
plt.plot(num_adapters, throughputs, marker='o', color='green', label="slora_no_mem")
num_adapters, throughputs, latencies = getData(slora_bmm)
plt.plot(num_adapters, throughputs, marker='o', color='orange', label="slora_bmm")
plt.xlabel("Number of Adapters")
plt.ylabel("Throughput (req/s)")
plt.title("Throughput vs Number of Adapters")
# 设置 x 轴的刻度，只显示 100 和 200
plt.yticks([0.0, 0.5, 1, 1.5])
plt.xticks([100, 200])
plt.legend()

# 绘制延迟图表
plt.subplot(2, 1, 2)
num_adapters, throughputs, latencies = getData(slora)
plt.plot(num_adapters, latencies, marker='o', color='blue', label="slora")
# plt.ylim(0, max(latencies) * 1.3)  # 设置 y 轴从 0 开始，给出 10% 的上限缓冲
num_adapters, throughputs, latencies = getData(slora_no_mem)
plt.plot(num_adapters, latencies, marker='o', color='green', label="slora_no_mem")
num_adapters, throughputs, latencies = getData(slora_bmm)
plt.plot(num_adapters, latencies, marker='o', color='orange', label="slora_bmm")
plt.xlabel("Number of Adapters")
plt.ylabel("Average Latency (s)")
plt.title("Average Latency vs Number of Adapters")
plt.xticks([100, 200])
plt.legend()

# 调整布局并显示图表
plt.tight_layout()
plt.show()
