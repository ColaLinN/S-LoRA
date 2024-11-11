import numpy as np
import matplotlib.pyplot as plt

def generate_long_tail_distribution(total_requests, min_length=1, max_length=100, alpha=2.5):
    # 使用幂律分布生成长尾数据
    lengths = (np.random.power(alpha, total_requests) * (max_length - min_length)) + min_length
    return lengths.astype(int)

# 生成 10,000 个请求长度样本
request_lengths = generate_long_tail_distribution(10000)

# 可视化结果
plt.hist(request_lengths, bins=200, density=False)  # density=False 表示 y 轴显示数量
plt.xlabel("Request Length")
plt.ylabel("Count")
plt.title("Long Tail Distribution")
plt.xlim(0, max(request_lengths))  # 限制 x 轴范围
plt.show()
