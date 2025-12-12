import numpy as np
import matplotlib.pyplot as plt

# 一共有多少个 token（这里假设还是 16x16 的 256 个 token）
n = 8
num_tokens = n * n

# 一共分成多少批（step）
num_steps = 8

# 整张图初始化为浅灰色：shape = (num_steps, num_tokens, 3)
data = np.ones((num_steps, num_tokens, 3)) * 0.9

# 每一批的颜色（批之间颜色不同）
colors = np.array([
    [0.95, 0.30, 0.30],  # 红
    [0.30, 0.55, 0.95],  # 蓝
    [0.30, 0.80, 0.30],  # 绿
    [0.95, 0.85, 0.30],  # 黄
    [0.95, 0.60, 0.25],  # 橙
    [0.75, 0.40, 0.85],  # 紫
    [0.60, 0.85, 0.60],  # 浅绿
    [0.65, 0.85, 0.95],  # 浅蓝
])

for step in range(num_steps):
    # 归一化时间 t ∈ (0,1]
    t = (step + 1) / num_steps

    # cos 调度：从少到多逐渐增大
    # 0.5*(1 - cos(pi * t)) 从 0 → 1 平滑增长
    frac = 0.5 * (1 - np.cos(np.pi * t))

    # 这一批中要上色的 token 数
    k = int(frac * num_tokens)

    # 随机选出 k 个 token 上色
    idxs = np.random.choice(num_tokens, k, replace=False)

    # 把这些 token 画成当前批的颜色
    data[step, idxs] = colors[step % len(colors)]

fig, ax = plt.subplots(figsize=(10, 3))

# 显示：纵轴是批次，横轴是一行 token
ax.imshow(data, vmin=0, vmax=1, aspect='auto')

# 去掉坐标刻度
ax.set_xticks([])
ax.set_yticks([])

# 画网格线方便看
for x in range(num_tokens + 1):
    ax.axvline(x - 0.5, color='white', linewidth=0.5)
for y in range(num_steps + 1):
    ax.axhline(y - 0.5, color='white', linewidth=0.5)

plt.savefig("token_cos_schedule.png", dpi=300, bbox_inches='tight')

