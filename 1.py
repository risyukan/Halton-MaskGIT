import numpy as np
import matplotlib.pyplot as plt

# 网格大小（和原图一样 20x20）
n = 16

# 用 0.9（接近白色的浅灰）填满
data = np.ones((n, n, 3)) * 0.9

num_colored = 256
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
# 随机选出若干个格子，赋予随机颜色
idxs = np.random.choice(n * n, num_colored, replace=False)
for idx in idxs:
    i = idx // n
    j = idx % n
    data[i, j] = colors[np.random.randint(len(colors))]

fig, ax = plt.subplots(figsize=(6, 6))

# 直接画 RGB 图像（不需要 cmap）
ax.imshow(data, vmin=0, vmax=1)

# 去掉刻度
ax.set_xticks([])
ax.set_yticks([])

# 去掉边框
ax.set_axis_off()

# 画白色网格线
for x in range(n + 1):
    ax.axhline(x - 0.5, color='white', linewidth=1)
    ax.axvline(x - 0.5, color='white', linewidth=1)

plt.savefig("full_color_grid.png", dpi=300, bbox_inches='tight')
