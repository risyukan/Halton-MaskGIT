import numpy as np
import matplotlib.pyplot as plt

# 原表里的 Token 保留率
keep_ratio = np.array([1.00, 0.75, 0.50, 0.33])
# 合并比率 = 1 - 保留率
merge_ratio = 1.0 - keep_ratio

# 你的方法（从表中读的相対速度）
ours_speedup = np.array([1.00, 1.30, 1.71, 2.22])

# ToMeSD 方法（从表中读的相対速度）
tomesd_speedup = np.array([1.00, 0.88, 1.05, 1.22])

plt.figure(figsize=(6, 4))

plt.plot(keep_ratio, ours_speedup, marker='o', linestyle='-', label='Ours')
plt.plot(keep_ratio, tomesd_speedup, marker='s', linestyle='--', label='ToMeSD')

plt.xlabel('Token keep ratio')
plt.ylabel('Speedup over baseline (×)')
plt.title('Token Merge Speedup vs. Merge Ratio')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("token_merge_speedup.png", dpi=300)
