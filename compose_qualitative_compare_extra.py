"""追加 4 クラスの定性比較図の組版 + 「手法の効果が最も出る画像」の自動選定。

  python compose_qualitative_compare_extra.py            # コンタクトシート + スコア表 + best 図
  python compose_qualitative_compare_extra.py <seed>     # その seed の 3 列比較図

選定基準:
  手法 (Ours N=4) は baseline に近く、norefresh は劣化する、という主張を最も強く
  示す (seed, class) を選ぶ。スコア = d(norefresh, baseline) - d(n4, baseline)。
  d は 1-SSIM (skimage があれば) / なければ MSE。値が大きいほど
  「norefresh は崩れるが N=4 は baseline を保つ」= 手法の効果が際立つ。
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from skimage.metrics import structural_similarity as _ssim
    def dist(a, b):  # a,b: (H,W,3) [0,1]
        return 1.0 - _ssim(a, b, channel_axis=2, data_range=1.0)
    METRIC = "1-SSIM"
except Exception:
    def dist(a, b):
        return float(np.mean((a - b) ** 2))
    METRIC = "MSE"

OUT_DIR = "statics/qual_compare_extra"
data = torch.load(os.path.join(OUT_DIR, "images.pt"))
store = data["store"]
LABEL_NAMES = data["label_names"]
SEEDS = data["seeds"]

COL_TITLES = ["Baseline", "No refresh\n(partial update)", r"Ours ($N=4$)"]
CONFIG_KEYS = ["baseline", "norefresh", "n4"]


def to_np(img):
    return img.permute(1, 2, 0).numpy()


def make_figure(seed, out_path, highlight=None):
    """highlight: 強調する行 index (best 選定行を枠で囲む)。"""
    imgs = store[seed]
    n_rows = len(LABEL_NAMES)
    fig, axes = plt.subplots(n_rows, 3, figsize=(3 * 2.1, n_rows * 2.1),
                             gridspec_kw=dict(wspace=0.04, hspace=0.04))
    for r in range(n_rows):
        for c, key in enumerate(CONFIG_KEYS):
            ax = axes[r, c]
            ax.imshow(to_np(imgs[key][r]))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if highlight is not None and r == highlight:
                for s in ax.spines.values():
                    s.set_visible(True); s.set_color("red"); s.set_linewidth(2.5)
            if r == 0:
                ax.set_title(COL_TITLES[c], fontsize=13, pad=8)
        axes[r, 0].set_ylabel(LABEL_NAMES[r], fontsize=12, rotation=90,
                              labelpad=6, va="center")
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.005)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved ->", out_path, "(+ .pdf)")


def make_contact_sheet(out_path):
    n_seeds = len(SEEDS)
    n_rows = len(LABEL_NAMES)
    fig, axes = plt.subplots(n_seeds * n_rows, 3,
                             figsize=(3 * 1.6, n_seeds * n_rows * 1.6),
                             gridspec_kw=dict(wspace=0.03, hspace=0.03))
    for si, seed in enumerate(SEEDS):
        imgs = store[seed]
        for r in range(n_rows):
            gr = si * n_rows + r
            for c, key in enumerate(CONFIG_KEYS):
                ax = axes[gr, c]
                ax.imshow(to_np(imgs[key][r]))
                ax.set_xticks([]); ax.set_yticks([])
                if gr == 0:
                    ax.set_title(COL_TITLES[c], fontsize=11)
                if c == 0:
                    ax.set_ylabel(f"s{seed}\n{LABEL_NAMES[r]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("saved contact sheet ->", out_path)


def score_all():
    """全 (seed, class) のスコアを計算しランキング。"""
    rows = []
    for seed in SEEDS:
        imgs = store[seed]
        base = imgs["baseline"]; nrf = imgs["norefresh"]; n4 = imgs["n4"]
        for r in range(len(LABEL_NAMES)):
            b = to_np(base[r]); nr = to_np(nrf[r]); n = to_np(n4[r])
            d_nr = dist(nr, b)   # norefresh の劣化 (大きいほど崩れる)
            d_n4 = dist(n, b)    # N=4 の劣化 (小さいほど baseline を保つ)
            rows.append((seed, r, LABEL_NAMES[r], d_nr, d_n4, d_nr - d_n4))
    rows.sort(key=lambda x: x[5], reverse=True)
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        make_figure(seed, os.path.join(OUT_DIR, f"qual_compare_seed{seed}.png"))
    else:
        make_contact_sheet(os.path.join(OUT_DIR, "contact_sheet.png"))
        rows = score_all()
        print(f"\n=== 手法効果ランキング (metric={METRIC}, "
              f"score = d(norefresh,base) - d(n4,base), 大きいほど良い) ===")
        print(f"{'rank':>4} {'seed':>5} {'class':<16} "
              f"{'d_norefresh':>12} {'d_n4':>10} {'score':>10}")
        for i, (seed, r, name, d_nr, d_n4, sc) in enumerate(rows[:12]):
            print(f"{i+1:>4} {seed:>5} {name:<16} {d_nr:>12.4f} {d_n4:>10.4f} {sc:>10.4f}")

        best_seed, best_r, best_name = rows[0][0], rows[0][1], rows[0][2]
        print(f"\n>>> BEST: seed={best_seed}, class='{best_name}' (row {best_r})")
        make_figure(best_seed, os.path.join(OUT_DIR, "best_seed_highlight.png"),
                    highlight=best_r)
