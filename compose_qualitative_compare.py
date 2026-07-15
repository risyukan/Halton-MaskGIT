"""定性比較図の組版。生成済み statics/qual_compare/images.pt を読み、
行=クラス, 列=Baseline / No refresh / N=4 の 3 列比較図を作る。

  python compose_qualitative_compare.py            # 全 seed のコンタクトシート
  python compose_qualitative_compare.py <seed>     # その seed で最終図を出力
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "statics/qual_compare"
data = torch.load(os.path.join(OUT_DIR, "images.pt"))
store = data["store"]
# 注: 元リポジトリのコメントは 681 を "race car" としていたが実際は laptop(notebook)。
LABEL_NAMES = ["goldfish", "tiger cat", "ship", "laptop"]
SEEDS = data["seeds"]

COL_TITLES = ["Baseline", "No refresh\n(partial update)", r"Ours ($N=4$)"]
CONFIG_KEYS = ["baseline", "norefresh", "n4"]


def to_np(img):  # (3,H,W) [0,1] -> (H,W,3)
    return img.permute(1, 2, 0).numpy()


def make_figure(seed, out_path):
    imgs = store[seed]
    n_rows = len(LABEL_NAMES)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.1, n_rows * 2.1),
                             gridspec_kw=dict(wspace=0.04, hspace=0.04))
    for r in range(n_rows):
        for c, key in enumerate(CONFIG_KEYS):
            ax = axes[r, c]
            ax.imshow(to_np(imgs[key][r]))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(COL_TITLES[c], fontsize=13, pad=8)
        # 行ラベル (クラス名) を左端に
        axes[r, 0].set_ylabel(LABEL_NAMES[r], fontsize=12, rotation=90,
                              labelpad=6, va="center")
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.005)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved ->", out_path, "(+ .pdf)")


def make_contact_sheet(out_path):
    """全 seed を縦に並べたコンタクトシート (seed 選び用)。"""
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        make_figure(seed, os.path.join(OUT_DIR, f"qual_compare_seed{seed}.png"))
    else:
        make_contact_sheet(os.path.join(OUT_DIR, "contact_sheet.png"))
