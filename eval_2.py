import os
import math
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import numpy as np
from scipy import linalg

from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler
from Metrics.inception_metrics import MultiInceptionMetrics

# ===================== 可调参数 =====================
CONFIG_PATH   = "Config/base_cls2img.yaml"
IMG_SIZE      = 384           # 256 或 384 —— 必须与真实统计一致！
VIT_SIZE      = "large"        # "tiny" / "small" / "base" / "large"
DTYPE         = "float32"      # "float32" 更稳
COMPILE       = False
TOME_KEEP     = 1.0           # 1.0 关闭 Token Merge；<1 开启
NUM_SAMPLES   = 50000          # 典型为 50k；快速测试可设小一些
BATCH_SIZE    = 32
INCEPT_CHUNKS = 10             # MultiInceptionMetrics 的分块数
REAL_STATS    = f"./saved_networks/ImageNet_{IMG_SIZE}_val_stats.pt"  # 真实统计路径
SAVE_EVERY    = 5000           # 每隔多少 fake 样本顺手做一次中间打印
SEED          = 42
# ===================================================

@torch.no_grad()
def fid_from_stats(mu_real, cov_real, mu_fake, cov_fake):
    """
    基于 pytorch-fid 的标准实现 (Numpy 版本)。
    输入应为 torch.Tensor 或 numpy array。
    """
    # 1. 转为 Numpy
    if isinstance(mu_real, torch.Tensor): mu_real = mu_real.cpu().numpy()
    if isinstance(cov_real, torch.Tensor): cov_real = cov_real.cpu().numpy()
    if isinstance(mu_fake, torch.Tensor): mu_fake = mu_fake.cpu().numpy()
    if isinstance(cov_fake, torch.Tensor): cov_fake = cov_fake.cpu().numpy()

    mu1, mu2 = mu_real, mu_fake
    sigma1, sigma2 = cov_real, cov_fake

    # 2. 计算均值差的平方
    diff = mu1 - mu2
    
    # 3. 计算协方差乘积的平方根
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # 4. 数值稳定性处理
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % 1e-6
        print(msg)
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # 如果存在极小的虚部（数值误差），取实部
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    # 5. 最终公式: ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def load_real_stats(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"未找到真实统计文件：{path}\n"
            f"请先运行 extract_real_stats.py 对 ImageNet val 提取并保存为：{path}\n"
            f"（img_size={IMG_SIZE} 必须一致）"
        )
    d = torch.load(path, map_location="cpu")
    return d["mu"].double(), d["cov"].double()

@torch.no_grad()  # <--- 修改 #1: 添加全局 no_grad
def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # 载入配置并设置
    args = load_args_from_file(CONFIG_PATH)
    args.device = device
    args.vit_size = VIT_SIZE
    args.img_size = IMG_SIZE
    args.compile  = COMPILE
    args.dtype    = DTYPE
    args.resume   = True
    args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"
    args.tome_keep_ratio = TOME_KEEP
    # 确保单卡评估时这些参数存在
    args.is_master = True
    args.is_multi_gpus = False

    # 下载权重（若本地不存在）
    hf_hub_download(repo_id="llvictorll/Halton-Maskgit",
                    filename=f"ImageNet_{args.img_size}_{args.vit_size}.pth",
                    local_dir="./saved_networks")
    hf_hub_download(repo_id="FoundationVision/LlamaGen",
                    filename="vq_ds16_c2i.pt",
                    local_dir="./saved_networks")

    # 初始化模型与采样器
    trainer = MaskGIT(args)
    trainer.vit.eval()
    trainer.ae.eval()

    sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0, w=0.5,
                            sched_pow=2, step=32, randomize=False, top_k=-1)

    # 载入真实统计
    mu_real, cov_real = load_real_stats(REAL_STATS)

    # Inception 特征收集器
    metrics = MultiInceptionMetrics(
        device=device, compute_manifold=False, num_classes=1000,
        num_inception_chunks=INCEPT_CHUNKS, manifold_k=3, model="inception",
    )

    # 生成并累计特征（分类条件：均匀采样 1000 类）
    num_generated = 0  # <--- 修改 #2: 使用更清晰的计数器
    pbar = tqdm(total=NUM_SAMPLES, desc="Generating & updating Inception features")
    
    while num_generated < NUM_SAMPLES:
        cur = min(BATCH_SIZE, NUM_SAMPLES - num_generated)
        labels = torch.randint(0, 1000, (cur,), device=device)

        fake_images = sampler(trainer=trainer, nb_sample=cur, labels=labels, verbose=False)[0]
        
        # <--- 修改 #3: 确保图像在正确的设备和范围
        # 假设 sampler 返回的是 [-1, 1] 范围，与真实数据一致
        # 如果不确定，可以打印检查: print(f"fake range: [{fake_images.min()}, {fake_images.max()}]")
        fake_images = fake_images.to(device)

        metrics.update(fake_images, image_type="fake")

        num_generated += cur
        pbar.update(cur)

        # <--- 修改 #4: 修正中间估算逻辑
        if num_generated > 0 and num_generated % SAVE_EVERY == 0:
            fake_features = torch.cat(metrics.fake_features, dim=0)
            mu_fake = fake_features.mean(dim=0)
            cov_fake = metrics.cov(fake_features, mu_fake)
            fid_mid = fid_from_stats(mu_real, cov_real, mu_fake.cpu(), cov_fake.cpu())
            pbar.set_postfix({"FID(est.)": f"{fid_mid:.3f}", "samples": num_generated})

    pbar.close()

    # 计算最终 FID
    fake_features = torch.cat(metrics.fake_features, dim=0)
    mu_fake = fake_features.mean(dim=0)
    cov_fake = metrics.cov(fake_features, mu_fake)

    fid = fid_from_stats(mu_real, cov_real, mu_fake.cpu(), cov_fake.cpu())
    
    print("\n========== FID RESULT ==========")
    print(f"IMG_SIZE: {IMG_SIZE} | NUM_SAMPLES: {NUM_SAMPLES} | BATCH_SIZE: {BATCH_SIZE} | ToMe keep: {TOME_KEEP}")
    print(f"FID: {fid:.4f}")
    print("================================")

if __name__ == "__main__":
    main()