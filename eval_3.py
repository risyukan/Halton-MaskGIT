import torch
import numpy as np
from scipy import linalg
from tqdm import tqdm

from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

# 导入你自己的 Metrics 类
from Metrics.inception_metrics import MultiInceptionMetrics

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    标准的 Numpy 版 FID 计算函数 (Fréchet Distance)
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # 计算协方差矩阵的平方根: (Sigma1 * Sigma2)^(1/2)
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # 数值稳定性处理：如果是复数结果，取实部
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def main():
    # --- 1. 配置与模型初始化 ---
    config_path = "Config/base_cls2img.yaml"
    args = load_args_from_file(config_path)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确保参数与训练时一致
    args.vit_size = "Large"
    args.img_size = 384
    args.compile = False
    args.dtype = "float32"
    args.resume = True
    args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"

    # 初始化 MaskGIT
    model = MaskGIT(args)
    # model.eval()  <--- 删除或注释掉这一行，因为 MaskGIT 类的实现有 bug

    # --- 手动将内部模型设置为 eval 模式 ---
    # 根据你的日志 "Size of model vit"，内部属性名很可能是 transformer 或 vit
    if hasattr(model, 'transformer'):
        model.transformer.eval()
        print("Set model.transformer to eval mode.")
    elif hasattr(model, 'vit'):
        model.vit.eval()
        print("Set model.vit to eval mode.")
        
    # 同时也把 VQGAN 设置为 eval
    if hasattr(model, 'vqgan'):
        model.vqgan.eval()
        print("Set model.vqgan to eval mode.")

    # 初始化 Sampler
    sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0, w=0.5,
                            sched_pow=2, step=32, randomize=False, top_k=-1)

    # --- 2. 初始化 Metrics ---
    # 使用与生成 .pt 文件时完全相同的配置
    print("正在初始化 Inception Metrics...")
    metrics = MultiInceptionMetrics(
        device=args.device, 
        compute_manifold=False, 
        num_classes=1000,
        num_inception_chunks=10, 
        manifold_k=3, 
        model="inception"
    )

    # --- 3. 生成图片并提取特征 ---
    total_gen = 50_000  # 建议生成 50k 张以获得准确结果
    batch_size = 32     # 根据显存调整
    
    print(f"开始生成 {total_gen} 张图片并提取特征...")
    
    num_batches = (total_gen + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating"):
            # 随机采样标签
            current_bs = min(batch_size, total_gen) # 逻辑简化，最后一次batch可能会多一点点，不影响
            labels = torch.randint(0, 1000, (batch_size,)).to(args.device)
            
            # MaskGIT 生成
            # 注意：Sampler 返回的是 tensor
            gen_images = sampler(trainer=model, nb_sample=batch_size, labels=labels, verbose=False)[0]
            
            # --- 关键点：数据范围归一化 ---
            # MaskGIT 通常输出 [-1, 1]。
            # InceptionMetrics 通常期望输入是 [-1, 1] 或者 [0, 1]。
            # 既然你的 data_loader 输出是用来跑 Metrics 的，
            # 假设 MaskGIT 输出的 range 应该直接传进去即可。
            # 如果 metrics 内部有 assert 0<=x<=1，则需在这里做 (x+1)/2
            
            # 这里的 image_type="fake" 是为了让 Metrics 类把特征存到 fake_features 列表里
            # 如果你的 Metrics 类不支持 "fake"，可以用 "real" 然后取 metrics.real_features
            metrics.update(gen_images, image_type="fake")

    # --- 4. 计算生成数据的统计信息 (Mu, Sigma) ---
    print("正在计算生成图片的统计数据...")
    
    # 假设 update 存入了 self.fake_features (参照你的 SampleAndEval 类逻辑)
    # 如果 Metrics 类里没有 fake_features，请检查代码，可能是复用了 real_features
    if hasattr(metrics, 'fake_features') and len(metrics.fake_features) > 0:
        fake_feats_list = metrics.fake_features
    else:
        # 如果类里没分 real/fake，可能默认存到了 real_features
        fake_feats_list = metrics.real_features

    fake_features = torch.cat(fake_feats_list, dim=0)
    
    # 截取到精确的 total_gen 数量
    fake_features = fake_features[:total_gen]

    # 计算均值和协方差
    mu_gen = fake_features.mean(dim=0).cpu().numpy()
    
    # 计算协方差 (使用 Metrics 自带的 helper 或者 numpy)
    # 既然你有 metrics.cov 方法，我们可以尝试用它，或者直接用 numpy
    # 为了保险，这里转换成 numpy 计算
    fake_features_np = fake_features.cpu().numpy()
    sigma_gen = np.cov(fake_features_np, rowvar=False)

    # --- 5. 加载参考统计信息 (.pt) ---
    stats_path = "./saved_networks/ImageNet_384_val_stats.pt"
    print(f"正在加载参考统计文件: {stats_path}")
    
    ref_stats = torch.load(stats_path, map_location="cpu")
    mu_real = ref_stats["mu"].numpy()
    sigma_real = ref_stats["cov"].numpy()

    # --- 6. 计算 FID ---
    print("正在计算最终 FID...")
    fid_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    
    print("-" * 30)
    print(f"FID Score: {fid_score:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()