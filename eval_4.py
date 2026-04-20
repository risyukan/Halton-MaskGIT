import os
import torch
import torch.distributed as dist
from tqdm import tqdm

from Utils.utils import load_args_from_file
from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler

from Metrics.inception_metrics import MultiInceptionMetrics


def main():
    # --- 1. 初始化分布式进程组 ---
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    is_main = (local_rank == 0)

    # --- 2. 配置与模型初始化 ---
    config_path = "Config/base_cls2img.yaml"
    args = load_args_from_file(config_path)
    # args.device 设为整数 local_rank，与 cls_trainer.py 的 DDP 模式一致
    args.device = local_rank
    args.is_multi_gpus = False   # eval 模式不需要 DDP 包裹生成模型
    args.is_master = is_main

    args.vit_size = "large"
    args.img_size = 384
    args.compile = False
    args.dtype = "float32"
    args.resume = True
    args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"

    model = MaskGIT(args)

    if hasattr(model, 'transformer'):
        model.transformer.eval()
        if is_main:
            print("Set model.transformer to eval mode.")
    elif hasattr(model, 'vit'):
        model.vit.eval()
        if is_main:
            print("Set model.vit to eval mode.")

    if hasattr(model, 'vqgan'):
        model.vqgan.eval()
        if is_main:
            print("Set model.vqgan to eval mode.")

    sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0, w=0.5,
                            sched_pow=2, step=32, randomize=False, top_k=-1)

    # --- 3. 初始化 Metrics（每个 rank 各自持有一份 Inception 模型）---
    if is_main:
        print("正在初始化 Inception Metrics...")
    metrics = MultiInceptionMetrics(
        device=local_rank,
        compute_manifold=False,
        num_classes=1000,
        num_inception_chunks=10,
        manifold_k=3,
        model="inception"
    )

    # --- 4. 每个 rank 各自生成 total_gen / world_size 张图片 ---
    total_gen = 50_000
    batch_size = 32

    per_rank = total_gen // world_size
    if local_rank == 0:
        per_rank += total_gen % world_size
    num_batches = (per_rank + batch_size - 1) // batch_size

    if is_main:
        print(f"开始生成 {total_gen} 张图片并提取特征（{world_size} 个 GPU）...")

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc=f"[rank{local_rank}] Generating", disable=not is_main):
            labels = torch.randint(0, 1000, (batch_size,), device=local_rank)
            gen_images = sampler(trainer=model, nb_sample=batch_size, labels=labels, verbose=False)[0]
            metrics.update(gen_images, image_type="fake")

    dist.barrier()

    # --- 5. 跨 rank 汇聚特征 ---
    if is_main:
        print("正在汇聚各 GPU 的特征...")

    fake_features = torch.cat(metrics.fake_features, dim=0)
    fake_logits   = torch.cat(metrics.fake_logits,   dim=0)

    # gather_and_concat 内部检查 dist.is_initialized()，自动汇聚所有 rank
    fake_features = metrics.gather_and_concat(fake_features)[:total_gen]
    fake_logits   = metrics.gather_and_concat(fake_logits)[:total_gen]

    # --- 6. 只在 rank 0 计算并打印指标 ---
    if is_main:
        print("正在计算生成图片的统计数据...")
        fake_features_mean = fake_features.mean(dim=0)
        fake_features_cov  = metrics.cov(fake_features, fake_features_mean)

        stats_path = "./saved_networks/ImageNet_384_val_stats.pt"
        print(f"正在加载参考统计文件: {stats_path}")
        ref_stats = torch.load(stats_path, map_location=f"cuda:{local_rank}")
        real_features_mean = ref_stats["mu"].to(local_rank)
        real_features_cov  = ref_stats["cov"].to(local_rank)

        print("正在计算最终 FID...")
        fid_score = metrics._compute_fid(
            real_features_mean, real_features_cov,
            fake_features_mean, fake_features_cov
        ).item()

        is_score = metrics.inception_score(fake_logits)

        print("-" * 30)
        print(f"FID Score: {fid_score:.4f}")
        print(f"IS  Score: {is_score:.4f}")
        print("-" * 30)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
