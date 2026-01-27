import time
import numpy as np

import torch
from Utils.utils import load_args_from_file
from Utils.viz import show_images_grid
from huggingface_hub import hf_hub_download

from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler


config_path = "Config/base_cls2img.yaml"        # Path to your config file
args = load_args_from_file(config_path)

# Update arguments
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select Network (Large 384 is the best, but the slowest)
args.vit_size = "large"  # "tiny", "small", "base", "large"
args.img_size = 384  # 256 or 384
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"

# Download the MaskGIT
hf_hub_download(repo_id="llvictorll/Halton-Maskgit",
                filename=f"ImageNet_{args.img_size}_{args.vit_size}.pth",
                local_dir="./saved_networks")

# Download VQGAN
hf_hub_download(repo_id="FoundationVision/LlamaGen",
                filename="vq_ds16_c2i.pt",
                local_dir="./saved_networks")


# Initialisation of the model
model = MaskGIT(args)

# select your scheduler (Halton is better)
sampler = HaltonSampler(sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0, w=2,
                        sched_pow=2, step=32, randomize=False, top_k=-1)

# [goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear]
labels = torch.LongTensor([1, 7, 282, 604, 724, 179, 681, 850]).to(args.device)
labels = labels.repeat(4)  # Repeat labels to increase batch size for latency test



def measure_inference_latency(model, sampler, labels, batch_size, num_runs=10):
    print(f"--- 开始测量 (设备: {args.device}) ---")
    
    # 1. 预热 (Warm-up)
    # GPU 在首次运行时需要初始化显存和编译计算图，耗时较长。
    # 我们先运行一次不计入统计，以保证后续测量准确。
    print("正在预热模型...")
    with torch.no_grad():
        _ = sampler(trainer=model, nb_sample=batch_size, labels=labels, verbose=False)
    
    # 确保预热完成
    if args.device.type == 'cuda':
        torch.cuda.synchronize()
        
    print(f"预热完成。开始运行 {num_runs} 次测试...")

    # 2. 测量循环
    timings = []
    
    # 使用 CUDA Event 进行高精度计时 (如果是 GPU)
    if args.device.type == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        for i in range(num_runs):
            starter.record()
            with torch.no_grad():
                _ = sampler(trainer=model, nb_sample=batch_size, labels=labels, verbose=False)
            ender.record()
            
            # 等待 GPU 完成当前流的所有操作
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000.0 # 转换为秒
            timings.append(curr_time)
            print(f"Run {i+1}/{num_runs}: {curr_time:.4f} 秒")
            
    else:
        # CPU 计时
        for i in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = sampler(trainer=model, nb_sample=batch_size, labels=labels, verbose=False)
            end_time = time.time()
            
            curr_time = end_time - start_time
            timings.append(curr_time)
            print(f"Run {i+1}/{num_runs}: {curr_time:.4f} 秒")

    # 3. 统计结果
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    fps = batch_size / avg_time

    print("\n--- 性能报告 ---")
    print(f"Batch Size: {batch_size}")
    print(f"平均推理延迟 (Batch): {avg_time:.4f} 秒 ± {std_time:.4f}")
    print(f"平均每张图延迟: {avg_time / batch_size:.4f} 秒")
    print(f"吞吐量 (Throughput): {fps:.2f} images/sec")
    
    return avg_time

# --- 执行测量 ---
# 确保 labels 的数量与 nb_sample 一致，或者根据 batch_size 截断/重复 labels
batch_size = 32
# 如果 labels 数量少于 batch_size，需要重复填充；如果多则截取。这里假设你是为了测试这8个 label
test_labels = labels[:batch_size] 

measure_inference_latency(model, sampler, test_labels, batch_size=batch_size, num_runs=5)