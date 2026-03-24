import torch
from torch.profiler import profile, record_function, ProfilerActivity
from Utils.utils import load_args_from_file
from huggingface_hub import hf_hub_download

from Trainer.cls_trainer import MaskGIT
from Sampler.halton_sampler import HaltonSampler


config_path = "Config/base_cls2img.yaml"
args = load_args_from_file(config_path)

# Update arguments
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select Network
args.vit_size = "large"
args.img_size = 384
args.compile = False
args.dtype = "float32"
args.resume = True
args.vit_folder = f"./saved_networks/ImageNet_{args.img_size}_{args.vit_size}.pth"

# Download checkpoints
hf_hub_download(
    repo_id="llvictorll/Halton-Maskgit",
    filename=f"ImageNet_{args.img_size}_{args.vit_size}.pth",
    local_dir="./saved_networks"
)

hf_hub_download(
    repo_id="FoundationVision/LlamaGen",
    filename="vq_ds16_c2i.pt",
    local_dir="./saved_networks"
)

# Init model
model = MaskGIT(args)


# Init sampler
sampler = HaltonSampler(
    sm_temp_min=1, sm_temp_max=1.2, temp_pow=1, temp_warmup=0,
    w=2, sched_pow=2, step=32, randomize=False, top_k=-1
)

# Labels
labels = torch.LongTensor([1, 7, 282, 604, 724, 179, 681, 850]).to(args.device)


def run_one_generation():
    with torch.no_grad():
        out = sampler(trainer=model, nb_sample=8, labels=labels, verbose=False)
    return out


# -----------------------------
# 1. Warmup
# -----------------------------
print("Warmup...")
for _ in range(2):
    _ = run_one_generation()

if torch.cuda.is_available():
    torch.cuda.synchronize()

print("Warmup done.\n")


# -----------------------------
# 2. Profile one run
# -----------------------------
activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
    with_stack=False
) as prof:
    with record_function("full_generation"):
        _ = run_one_generation()

if torch.cuda.is_available():
    torch.cuda.synchronize()


# -----------------------------
# 3. Print profiler result
# -----------------------------
sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"

print(prof.key_averages().table(
    sort_by=sort_key,
    row_limit=50
))