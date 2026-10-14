"""确认 PyTorch 能用上显卡：版本、CUDA、显卡架构，再跑一次 fp16 卷积。"""
import torch

print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
assert torch.cuda.is_available(), "CUDA 不可用：检查驱动，或者装的是不是 CPU 版"
dev = torch.device("cuda")
cap = torch.cuda.get_device_capability()
print("GPU", torch.cuda.get_device_name(), f"| sm_{cap[0]}{cap[1]}", "| 显存 %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
arch = f"sm_{cap[0]}{cap[1]}"
assert arch in torch.cuda.get_arch_list(), f"这个 PyTorch 没编 {arch} 的内核：{torch.cuda.get_arch_list()}"

# 一层 3×3 卷积：720p、64 通道，fp16
x = torch.randn(1, 64, 720, 1280, device=dev, dtype=torch.half)
conv = torch.nn.Conv2d(64, 64, 3, padding=1).to(dev).half()
with torch.no_grad():
    for _ in range(5):
        conv(x)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        y = conv(x)
    end.record()
    torch.cuda.synchronize()
ms = start.elapsed_time(end) / 20
flops = 2 * 64 * 64 * 9 * 720 * 1280
print(f"3x3 conv 64→64 @720p fp16: {ms:.3f} ms ({flops / ms / 1e9:.1f} TFLOPS)")
print("OK")
