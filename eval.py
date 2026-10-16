"""第三天：评估。双三次、模型 A、模型 B 在测试集 40 张整图上的 PSNR、SSIM、分区域 PSNR，以及推理耗时。

用法：python eval.py
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from dataset import DATA, SRDataset
from model import SRNet

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
REGIONS = ["其他", "天空", "水面", "镜子"]  # region.npy 里的 0、1、2、3（tools/make_masks.py 生成）


def load(name, dev):
    """按名字建网络，再把训练好的权重读进来"""
    model = SRNet(name=="B").to(dev)  # 建一个和训练时一样的网络，搬到 GPU
    model.load_state_dict(torch.load(os.path.join(RUNS,name + ".pth")))  # 读 runs/{name}.pth
    return model.eval()


def upscale(method, lr, gb):
    """method 是 "双三次" 或者一个网络"""
    if method == "双三次":
        return F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
    return method(lr, gb)


def quality(methods, dev):
    """每种方法：整图 PSNR、SSIM（40 张各算再平均）；分区域 PSNR（所有图里这个区域的像素合起来算）"""
    ds = SRDataset("test", patch=None)
    psnrs = {n: [] for n in methods}
    ssims = {n: [] for n in methods}
    err_sum = {n: np.zeros(4) for n in methods}  # 每个区域：平方误差加起来
    count = np.zeros(4)  # 每个区域：一共多少像素
    with torch.no_grad():
        for k, (lr, gb, hr) in enumerate(DataLoader(ds, batch_size=1)):
            lr, gb, hr = lr.to(dev), gb.to(dev), hr.to(dev)
            region = np.load(os.path.join(DATA, "test", f"{ds.ids[k]:04d}_region.npy"))
            region = torch.from_numpy(region).to(dev)  # (H, W)，每个像素 0～3
            for r in range(4):
                count[r] += (region == r).sum().item()  # 这张图里区域 r 有多少像素
            for name, method in methods.items():
                sr = upscale(method, lr, gb).clamp(0, 1)
                se = ((sr-hr)**2).mean(1)[0]  # 每个像素的平方误差，RGB 三个通道取平均：(H, W)
                psnrs[name].append(10*torch.log10(1/se.mean()).item())  # 整图 PSNR
                ssims[name].append(ssim(sr, hr, data_range=1.0).item())
                for r in range(4):
                    err_sum[name][r] += se[region==r].sum().item()  # 区域 r 里所有像素的平方误差之和
    print("| 方法 | PSNR | SSIM | " + " | ".join(REGIONS) + " |")
    print("|---|---|---|" + "---|" * 4)
    for name in methods:
        region_psnr = 10*np.log10(count / err_sum[name])  # 每个区域：先合起来算平均平方误差，再换成 PSNR
        cells = [f"{np.mean(psnrs[name]):.2f}", f"{np.mean(ssims[name]):.4f}"] + [f"{v:.2f}" for v in region_psnr]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("各区域占全部像素：" + "、".join(f"{n} {c / count.sum() * 100:.1f}%" for n, c in zip(REGIONS, count)))


def timing(methods, dev):
    """720p → 2K 放大一次要多少毫秒（半精度 fp16，只算 GPU 计算，不含读数据）"""
    lr = torch.rand(1, 3, 720, 1280, device=dev).half()
    gb = torch.rand(1, 9, 1440, 2560, device=dev).half()
    for name, method in methods.items():
        if method != "双三次":
            method = method.half()  # 网络的权重也换成半精度
        with torch.no_grad():
            for _ in range(10):  # 预热：头几次要挑算法、分配显存，不算在内
                upscale(method, lr, gb)
            start, end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)  # 两个能记时间的 CUDA 事件
            start.record()
            for _ in range(50):
                upscale(method, lr, gb)
            end.record()
            torch.cuda.synchronize()   # CPU 在这里等 GPU 把上面排的活全部干完
        params = 0 if method == "双三次" else sum(p.numel() for p in method.parameters())
        print(f"{name}：{start.elapsed_time(end) / 50:.2f} ms，参数 {params}")


if __name__ == "__main__":
    dev = torch.device("cuda")
    methods = {"双三次": "双三次", "模型 A": load("A", dev), "模型 B": load("B", dev)}
    quality(methods, dev)
    timing(methods, dev)
