"""s193：单帧 HDR 基线的评估。双三次、2K 直接画（不抗锯齿）、模型 A、模型 B 在测试集 4 段 × 32 帧 = 128 帧整图上：
  屏幕空间 PSNR / SSIM：两边都过 tonemap()（× 曝光 → ACES → sRGB，和色调映射 Pass 一样，不加 Bloom），每帧各算再平均
  分区域 PSNR：屏幕空间，所有帧里这个区域的像素合起来算
  log 空间 L1：|encode(结果) − encode(GT)| 的平均，就是训练时的损失
"2K 直接画（不抗锯齿）"不是超分，是同一时刻按 2K 不抖地画的一帧（noaa.npy）：看普通的 2K 渲染离 GT 有多远。

用法：python eval_hdr.py（约 2 分钟）。写完先跑 python tools/check_eval_hdr.py。
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from dataset import SEQ, HDRDataset, to_tensor_Hdr
from hdr import decode, encode, tonemap
from model import SRNet

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
REGIONS = ["其他", "天空", "水面", "镜子"]  # hrregion.npy 里的 0、1、2、3


def load(name, dev):
    """按名字建网络，再把训练好的权重读进来"""
    model = SRNet(name == "B").to(dev)
    # 空1：train_hdr.py 存的权重叫什么名字？（现在读的是 Day 2 那两个 8 位 sRGB 训出来的）
    model.load_state_dict(torch.load(os.path.join(RUNS,"hdr_" + name + ".pth")))
    return model.eval()


def upscale(method, lr, gb, base):
    """method 是 "双三次"、"2K 直接画" 或一个网络；返回 (1, 3, 1440, 2560) 的线性 HDR。base 是这一帧文件名的前缀，比如 ".../seq/test/12_07_" """
    if method == "双三次":
        return F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
    if method == "2K 直接画":
        # 空2：不是放大出来的，是同一时刻按 2K 画的一帧，存在这一帧的 noaa.npy 里。
        #      读进来 → (3, H, W) → 前面加一维 → 放到和 lr 同一个设备上（现在是拿双三次凑数）
        res = to_tensor_Hdr(np.load(base + "noaa.npy"))[None].to(lr.device)
        return res
    # 空3：网络在 log 空间里，和 train_hdr.py 的 evaluate 一样
    return decode(method(encode(lr), gb))


def quality(methods, dev, every=1):
    """every：每几帧取一帧（检查脚本用 every=16 省时间）。返回 {方法: (PSNR, SSIM, [4 个区域的 PSNR], log 空间 L1)}"""
    ds = HDRDataset("test", patch=None, every=every)
    psnrs = {n: [] for n in methods}
    ssims = {n: [] for n in methods}
    l1s = {n: [] for n in methods}
    err_sum = {n: np.zeros(4) for n in methods}  # 每个区域：平方误差加起来
    count = np.zeros(4)  # 每个区域：一共多少像素
    with torch.no_grad():
        for k, (lr, gb, hr) in enumerate(DataLoader(ds, batch_size=1, num_workers=4)):
            lr, gb, hr = lr.to(dev), gb.to(dev), hr.to(dev)
            base = os.path.join(SEQ, "test", ds.ids[k] + "_")
            region = torch.from_numpy(np.load(base + "hrregion.npy")).to(dev)  # 2K 的区域掩码 (H, W)，每个像素 0～3
            for r in range(4):
                count[r] += (region == r).sum().item()
            for name, method in methods.items():
                sr = upscale(method, lr, gb, base).clamp(min=0)  # 线性 HDR 没有上限，只夹掉负数
                # 空4：屏幕空间：sr、hr 都换成屏幕上的值，下面的平方误差和 SSIM 都在屏幕空间算
                t, t_hr =tonemap(sr),tonemap(hr)
                se = ((t - t_hr) ** 2).mean(1)[0]  # 每个像素的平方误差，RGB 取平均：(H, W)
                psnrs[name].append(10 * torch.log10(1 / se.mean()).item())
                ssims[name].append(ssim(t, t_hr, data_range=1.0).item())
                # 空5：log 空间的 L1（训练时的损失），这一帧的平均
                l1s[name].append((encode(sr) - encode(hr)).abs().mean().item())
                for r in range(4):
                    err_sum[name][r] += se[region == r].sum().item()
    rows = {}
    print(f"测试集 {len(ds)} 帧，屏幕空间（× 2^EV → ACES → sRGB，不加 Bloom）")
    print("| 方法 | PSNR | SSIM | " + " | ".join(REGIONS) + " | log 空间 L1 |")
    print("|---|---|---|" + "---|" * 4 + "---|")
    for name in methods:
        region_psnr = 10 * np.log10(count / np.maximum(err_sum[name], 1e-12))
        rows[name] = (np.mean(psnrs[name]), np.mean(ssims[name]), list(region_psnr), np.mean(l1s[name]))
        cells = [f"{rows[name][0]:.2f}", f"{rows[name][1]:.4f}"] + [f"{v:.2f}" if count[i] else "—" for i, v in enumerate(region_psnr)]
        print(f"| {name} | " + " | ".join(cells) + f" | {rows[name][3]:.4f} |")
    print("各区域占全部像素：" + "、".join(f"{n} {c / count.sum() * 100:.1f}%" for n, c in zip(REGIONS, count)))
    seqs = sorted({i[:2] for i in ds.ids})
    print("按段（屏幕空间 PSNR，方法顺序同上）：" + "；".join(
        f"{s} " + " / ".join(f"{np.mean([p for i, p in zip(ds.ids, psnrs[n]) if i[:2] == s]):.2f}" for n in methods) for s in seqs))
    return rows


if __name__ == "__main__":
    dev = torch.device("cuda")
    methods = {"双三次": "双三次", "2K 直接画": "2K 直接画", "模型 A": load("A", dev), "模型 B": load("B", dev)}
    quality(methods, dev)
