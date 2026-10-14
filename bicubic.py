"""第一天：双三次放大的基线 PSNR（测试集 40 张整图）。"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SRDataset


def psnr(sr, hr):
    """两张 0..1 的图 → PSNR（dB）。PSNR = 10·log10(MAX² / MSE)，MAX = 像素可能的最大值"""
    mse = 空9
    return 空10


if __name__ == "__main__":
    dev = torch.device("cuda")
    loader = DataLoader(SRDataset("test", patch=None), batch_size=1)  # 整图，按编号顺序
    scores = []
    with torch.no_grad():
        for lr, gb, hr in loader:
            lr, hr = 空11
            sr = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=空12)
            sr = 空13
            scores.append(psnr(sr, hr).item())
    s = torch.tensor(scores)
    # 测试集前 33 张是随机机位，后 7 张是 demo 机位（p1–p7）
    print(f"双三次：全部 {s.mean():.2f} dB | 随机机位 {s[:33].mean():.2f} | demo 机位 {s[33:].mean():.2f} | 最差 {s.min():.2f}  最好 {s.max():.2f}")
