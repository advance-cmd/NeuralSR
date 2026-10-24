"""对齐检查（s193）：用运动矢量把上一帧拉到这一帧（backward warping），量和这一帧差多少。

用法：python tools/check_align.py [段号 ...]        默认测试集的 12 13 14 15；写 docs/align_check.png

三种"用上一帧猜这一帧"：
  不对齐        直接拿上一帧同一个像素
  只用速度      到上一帧的 uv − 速度 处取（双线性）
  速度 + 抖动差  再加上两帧抖动的差（两帧的采样点本来就差了亚像素）
误差在屏幕空间算（× 曝光 → ACES → sRGB，和色调映射 Pass 一样，不加 Bloom），按区域合起来算 PSNR；
上一帧位置出了屏幕的像素不算。warp 用项目根目录的 warp.py（backward_warp）。
"""
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from warp import backward_warp  # noqa: E402

SEQ = os.path.join(ROOT, "data", "seq")
REGIONS = ["其他", "天空", "水面", "镜子"]
METHODS = ["不对齐", "只用速度", "速度 + 抖动差"]


def tonemap(x):
    """和 Tonemap.hlsl 一样：× 2^EV（EV = 1）→ ACES（Narkowicz）→ sRGB 编码，不加 Bloom。x：线性 HDR"""
    x = x.clamp(min=0) * 2.0
    y = ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0, 1)
    return torch.where(y <= 0.0031308, 12.92 * y, 1.055 * y.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def load(split, k, f, name):
    return torch.from_numpy(np.load(os.path.join(SEQ, split, f"{k:02d}_{f:02d}_{name}.npy")).astype(np.float32))


def run(k, split, jit, dev):
    """返回 se[方法][类别] = (平方误差和, 像素数)；类别 = 4 个区域 + "其他里在动的"；以及速度中位数（像素/帧）、出屏比例"""
    se = np.zeros((3, 5, 2))
    speeds, outside = [], []
    prev = None
    for f in range(32):
        cur = {n: load(split, k, f, n) for n in ("color", "velocity", "region")}
        if prev is not None:
            c = cur["color"].permute(2, 0, 1)[None].to(dev)
            p = prev["color"].permute(2, 0, 1)[None].to(dev)
            v = cur["velocity"].permute(2, 0, 1)[None].to(dev)
            jp = torch.tensor([jit[(k, f - 1)]], device=dev)
            jc = torch.tensor([jit[(k, f)]], device=dev)
            zero = torch.zeros_like(jc)
            guesses = [p,
                       backward_warp(p, v, zero, zero)[0],
                       backward_warp(p, v, jp, jc)[0]]
            valid = backward_warp(p, v, jp, jc)[1][0, 0]
            reg = cur["region"].to(dev).long()
            vpx = (v[0] * torch.tensor([1280.0, 720.0], device=dev).view(2, 1, 1)).norm(dim=0)
            moving = (reg == 0) & (vpx > 0.5)
            t = tonemap(c)
            for m, g in enumerate(guesses):
                e = ((tonemap(g) - t) ** 2).mean(1)[0]
                for r in range(4):
                    mask = valid & (reg == r)
                    se[m, r] += [e[mask].sum().item(), mask.sum().item()]
                mask = valid & moving
                se[m, 4] += [e[mask].sum().item(), mask.sum().item()]
            speeds.append(vpx[reg == 0].median().item())
            outside.append(1 - valid.float().mean().item())
        prev = cur
    return se, float(np.median(speeds)), float(np.mean(outside))


def main():
    seqs = [int(a) for a in sys.argv[1:]] or [12, 13, 14, 15]
    info = {int(r["seq"]): r for r in csv.DictReader(open(os.path.join(SEQ, "seqs.csv"), encoding="utf-8"))}
    jit = {(int(r["seq"]), int(r["frame"])): (float(r["jx"]), float(r["jy"]))
           for r in csv.DictReader(open(os.path.join(SEQ, "frames.csv"), encoding="utf-8"))}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    total = np.zeros((3, 5, 2))
    print("屏幕空间 PSNR（dB），每段 31 对相邻帧合起来算；— = 这段里没有这种像素")
    print(f"{'段':4s} {'类型':10s} {'方法':12s} " + " ".join(f"{n:>6s}" for n in REGIONS + ["在动的"]))
    for k in seqs:
        r = info[k]
        se, speed, out = run(k, r["split"], jit, dev)
        total += se
        for m in range(3):
            cells = [f"{10 * np.log10(se[m, c, 1] / se[m, c, 0]):6.2f}" if se[m, c, 1] > 0 and se[m, c, 0] > 0 else "     —" for c in range(5)]
            tag = f"{r['kind']} {speed:.1f}px" if m == 0 else ""
            print(f"{k:<4d} {tag:10s} {METHODS[m]:12s} " + " ".join(cells))
        print(f"     出屏 {out * 100:.1f}%")
    print("合计")
    for m in range(3):
        cells = [f"{10 * np.log10(total[m, c, 1] / total[m, c, 0]):6.2f}" for c in range(5)]
        print(f"{'':15s} {METHODS[m]:12s} " + " ".join(cells))


if __name__ == "__main__":
    main()
