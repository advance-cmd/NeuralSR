"""检查项目根目录的 hdr.py（encode / decode / tonemap）。

用法：python tools/check_hdr.py
  1. 小测试：曝光、log 空间来回一趟、色调映射和 Tonemap.hlsl 一样、反向传播没有 NaN
  2. 真数据：测试集 4 段各取第 0 帧，双三次放大后的屏幕空间 PSNR 要和参考答案一样
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import hdr  # noqa: E402
from hdr import decode, encode, tonemap  # noqa: E402

SEQ = os.path.join(ROOT, "data", "seq", "test")
REF = 32.14
ok = True


def check(name, cond, hint=""):
    global ok
    print(("  ✅ " if cond else "  ❌ ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)


def aces(x):
    """Tonemap.hlsl 的 ACESFitted，numpy 版"""
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def srgb(c):
    """Tonemap.hlsl 的 LinearToSRGB，numpy 版"""
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.maximum(c, 1e-8) ** (1 / 2.4) - 0.055)


def main():
    print("1. 小测试")
    check(f"曝光 EXPOSURE = {hdr.EXPOSURE}", abs(hdr.EXPOSURE - 2.0) < 1e-6, "空1：mExposureEV = 1，2 的 1 次方")

    x = torch.tensor([0.0, 0.039, 1.0, 100.0])
    y = encode(x)
    check("encode：0 → 0，1 → log 2，100 → log 101", torch.allclose(y, torch.log(1 + x)), "空2：log(1 + x)")
    print("     线性 " + "  ".join(f"{v:7.3f}" for v in x.tolist()) + "\n     log  " + "  ".join(f"{v:7.3f}" for v in y.tolist()))
    check("encode：负数先夹成 0", bool((encode(torch.tensor([-0.5, -2.0])) == 0).all()), "空2：log(1 + x) 在 x ≤ −1 时是 −inf / NaN，要先 clamp")

    xs = torch.logspace(-4, 2.5, 50)
    check("decode(log(1 + x)) == x（0.0001 到 316）", torch.allclose(decode(torch.log1p(xs)), xs, rtol=1e-5, atol=1e-7), "空3：encode 的反函数")
    check("decode：负的 y → 0", bool((decode(torch.tensor([-0.3, -5.0])) == 0).all()), "空3：e^y − 1 在 y < 0 时是负数，要夹成 0")

    t = np.array([-1.0, 0.0, 0.001, 0.01, 0.039, 0.1, 0.5, 1.0, 10.0, 1000.0], dtype=np.float32)
    mine = tonemap(torch.from_numpy(t)).numpy()
    same = lambda a: np.abs(mine - a).max() < 1e-5
    e = float(hdr.EXPOSURE)
    xe = np.maximum(t, 0) * e  # 用你自己的曝光，空1 错了也能接着查后面的空
    srgb_twice = lambda c: np.where(np.maximum(c, 1e-8) ** (1 / 2.4) <= 0.0031308, 12.92 * np.maximum(c, 1e-8) ** (1 / 2.4),
                                    1.055 * np.maximum(c, 1e-8) ** (1 / 2.4 / 2.4) - 0.055)
    good = same(srgb(aces(np.maximum(t, 0) * 2.0)))
    if good:
        hint = ""
    elif e != 2.0 and same(srgb(aces(xe))):
        hint = "只差空1 的曝光，其余都对"
    elif same(srgb(aces(np.maximum(t, 0)))):
        hint = "空4：忘了乘曝光"
    elif abs(mine[0] - mine[1]) > 1e-6:
        hint = "空4：负数要先夹成 0（输入 −1 和 0 结果不一样）"
    elif same(aces(xe)):
        hint = "空6：还没做 sRGB 编码"
    elif same(srgb(np.clip(xe, 0, 1))):
        hint = "空5：还没过 ACES 曲线"
    elif same(srgb_twice(aces(xe))):
        hint = "空6：y_pow 已经是开过 1/2.4 次方的 y 了。判断条件和 12.92 那一段要用 y，亮的那一段直接用 y_pow，不要再开一次方"
    else:
        hint = "对照 Tonemap.hlsl 的 ACESFitted、LinearToSRGB 一行一行看"
    check("tonemap 和 Tonemap.hlsl 一样（× 2 → ACES → sRGB）", good, hint)
    print("     线性   " + "  ".join(f"{v:6.3f}" for v in t[1:].tolist()) + "\n     屏幕上 " + "  ".join(f"{v:6.3f}" for v in mine[1:].tolist()))

    g = torch.tensor([0.0, 0.0005, 0.3, 5.0], requires_grad=True)
    tonemap(g).sum().backward()
    check("tonemap 反向传播没有 NaN", bool(torch.isfinite(g.grad).all()), "空6：亮的那一段用已经写好的 y_pow，不要自己再写 y ** (1 / 2.4)")

    print("2. 真数据（测试集 12–15 段的第 0 帧，双三次放大，屏幕空间 PSNR）")
    se, n = 0.0, 0
    for k in (12, 13, 14, 15):
        load = lambda name: torch.from_numpy(np.load(os.path.join(SEQ, f"{k:02d}_00_{name}.npy")).astype(np.float32)).permute(2, 0, 1)[None]
        lr, gt = load("color"), load("gt")
        sr = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(min=0)
        with torch.no_grad():
            se += ((tonemap(sr) - tonemap(gt)) ** 2).sum().item()
            n += gt.numel()
        if k == 12:
            print(f"     第 12 段第 0 帧 GT：线性最大 {gt.max().item():.1f} → log 空间最大 {encode(gt).max().item():.2f}")
    psnr = 10 * math.log10(n / se)
    check(f"屏幕空间 PSNR {psnr:.2f} dB（参考答案 {REF:.2f}）", abs(psnr - REF) < 0.01, "小测试都过了还对不上的话，把输出贴给我")
    print("全部通过" if ok else "还有没过的")


if __name__ == "__main__":
    main()
