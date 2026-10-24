"""检查项目根目录的 warp.py（backward_warp）。

用法：python tools/check_warp.py
  1. 小测试：速度和抖动都是 0 → 原样；整像素平移；只有抖动差
  2. 真数据：测试集第 13（转头）、14（横移）段，其他区域的屏幕空间 PSNR 要和参考答案一样（27.25 dB）
     对不上时，按沙盒里量过的常见错法猜一下是哪一种
"""
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from warp import backward_warp  # noqa: E402
from check_align import SEQ, load, tonemap  # noqa: E402

REF = 27.25
BUGS = [(16.21, "uv + 速度：方向反了（速度 = 这一帧 − 上一帧，上一帧要减）"),
        (21.59, "抖动差反了：要的是 上一帧抖动 − 这一帧抖动"),
        (10.22, "抖动差没除以宽高（像素直接加到了 uv 上）"),
        (26.14, "align_corners=True（−1 应该是左边那条边，不是第一个像素的中心）"),
        (10.21, "没换成 −1..1（少了 ×2 − 1）"),
        (23.67, "像素中心没加 0.5"),
        (10.75, "u、v 顺序反了（通道 0 要放横向的 x）"),
        (24.00, "没加抖动差（只用速度）"),
        (17.30, "等于没对齐（速度没用上）")]
ok = True


def check(name, cond, hint=""):
    global ok
    print(("  ✅ " if cond else "  ❌ ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)


def main():
    print("1. 小测试")
    torch.manual_seed(0)
    img = torch.rand(1, 3, 6, 8)
    zero2 = torch.zeros(1, 2)
    out, valid = backward_warp(img, torch.zeros(1, 2, 6, 8), zero2, zero2)
    check("速度、抖动都是 0 → 原样", (out - img).abs().max() < 1e-5, "空1（像素中心）或空4（×2 − 1）或空5")
    check("valid 全是 True", bool(valid.all()), "空1：uv 应该在 0..1 里；画面几项都过、只有 valid 错的话，是空4 用了 += / *= 原地改（连 prev_uv 一起改了）")
    vel = torch.zeros(1, 2, 6, 8)
    vel[:, 0] = 2 / 8  # 这一帧的东西在上一帧的左边 2 个像素
    out, valid = backward_warp(img, vel, zero2, zero2)
    check("速度 = 向右 2 像素 → 第 x 列取上一帧第 x − 2 列", (out[..., 2:] - img[..., :-2]).abs().max() < 1e-5, "空3 的正负号，或空1 的 u、v 顺序")
    check("最左 2 列的上一帧在屏幕外 → valid = False", bool((~valid[..., :2]).all() and valid[..., 2:].all()), "空3，或空4 原地改了 prev_uv")
    out, _ = backward_warp(img, torch.zeros(1, 2, 6, 8), torch.tensor([[0.0, 1.0]]), zero2)
    check("上一帧抖动 (0, 1) 像素 → 第 y 行取上一帧第 y + 1 行", (out[:, :, :-1] - img[:, :, 1:]).abs().max() < 1e-5, "空2：抖动差的方向，或没除以宽高")
    vel = torch.zeros(1, 2, 6, 8)
    vel[:, 0] = -0.5 / 8  # 上一帧在右边半个像素
    out, _ = backward_warp(img, vel, zero2, zero2)
    check("速度 = 向左半个像素 → 第 x、x + 1 列各一半（双线性）", (out[..., :-1] - (img[..., :-1] + img[..., 1:]) / 2).abs().max() < 1e-5,
          "前面几项都过了的话，是空5 的 mode")

    print("2. 真数据（第 13、14 段，31 × 2 对相邻帧）")
    jit = {(int(r["seq"]), int(r["frame"])): (float(r["jx"]), float(r["jy"]))
           for r in csv.DictReader(open(os.path.join(SEQ, "frames.csv"), encoding="utf-8"))}
    s, cnt = 0.0, 0
    with torch.no_grad():
        for k in (13, 14):
            for f in range(1, 32):
                c, p, v, reg = (load("test", k, f, "color"), load("test", k, f - 1, "color"),
                                load("test", k, f, "velocity"), load("test", k, f, "region"))
                g, _ = backward_warp(p.permute(2, 0, 1)[None], v.permute(2, 0, 1)[None],
                                     torch.tensor([jit[(k, f - 1)]]), torch.tensor([jit[(k, f)]]))
                e = ((tonemap(g[0].permute(1, 2, 0)) - tonemap(c)) ** 2).mean(-1)
                m = reg == 0
                s += e[m].sum().item()
                cnt += m.sum().item()
    psnr = 10 * np.log10(cnt / s)
    near = min(BUGS, key=lambda b: abs(b[0] - psnr))
    check(f"其他区域 PSNR {psnr:.2f} dB（参考答案 {REF:.2f}）", abs(psnr - REF) < 0.05,
          f"最像这种错法：{near[1]}（沙盒里 {near[0]:.2f}）")
    print("全部通过" if ok else "还有没过的")


if __name__ == "__main__":
    main()
