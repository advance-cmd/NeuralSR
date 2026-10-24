"""对齐检查的配图（s193）：docs/align_check.png —— 同一块局部：这一帧，以及三种"用上一帧猜这一帧"的误差。

用法：python tools/make_align_fig.py [段号 帧号]        默认第 14 段（横移）第 16 帧
warp 用项目根目录的 warp.py（backward_warp）；误差在屏幕空间算（和 check_align.py 一样）。
"""
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_align import SEQ, load, tonemap, backward_warp  # noqa: E402

CROP = 200  # 720p 像素，放大 2 倍显示


def main():
    k, f = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (14, 16)
    jit = {(int(r["seq"]), int(r["frame"])): (float(r["jx"]), float(r["jy"]))
           for r in csv.DictReader(open(os.path.join(SEQ, "frames.csv"), encoding="utf-8"))}
    split = {int(r["seq"]): r["split"] for r in csv.DictReader(open(os.path.join(SEQ, "seqs.csv"), encoding="utf-8"))}[k]
    c, p, v, reg = (load(split, k, f, "color"), load(split, k, f - 1, "color"),
                    load(split, k, f, "velocity"), load(split, k, f, "region"))
    P, V = p.permute(2, 0, 1)[None], v.permute(2, 0, 1)[None]
    jp, jc = torch.tensor([jit[(k, f - 1)]]), torch.tensor([jit[(k, f)]])
    zero = torch.zeros_like(jc)
    guesses = [("不对齐", P), ("只用速度", backward_warp(P, V, zero, zero)[0]), ("速度 + 抖动差", backward_warp(P, V, jp, jc)[0])]
    t = tonemap(c)
    errs = [((tonemap(g[0].permute(1, 2, 0)) - t) ** 2).mean(-1).sqrt() for _, g in guesses]
    # 局部：误差（不对齐）最大、又以"其他"区域为主的一块
    e0 = errs[0].numpy() * (reg.numpy() == 0)
    ii = torch.nn.functional.avg_pool2d(torch.tensor(e0)[None, None], CROP, 20)[0, 0]
    y0, x0 = np.unravel_index(int(ii.argmax()), ii.shape)
    y0, x0 = y0 * 20, x0 * 20
    sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 19)
    tiles = [("这一帧（720p，带抖动）", (t[sl].numpy() * 255).astype(np.uint8))]
    for (name, _), e in zip(guesses, errs):
        m = reg.numpy() == 0
        psnr = 10 * np.log10(1 / (e.numpy()[m] ** 2).mean())
        tiles.append((f"{name}：误差 ×4（其他 {psnr:.1f} dB）", (np.clip(e[sl].numpy() * 4, 0, 1) * 255).astype(np.uint8)))
    out = Image.new("RGB", (4 * CROP * 2 + 30, CROP * 2 + 40), (255, 255, 255))
    d = ImageDraw.Draw(out)
    for i, (name, a) in enumerate(tiles):
        im = Image.fromarray(a if a.ndim == 3 else np.stack([a] * 3, -1)).resize((CROP * 2, CROP * 2), Image.NEAREST)
        out.paste(im, (i * (CROP * 2 + 10), 40))
        d.text((i * (CROP * 2 + 10) + 4, 6), name, font=font, fill=(0, 0, 0))
    path = os.path.join(ROOT, "docs", "align_check.png")
    out.save(path)
    print(f"seq {k} frame {f}, 局部 x {x0}..{x0 + CROP} y {y0}..{y0 + CROP} → {path}")


if __name__ == "__main__":
    main()
