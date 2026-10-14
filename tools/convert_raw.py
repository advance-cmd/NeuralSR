"""把 DX12 程序读回的原始 .npy 转成数据集（PNG + 深度 .npy），按 cams.csv 分到 train / test。

用法：python tools/convert_raw.py <raw_hr 目录> <raw_lr 目录>

raw 目录里每个机位 4 个文件（{id:04d}_*.npy，由 LearnDX12 的导数据模式写出）：
  color  (H, W, 4) uint8    后台缓冲 = 屏幕上的成品画面（色调映射之后，sRGB 编码）
  gb0    (H, W, 4) uint8    G-Buffer0：rgb 基础色（sRGB 编码），a 金属度
  gb1    (H, W, 4) float16  G-Buffer1：xyz 世界空间法线，w 粗糙度
  depth  (H, W)    uint32   深度缓冲（光照 Pass 刚结束时读的，只有不透明物体），低 24 位是 D24 深度
"""
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
NEAR, FAR = 1.0, 1000.0  # main.cpp：mCamera.SetLens(0.25π, aspect, 1, 1000)


def view_depth(raw):
    """D24 深度（0..1，非线性）→ 观察空间深度（世界单位）；天空 = 0。"""
    d = (raw & 0xFFFFFF).astype(np.float64) / 16777215.0
    z = FAR * NEAR / (FAR - d * (FAR - NEAR))
    z[d >= 1.0] = 0.0
    return z.astype(np.float16)


def convert(job):
    i, split, raw_hr, raw_lr = job
    out = os.path.join(DATA, split)
    src = lambda d, n: np.load(os.path.join(d, f"{i:04d}_{n}.npy"))
    dst = lambda n: os.path.join(out, f"{i:04d}_{n}")

    depth = view_depth(src(raw_hr, "depth"))
    sky = depth == 0
    gb0, gb1 = src(raw_hr, "gb0"), src(raw_hr, "gb1").astype(np.float32)

    albedo = gb0[..., :3].copy()
    albedo[sky] = 0
    normal = np.clip(np.rint((gb1[..., :3] * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    normal[sky] = 0
    mr = np.zeros_like(albedo)
    mr[..., 0] = gb0[..., 3]                                              # 金属度
    mr[..., 1] = np.clip(np.rint(gb1[..., 3] * 255), 0, 255).astype(np.uint8)  # 粗糙度
    mr[sky] = 0

    Image.fromarray(src(raw_hr, "color")[..., :3]).save(dst("hr.png"))
    Image.fromarray(src(raw_lr, "color")[..., :3]).save(dst("lr.png"))
    Image.fromarray(albedo).save(dst("albedo.png"))
    Image.fromarray(normal).save(dst("normal.png"))
    Image.fromarray(mr).save(dst("mr.png"))
    np.save(dst("depth.npy"), depth)
    return i


def main():
    raw_hr, raw_lr = sys.argv[1], sys.argv[2]
    with open(os.path.join(DATA, "cams.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for split in ("train", "test"):
        os.makedirs(os.path.join(DATA, split), exist_ok=True)
    jobs = [(int(r["id"]), r["split"], raw_hr, raw_lr) for r in rows]
    with ProcessPoolExecutor(max_workers=8) as ex:
        for n, _ in enumerate(ex.map(convert, jobs), 1):
            if n % 40 == 0 or n == len(jobs):
                print(f"{n} / {len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
