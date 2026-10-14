"""把每个样本的 PNG 再存一份不压缩的 .npy，训练时用 np.load(..., mmap_mode="r") 只读裁出来的那一块。

用法：python tools/pack_npy.py
写到 PNG 旁边：{split}/{id:04d}_lr.npy / _hr.npy / _albedo.npy / _normal.npy（形状和 PNG 一样，uint8）、
_mr.npy（只留两个通道：金属度、粗糙度）。深度本来就是 .npy，不用转。

为什么：解一张 2K 的 PNG 要 20–70 ms，一个样本 5 张约 200 ms；
而训练每次只用其中一小块，从不压缩的 .npy 里按行切只要零点几毫秒。
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def pack(base):
    for name in ("lr", "hr", "albedo", "normal", "mr"):
        a = np.asarray(Image.open(base + name + ".png"))
        if name == "mr":
            a = np.ascontiguousarray(a[..., :2])
        np.save(base + name + ".npy", a)
    return base


def main():
    bases = []
    for split in ("train", "test"):
        d = os.path.join(DATA, split)
        bases += [os.path.join(d, f[:5]) for f in sorted(os.listdir(d)) if f.endswith("_hr.png")]
    with ProcessPoolExecutor(max_workers=8) as ex:
        for n, _ in enumerate(ex.map(pack, bases), 1):
            if n % 40 == 0 or n == len(bases):
                print(f"{n} / {len(bases)}", flush=True)


if __name__ == "__main__":
    main()
