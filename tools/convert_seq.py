"""把导出程序写的一段序列（原始 .npy）转成多帧数据集的格式，放进 data/seq/{split}/，再删掉原始文件。

用法：python tools/convert_seq.py <seq 号> <lr 原始目录> <gt 原始目录>

原始文件（{seq:02d}_{frame:02d}_*.npy，tools/export/patch_seq.py 写出）：
  lr：color (720,1280,3) f16 | velocity (720,1280,2) f16 | depthraw (720,1280) u4 | jitter_lr.csv
  gt：gt (1440,2560,3) f16 | gb0 (…,4) u1 | gb1 (…,4) f16 | depthraw u4 | noaa (…,3) f16（只有测试集）
数据集文件见 data/README.md 的"多帧数据集"一节。
"""
import csv
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_masks as mm  # noqa: E402  区域掩码（按机位 + 深度反推水面、镜子、天空）

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = os.path.join(ROOT, "data", "seq")
NEAR, FAR = 1.0, 1000.0  # main.cpp：mCamera.SetLens(0.25π, aspect, 1, 1000)


def view_depth(raw):
    """D24 深度（0..1，非线性）→ 观察空间深度（世界单位）；天空 = 0。"""
    d = (raw & 0xFFFFFF).astype(np.float64) / 16777215.0
    z = FAR * NEAR / (FAR - d * (FAR - NEAR))
    z[d >= 1.0] = 0.0
    return z


def region(eye, tgt, depth, w, h):
    mm.W, mm.H = w, h
    return mm.region_map(eye, tgt, depth)


def main():
    k, raw_lr, raw_gt = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    info = {int(r["seq"]): r for r in csv.DictReader(open(os.path.join(SEQ, "seqs.csv"), encoding="utf-8"))}[k]
    split = info["split"]
    out = os.path.join(SEQ, split)
    os.makedirs(out, exist_ok=True)
    cams = {}
    for line in open(os.path.join(SEQ, f"cams_{k:02d}.txt")):
        v = line.strip().split(",")
        if v[2] == "1":
            cams[int(v[1])] = np.array([float(x) for x in v[4:10]])
    jit = {int(r[1]): (float(r[2]), float(r[3])) for r in csv.reader(open(os.path.join(raw_lr, "jitter_lr.csv"))) if int(r[0]) == k}
    rows = []
    for f, cam in sorted(cams.items()):
        src = lambda d, n: os.path.join(d, f"{k:02d}_{f:02d}_{n}.npy")
        dst = lambda n: os.path.join(out, f"{k:02d}_{f:02d}_{n}.npy")
        eye, tgt = cam[:3], cam[3:]
        # 720p：颜色、速度原样搬过去；深度换成观察空间深度
        shutil.move(src(raw_lr, "color"), dst("color"))
        shutil.move(src(raw_lr, "velocity"), dst("velocity"))
        d = view_depth(np.load(src(raw_lr, "depthraw")))
        np.save(dst("depth"), d.astype(np.float16))
        np.save(dst("region"), region(eye, tgt, d, 1280, 720))
        # 2K：GT 原样；G-Buffer 打包成和单帧数据集一样的 uint8
        shutil.move(src(raw_gt, "gt"), dst("gt"))
        hd = view_depth(np.load(src(raw_gt, "depthraw")))
        sky = hd == 0
        gb0, gb1 = np.load(src(raw_gt, "gb0")), np.load(src(raw_gt, "gb1")).astype(np.float32)
        albedo = gb0[..., :3].copy()
        normal = np.clip(np.rint((gb1[..., :3] * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
        mr = np.stack([gb0[..., 3], np.clip(np.rint(gb1[..., 3] * 255), 0, 255).astype(np.uint8)], -1)
        for a in (albedo, normal, mr):
            a[sky] = 0
        np.save(dst("albedo"), albedo)
        np.save(dst("normal"), normal)
        np.save(dst("mr"), mr)
        np.save(dst("hrdepth"), hd.astype(np.float16))
        if split == "test":
            np.save(dst("hrregion"), region(eye, tgt, hd, 2560, 1440))
            shutil.move(src(raw_gt, "noaa"), dst("noaa"))
        for n in ("depthraw",):
            os.remove(src(raw_lr, n))
        for n in ("gb0", "gb1", "depthraw"):
            os.remove(src(raw_gt, n))
        rows.append([k, f, split, f"{jit[f][0]:.6f}", f"{jit[f][1]:.6f}"] + [f"{x:.4f}" for x in cam])
    # frames.csv：所有段合在一起，同一段重跑就覆盖
    path = os.path.join(SEQ, "frames.csv")
    head = ["seq", "frame", "split", "jx", "jy", "ex", "ey", "ez", "tx", "ty", "tz"]
    old = [r for r in csv.reader(open(path, encoding="utf-8"))][1:] if os.path.exists(path) else []
    old = [r for r in old if int(r[0]) != k]
    with open(path, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(head)
        w.writerows(sorted(old + rows, key=lambda r: (int(r[0]), int(r[1]))))
    print(f"seq {k} ({split}): {len(rows)} 帧 → {out}")


if __name__ == "__main__":
    main()
