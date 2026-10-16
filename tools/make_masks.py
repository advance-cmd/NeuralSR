"""给测试集每张图标出区域：0 = 其他（G-Buffer 里有的不透明物体），1 = 天空，2 = 水面，3 = 镜子。

用法：python tools/make_masks.py        写 data/test/{id}_region.npy（2560×1440 uint8）和 data/regions_preview.jpg

水面、镜子、天空都是前向画的，不在 G-Buffer 里，所以要靠几何反推：
  每个像素按相机参数还原出一条视线，再用深度（光照 Pass 后读的，只有不透明物体）算出不透明表面的位置；
  视线在碰到不透明表面之前，先穿过水面（y = 0，128×128 的波浪网格）或镜子（12×10 的矩形），就算水面或镜子；
  什么都没碰到（深度 = 0）就是天空。
相机和场景参数抄自 LearnDX12 的 main.cpp：SetLens(0.25π, 宽/高, 1, 1000)、LookAt(眼睛, 目标, (0,1,0))、
Waves(128, 128, 1.0)、镜子 RotationY(0.9608) * Translation(11, 5, -20)。
"""
import csv
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
W, H, FOV = 2560, 1440, 0.25 * np.pi
WATER_HALF = 63.5  # 波浪网格 128 个格点、间距 1，中心在原点
MIRROR_C = np.array([11.0, 5.0, -20.0])
TH = 0.9608
MIRROR_X = np.array([np.cos(TH), 0.0, -np.sin(TH)])  # 镜子局部 x 轴在世界里的方向（DirectX 行向量约定）
MIRROR_Y = np.array([0.0, 1.0, 0.0])
MIRROR_Z = np.array([np.sin(TH), 0.0, np.cos(TH)])  # 局部 -z 是正面


def rays(eye, target):
    """每个像素的视线方向 d：世界坐标 = eye + d * 观察空间深度。返回 (H, W, 3)"""
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross([0.0, 1.0, 0.0], f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    t = np.tan(FOV / 2)
    x = ((np.arange(W) + 0.5) / W * 2 - 1) * t * (W / H)
    y = (1 - (np.arange(H) + 0.5) / H * 2) * t
    return x[None, :, None] * r + y[:, None, None] * u + f


def region_map(eye, target, depth):
    d = rays(eye, target)
    z = np.where(depth > 0, depth, np.inf)  # 到不透明表面的"距离"（按观察空间深度量）
    near = z.copy()
    region = np.where(depth > 0, 0, 1).astype(np.uint8)
    # 水面：视线和 y = 0 平面的交点
    with np.errstate(divide="ignore", invalid="ignore"):
        tw = np.where(d[..., 1] < 0, -eye[1] / d[..., 1], np.inf)
    p = eye + d * np.where(np.isfinite(tw), tw, 0)[..., None]
    hit = np.isfinite(tw) & (tw > 0) & (np.abs(p[..., 0]) <= WATER_HALF) & (np.abs(p[..., 2]) <= WATER_HALF) & (tw < near)
    region[hit], near[hit] = 2, tw[hit]
    # 镜子：只有站在正面（局部 -z 一侧）才看得见
    if np.dot(eye - MIRROR_C, MIRROR_Z) < 0:
        dn = d @ MIRROR_Z
        with np.errstate(divide="ignore", invalid="ignore"):
            tm = np.where(np.abs(dn) > 1e-9, np.dot(MIRROR_C - eye, MIRROR_Z) / dn, np.inf)
        q = eye + d * np.where(np.isfinite(tm), tm, 0)[..., None] - MIRROR_C
        hit = np.isfinite(tm) & (tm > 0) & (np.abs(q @ MIRROR_X) <= 6) & (np.abs(q @ MIRROR_Y) <= 5) & (tm < near)
        region[hit] = 3
    return region


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(DATA, "cams.csv"), encoding="utf-8")) if r["split"] == "test"]
    colors = np.array([[0, 0, 0], [80, 140, 255], [0, 200, 200], [255, 120, 0]], np.uint8)
    tiles = []
    for r in rows:
        i = int(r["id"])
        eye = np.array([float(r[k]) for k in ("ex", "ey", "ez")])
        target = np.array([float(r[k]) for k in ("tx", "ty", "tz")])
        depth = np.load(os.path.join(DATA, "test", f"{i:04d}_depth.npy")).astype(np.float64)
        reg = region_map(eye, target, depth)
        np.save(os.path.join(DATA, "test", f"{i:04d}_region.npy"), reg)
        share = np.bincount(reg.ravel(), minlength=4) / reg.size * 100
        print(f"{i} {r['name']:<12} 其他 {share[0]:5.1f}%  天空 {share[1]:5.1f}%  水面 {share[2]:5.1f}%  镜子 {share[3]:5.1f}%")
        hr = np.load(os.path.join(DATA, "test", f"{i:04d}_hr.npy"))[::8, ::8].astype(np.float32)
        mix = hr * 0.5 + colors[reg[::8, ::8]] * 0.5
        mix[reg[::8, ::8] == 0] = hr[reg[::8, ::8] == 0]
        tiles.append(mix.astype(np.uint8))
    sheet = np.concatenate([np.concatenate(tiles[k:k + 8], 1) for k in range(0, 40, 8)], 0)
    Image.fromarray(sheet).save(os.path.join(DATA, "regions_preview.jpg"), quality=88)
    print("预览：data/regions_preview.jpg（蓝 = 天空，青 = 水面，橙 = 镜子）")


if __name__ == "__main__":
    main()
