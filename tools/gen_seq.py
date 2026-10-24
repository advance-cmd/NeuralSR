"""生成多帧数据集的相机路径（s193，固定种子，可复现）。

输出：
  data/seq/seqs.csv     每段一行：seq,split,kind,poi,balls,speed（kind 的速度单位见 KINDS）
  data/seq/cams_XX.txt  给 DX12 导出程序读的第 XX 段：每行 seq,frame,export,balls,ex,ey,ez,tx,ty,tz
                        frame = -2、-1 是预滚帧（不导出，只让第 0 帧的"上一帧"有意义），0..31 导出

每段 32 帧（正好把 2× 超分的 32 个抖动相位走一圈）。路径有 5 种：
  static 不动（只剩抖动、波浪、测试球）| pan 原地转头 | strafe 横移（眼睛和目标一起平移）
  orbit 绕目标转（视差 + 遮挡变化最多）| dolly 往前推
划分和单帧数据集一样：眼睛相对场景中心 (8, -25) 的方位角在 [0°, 60°) 的归测试集，而且整段都要在这个扇区里；
训练段整段都不能进这个扇区。
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_cams as gc  # noqa: E402  复用目标点、地形、遮挡、视线检查

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "seq")
N_TRAIN, N_TEST = 12, 4
FRAMES, PREROLL = 32, 2
KINDS = {  # 每帧的速度范围
    "static": (0.0, 0.0),
    "pan": (0.1, 0.5),      # 度 / 帧（偏航）
    "strafe": (0.03, 0.15),  # 世界单位 / 帧
    "orbit": (0.2, 0.8),    # 度 / 帧（绕目标）
    "dolly": (0.03, 0.12),  # 世界单位 / 帧
}


def path(kind, eye0, tgt0, speed, sign, t):
    eye0, tgt0 = np.array(eye0), np.array(tgt0)
    if kind == "static":
        return eye0, tgt0
    if kind == "pan":
        a = math.radians(speed * t * sign)
        d = tgt0 - eye0
        x, z = d[0] * math.cos(a) - d[2] * math.sin(a), d[0] * math.sin(a) + d[2] * math.cos(a)
        return eye0, eye0 + np.array([x, d[1], z])
    if kind == "strafe":
        f = tgt0 - eye0
        r = np.cross([0.0, 1.0, 0.0], f)
        r /= np.linalg.norm(r)
        return eye0 + r * speed * t * sign, tgt0 + r * speed * t * sign
    if kind == "orbit":
        a = math.radians(speed * t * sign)
        d = eye0 - tgt0
        x, z = d[0] * math.cos(a) - d[2] * math.sin(a), d[0] * math.sin(a) + d[2] * math.cos(a)
        return tgt0 + np.array([x, d[1], z]), tgt0
    if kind == "dolly":
        f = tgt0 - eye0
        f /= np.linalg.norm(f)
        return eye0 + f * speed * t * sign, tgt0
    raise ValueError(kind)


def valid(eye, tgt, in_test):
    x, y, z = eye
    if abs(x) > 75 or abs(z) > 75 or y < gc.ground(x, z) + 1.0:
        return False
    if any(math.hypot(x - c[0], z - c[1]) < rad and y < top for c, rad, top in gc.OBSTACLES):
        return False
    a = gc.azimuth(eye)
    if (gc.TEST_SECTOR[0] <= a < gc.TEST_SECTOR[1]) != in_test:
        return False
    return gc.visible(eye, tgt)


def main():
    rng = np.random.default_rng(20261024)
    kinds = list(KINDS)
    want = [("train", kinds[i % len(kinds)]) for i in range(N_TRAIN)] + [("test", k) for k in ("static", "pan", "strafe", "orbit")]
    os.makedirs(OUT, exist_ok=True)
    seqs = []
    for split, kind in want:
        in_test = split == "test"
        for _ in range(20000):
            poi, tgt = gc.pick_target(rng)
            if split == "test" and kind == "static" and poi != "spheres":
                continue  # 测试集的静止段专门看测试球在动（物体自己的运动）
            eye = gc.pick_eye(rng, tgt)
            if eye is None:
                continue
            lo, hi = KINDS[kind]
            speed = float(rng.uniform(lo, hi))
            sign = 1 if rng.random() < 0.5 else -1
            frames = [path(kind, eye, tgt, speed, sign, t) for t in range(-PREROLL, FRAMES)]
            if all(valid(e, g, in_test) for e, g in frames[::5] + [frames[-1]]):
                break
        else:
            raise RuntimeError(f"找不到 {split} {kind}")
        balls = int(poi == "spheres" or rng.random() < 0.3)
        seqs.append((split, kind, poi, balls, speed * sign, frames))

    with open(os.path.join(OUT, "seqs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "split", "kind", "poi", "balls", "speed"])
        for k, (split, kind, poi, balls, speed, _) in enumerate(seqs):
            w.writerow([k, split, kind, poi, balls, f"{speed:.3f}"])
    for k, (split, kind, poi, balls, speed, frames) in enumerate(seqs):
        with open(os.path.join(OUT, f"cams_{k:02d}.txt"), "w", encoding="ascii", newline="\n") as f:
            for t, (e, g) in zip(range(-PREROLL, FRAMES), frames):
                f.write(f"{k},{t},{int(t >= 0)},{balls}," + ",".join(f"{v:.4f}" for v in (*e, *g)) + "\n")
        print(f"{k:2d} {split:5s} {kind:6s} {poi:8s} balls {balls} speed {speed:+.3f}  eye0 {np.round(frames[PREROLL][0], 1)}")


if __name__ == "__main__":
    main()
