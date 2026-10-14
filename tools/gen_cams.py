"""生成导数据用的机位表（固定种子，可复现）。

输出：
  data/cams.csv  每行 id,split,name,ex,ey,ez,tx,ty,tz
  data/cams.txt  给 DX12 程序读的机位表，每行 ex,ey,ez,tx,ty,tz（行号 = id）

划分：眼睛相对场景中心 (8, -25) 的方位角落在 [0°, 60°) 的归测试集（训练时没见过这个方向），
      再加上 demo 用的 7 个参考机位；其余归训练集。
"""
import csv
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data")
N_TRAIN, N_TEST_RANDOM = 200, 33
CENTER = np.array([8.0, -25.0])
TEST_SECTOR = (0.0, 60.0)  # 度

PRESETS = [  # main.cpp 的 kCameraPresets
    ("p1 开场全景", (34.0, 5.0, -44.0), (-4.0, 2.0, -20.0)),
    ("p2 水面与倒影", (35.0, 14.0, -62.0), (0.0, 2.0, -22.0)),
    ("p3 铁丝网、镜子和木箱", (-14.0, 4.0, -46.0), (12.0, 3.0, -18.0)),
    ("p4 镜子特写", (-2.0, 6.0, -32.0), (11.0, 4.5, -20.0)),
    ("p5 山坡、树影和雾", (-30.0, 22.0, -75.0), (8.0, 2.0, -20.0)),
    ("p6 法线贴图地板", (6.0, 2.5, -42.0), (-2.0, 0.0, -18.0)),
    ("p7 PBR 测试球", (19.0, 4.0, -19.0), (19.0, 1.0, -34.0)),
]

# 目标点：(名字, 权重, 中心, 抖动半径)；hills / water 在地形上随机取
POIS = [
    ("fence", 0.15, (0.0, 3.0, -20.0), 3.0),
    ("crates", 0.15, (6.0, 2.0, -23.5), 2.0),
    ("mirror", 0.10, (11.0, 4.5, -20.0), 2.0),
    ("floor", 0.15, (1.0, 0.3, -27.0), 5.0),
    ("spheres", 0.10, (19.0, 1.2, -34.0), 4.0),
    ("hills", 0.20, None, 0.0),
    ("water", 0.15, None, 0.0),
]
OBSTACLES = [  # 眼睛不能放进这些东西里：(中心 xz, 半径, 高度上限)
    ((0.0, -20.0), 4.5, 9.0),     # 铁丝网
    ((5.0, -22.0), 2.0, 3.5),     # 木箱
    ((6.75, -25.25), 2.0, 4.5),   # 木箱
    ((11.0, -20.0), 2.5, 9.0),    # 镜子
]


def hills(x, z):
    return 0.3 * (z * math.sin(0.1 * x) + x * math.cos(0.1 * z))


def ground(x, z):
    return max(hills(x, z), 0.0)  # 水面以下按水面算


def azimuth(eye):
    d = np.array([eye[0], eye[2]]) - CENTER
    return math.degrees(math.atan2(d[1], d[0])) % 360.0


def pick_target(rng):
    w = np.array([p[1] for p in POIS])
    i = rng.choice(len(POIS), p=w / w.sum())
    name, _, c, r = POIS[i]
    if name == "hills":
        while True:
            x, z = rng.uniform(-60, 60, 2)
            if hills(x, z) > 1.0:
                return name, (x, hills(x, z) + 0.5, z)
    if name == "water":
        while True:
            x, z = rng.uniform(-55, 55, 2)
            if hills(x, z) < -0.5:
                return name, (x, 0.0, z)
    off = rng.uniform(-r, r, 3) * np.array([1.0, 0.3, 1.0])
    return name, (c[0] + off[0], max(c[1] + off[1], 0.2), c[2] + off[2])


def visible(eye, tgt):
    """眼睛到目标的连线不能钻进山里（离目标最近的 10% 不查，目标本来就贴着地面）。"""
    for t in np.linspace(0.0, 0.9, 24):
        p = np.array(eye) * (1 - t) + np.array(tgt) * t
        if p[1] < hills(p[0], p[2]) + 0.3:
            return False
    return True


def pick_eye(rng, tgt):
    for _ in range(200):
        r = math.exp(rng.uniform(math.log(4.0), math.log(40.0)))
        a = rng.uniform(0, 2 * math.pi)
        x, z = tgt[0] + r * math.cos(a), tgt[2] + r * math.sin(a)
        if abs(x) > 75 or abs(z) > 75:
            continue
        y = ground(x, z) + math.exp(rng.uniform(math.log(1.2), math.log(15.0)))
        eye = (x, y, z)
        if any(math.hypot(x - c[0], z - c[1]) < rad and y < top for c, rad, top in OBSTACLES):
            continue
        horiz = math.hypot(x - tgt[0], z - tgt[2])
        if math.degrees(math.atan2(y - tgt[1], horiz)) > 50 or y < tgt[1] - 3.0:
            continue
        if not visible(eye, tgt):
            continue
        return eye
    return None


def main():
    rng = np.random.default_rng(20261014)
    train, test = [], []
    while len(train) < N_TRAIN or len(test) < N_TEST_RANDOM:
        name, tgt = pick_target(rng)
        eye = pick_eye(rng, tgt)
        if eye is None:
            continue
        a = azimuth(eye)
        in_test = TEST_SECTOR[0] <= a < TEST_SECTOR[1]
        bucket, cap = (test, N_TEST_RANDOM) if in_test else (train, N_TRAIN)
        if len(bucket) < cap:
            bucket.append((name, eye, tgt))

    rows = [("train", n, e, t) for n, e, t in train] + [("test", n, e, t) for n, e, t in test]
    rows += [("test", n, e, t) for n, e, t in PRESETS]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cams.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "name", "ex", "ey", "ez", "tx", "ty", "tz"])
        for i, (split, n, e, t) in enumerate(rows):
            w.writerow([i, split, n] + [f"{v:.3f}" for v in (*e, *t)])
    with open(os.path.join(OUT, "cams.txt"), "w", encoding="ascii", newline="\n") as f:
        for _, _, e, t in rows:
            f.write(",".join(f"{v:.3f}" for v in (*e, *t)) + "\n")
    print(f"{len(rows)} cams: train {len(train)}, test {len(test)} + {len(PRESETS)} presets")
    for k in ("train", "test"):
        names = [n for s, n, _, _ in rows if s == k and not n.startswith("p")]
        print(" ", k, {p[0]: names.count(p[0]) for p in POIS})


if __name__ == "__main__":
    main()
