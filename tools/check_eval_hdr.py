"""检查 eval_hdr.py（正式出表之前先跑这个）。

用法：python tools/check_eval_hdr.py
  1. upscale：网络那条路（log 空间进出）、"2K 直接画"读的是 noaa.npy
  2. quality：每 16 帧取 1 帧（8 帧），双三次 / 2K 直接画 / 权重全 0 的模型 B 的分数要和参考答案一样（不用训练好的权重）
  3. load：读的是 train_hdr.py 存的权重（训练完才查）
"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import eval_hdr  # noqa: E402
from dataset import SEQ, HDRDataset  # noqa: E402
from hdr import decode, encode  # noqa: E402
from model import SRNet  # noqa: E402

# 参考答案：8 帧（every=16）上的 (PSNR, SSIM, log L1)
REF = {"双三次": (31.0547, 0.88901, 0.00331), "2K 直接画": (39.7327, 0.98942, 0.00077), "全 0 的 B": (31.1339, 0.88921, 0.00331)}
REF_LINEAR_PSNR = {"双三次": 33.3611}  # 空4 没过 tonemap 时双三次那一行的 PSNR
ok = True


def check(name, cond, hint=""):
    global ok
    print(("  ✅ " if cond else "  ❌ ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)


def main():
    dev = torch.device("cuda")
    z = SRNet(True).to(dev).eval()
    with torch.no_grad():
        for p in z.parameters():
            p.zero_()

    print("1. upscale（测试集 12_00）")
    lr, gb, hr = HDRDataset("test", patch=None)[0]
    lr, gb = lr[None].to(dev), gb[None].to(dev)
    base = os.path.join(SEQ, "test", "12_00_")
    up = lambda x: F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
    with torch.no_grad():
        got = eval_hdr.upscale(z, lr, gb, base)
        want = decode(up(encode(lr)))
        if torch.allclose(got, want, atol=1e-4):
            hint = ""
        elif torch.allclose(got, up(lr), atol=1e-4):
            hint = "空3：颜色要先 encode 再进网络，出来再 decode"
        elif torch.allclose(got, up(encode(lr)), atol=1e-4):
            hint = "空3：网络的输出还在 log 空间，要 decode"
        else:
            hint = "空3：对照 train_hdr.py 的 evaluate"
        check("网络：权重全 0 时 = log 空间里双三次放大，再 decode 回线性", not hint, hint)

        got = eval_hdr.upscale("2K 直接画", lr, gb, base)
        noaa = torch.from_numpy(np.load(base + "noaa.npy").astype(np.float32)).permute(2, 0, 1)[None]
        if got.shape == up(lr).shape and got.device == lr.device and torch.allclose(got, up(lr)):
            hint = "空2：还是双三次凑的数，要读 noaa.npy"
        elif got.dim() == 3:
            hint = "空2：前面少了一维：(3, H, W) → (1, 3, H, W)，用 [None]"
        elif got.device != lr.device:
            hint = "空2：要放到和 lr 同一个设备上（lr.device）"
        elif torch.allclose(got.cpu() * 255, noaa, rtol=1e-3):
            hint = "空2：HDR 不要除以 255，用 to_tensor_Hdr"
        elif not torch.equal(got.cpu(), noaa):
            hint = "空2：读 base + \"noaa.npy\""
        else:
            hint = ""
        check(f"2K 直接画：形状 {tuple(got.shape)}，在 {got.device}，和 noaa.npy 一样", not hint, hint)
    if not ok:
        print("还有没过的")
        return

    print("2. quality（每 16 帧取 1 帧，8 帧）")
    rows = eval_hdr.quality({"双三次": "双三次", "2K 直接画": "2K 直接画", "全 0 的 B": z}, dev, every=16)
    if "ref" in sys.argv[1:]:
        print({n: tuple(round(float(v), 5) for v in (r[0], r[1], r[3])) for n, r in rows.items()})
        return
    for name, (p, s, l1) in REF.items():
        r = rows[name]
        if abs(r[0] - p) < 0.01 and abs(r[1] - s) < 1e-3 and abs(r[3] - l1) < 1e-4:
            hint = ""
        elif name == "双三次" and abs(r[0] - REF_LINEAR_PSNR[name]) < 0.01:
            hint = "空4：sr、hr 都要过 tonemap（现在是在线性 HDR 里算的）"
        elif r[3] == 0:
            hint = "空5：还没算 log 空间的 L1"
        elif abs(r[0] - p) < 0.01 and abs(r[1] - s) < 1e-3:
            hint = "空5：log 空间 L1 = |encode(sr) − encode(hr)| 的平均"
        else:
            hint = "空4、空5 再对照注释看一遍"
        check(f"{name}：PSNR {r[0]:.2f}、SSIM {r[1]:.4f}、log L1 {r[3]:.4f}（参考 {p:.2f}、{s:.4f}、{l1:.4f}）", not hint, hint)

    print("3. load")
    for name in ("A", "B"):
        path = os.path.join(eval_hdr.RUNS, f"hdr_{name}.pth")
        if not os.path.exists(path):
            print(f"  （runs/hdr_{name}.pth 还没训练出来，先跳过）")
            continue
        m = eval_hdr.load(name, dev)
        want = torch.load(path)
        same = all(torch.equal(v.to(dev), m.state_dict()[k]) for k, v in want.items())
        check(f"load(\"{name}\") 读的是 runs/hdr_{name}.pth", same, "空1：train_hdr.py 存的是 hdr_A.pth、hdr_B.pth")
    print("全部通过，可以出正式的表了：python eval_hdr.py" if ok else "还有没过的")


if __name__ == "__main__":
    main()
