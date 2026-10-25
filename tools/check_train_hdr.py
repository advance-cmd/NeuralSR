"""检查 train_hdr.py（训练前先跑这个，免得训了十几分钟才发现写错）。

用法：python tools/check_train_hdr.py
  1. step_loss：权重全 0 的模型 B（输出 = 输入的双三次放大），在一批随机数据上的损失要等于 log 空间的 L1；梯度能传回去
  2. evaluate：用的数据对不对（整帧、每 4 帧取 1 帧），权重全 0 的模型 B 的分数要和参考答案一样（约 10 秒）
  3. 训练循环用的是训练集
"""
import copy
import os
import re
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train_hdr  # noqa: E402
from dataset import HDRDataset  # noqa: E402
from hdr import decode, encode  # noqa: E402
from model import SRNet  # noqa: E402

REF = 31.250
BUGS = [(30.880, "空4：忘了 decode（网络输出还在 log 空间，就拿去 tonemap 了）"),
        (31.169, "空4：颜色没 encode 也没 decode（网络在线性 HDR 里放大，那是另一种训法）"),
        (33.136, "空5：没过 tonemap（在线性 HDR 里算的 PSNR，不是屏幕上看到的）"),
        (8.299, "空5：只给 sr 过了 tonemap，hr 也要过")]
ok = True


def check(name, cond, hint=""):
    global ok
    print(("  ✅ " if cond else "  ❌ ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)


def zeroed(m):
    z = copy.deepcopy(m)
    with torch.no_grad():
        for p in z.parameters():
            p.zero_()
    return z


def main():
    print("1. step_loss（一批随机的 HDR 数据，权重全 0 的模型 B：输出 = 输入的双三次放大）")
    torch.manual_seed(0)
    lr, gb, hr = torch.rand(2, 3, 32, 32) * 4, torch.rand(2, 9, 64, 64), torch.rand(2, 3, 64, 64) * 4
    up = lambda x: F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
    m = SRNet(True)
    got = train_hdr.step_loss(zeroed(m), lr, gb, hr).item()
    want = (up(encode(lr)) - encode(hr)).abs().mean().item()
    alts = [((up(lr) - encode(hr)).abs().mean().item(), "空1：颜色要先 encode 再进网络"),
            ((up(encode(lr)) - hr).abs().mean().item(), "空2：目标也要 encode，两边都在 log 空间里比"),
            ((up(lr) - hr).abs().mean().item(), "空1、空2：这是线性 HDR 里的 L1，要在 log 空间"),
            ((decode(up(encode(lr))) - hr).abs().mean().item(), "空2：训练时不 decode，直接拿网络的输出和 encode(hr) 比"),
            (((up(encode(lr)) - encode(hr)) ** 2).mean().item(), "空2：L1 是差的绝对值的平均，不是平方")]
    near = min(alts, key=lambda a: abs(a[0] - got))
    check(f"损失 {got:.5f}（应该是 {want:.5f}）", abs(got - want) < 1e-5,
          near[1] if abs(near[0] - got) < 1e-5 else "对照 train.py 的 L1，再看空1、空2 的注释")
    m2 = copy.deepcopy(m)
    try:
        train_hdr.step_loss(m2, lr, gb, hr).backward()
        flows = all(p.grad is not None and p.grad.abs().sum() > 0 for p in m2.parameters() if p.dim() > 1)
    except (RuntimeError, AttributeError):
        flows = False
    check("梯度能传回每一层", flows, "step_loss 里不要用 torch.no_grad()，也不要 .item() / .detach()，要返回张量")

    print("2. evaluate")
    seen = {}

    class Probe(Exception):
        pass

    def spy(split, patch=64, repeat=16, every=1):
        """先只记下 evaluate 要的数据参数，不真的去读"""
        seen.update(split=split, patch=patch, every=every)
        raise Probe

    train_hdr.HDRDataset = spy
    try:
        train_hdr.evaluate(SRNet(True), torch.device("cpu"))
    except Probe:
        pass
    finally:
        train_hdr.HDRDataset = HDRDataset
    good = seen.get("split") == "test" and seen.get("patch") is None and seen.get("every") == 4
    check(f"数据：HDRDataset({seen.get('split')!r}, patch={seen.get('patch')}, every={seen.get('every')})", good,
          "空3：测试集 \"test\"、整帧 patch=None、every=4")
    if good:
        dev = torch.device("cuda")
        s = train_hdr.evaluate(zeroed(SRNet(True)).to(dev), dev)
        if s != s:
            hint = "空4：decode 前没 encode（e 的几百次方 → inf → NaN）"
        else:
            near = min(BUGS, key=lambda b: abs(b[0] - s))
            hint = near[1] if abs(near[0] - s) < 0.01 else "空4、空5 再对照注释看一遍"
        check(f"权重全 0 的模型 B：{s:.3f} dB（参考答案 {REF:.3f}，就是在 log 空间里双三次放大）", abs(s - REF) < 0.01, hint)

    print("3. 训练循环")
    src = open(os.path.join(ROOT, "train_hdr.py"), encoding="utf-8").read()
    line = next((l for l in src.splitlines() if re.match(r"\s*loader\s*=", l)), "")
    check("loader 用的是 HDRDataset(\"train\")", re.search(r"HDRDataset\(\s*[\"']train[\"']", line) is not None, "空6")
    print("全部通过，可以训练了：python train_hdr.py B" if ok else "还有没过的")


if __name__ == "__main__":
    main()
