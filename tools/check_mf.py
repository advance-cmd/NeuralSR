"""检查 s194 多帧网络的代码。每写完一块跑一次：
  python tools/check_mf.py 1   第 1 块：dataset.py 的 SeqDataset（约 5 秒）
  python tools/check_mf.py 2   第 2 块：model.py 的 MFNet、multiframe.py（几秒）
  python tools/check_mf.py 3   第 3 块：eval_mf.py 的 sequence_psnr、train_mf.py（约 1 分钟）
不写数字就三块都查。
"""
import copy
import os
import random
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SEQ = os.path.join(ROOT, "data", "seq")
REF_ZERO = 30.711  # 权重全 0 的 MFNet（= log 空间里双三次放大），4 段 × 32 帧
REF_ACCUM = 31.451  # 下面的 Accum（不用网络，TAA 那样按掩码把历史和双三次混起来）
ok = True


def check(name, cond, hint=""):
    global ok
    print(("  ✅ " if cond else "  ❌ ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)
    return bool(cond)


def zeroed(m):
    z = copy.deepcopy(m)
    with torch.no_grad():
        for p in z.parameters():
            p.zero_()
    return z


def raw(split, fid, name):
    return np.load(os.path.join(SEQ, split, f"{fid}_{name}.npy")).astype(np.float32)


def part1():
    print("1. SeqDataset")
    from dataset import SeqDataset
    ds = SeqDataset("train", frames=4)
    check(f"训练集 frames=4：{len(ds.clips)} 个起点（12 段 × 29）", len(ds.clips) == 12 * 29,
          "空1：每段从第 0 帧到第 32 − frames 帧都能当起点，range 的上界要 + 1")
    check("第一个起点 ('00', 0)、最后一个 ('11', 28)", ds.clips[0] == ("00", 0) and ds.clips[-1] == ("11", 28),
          "空1：看 range 的上界；起点是 (段号字符串, 帧号整数)，段在外层循环")
    check(f"len = 起点数 × 16 = {len(ds)}", len(ds) == len(ds.clips) * 16)
    dt = SeqDataset("test", frames=1, patch=None)
    check("测试集 frames=1、整帧：128 个，按段、按帧排好", len(dt.clips) == 128 and dt.clips[31] == ("12", 31) and dt.clips[32] == ("13", 0))

    # 裁块：把 random.randint 换成一串固定的数，看几帧是不是用的同一个位置
    ds = SeqDataset("test", frames=3, patch=64)
    seq_vals = iter([100, 200, 37, 333, 5, 7, 80, 90, 11, 13])
    real = random.randint
    random.randint = lambda a, b: next(seq_vals)
    try:
        item = ds[ds.clips.index(("13", 5))]
    except RuntimeError as e:
        check("取一个样本（3 帧、64×64 的块）", False, f"{e}" + ("（gb 的几张图大小不一样：albedo、normal、mr 是 2K 的，"
              "1/深度也要 2K 的：hrdepth + crop2，和 HDRDataset 一样）" if "torch.cat" in str(e) or "Sizes of tensors" in str(e) else ""))
        return
    finally:
        random.randint = real
    check("返回 5 个张量", isinstance(item, (tuple, list)) and len(item) == 5, "按 lr, gb, hr, velocity, jitter 的顺序返回")
    if len(item) != 5:
        return
    lr, gb, hr, vel, jit = item
    shapes = [tuple(t.shape) if torch.is_tensor(t) else type(t).__name__ for t in item]
    want = [(3, 3, 64, 64), (3, 9, 128, 128), (3, 3, 128, 128), (3, 2, 64, 64), (3, 2)]
    if not all(torch.is_tensor(t) for t in item):
        check(f"形状 {shapes}", False, "空4：每个列表用 torch.stack 摞成一个张量，第 0 维是第几帧")
        return
    why = {0: "lr 是 720p 的东西，用 crop，边长 p（P = 2p 是 2K 的边长）",
           3: "velocity 和 color 一样是 720p 的（720×1280），用 crop，边长 p"}
    bad = [k for k in range(5) if shapes[k] != want[k]]
    check(f"形状 {shapes}", not bad, f"应该是 {want}。" + "；".join(why.get(k, f"第 {k} 个不对") for k in bad))
    if bad:
        return
    check("都是 float32", all(t.dtype == torch.float32 for t in item))
    pos = None
    for y, x in [(100, 200), (200, 100)]:
        c0 = raw("test", "13_05", "color")[y:y + 64, x:x + 64]
        if np.abs(lr[0].permute(1, 2, 0).numpy() - c0).max() < 1e-6:
            pos = (y, x)
    if not check("第 0 帧的 lr 是 color.npy 里那一块", pos is not None, "看裁块：lr 用 crop，从抽到的 (y, x) 开始取 p 行 p 列"):
        return
    y, x = pos
    same = all(np.abs(lr[k].permute(1, 2, 0).numpy() - raw("test", f"13_{5 + k:02d}", "color")[y:y + 64, x:x + 64]).max() < 1e-6
               for k in range(3))
    check("3 帧用的是同一个裁块位置、依次是第 5、6、7 帧", same, "空2：y、x 要在 for 循环外面抽一次；每帧重抽，几帧就对不上了")
    Y, X = y * 2, x * 2
    gt_ok = np.abs(hr[0].permute(1, 2, 0).numpy() - raw("test", "13_05", "gt")[Y:Y + 128, X:X + 128]).max() < 1e-6
    check("hr 是 gt.npy 里对应的 2K 块（左上角 ×2、边长 ×2）", gt_ok, "2K 的东西用 crop2")
    d = raw("test", "13_05", "hrdepth")[Y:Y + 128, X:X + 128]
    inv = np.where(d > 0, 1 / np.maximum(d, 1e-8), 0)
    check("gb 的第 9 个通道 = 2K 的 1/深度", np.abs(gb[0, 8].numpy() - inv).max() < 1e-4, "和 HDRDataset 一样：hrdepth，天空给 0")
    v = raw("test", "13_05", "velocity")[y:y + 64, x:x + 64]
    want_v = v * np.array([1280 / 64, 720 / 64], dtype=np.float32)
    got_v = vel[0].permute(1, 2, 0).numpy()
    if np.abs(got_v - v).max() < 1e-6:
        hint = "空3：速度存的是整屏的 uv，裁成 64×64 以后要换成块的 uv：u × 1280 / 64、v × 720 / 64"
    elif np.abs(got_v - v * np.array([64 / 1280, 64 / 720], dtype=np.float32)).max() < 1e-6:
        hint = "空3：乘反了。整屏 1 个 uv 单位 = 1280 像素，块的 1 个 uv 单位只有 64 像素，所以数值要变大"
    elif np.abs(got_v - v * 64).max() < 1e-4:
        hint = "空3：乘的是 p 本身。整屏 uv × 1280 = 像素，像素 ÷ p = 块的 uv，合起来是 u × 1280 / p、v × 720 / p"
    else:
        hint = "空3：u 乘 1280 / p、v 乘 720 / p（先 u 后 v，和速度的两个通道一样）"
    check("velocity 换成了块的 uv", np.abs(got_v - want_v).max() < 1e-4, hint)
    import csv
    rows = {(r["seq"], r["frame"]): (float(r["jx"]), float(r["jy"])) for r in csv.DictReader(open(os.path.join(SEQ, "frames.csv"), encoding="utf-8"))}
    check("jitter 是 frames.csv 里第 5、6、7 帧的抖动", torch.allclose(jit, torch.tensor([rows[("13", str(f))] for f in (5, 6, 7)])))

    try:
        lr, gb, hr, vel, jit = dt[33]
    except RuntimeError as e:
        check("整帧（patch=None）能取出来", False, f"{e}（整帧时 crop 不裁，720p 和 2K 的东西大小不一样，拼 gb 的每张图都要是 2K 的）")
        return
    check("整帧（patch=None）：形状对、速度不变", tuple(lr.shape) == (1, 3, 720, 1280) and tuple(hr.shape) == (1, 3, 1440, 2560)
          and np.abs(vel[0].permute(1, 2, 0).numpy() - raw("test", "13_01", "velocity")).max() < 1e-6,
          "整帧不裁：crop 什么都不做，速度乘 1")


class RefNet(nn.Module):
    """参考答案的 forward，拿用户的权重算一遍对照"""

    def __init__(self, user):
        super().__init__()
        self.user = user

    def forward(self, lr, gb, hist, mask, jitter):
        n, _, h, w = lr.shape
        j = jitter.view(n, 2, 1, 1).expand(n, 2, h, w)
        u = self.user.unshuffle
        x = torch.cat([lr, u(gb), u(hist), u(mask), j], dim=1)
        up = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
        return up + self.user.shuffle(self.user.body(x))


def part2():
    print("2. MFNet")
    from model import MFNet
    m = MFNet()
    n = sum(p.numel() for p in m.parameters())
    first = m.body[0].in_channels if hasattr(m, "body") else None
    if first != 57:
        hint = "空5：3 + 9×4 + 3×4 + 1×4 + 2：2K 的东西 PixelUnshuffle 以后通道 ×4，抖动铺成 2 张图"
    elif m.body[0].out_channels != 32:
        hint = (f"第一层是 Conv2d(cin, {m.body[0].out_channels}, …)：输出应该是 c（32）张特征图，"
                "第二层 Conv2d(c, c, …) 才接得上。除了 cin，body 和模型 B 一模一样")
    else:
        hint = "除了 cin，body 和模型 B 一模一样：对照 SRNet 的 body 看"
    check(f"参数 {n}（应该是 67596），第一层输入 {first} 个通道（应该是 57）", n == 67596 and first == 57, hint)
    check("最后一层是 Conv2d，后面没有 ReLU", isinstance(m.body[-1], nn.Conv2d),
          "detail 要能是负数：双三次比 GT 亮的地方要往下减。ReLU 把负数全变成 0，所以和模型 B 一样，最后一层后面不加 ReLU")
    if first != 57:
        return
    torch.manual_seed(0)
    lr, gb, hist = torch.rand(2, 3, 16, 16), torch.rand(2, 9, 32, 32), torch.rand(2, 3, 32, 32)
    mask, jit = (torch.rand(2, 1, 32, 32) > 0.5).float(), torch.tensor([[0.25, -0.1], [-0.4, 0.3]])
    try:
        out = m(lr, gb, hist, mask, jit)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "expand" in msg:
            hint = "空6：jitter 是 (n, 2)，先 view(n, 2, 1, 1) 变成每个样本 2 张 1×1 的图，再 expand(n, 2, h, w) 铺满"
        elif "channels" in msg:
            hint = "body 的层与层接不上：每一层的输入通道 = 上一层的输出通道"
        else:
            hint = "空6：jitter.view(n, 2, 1, 1).expand(n, 2, h, w)"
        check("forward 能跑", False, f"{type(e).__name__}: {msg}（{hint}）")
        return
    check(f"输出形状 {tuple(out.shape)}", tuple(out.shape) == (2, 3, 32, 32))
    up = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
    check("权重全 0 时 = 双三次放大", torch.allclose(zeroed(m)(lr, gb, hist, mask, jit), up, atol=1e-6))
    same = torch.allclose(out, RefNet(m)(lr, gb, hist, mask, jit), atol=1e-5)
    swapped = torch.allclose(m(lr, gb, hist, mask, jit.flip(0)), RefNet(m)(lr, gb, hist, mask, jit), atol=1e-5)
    check("拼起来的顺序和打出来的一样，两个样本各用各的抖动", same,
          "空6：view 成 (n, 2, 1, 1) 再 expand，不要把 batch 里的抖动混了" if swapped else
          "torch.cat 的顺序按 [lr, unshuffle(gb), unshuffle(hist), unshuffle(mask), j]")

    print("3. multiframe.history_inputs")
    import multiframe as mf
    gb = torch.zeros(1, 9, 32, 32)
    gb[:, 8] = 0.1  # 深度 10
    h0, m0 = mf.history_inputs(None, torch.zeros(1, 2, 16, 16), gb)
    check("第一帧（prev = None）：历史 (1, 3, 32, 32) 全 0、掩码 (1, 1, 32, 32) 全 0",
          tuple(h0.shape) == (1, 3, 32, 32) and tuple(m0.shape) == (1, 1, 32, 32) and h0.abs().sum() == 0 and m0.abs().sum() == 0, "空7")
    out = torch.rand(1, 3, 32, 32)
    try:
        hist, mask = mf.history_inputs((out, gb[:, 8:9].clone()), torch.zeros(1, 2, 16, 16), gb)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        hint = ("空9：两个抖动的形状是 (n, 2)（backward_warp 里会 view(n, 2, 1, 1)），不是速度图那么大：torch.zeros(n, 2, device=gb.device)"
                if "invalid for input of size" in msg else "空8–11 对照注释看一遍")
        check("有上一帧时能跑", False, f"{type(e).__name__}: {msg}（{hint}）")
        return
    if not check(f"历史 {tuple(hist.shape)}、掩码 {tuple(mask.shape)}（应该是 (1, 3, 32, 32)、(1, 1, 32, 32)）",
                 tuple(hist.shape) == (1, 3, 32, 32) and tuple(mask.shape) == (1, 1, 32, 32),
                 "空10 上面那行：这一帧的 1/深度是 gb[:, 8:9]：逗号前的 : 是 batch 全要，逗号后的 8:9 是第 8 个通道（保留这一维）"):
        return
    check("不动：历史 = 上一帧的输出，掩码全 1", torch.allclose(hist, out, atol=1e-6) and mask.min() == 1,
          "空8–11：速度 0、深度一样，应该原样拿过来")
    check("掩码是 0/1 的 float", mask.dtype == torch.float32, "空11：(valid & same).float()")
    vel = torch.zeros(1, 2, 16, 16)
    vel[:, 0] = 2 / 16  # 往右动了 1/8 个屏幕 = 2K 的 4 个像素
    hist, mask = mf.history_inputs((out, gb[:, 8:9].clone()), vel, gb)
    shift = None
    for s in range(1, 12):
        if torch.allclose(hist[..., s:], out[..., :-s], atol=1e-5):
            shift = s
    check(f"整幅往右动 1/8 屏：历史从左边 4 个 2K 像素处取（实际 {shift}）", shift == 4,
          "空8：uv 不随分辨率变，放大速度图时值不用 ×2" if shift == 8 else "空8：F.interpolate 双线性放大 2 倍；空9：两个抖动都给 0")
    check("左边 4 列在上一帧出了屏幕 → 掩码 0，其余 1", mask[..., :4].max() == 0 and mask[..., 4:].min() == 1, "空11：要和 valid 一起")
    prev_inv = gb[:, 8:9].clone()
    cur = gb.clone()
    cur[:, 8, :8, :8] = 0.2  # 深度 5：前面冒出来一个东西
    cur[:, 8, :8, 8:16] = 0.105  # 深度 9.5：差 5%，还算同一个表面
    cur[:, 8, 8:16, :8] = 0  # 这一帧是天空，上一帧不是
    prev_inv[:, :, 16:, 16:] = 0
    cur[:, 8, 16:, 16:] = 0  # 两帧都是天空
    _, mask = mf.history_inputs((out, prev_inv), torch.zeros(1, 2, 16, 16), cur)
    check("深度 10 → 5：掩码 0", mask[0, 0, :8, :8].max() == 0, "空10：相对差 > DEPTH_TOL 就不是同一个表面")
    check("深度 10 → 9.5（差 5%）：掩码 1", mask[0, 0, :8, 8:16].min() == 1, "空10：比的是相对差：|差| ≤ DEPTH_TOL × 两个里大的那个")
    check("上一帧不是天空、这一帧是：掩码 0", mask[0, 0, 8:16, :8].max() == 0)
    check("两帧都是天空（1/深度 都是 0）：掩码 1", mask[0, 0, 16:, 16:].min() == 1, "空10：用 ≤，0 ≤ 0 才成立")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        g, v = cur.to(dev), torch.zeros(1, 2, 16, 16, device=dev)
        try:
            h0, m0 = mf.history_inputs(None, v, g)
            on_gpu = h0.is_cuda and m0.is_cuda
            mf.history_inputs((out.to(dev), prev_inv.to(dev)), v, g)
        except RuntimeError:
            on_gpu = False
        check("数据在 GPU 上时也能跑，历史和掩码也在 GPU 上", on_gpu,
              "空7、空9：torch.zeros 要加 device=gb.device。训练时数据在 GPU 上，不加就建在 CPU 上，"
              "一拼就报 Expected all tensors to be on the same device")

    print("4. run_frame、clip_loss")
    from hdr import encode
    torch.manual_seed(1)
    lr = torch.rand(2, 3, 3, 16, 16) * 3  # batch 2、3 帧
    gb = torch.rand(2, 3, 9, 32, 32)
    hr = torch.rand(2, 3, 3, 32, 32) * 3
    vel = (torch.rand(2, 3, 2, 16, 16) - 0.5) * 0.1
    jit = torch.rand(2, 3, 2) - 0.5
    net = MFNet()
    o, prev, m1 = mf.run_frame(net, None, lr[:, 0], gb[:, 0], vel[:, 0], jit[:, 0])
    check("run_frame 返回的 prev = (这一帧的输出, 这一帧的 1/深度)",
          isinstance(prev, tuple) and len(prev) == 2 and prev[0] is o and torch.equal(prev[1], gb[:, 0, 8:9]),
          "空12：prev 是一个元组，里面两样，要加括号：return out, (out, 1/深度), mask")
    up = lambda x: F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
    want = sum((up(encode(lr[:, k])) - encode(hr[:, k])).abs().mean() for k in range(3)) / 3
    try:
        got = mf.clip_loss(zeroed(net), lr, gb, hr, vel, jit)
        check(f"权重全 0：clip_loss {got.item():.5f}（应该是 {want.item():.5f}）", abs(got.item() - want.item()) < 1e-5,
              "空13：第 k 帧是 lr[:, k]（第 1 维），损失除以帧数")
        net2 = copy.deepcopy(net)
        mf.clip_loss(net2, lr, gb, hr, vel, jit).backward()
        check("梯度能传回每一层", all(p.grad is not None and p.grad.abs().sum() > 0 for p in net2.parameters() if p.dim() > 1))
    except Exception as e:  # noqa: BLE001
        check("clip_loss 能跑", False, f"{type(e).__name__}: {e}（空13：每个张量都取第 k 帧：lr[:, k]、gb[:, k]、velocity[:, k]、jitter[:, k]）")


class Accum(nn.Module):
    """不用网络：掩码 = 1 的地方 0.9 × 历史 + 0.1 × 双三次（TAA 那样），掩码 = 0 的地方只用双三次"""

    def forward(self, lr, gb, hist, mask, jitter):
        up = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
        return mask * (0.9 * hist + 0.1 * up) + (1 - mask) * up


def part3():
    print("5. eval_mf.sequence_psnr（两遍，每遍约 30 秒）")
    import eval_mf
    from model import MFNet
    dev = torch.device("cuda")
    try:
        s0 = eval_mf.sequence_psnr(zeroed(MFNet()).to(dev), dev)
    except Exception as e:  # noqa: BLE001
        check("sequence_psnr 能跑", False, f"{type(e).__name__}: {e}（空16：每个张量去掉第 1 维 t[:, 0]；空18：run_frame 的参数顺序）")
        return
    if abs(s0 - 29.471) < 0.01:
        hint = "空19：用成了 encode（线性 → log），这里要反过来：decode（log → 线性）再 tonemap"
    else:
        hint = "空19：网络输出在 log 空间，先 decode 再 tonemap"
    check(f"权重全 0 的 MFNet：{s0:.3f} dB（参考 {REF_ZERO:.3f}）", abs(s0 - REF_ZERO) < 0.01, hint)
    s1 = eval_mf.sequence_psnr(Accum().to(dev), dev)
    check(f"TAA 式累积（不用网络）：{s1:.3f} dB（参考 {REF_ACCUM:.3f}）", abs(s1 - REF_ACCUM) < 0.01,
          "空17：每段第 0 帧要把 prev 清成 None，不然上一段最后一帧的画面会被拉进来" if abs(s0 - REF_ZERO) < 0.01 else "先让上一条过")

    print("6. train_mf.py")
    src = open(os.path.join(ROOT, "train_mf.py"), encoding="utf-8").read()
    line = next((l for l in src.splitlines() if re.match(r"\s*loader\s*=", l)), "")
    check("loader 用的是 SeqDataset(\"train\", frames=FRAMES)", re.search(r"SeqDataset\(\s*[\"']train[\"']\s*,\s*frames\s*=\s*FRAMES", line) is not None,
          "空14：SeqDataset 的第一个参数是划分 split，写 SeqDataset(FRAMES) 会把 4 当成划分名；要写 SeqDataset(\"train\", frames=FRAMES)"
          if re.search(r"SeqDataset\(\s*FRAMES", line) else "空14")
    check("损失用的是 clip_loss", re.search(r"loss\s*=\s*clip_loss\(", src) is not None, "空15")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["1", "2", "3"]
    for p in parts:
        {"1": part1, "2": part2, "3": part3}[p]()
    print("全部通过" if ok else "还有没过的")
