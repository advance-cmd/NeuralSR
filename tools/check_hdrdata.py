"""检查 dataset.py 里的 HDRDataset（多帧数据集里每一帧单独当一个样本）。

用法：python tools/check_hdrdata.py
  1. 编号：训练 12 段 × 32 帧、测试 every=4、长度
  2. 整帧（测试集 12_00）：形状、和原始 .npy 一个像素一个像素对得上（没除以 255、通道顺序对、目标是 gt）、G-Buffer 的 1/深度
  3. 裁块：形状、LR 块和 HR 块对得上
  4. DataLoader：一个 batch 的形状和用时
"""
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dataset import HDRDataset, SEQ  # noqa: E402
from hdr import tonemap  # noqa: E402

ok = True


def check(name, cond, hint=""):
    global ok
    print(("  ✅ " if cond else "  ❌ ") + name + ("" if cond else f"   ← {hint}"))
    ok &= bool(cond)


def try_get(ds, k):
    """取第 k 个样本；出错时按报错猜是哪个空"""
    try:
        return ds[k]
    except FileNotFoundError as e:
        name = os.path.basename(str(e.filename or e))
        hint = {"lr": "空3", "hr": "空6", "noaa": "空6：noaa 只有测试集有，而且目标是 gt"}.get(name.rsplit("_", 1)[-1][:-4], "空2（前缀）或空3、5、6（文件名）")
        check(f"读文件：找不到 {name}", False, f"{hint}：data/README.md 最后一节的表里有这个划分的文件名")
    except RuntimeError as e:
        if "Sizes of tensors must match" in str(e):
            check("拼 G-Buffer：1/深度和别的通道尺寸不一样", False, "空5：这个划分里有两个深度文件，720p 的和 2K 的，G-Buffer 要 2K 的")
        else:
            check(f"取样本出错：{e}", False, "把报错贴给我")
    except ZeroDivisionError:
        check("取样本：ids 是空的（i % 0）", False, "空1")
    return None


def main():
    print("1. 编号")
    train, test4 = HDRDataset("train"), HDRDataset("test", patch=None, every=4)
    check(f"训练集 {len(train.ids)} 帧（12 段 × 32）", len(train.ids) == 384, "空1：按 *_gt.npy 列，一帧只算一次")
    check("编号是 \"段_帧\" 字符串，排好序", len(train.ids) > 0 and all(isinstance(s, str) and len(s) == 5 for s in train.ids)
          and train.ids == sorted(train.ids) and train.ids[0] == "00_00", "空1：f[:5]，sorted")
    want = [f"{k}_{f:02d}" for k in (12, 13, 14, 15) for f in range(0, 32, 4)]
    check(f"测试集 every=4：{len(test4.ids)} 帧，12_00、12_04 …", test4.ids == want, "空1：列表[::every] 每 every 个取一个")
    check(f"len(训练集) = 384 × 16 = {len(train)}", len(train) == 384 * 16, "空1")
    if not ok:
        print("还有没过的")
        return

    print("2. 整帧（测试集 12_00，patch=None）")
    full = HDRDataset("test", patch=None)
    got = try_get(full, 0)
    if got is None:
        print("还有没过的")
        return
    lr, gb, hr = got
    check(f"形状 lr {tuple(lr.shape)}、gb {tuple(gb.shape)}、hr {tuple(hr.shape)}",
          lr.shape == (3, 720, 1280) and gb.shape == (9, 1440, 2560) and hr.shape == (3, 1440, 2560), "空4：permute(2, 0, 1)")
    check(f"类型 {lr.dtype}、{hr.dtype}", lr.dtype == torch.float32 and hr.dtype == torch.float32, "空4：float16 → float32")
    raw = lambda n: torch.from_numpy(np.load(os.path.join(SEQ, "test", f"12_00_{n}.npy")).astype(np.float32))
    color, gt = raw("color"), raw("gt")
    def hdr_hint(t, a, name):
        """t 是 HDRDataset 给的，a 是原始 (H, W, C) 数组；一样就返回空字符串，不一样就猜是哪个空"""
        want = a.permute(2, 0, 1)
        if torch.equal(t, want):
            return ""
        if torch.allclose(t * 255, want, rtol=1e-3):
            return "空4：HDR 不要除以 255（to_tensor 是给 0..255 的 uint8 用的）"
        if torch.equal(t, a.reshape(want.shape)):
            return "空4：view / reshape 只改形状、不搬数据，要 permute(2, 0, 1)"
        return name

    if lr.shape == (3, 720, 1280):
        hint = hdr_hint(lr, color, "空3：输入应该是 color")
        check(f"lr 和 12_00_color.npy 一样（最亮 {lr.max().item():.1f}，HDR 本来就会大于 1）", not hint, hint)
    if hr.shape == (3, 1440, 2560):
        hint = hdr_hint(hr, gt, "空6：目标是 gt（64 次抖动平均）")
        check("hr 和 12_00_gt.npy 一样", not hint, hint)
    d = raw("hrdepth")
    check("gb 最后一个通道 = 1 / 2K 深度（天空 0）", torch.allclose(gb[8], torch.where(d > 0, 1 / d, 0)), "空5：hrdepth")
    check("没有 NaN / inf", bool(torch.isfinite(lr).all() and torch.isfinite(hr).all() and torch.isfinite(gb).all()), "把输出贴给我")

    print("3. 裁块（训练集，随机 20 块）")
    random.seed(0)
    shapes, se, n = set(), 0.0, 0
    for _ in range(20):
        got = try_get(train, random.randrange(len(train)))
        if got is None:
            print("还有没过的")
            return
        lr, gb, hr = got
        shapes.add((tuple(lr.shape), tuple(gb.shape), tuple(hr.shape)))
        up = F.interpolate(lr[None], scale_factor=2, mode="bicubic", align_corners=False).clamp(min=0)
        se += ((tonemap(up) - tonemap(hr[None])) ** 2).sum().item()
        n += hr.numel()
    check(f"块的形状 {sorted(shapes)}", shapes == {((3, 64, 64), (9, 128, 128), (3, 128, 128))}, "裁块那段是写好的，前面几项先过")
    if n:
        psnr = 10 * math.log10(n / se)
        check(f"LR 块放大后和 HR 块对得上：屏幕空间 PSNR {psnr:.1f} dB（> 25）", psnr > 25, "LR、HR 裁的不是同一块")

    if not ok:
        print("还有没过的")
        return

    print("4. DataLoader（batch 16，4 个进程）")
    loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=4)
    t0 = time.time()
    lr, gb, hr = next(iter(loader))
    check(f"一个 batch：{tuple(lr.shape)} {tuple(gb.shape)} {tuple(hr.shape)}，{time.time() - t0:.1f} 秒（含启动进程）",
          lr.shape == (16, 3, 64, 64) and gb.shape == (16, 9, 128, 128) and hr.shape == (16, 3, 128, 128))
    print("全部通过" if ok else "还有没过的")


if __name__ == "__main__":
    main()
