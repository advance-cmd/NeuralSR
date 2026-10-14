"""检查第一天的 dataset.py：形状、取值范围、裁块对齐、通道顺序、DataLoader。

用法：python tools/check_day1.py
另外存一张 data/day1_check.png：每行一个随机块，依次是 双三次放大的 LR | HR | 基础色 | 法线 | 1/深度。
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ROOT)
from dataset import SRDataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

failed = 0


def check(ok, msg, hint=""):
    global failed
    print(("  通过  " if ok else "  没过  ") + msg + ("" if ok else f"\n        → {hint}"))
    failed += not ok
    return ok


def psnr(a, b):
    return (10 * torch.log10(1 / ((a - b) ** 2).mean())).item()


def main():
    print("[1] 训练集裁块")
    ds = SRDataset("train")
    p = ds.patch
    check(len(ds) == len(ds.ids) * ds.repeat, f"len = {len(ds)}", "一轮 = 每张图裁 repeat 块")
    lr, gb, hr = ds[0]
    shapes_ok = check(tuple(lr.shape) == (3, p, p) and tuple(gb.shape) == (9, 2 * p, 2 * p) and tuple(hr.shape) == (3, 2 * p, 2 * p),
                      f"形状 lr {tuple(lr.shape)}  gb {tuple(gb.shape)}  hr {tuple(hr.shape)}",
                      f"应该是 (3, {p}, {p}) / (9, {2 * p}, {2 * p}) / (3, {2 * p}, {2 * p})")
    check(lr.dtype == gb.dtype == hr.dtype == torch.float32, f"dtype {lr.dtype} {gb.dtype} {hr.dtype}", "三个都应该是 float32")
    range_ok = check(0 <= lr.min() and lr.max() <= 1 and 0 <= hr.min() and hr.max() <= 1, "lr / hr 在 0..1", "uint8 要除以 255")
    check(0 <= gb.min() and gb.max() <= 1, "gb 在 0..1（含 1/深度）", "看看是不是哪个通道没除以 255，或者天空的 1/深度成了 inf")
    check(not torch.equal(ds[0][2], ds[0][2]), "同一个下标每次裁的位置不一样", "裁块的位置要在 __getitem__ 里随机")
    if shapes_ok:
        bad = [tuple(t.shape) for k in range(300) for t in ds[k][:1] if tuple(t.shape) != (3, p, p)]
        shapes_ok = check(not bad, "连取 300 块，大小都一样",
                          f"有 {len(bad)} 块变小了，比如 lr {bad[0] if bad else ''}：随机坐标的上限大了，块伸出了图像边缘"
                          "（numpy 切片越界不报错，只会切短；DataLoader 拼 batch 时才报错）")
    if shapes_ok and range_ok:
        torch.manual_seed(0)
        scores = []
        for k in range(60):
            lr, gb, hr = ds[k * 37 % len(ds)]
            up = F.interpolate(lr[None], scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
            scores.append(psnr(up, hr[None]))
        m = float(np.mean(scores))
        check(m > 30, "LR 块和 HR 块对得上（双三次放大后很接近）",
              f"60 个块平均只有 {m:.1f} dB：HR 的裁块坐标可能没乘 2，或者裁的不是同一张图")

    print("[2] 测试集整图（patch=None）")
    dt = SRDataset("test", patch=None)
    check(len(dt) == 40, f"len = {len(dt)}", "patch=None 时每张图只取一次")
    lr, gb, hr = dt[7]  # 这张有 30% 是天空
    if check(tuple(lr.shape) == (3, 720, 1280) and tuple(gb.shape) == (9, 1440, 2560) and tuple(hr.shape) == (3, 1440, 2560),
             f"形状 lr {tuple(lr.shape)}  gb {tuple(gb.shape)}  hr {tuple(hr.shape)}", "整图不裁"):
        base = os.path.join(ROOT, "data", "test", f"{dt.ids[7]:04d}_")
        ref = lambda n: torch.from_numpy(np.load(base + n + ".npy")).permute(2, 0, 1).float() / 255
        d = torch.from_numpy(np.load(base + "depth.npy").astype(np.float32))
        inv = torch.where(d > 0, 1 / d, 0)
        close = lambda a, b: torch.allclose(a, b, atol=1e-3)
        if not check(close(lr, ref("lr")) and close(hr, ref("hr")), "lr / hr 和原图逐像素一致",
                     "没除以 255" if close(lr, ref("lr") * 255) else "多半是 (H, W, C) → (C, H, W) 用了 view / reshape，要用 permute"):
            pass
        check(bool(torch.isfinite(gb).all()), "天空像素的 gb 没有 inf / nan", "天空深度 = 0，1/0 = inf：天空要单独给 0")
        for name, sl, want in (("基础色", slice(0, 3), ref("albedo")), ("法线", slice(3, 6), ref("normal")),
                               ("金属度 / 粗糙度", slice(6, 8), ref("mr")), ("1/深度", slice(8, 9), inv[None])):
            check(close(gb[sl], want), f"gb 第 {sl.start}" + (f"–{sl.stop - 1}" if sl.stop - sl.start > 1 else "") + f" 通道 = {name}", "顺序：基础色 3、法线 3、金属度 / 粗糙度 2、1/深度 1")

    print("[3] DataLoader（num_workers=2）")
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    try:
        lr, gb, hr = next(iter(loader))
    except RuntimeError as e:
        check(False, "DataLoader 拼 batch", str(e).splitlines()[0][:200])
        print(f"{failed} 项没过")
        return
    if not check(tuple(lr.shape) == (4, 3, p, p) and tuple(hr.shape) == (4, 3, 2 * p, 2 * p), f"一批 lr {tuple(lr.shape)}  gb {tuple(gb.shape)}  hr {tuple(hr.shape)}",
          "DataLoader 会在最前面加一维 batch"):
        print(f"{failed} 项没过")
        return

    # 看图：每行一个块
    up = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
    inv = gb[:, 8:9]
    inv = (inv / inv.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)).expand(-1, 3, -1, -1)
    rows = torch.cat([torch.cat([up, hr, gb[:, 0:3], gb[:, 3:6], inv], dim=3)[k] for k in range(4)], dim=1)
    img = (rows.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    out = os.path.join(ROOT, "data", "day1_check.png")
    Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(out)
    print(f"\n看图：{out}（每行：双三次放大的 LR | HR | 基础色 | 法线 | 1/深度）")
    print("全部通过" if failed == 0 else f"{failed} 项没过")


if __name__ == "__main__":
    main()
