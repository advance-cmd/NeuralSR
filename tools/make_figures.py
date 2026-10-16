"""拼 README 用的图（docs/*.jpg）：局部放大对比、误差图、Bloom 对不上的例子。

做法：40 张测试图都跑一遍双三次 / A / B，把每张图切成 160×160 的小块（步长 80）算 PSNR，
按区域占比和 PSNR 差自动挑块：不透明物体里 B 比 A 赢得最多的、天空轮廓、水面、镜子。

用法：python tools/make_figures.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dataset import DATA, SRDataset  # noqa: E402
from model import SRNet  # noqa: E402

OUT = os.path.join(ROOT, "docs")
S, STRIDE = 160, 80  # 小块边长、步长（2K 像素）
T = 320  # 每格显示边长：小块最近邻放大 2 倍
BG, FG, DIM = (24, 24, 24), (235, 235, 235), (160, 160, 160)
FONT = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 22)
FONT_S = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18)
NAMES = ["双三次", "模型 A", "模型 B"]


def load(name, dev):
    m = SRNet(name == "B").to(dev)
    m.load_state_dict(torch.load(os.path.join(ROOT, "runs", name + ".pth")))
    return m.eval()


def upscale_all(ds, nets, k, dev):
    lr, gb, hr = (t[None].to(dev) for t in ds[k])
    with torch.no_grad():
        outs = [F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)] + [n(lr, gb) for n in nets]
    return lr, hr, [o.clamp(0, 1) for o in outs]


def psnr(mse):
    return 10 * np.log10(1 / np.maximum(mse, 1e-10))


def region_of(ds, k):
    return np.load(os.path.join(DATA, "test", f"{ds.ids[k]:04d}_region.npy"))


def window_stats(ds, nets, dev):
    """每张图：每个小块三种方法的 PSNR (3, gh, gw)、四个区域的占比 (4, gh, gw)；整图 PSNR、区域占比；
    每个区域的平方误差之和 (3, 4) 和像素数 (4)（分区 PSNR 用）"""
    rows = []
    for k in range(len(ds)):
        lr, hr, outs = upscale_all(ds, nets, k, dev)
        region = torch.from_numpy(region_of(ds, k)).to(dev)
        pool = lambda x: F.avg_pool2d(x[None, None].float(), S, STRIDE)[0, 0].cpu().numpy()
        ses = [((o - hr) ** 2).mean(1)[0] for o in outs]
        rows.append(dict(k=k,
                         lum=pool(hr.mean(1)[0]),  # 每个小块的平均亮度（真值）
                         win=psnr(np.stack([pool(se) for se in ses])),
                         frac=np.stack([pool(region == r) for r in range(4)]),
                         whole=psnr(np.array([se.mean().item() for se in ses])),
                         share=np.array([(region == r).float().mean().item() for r in range(4)]),
                         err_sum=np.array([[se[region == r].sum().item() for r in range(4)] for se in ses]),
                         count=np.array([(region == r).sum().item() for r in range(4)])))
    return rows


def print_glare_split(ds, rows):
    """分区 PSNR 按"水面上有没有太阳反光"分两组再算一遍（README 表 2）"""
    glare = [r for r in rows if has_glare(r)]
    print("水面上有太阳反光的图：" + "、".join(f"{ds.ids[r['k']]:04d}" for r in glare))
    print("| 测试图 | 张数 | 方法 | 其他 | 天空 | 水面 | 镜子 |")
    print("|---|---|---|---|---|---|---|")
    for label, group in [("无反光", [r for r in rows if not has_glare(r)]), ("有反光", glare)]:
        err = sum(r["err_sum"] for r in group)
        count = sum(r["count"] for r in group)
        for m, name in enumerate(NAMES):
            cells = " | ".join(f"{v:.2f}" for v in psnr(err[m] / count))
            print(f"| {label} | {len(group)} | {name} | {cells} |")


def has_glare(r):
    """水面上有太阳反光（很亮的水面小块）的图"""
    return bool(np.any((r["frac"][2] > 0.5) & (r["lum"] > 0.7)))


def pick(rows, ok, score, used, skip=lambda r: False):
    """所有图里满足 ok 的小块，取 score 最大的一个（每张图最多被挑一次；skip(r) 为真的图不要）"""
    best = None
    for r in rows:
        if r["k"] in used or skip(r):
            continue
        m = ok(r["win"], r["frac"])
        if not m.any():
            continue
        s = np.where(m, score(r["win"], r["frac"]), -np.inf)
        i = np.unravel_index(np.argmax(s), s.shape)
        if best is None or s[i] > best[0]:
            best = (s[i], r["k"], (int(i[0]) * STRIDE, int(i[1]) * STRIDE), r["win"][:, i[0], i[1]])
    used.add(best[1])
    return best[1:]


def to_pil(t):
    """(1, 3, H, W) 0..1 → PIL"""
    return Image.fromarray((t[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))


def crops_figure(ds, nets, dev, picks):
    cols = ["位置", "720p 输入", "双三次", "模型 A（只用颜色）", "模型 B（+ G-Buffer）", "2K 真值"]
    head, cap = 44, 34
    sheet = Image.new("RGB", (T * len(cols), head + (T + cap) * len(picks)), BG)
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        d.text((c * T + T // 2, head // 2), name, fill=FG, font=FONT, anchor="mm")
    for row, (label, k, (y, x), p) in enumerate(picks):
        lr, hr, outs = upscale_all(ds, nets, k, dev)
        top = head + row * (T + cap)
        th = to_pil(hr).resize((T, T * 1440 // 2560), Image.BILINEAR)
        ImageDraw.Draw(th).rectangle([x * T // 2560 - 1, y * T // 2560 - 1, (x + S) * T // 2560, (y + S) * T // 2560],
                                     outline=(255, 60, 60), width=2)
        sheet.paste(th, (0, top))
        d.text((10, top + th.height + 12), label, fill=FG, font=FONT)
        d.text((10, top + th.height + 46), f"test {ds.ids[k]:04d}", fill=DIM, font=FONT_S)
        tiles = [to_pil(lr).crop((x // 2, y // 2, (x + S) // 2, (y + S) // 2))] + \
                [to_pil(o).crop((x, y, x + S, y + S)) for o in outs] + [to_pil(hr).crop((x, y, x + S, y + S))]
        for c, im in enumerate(tiles, start=1):
            sheet.paste(im.resize((T, T), Image.NEAREST), (c * T, top))
            if 2 <= c <= 4:
                d.text((c * T + T // 2, top + T + cap // 2), f"{p[c - 2]:.2f} dB", fill=FG, font=FONT_S, anchor="mm")
    return sheet


def error_pil(o, hr):
    """每个像素的误差 → 热度图：0 = 黑，误差 0.06（约 15/255）及以上 = 最亮"""
    se = ((o - hr) ** 2).mean(1)[0]
    v = (se.sqrt() / 0.06).clamp(0, 1).cpu().numpy()
    return Image.fromarray((colormaps["inferno"](v)[..., :3] * 255).astype(np.uint8)), psnr(se.mean().item())


def error_figure(ds, nets, dev, k):
    """2K 真值 / 双三次误差 / 模型 B 误差（同一色标，越亮错得越多）"""
    lr, hr, outs = upscale_all(ds, nets, k, dev)
    w, h, head = 960, 540, 44
    sheet = Image.new("RGB", (w * 3, h + head), BG)
    d = ImageDraw.Draw(sheet)
    panels = [("2K 真值", to_pil(hr))]
    for name, o in [("双三次误差", outs[0]), ("模型 B 误差", outs[2])]:
        im, p = error_pil(o, hr)
        panels.append((f"{name}（整图 {p:.2f} dB）", im))
    for c, (name, im) in enumerate(panels):
        sheet.paste(im.resize((w, h), Image.BOX), (c * w, head))
        d.text((c * w + w // 2, head // 2), name, fill=FG, font=FONT, anchor="mm")
    return sheet


def bloom_figure(ds, nets, dev, k, box):
    """720p 输入 / 模型 B / 2K 真值 / 模型 B 误差，同一块区域：Bloom 光晕在 720p 下更大"""
    lr, hr, outs = upscale_all(ds, nets, k, dev)
    x0, y0, x1, y1 = box
    w = 640
    h = w * (y1 - y0) // (x1 - x0)
    head = 44
    sheet = Image.new("RGB", (w * 4, h + head), BG)
    d = ImageDraw.Draw(sheet)
    lr_up = to_pil(F.interpolate(lr, scale_factor=2, mode="nearest"))
    err, p = error_pil(outs[2], hr)
    panels = [("720p 输入", lr_up), ("模型 B", to_pil(outs[2])), ("2K 真值", to_pil(hr)), (f"模型 B 误差（{p:.2f} dB）", err)]
    for c, (name, im) in enumerate(panels):
        sheet.paste(im.crop(box).resize((w, h), Image.BILINEAR), (c * w, head))
        d.text((c * w + w // 2, head // 2), name, fill=FG, font=FONT, anchor="mm")
    return sheet


def main():
    dev = torch.device("cuda")
    ds = SRDataset("test", patch=None)
    nets = [load("A", dev), load("B", dev)]
    rows = window_stats(ds, nets, dev)

    used = set()
    detail_ok = lambda p, f: (f[0] > 0.97) & (p[0] < 30)
    picks = [("细节 ①", *pick(rows, detail_ok, lambda p, f: p[2] - p[1], used)),
             ("细节 ②", *pick(rows, detail_ok, lambda p, f: p[2] - p[1], used)),
             ("天空轮廓", *pick(rows, lambda p, f: (f[1] > 0.3) & (f[1] < 0.7) & (f[0] + f[1] > 0.97),
                              lambda p, f: p[2] - p[1], used)),
             # 水面：有太阳反光的图，Bloom 光晕在 720p 和 2K 下大小不同（单独见 bloom_mismatch.jpg），这里只挑普通波纹
             ("浅水", *pick(rows, lambda p, f: (f[2] > 0.97) & (p[0] < 30), lambda p, f: -p[0], used, skip=has_glare)),
             ("镜子", *pick(rows, lambda p, f: f[3] > 0.9, lambda p, f: -p[0], used))]
    for label, k, (y, x), p in picks:
        print(f"{label}: test {ds.ids[k]:04d} 块 (x={x}, y={y})  双三次 {p[0]:.2f} / A {p[1]:.2f} / B {p[2]:.2f}")
    print_glare_split(ds, rows)

    # 误差图：水面占 15% 以上、不透明物体占 40% 以上的图里，B 比双三次整图提升最多的一张
    cand = [r for r in rows if r["share"][2] > 0.15 and r["share"][0] > 0.4]
    err_k = max(cand, key=lambda r: r["whole"][2] - r["whole"][0])["k"]
    print(f"误差图: test {ds.ids[err_k]:04d}  整图 双三次 {rows[err_k]['whole'][0]:.2f} / B {rows[err_k]['whole'][2]:.2f}")

    os.makedirs(OUT, exist_ok=True)
    crops_figure(ds, nets, dev, picks).save(os.path.join(OUT, "compare_crops.jpg"), quality=92)
    error_figure(ds, nets, dev, err_k).save(os.path.join(OUT, "error_maps.jpg"), quality=90)
    k229 = ds.ids.index(229)
    print(f"Bloom: test 0229  整图 双三次 {rows[k229]['whole'][0]:.2f} / B {rows[k229]['whole'][2]:.2f}")
    bloom_figure(ds, nets, dev, k229, (0, 0, 2560, 1440)).save(os.path.join(OUT, "bloom_mismatch.jpg"), quality=90)
    print("写好了：docs/compare_crops.jpg、docs/error_maps.jpg、docs/bloom_mismatch.jpg")


if __name__ == "__main__":
    main()
