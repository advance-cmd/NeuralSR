"""拼一张数据预览图：每行一个机位，列 = 720p 输入（最近邻放大）/ 2K 目标 / 基础色 / 法线 / 深度 / 金属度·粗糙度。"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 480, 270


def tile(img):
    return img.resize((W, H), Image.BILINEAR)


def depth_vis(z):
    v = np.zeros(z.shape, np.uint8)
    m = z > 0
    lz = np.log(z[m].astype(np.float32))
    v[m] = (255 * (1 - (lz - lz.min()) / max(lz.max() - lz.min(), 1e-6))).astype(np.uint8)  # 近亮远暗，天空黑
    return Image.fromarray(v).convert("RGB")


def main(samples):
    cols = ["lr (input)", "hr (target)", "albedo", "normal", "depth", "mr"]
    sheet = Image.new("RGB", (W * len(cols), (H + 22) * len(samples) + 22), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for c, name in enumerate(cols):
        draw.text((c * W + 6, 5), name, fill=(230, 230, 230), font=font)
    for r, (split, i) in enumerate(samples):
        p = lambda n: os.path.join(ROOT, "data", split, f"{i:04d}_{n}")
        lr = Image.open(p("lr.png"))
        imgs = [lr.resize((2560, 1440), Image.NEAREST), Image.open(p("hr.png")), Image.open(p("albedo.png")),
                Image.open(p("normal.png")), depth_vis(np.load(p("depth.npy"))), Image.open(p("mr.png"))]
        y = 22 + r * (H + 22)
        draw.text((6, y + H + 4), f"{split} {i:04d}", fill=(230, 230, 230), font=font)
        for c, im in enumerate(imgs):
            sheet.paste(tile(im.convert("RGB")), (c * W, y))
    out = os.path.join(ROOT, "data", "preview.jpg")
    sheet.save(out, quality=88)
    print(out)


if __name__ == "__main__":
    args = sys.argv[1:] or ["train:0", "train:57", "test:205", "test:234"]
    main([(a.split(":")[0], int(a.split(":")[1])) for a in args])
