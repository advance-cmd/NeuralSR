"""Backward warping（Xiao et al. 2020）：用这一帧的运动矢量，到上一帧里取颜色，把上一帧"拉"到这一帧。

s193 对齐检查（tools/check_align.py）和 s194 的多帧网络都用它。写完跑 python tools/check_warp.py。
"""
import torch
import torch.nn.functional as F


def backward_warp(prev, velocity, jitter_prev, jitter_cur):
    """
    prev        (N, C, H, W) 上一帧（任意通道：颜色、深度、上一帧的网络输出都行）
    velocity    (N, 2, H, W) 这一帧的速度缓冲：这一帧的 uv − 上一帧的 uv（不含抖动）
    jitter_*    (N, 2)       两帧各自的抖动（像素，x 向右、y 向下：整幅画面往右下移了这么多）
    返回 warped (N, C, H, W) {上一帧被拉到这一帧以后的样子}、valid (N, 1, H, W) {上一帧的位置在屏幕里 = True}
    """
    n, c, h, w = prev.shape
    dev = prev.device
    ys, xs = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")  # 每个像素的行号、列号

    # 空1：每个像素中心的 uv（0..1）。通道 0 是 u（横向），通道 1 是 v（竖向）
    uv = torch.stack([(xs+0.5) / w, (ys+0.5)/h])[None]  # (1, 2, H, W)

    size = torch.tensor([w, h], device=dev, dtype=prev.dtype).view(1, 2, 1, 1)
    # 空2：两帧抖动的差，换成 uv
    dj = jitter_prev.view(n,2,1,1) - jitter_cur.view(n,2,1,1)
    dj /= size

    # 空3：这个像素上一帧在哪（uv）
    prev_uv = uv + dj - velocity

    # 空4：grid_sample 要的坐标：范围 −1..1，形状 (N, H, W, 2)
    grid = (prev_uv * 2 - 1).permute(0, 2, 3, 1)

    # 空5：在上一帧里双线性地取
    warped = F.grid_sample(prev, grid, mode="bilinear", padding_mode="border", align_corners=False)

    valid = ((prev_uv >= 0) & (prev_uv <= 1)).all(1, keepdim=True)
    return warped, valid
