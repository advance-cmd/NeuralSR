"""s194：多帧网络每一帧要做的事——把上一帧的输出拉到这一帧、找出新露出来的区域（遮挡掩码），再跑网络。
训练（train_mf.py）和评估（eval_mf.py）都用这里的函数。写完跑 python tools/check_mf.py 2
"""
import torch
import torch.nn.functional as F

from hdr import encode
from warp import backward_warp

DEPTH_TOL = 0.1  # 拉过来的深度和这一帧的差超过 10%，就当这里上一帧被别的东西挡着
LAMBDA = 0.5  # 一致性损失的权重（s195）。对照（各 5000 步）：0 → 0.5 时 PSNR +0.23 dB、闪烁 −17%；λ = 2 递归发散（第 12 段输出越滚越大，出 nan）

def history_inputs(prev,velocity,gb):
    """prev：上一帧留下的 (输出, 1/深度)，形状 (N, 3, 2h, 2w)、(N, 1, 2h, 2w)，输出在 log 空间；第一帧是 None。
        velocity (N, 2, h, w)：这一帧的速度（uv）；gb (N, 9, 2h, 2w)：这一帧的 G-Buffer。
        返回 hist (N, 3, 2h, 2w)：拉到这一帧的上一帧输出；mask (N, 1, 2h, 2w)：1 = 历史能用，0 = 新露出来的 / 上一帧在屏幕外"""
    n,c,H,W = gb.shape
    if prev is None:
        return torch.zeros(n,3,H,W,device=gb.device),torch.zeros(n,1,H,W,device=gb.device) # 第一帧没有上一帧：历史全给 0，掩码也全是 0（"这里没有能用的历史"）
    out, prev_invDepth = prev
    # 速度是720p的图，双线性放大到2K
    v2 = F.interpolate(velocity,scale_factor=2,mode="bilinear",align_corners=False)
    # 这里拉的不是原始颜色，而是网络上一帧的输出 out。网络是朝着 GT 训练的，而 GT 是"在固定的像素中心上，平均 64 次抖动"得到的，本身不偏。
    zero = torch.zeros(n,2,device=gb.device)
    hist, valid = backward_warp(out,v2,zero,zero)
    # 上一帧的 1/深度也用同一个速度拉过来，和这一帧的 1/深度（G-Buffer 最后一个通道）比
    prev_d , valid_d = backward_warp(prev_invDepth,v2,zero,zero)
    cur_d = gb[:,8:9]
    # 两个 1/深度的相对差 ≤ DEPTH_TOL，才算同一个表面（天空两边都是 0，也要算）
    same = (prev_d - cur_d).abs() <= DEPTH_TOL * torch.maximum(prev_d, cur_d)
    # 历史能用 = 上一帧在屏幕里，并且是同一个表面；变成 0/1 的 float
    return hist, (valid & same).float()

def run_frame(model,prev,lr,gb,velocity,jitter):
    """跑一帧。lr 是线性 HDR。返回 out（log 空间）、留给下一帧的 prev(这一帧的输出 out, 这一帧的 1/深度 gb[:, 8:9])、这一帧的 mask"""
    hist, mask = history_inputs(prev,velocity,gb)
    out = model(encode(lr),gb,hist,mask,jitter)
    # 留给下一帧的 prev：这一帧的输出，和这一帧的 1/深度
    return out,(out,gb[:,8:9]),mask

def clip_loss(model,lr,gb,hr,velocity,jitter):
    """一个样本 K 帧：张量的第 1 维是第几帧（第 0 维是 batch）。从"没有历史"开始一帧一帧往下跑。
        损失 = (每帧 log 空间 L1 + LAMBDA × 每对相邻帧的一致性损失) / 帧数
        一致性损失（s195）：网络两帧之间的变化 out_k − 拉过来的 out_{k−1}，要和 GT 的变化 gt_k − 拉过来的 gt_{k−1} 一样；只算掩码 = 1 的地方"""
    prev, loss, tloss = None, 0, 0
    K = lr.shape[1]
    for k in range(K):
        prev_out = prev[0] if prev is not None else None
        out,prev,mask = run_frame(model,prev,lr[:,k],gb[:,k],velocity[:,k],jitter[:,k])
        target = encode(hr[:, k])  # gt_k，也换到 log 空间：(16, 3, 128, 128)
        loss = loss + (out-target).abs().mean()
        if k > 0:  # 第 0 帧没有上一帧
            v2 = F.interpolate(velocity[:,k],scale_factor=2,mode="bilinear",align_corners=False)  # 这一帧的速度放大到 2K
            zero = torch.zeros(lr.shape[0],2,device=lr.device)
            out_w,out_w_valid = backward_warp(prev_out,v2,zero,zero)    # 拉过来的 out_{k−1}
            gt_w,gt_w_valid = backward_warp(encode(hr[:,k-1]),v2,zero,zero) # 拉过来的 gt_{k−1}
            d_out, d_gt = out - out_w, target - gt_w
            tloss = tloss + (mask * (d_out - d_gt).abs()).mean()  # mask：(16, 1, 128, 128)，广播到 RGB 三个通道
    return (loss + LAMBDA * tloss) / K  # 循环后：两项按 λ 加起来，再除以帧数