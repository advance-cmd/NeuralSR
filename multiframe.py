"""s194：多帧网络每一帧要做的事——把上一帧的输出拉到这一帧、找出新露出来的区域（遮挡掩码），再跑网络。
训练（train_mf.py）和评估（eval_mf.py）都用这里的函数。写完跑 python tools/check_mf.py 2
"""
import torch
import torch.nn.functional as F

from hdr import encode
from warp import backward_warp

DEPTH_TOL = 0.1  # 拉过来的深度和这一帧的差超过 10%，就当这里上一帧被别的东西挡着

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
        损失 = 每帧 log 空间 L1 的平均"""
    prev, loss = None,0
    K = lr.shape[1]
    for k in range(K):
        #取出第 k 帧（第 1 维的第 k 个）
        out,prev,mask = run_frame(model,prev,lr[:,k],gb[:,k],velocity[:,k],jitter[:,k])
        loss = loss + (out-encode(hr[:,k])).abs().mean()
    return loss / K