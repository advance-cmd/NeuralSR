"""第二天：超分网络。use_gb=False 是模型 A（只用颜色），True 是模型 B（颜色 + G-Buffer）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SRNet(nn.Module):
    def __init__(self, use_gb=False, c=32):
        """use_gb：要不要用 G-Buffer；c：中间每层几个通道（几张特征图）"""
        super().__init__()  # nn.Module 的规矩：先调父类的 __init__
        self.use_gb = use_gb
        cin = 39 if self.use_gb else 3  # 第一层的输入通道数
        self.unshuffle = nn.PixelUnshuffle(2)  # (B, 9, 2H, 2W) → (B, 36, H, W)
        self.body = nn.Sequential(
            nn.Conv2d(cin, c, 5, padding=2), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c, 12, 3, padding=1))  # 最后一层：给 PixelShuffle 准备
        self.shuffle = nn.PixelShuffle(2)  # (B, 12, H, W) → (B, 3, 2H, 2W)

    def forward(self, lr, gb=None):
        """lr：(B, 3, H, W)；gb：(B, 9, 2H, 2W)；返回 (B, 3, 2H, 2W)"""
        if self.use_gb:
            x = torch.cat(
                [lr,
                 self.unshuffle(gb)],
                dim=1
            )  # 颜色和 G-Buffer 按通道拼起来
        else:
            x = lr
        detail = self.shuffle(self.body(x))  # 网络算出的"补充细节"：(B, 3, 2H, 2W)
        up = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
        return up + detail  # 双三次放大 + 补充细节

class MFNet(nn.Module):
    """s194 多帧：模型 B 的输入再加三样——
    拉过来的上一帧输出（log 空间）、
    遮挡掩码、
    这一帧的抖动。
        整个网络还是在 720p 的格子上算：2K 的东西都先 PixelUnshuffle 成 4 倍通道"""
    def __init__(self,c=32):
        super().__init__()
        # 第一层的输入通道数：颜色 3 + G-Buffer 9×4 + 上一帧输出 3×4 + 掩码 1×4 + 抖动 2
        cin = 3 + 4*9 + 3*4 + 1*4 + 2
        self.unshuffle = nn.PixelUnshuffle(2)
        # LeakyReLU：负数乘 0.1 而不是变成 0。用 ReLU 时第一层 32 个通道有 20 个死掉（2500 步 32.65 dB），换了以后 33.96 dB
        self.body = nn.Sequential(
            nn.Conv2d(cin, c, 5, padding=2), nn.LeakyReLU(0.1),
            nn.Conv2d(c,c,3,padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(c,c,3,padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(c,12,3,padding=1),
        )
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, lr, gb, hist, mask, jitter):
        """lr (B, 3, H, W)、hist (B, 3, 2H, 2W) 都在 log 空间；gb (B, 9, 2H, 2W)；mask (B, 1, 2H, 2W)；jitter (B, 2)。返回 (B, 3, 2H, 2W)"""
        n,c,h,w = lr.shape
        # 抖动是每帧两个数，铺满成两张 (h, w) 的图，才能和别的按通道拼
        j = jitter.view(n,2,1,1).expand(n,2,h,w)
        x = torch.cat([lr,self.unshuffle(gb),self.unshuffle(hist),self.unshuffle(mask),j],dim=1)
        detail = self.shuffle(self.body(x))
        up = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
        return up + detail

if __name__ == "__main__":
    lr = torch.rand(2, 3, 64, 64)
    gb = torch.rand(2, 9, 128, 128)
    for use_gb in (False, True):
        m = SRNet(use_gb)
        n = sum(p.numel() for p in m.parameters())
        print("模型", "B" if use_gb else "A", "输出", tuple(m(lr, gb).shape), "参数", n)
