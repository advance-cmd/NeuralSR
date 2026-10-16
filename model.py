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


if __name__ == "__main__":
    lr = torch.rand(2, 3, 64, 64)
    gb = torch.rand(2, 9, 128, 128)
    for use_gb in (False, True):
        m = SRNet(use_gb)
        n = sum(p.numel() for p in m.parameters())
        print("模型", "B" if use_gb else "A", "输出", tuple(m(lr, gb).shape), "参数", n)
