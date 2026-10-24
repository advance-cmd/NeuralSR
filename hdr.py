"""HDR 的两个空间（s193）。

网络在 log 空间里学：encode(x) = log(1 + x)。HDR 值从 0 到一百（太阳），亮的地方被压扁，暗的地方几乎不变。
评估在屏幕空间：tonemap() 和渲染器的色调映射 Pass（LearnDX12/HLSL/Tonemap.hlsl）一样，× 曝光 → ACES → sRGB 编码，
结果 0..1，就是屏幕上看到的（不加 Bloom：Bloom 会在超分之后再做）。
写完跑 python tools/check_hdr.py。
"""
import torch

# 空1：曝光倍数。main.cpp 里 mExposureEV = 1，传给 Tonemap.hlsl 的是 exp2f(mExposureEV)
EXPOSURE = 2.0


def encode(x):
    """线性 HDR → 网络用的 log 空间"""
    x = torch.log1p(x.clamp(min=0))
    return x


def decode(y):
    """log 空间 → 线性 HDR（encode 的反函数）"""
    y = torch.expm1(y).clamp(min=0)
    return y


def tonemap(x):
    """线性 HDR → 屏幕上的 0..1（sRGB 编码），和 Tonemap.hlsl 的 ACESFitted + LinearToSRGB 一样"""
    # 负数夹成 0，再乘曝光
    x = x.clamp(min=0)*EXPOSURE
    # ACES 拟合曲线（Tonemap.hlsl 的 ACESFitted，a = 2.51、b = 0.03、c = 2.43、d = 0.59、e = 0.14），saturate → 夹到 0..1
    x = (x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14)
    y = x.clamp(0, 1)
    # 已写好：sRGB 曲线里亮的那一段要开 1/2.4 次方。先把 0 夹成 1e-8：0 的小数次方，反向传播时导数是无穷大，会变成 NaN
    y_pow = y.clamp(min=1e-8) ** (1 / 2.4)
    # 线性 → sRGB 编码（Tonemap.hlsl 的 LinearToSRGB）：判断条件和暗的那一段用 y，亮的那一段的 pow(c, 1/2.4) 就是 y_pow
    return torch.where(y <= 0.0031308,12.92*y,1.055*y_pow-0.055)
