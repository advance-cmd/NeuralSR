"""s193：单帧 HDR 基线的训练。python train_hdr.py A（只用颜色）或 python train_hdr.py B（颜色 + G-Buffer）
→ runs/hdr_A.pth、runs/hdr_B.pth，日志自己重定向到 runs/hdr_A.log 这样的文件。
和 train.py 只差三处：数据换成 HDRDataset；颜色进网络前 encode、出来 decode（网络在 log 空间里学）；评估在屏幕空间。
写完先跑 python tools/check_train_hdr.py，过了再训练。
"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from bicubic import psnr
from dataset import HDRDataset
from hdr import decode, encode, tonemap
from model import SRNet

STEPS = 20000  # 一共训练多少步
LEARNING_RATE = 1e-3  # 学习率（不叫 lr，免得和低分辨率图 lr 混）
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")  # 训练好的权重存这里


def step_loss(model, lr, gb, hr):
    """一步训练的损失。lr、hr 是线性 HDR，gb 是 0..1 的 G-Buffer。单独写成函数，检查脚本才能测"""
    # 空1：网络的输入、输出都在 log 空间：颜色先 encode 再进网络（G-Buffer 本来就在 0..1，不用 encode）
    sr = model(encode(lr), gb)
    # 空2：L1 也在 log 空间算：网络的输出和"目标的 log"比
    return (sr -encode(hr)).abs().mean()


def evaluate(model, dev):
    """测试集每 4 帧取 1 帧（4 段 × 8 帧 = 32 帧整图），屏幕空间 PSNR 的平均"""
    model.eval()
    scores = []
    with torch.no_grad():
        # 空3：测试集、整帧（不裁块）、每 4 帧取 1 帧
        for lr, gb, hr in DataLoader(HDRDataset("test", patch=None, every=4), batch_size=1, num_workers=4):
            lr, gb, hr = lr.to(dev), gb.to(dev), hr.to(dev)
            # 空4：和空1 一样，颜色先 encode 再进网络；网络的输出在 log 空间 → decode 换回线性 HDR
            sr = decode(model(encode(lr), gb))
            # 空5：在屏幕空间算 PSNR（bicubic.py 的 psnr 要两张 0..1 的图）
            scores.append(psnr(tonemap(sr),tonemap(hr)).item())
    model.train()
    return sum(scores) / len(scores)


if __name__ == "__main__":
    name = sys.argv[1]  # "A" 或 "B"
    dev = torch.device("cuda")
    torch.manual_seed(0)  # 固定随机数，两次训练的初始权重一样
    model = SRNet(name == "B").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 空6：训练集（64×64 的块，默认参数就行）
    loader = DataLoader(HDRDataset("train"), batch_size=16, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
    os.makedirs(RUNS, exist_ok=True)
    step, total, t0 = 0, 0.0, time.time()
    while step < STEPS:
        for lr, gb, hr in loader:
            lr, gb, hr = lr.to(dev), gb.to(dev), hr.to(dev)
            loss = step_loss(model, lr, gb, hr)
            opt.zero_grad()  # 清掉上一步留下的梯度
            loss.backward()  # 反向传播：算出每个权重的梯度
            opt.step()  # 按梯度把每个权重挪一小步
            step += 1
            total += loss.item()
            if step % 1000 == 0:
                print(f"第 {step} 步  loss {total / 1000:.5f}  {time.time() - t0:.0f} 秒", flush=True)
                total = 0.0
            if step % 5000 == 0:
                print(f"  测试集（32 帧）屏幕空间 PSNR {evaluate(model, dev):.2f} dB（权重全 0 = 双三次 31.25）", flush=True)
            if step == STEPS:
                break
    torch.save(model.state_dict(), os.path.join(RUNS, f"hdr_{name}.pth"))
    print(f"存好了：runs/hdr_{name}.pth，共 {time.time() - t0:.0f} 秒")
