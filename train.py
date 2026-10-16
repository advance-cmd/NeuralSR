"""第二天：训练。python train.py A（只用颜色）或 python train.py B（颜色 + G-Buffer）"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from bicubic import psnr
from dataset import SRDataset
from model import SRNet

STEPS = 20000  # 一共训练多少步
LEARNING_RATE = 1e-3  # 学习率（不叫 lr，免得和低分辨率图 lr 混）
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")  # 训练好的权重存这里


def evaluate(model, dev):
    """测试集 40 张整图的平均 PSNR"""
    model.eval()
    scores = []
    with torch.no_grad():
        for lr, gb, hr in DataLoader(SRDataset("test", patch=None), batch_size=1):
            lr, gb, hr = lr.to(dev), gb.to(dev), hr.to(dev)
            sr = model(lr,gb).clamp(min=0, max=1)
            scores.append(psnr(sr, hr).item())
    model.train()
    return sum(scores) / len(scores)


if __name__ == "__main__":
    name = sys.argv[1]  # "A" 或 "B"
    dev = torch.device("cuda")
    torch.manual_seed(0)  # 固定随机数，两次训练的初始权重一样
    model = SRNet(name == "B").to(dev)  # 建网络并搬到 GPU：B 才用 G-Buffer
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(SRDataset("train"), batch_size=16, shuffle=True, num_workers=8,
                        drop_last=True, persistent_workers=True)
    os.makedirs(RUNS, exist_ok=True)
    step, total, t0 = 0, 0.0, time.time()
    while step < STEPS:
        for lr, gb, hr in loader:
            lr, gb, hr = lr.to(dev), gb.to(dev), hr.to(dev)
            sr = model(lr,gb)  # 前向：网络的输出
            loss = (sr-hr).abs().mean()  # L1：差的绝对值的平均
            opt.zero_grad()  # 清掉上一步留下的梯度
            loss.backward()  # 反向传播：算出每个权重的梯度
            opt.step()  # 按梯度把每个权重挪一小步
            step += 1
            total += loss.item()
            if step % 200 == 0:
                print(f"第 {step} 步  loss {total / 200:.5f}  {time.time() - t0:.0f} 秒", flush=True)
                total = 0.0
            if step % 5000 == 0:
                print(f"  测试集 PSNR {evaluate(model, dev):.2f} dB（双三次 31.37）", flush=True)
            if step == STEPS:
                break
    torch.save(model.state_dict(), os.path.join(RUNS, name + ".pth"))
    print(f"存好了：runs/{name}.pth，共 {time.time() - t0:.0f} 秒")
