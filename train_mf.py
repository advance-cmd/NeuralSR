"""s194：多帧网络的训练。python train_mf.py → runs/mf.pth（约 30 分钟），日志自己重定向到 runs/mf.log。
和 train_hdr.py 的差别：一个样本是同一段里连续 FRAMES 帧；每帧把上一帧的输出拉过来当输入（multiframe.clip_loss）；评估整段跑。
写完先跑 python tools/check_mf.py 3，过了再训练。
"""
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import SeqDataset
from eval_mf import sequence_psnr
from model import MFNet
from multiframe import clip_loss

STEPS = 10000  # 一步 = 16 个样本 × FRAMES 帧，每步的帧数是 train_hdr 的 4 倍
FRAMES = 4  # 一个样本几帧
LEARNING_RATE = 1e-3
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

if __name__ == "__main__":
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = MFNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(SeqDataset("train", frames=FRAMES), batch_size=16,num_workers=8, shuffle=True, drop_last=True)
    os.makedirs(RUNS, exist_ok=True)
    step,total,t0=0,0.0,time.time()
    while step < STEPS:
        for lr,gb,hr,velocity,jitter in loader:
            lr,gb,hr,velocity,jitter = lr.to(dev),gb.to(dev),hr.to(dev),velocity.to(dev),jitter.to(dev)
            loss = clip_loss(model, lr, gb, hr, velocity, jitter)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            total +=loss.item()
            if step % 500 == 0:
                print(f"第 {step} 步  loss {total / 500:.5f}  {time.time() - t0:.0f} 秒", flush=True)
                total = 0.0
            if step % 5000 == 0:
                print(f"  测试集 4 段整段（128 帧）屏幕空间 PSNR {sequence_psnr(model, dev):.2f} dB", flush=True)
            if step == STEPS:
                break
    torch.save(model.state_dict(), os.path.join(RUNS, "mf.pth"))
    print(f"存好了：runs/mf.pth，共 {time.time() - t0:.0f} 秒")