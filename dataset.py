"""第一天：读数据集。每个样本返回 (lr, gb, hr) 三个张量，都是 (C, H, W) float32，值 0..1。
数据格式见 data/README.md。"""
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def to_tensor(a):
    """(H, W, C) uint8 的 numpy 数组 → (C, H, W) float32 张量，值 0..1"""
    t = torch.from_numpy(np.array(a))  # np.array：把 mmap 里这一块真正读进内存
    return t.permute(2, 0, 1).float() / 255

def to_tensor_Hdr(a):
    """(H, W, C) float16 的 HDR 数组 → (C, H, W) float32 张量，值不变（HDR 不除以 255）"""
    t = torch.from_numpy(np.array(a))  # np.array：把 mmap 里这一块真正读进内存
    return t.permute(2, 0, 1).float()

class SRDataset(Dataset):
    def __init__(self, split, patch=64, repeat=16):
        """split：'train' 或 'test'；patch：LR 裁块的边长，None = 整张图；repeat：一轮里每张图裁几块"""
        self.dir = os.path.join(DATA, split)
        self.ids = sorted(int(f[:4]) for f in os.listdir(self.dir) if f.endswith("_lr.npy"))  # 这个划分里的图片编号
        self.patch = patch
        self.repeat = repeat if patch else 1

    def __len__(self):
        return len(self.ids)*self.repeat

    def __getitem__(self, i):
        base = os.path.join(self.dir, f"{self.ids[i%len(self.ids)]:04d}_")
        load = lambda name: np.load(base + name + ".npy", mmap_mode="r")  # 只映射文件，还没读
        lr = load("lr")  # (720, 1280, 3)
        p = self.patch
        if p:
            y = random.randint(0, lr.shape[0] - p)  # LR 块的左上角
            x = random.randint(0, lr.shape[1] - p)
            lr = lr[y:y + p, x:x + p]
            Y, X, P = y*2,x*2,p*2  # 对应的 HR 块：左上角和边长
            crop = lambda a: a[Y:Y + P, X:X + P]
        else:
            crop = lambda a: a  # 整张图，不裁
        hr = crop(load("hr"))
        depth = torch.from_numpy(np.array(crop(load("depth")), dtype=np.float32))  # (H, W)，天空 = 0
        inv_depth = torch.where(depth > 0,1/depth,0)  # 1/深度，天空给 0
        gb = torch.cat([to_tensor(crop(load("albedo"))),  # 3 通道：基础色
                        to_tensor(crop(load("normal"))),  # 3：法线
                        to_tensor(crop(load("mr"))),  # 2：金属度、粗糙度
                        inv_depth[None]],  # 1：1/深度
                       dim=0)
        return to_tensor(lr), gb, to_tensor(hr)


SEQ = os.path.join(DATA, "seq")


class HDRDataset(Dataset):
    """多帧数据集里的每一帧单独当一个样本（s193 的单帧 HDR 基线）。返回 (lr, gb, hr)：
    lr (3, 720, 1280) 带抖动的线性 HDR；gb 和 SRDataset 一样的 9 通道 2K G-Buffer；hr (3, 1440, 2560) GT（2K、64 次抖动平均，线性 HDR）
    文件名和内容见 data/README.md 最后一节"多帧数据集"。写完跑 python tools/check_hdrdata.py"""

    def __init__(self, split, patch=64, repeat=16, every=1):
        """every：每几帧取一帧（评估时用 every=4 省时间）"""
        self.dir = os.path.join(SEQ, split)
        # 空1：这个划分里每一帧的编号，是"段_帧"这样的字符串，比如 "12_07"。按 GT 文件列出来、排好序，再每 every 帧取一帧
        self.ids = sorted(f[:5] for f in os.listdir(self.dir) if f.endswith("_gt.npy"))[::every]
        self.patch = patch
        self.repeat = repeat if patch else 1

    def __len__(self):
        return len(self.ids) * self.repeat

    def __getitem__(self, i):
        # 空2：这一帧所有文件名的公共前缀，比如 ".../seq/test/12_07_"（编号已经是字符串了）
        base = os.path.join(self.dir, f"{self.ids[i%len(self.ids)]}_")
        load = lambda name: np.load(base + name + ".npy", mmap_mode="r")
        # 空3：输入用哪个文件（现在写的是单帧数据集的名字）
        lr = load("color")  # (720, 1280, 3) float16
        p = self.patch
        if p:
            y = random.randint(0, lr.shape[0] - p)
            x = random.randint(0, lr.shape[1] - p)
            lr = lr[y:y + p, x:x + p]
            Y, X, P = y * 2, x * 2, p * 2
            crop = lambda a: a[Y:Y + P, X:X + P]
        else:
            crop = lambda a: a
        # 空4：HDR 的 numpy 数组 → (C, H, W) float32 张量。现在是照抄 to_tensor，它对 HDR 多做了一步
        hdr = lambda a: to_tensor_Hdr(a)
        # 空5：2K 的深度用哪个文件（现在写的是单帧数据集的名字，这个划分里也有同名文件，但尺寸不对）
        depth = torch.from_numpy(np.array(crop(load("hrdepth")), dtype=np.float32))
        inv_depth = torch.where(depth > 0, 1 / depth, 0)
        gb = torch.cat([to_tensor(crop(load("albedo"))), to_tensor(crop(load("normal"))), to_tensor(crop(load("mr"))),
                        inv_depth[None]], dim=0)
        # 空6：目标用哪个文件（现在写的是单帧数据集的名字）
        return hdr(lr), gb, hdr(crop(load("gt")))

SEQ_LEN = 32 # 每段 32 帧

def jitters(split):
    """frames.csv → {"12_07": (jx, jy)}：这个划分里每一帧的抖动（720p 像素，x 向右、y 向下）"""
    rows = csv.DictReader(open(os.path.join(SEQ, "frames.csv"), encoding="utf-8"))
    return {f"{int(r['seq']):02d}_{int(r['frame']):02d}" : (float(r["jx"]),float(r["jy"])) for r in rows if r["split"] == split}

class SeqDataset(Dataset):
    """s194：同一段里连续 frames 帧、同一个裁块位置，当一个样本。返回 5 个张量，第 0 维都是"第几帧"（K = frames）：
        lr (K, 3, h, w) 带抖动的线性 HDR；
        gb (K, 9, 2h, 2w) 2K G-Buffer；
        hr (K, 3, 2h, 2w)
        GT；
        velocity (K, 2, h, w) 速度（已经换成这个块的 uv）；
        jitter (K, 2) 抖动（720p 像素）
        patch=None 是整帧：评估用 frames=1，按顺序就是每段从第 0 帧到第 31 帧。写完跑 python tools/check_mf.py 1"""
    def __init__(self,split,frames=4,patch=64,repeat=16):
        self.dir = os.path.join(SEQ, split)
        self.jitter = jitters(split)
        seqs = sorted({f[:2] for f in os.listdir(self.dir) if f.endswith("_gt.npy")})  # 段号，比如 "12"
        self.clips =[(seq, f) for seq in seqs for f in range(0,SEQ_LEN - frames + 1)]
        self.frames = frames
        self.patch = patch
        self.repeat = repeat if patch else 1

    def __len__(self):
        return len(self.clips)*self.repeat

    def __getitem__(self, i):
        s, start = self.clips[i % len(self.clips)]
        p = self.patch
        if p:
            y = random.randint(0, 720 - p)
            x = random.randint(0, 1280 - p)
            Y,X,P = y * 2,x * 2,p * 2
            crop = lambda a : a[y:y + p, x:x + p]
            crop2 = lambda a: a[Y:Y + P, X:X + P]
            scale = torch.tensor([1280 / p, 720 / p]).view(2,1,1)
        else:
            crop = crop2 = lambda a: a
            scale = torch.ones(2, 1, 1)
        lrs, gbs, hrs, vels, jits = [], [], [], [], []
        for f in range(start,start + self.frames):
            fid = f"{s}_{f:02d}"
            load = lambda name: np.load(os.path.join(self.dir, fid + "_" + name + ".npy"), mmap_mode="r")
            depth = torch.from_numpy(np.array(crop2(load("hrdepth")), dtype=np.float32))
            inv_depth = torch.where(depth > 0,1/depth,0)
            gbs.append(torch.cat([to_tensor(crop2(load("albedo"))), to_tensor(crop2(load("normal"))),
                                  to_tensor(crop2(load("mr"))), inv_depth[None]], dim=0))
            lrs.append(to_tensor_Hdr(crop(load("color"))))
            hrs.append(to_tensor_Hdr(crop2(load("gt"))))
            vels.append(to_tensor_Hdr(crop(load("velocity")))* scale)
            jits.append(torch.tensor(self.jitter[fid]))
        return torch.stack(lrs), torch.stack(gbs), torch.stack(hrs), torch.stack(vels), torch.stack(jits)

if __name__ == "__main__":
    ds = SRDataset("train")
    lr, gb, hr = ds[0]
    print(len(ds), lr.shape, gb.shape, hr.shape, lr.dtype)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    lr, gb, hr = next(iter(loader))
    print(lr.shape, gb.shape, hr.shape)
