"""第一天：读数据集。每个样本返回 (lr, gb, hr) 三个张量，都是 (C, H, W) float32，值 0..1。
数据格式见 data/README.md。"""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def to_tensor(a):
    """(H, W, C) uint8 的 numpy 数组 → (C, H, W) float32 张量，值 0..1"""
    t = torch.from_numpy(np.array(a))  # np.array：把 mmap 里这一块真正读进内存
    return 空1


class SRDataset(Dataset):
    def __init__(self, split, patch=64, repeat=16):
        """split：'train' 或 'test'；patch：LR 裁块的边长，None = 整张图；repeat：一轮里每张图裁几块"""
        self.dir = os.path.join(DATA, split)
        self.ids = sorted(int(f[:4]) for f in os.listdir(self.dir) if f.endswith("_lr.npy"))  # 这个划分里的图片编号
        self.patch = patch
        self.repeat = repeat if patch else 1

    def __len__(self):
        return 空2

    def __getitem__(self, i):
        base = os.path.join(self.dir, f"{self.ids[空3]:04d}_")
        load = lambda name: np.load(base + name + ".npy", mmap_mode="r")  # 只映射文件，还没读
        lr = load("lr")  # (720, 1280, 3)
        p = self.patch
        if p:
            y = random.randint(0, 空4)  # LR 块的左上角
            x = random.randint(0, 空5)
            lr = lr[y:y + p, x:x + p]
            Y, X, P = 空6  # 对应的 HR 块：左上角和边长
            crop = lambda a: a[Y:Y + P, X:X + P]
        else:
            crop = lambda a: a  # 整张图，不裁
        hr = crop(load("hr"))
        depth = torch.from_numpy(np.array(crop(load("depth")), dtype=np.float32))  # (H, W)，天空 = 0
        inv_depth = 空7  # 1/深度，天空给 0
        gb = torch.cat([to_tensor(crop(load("albedo"))),  # 3 通道：基础色
                        to_tensor(crop(load("normal"))),  # 3：法线
                        to_tensor(crop(load("mr"))),  # 2：金属度、粗糙度
                        空8],  # 1：1/深度
                       dim=0)
        return to_tensor(lr), gb, to_tensor(hr)


if __name__ == "__main__":
    ds = SRDataset("train")
    lr, gb, hr = ds[0]
    print(len(ds), lr.shape, gb.shape, hr.shape, lr.dtype)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    lr, gb, hr = next(iter(loader))
    print(lr.shape, gb.shape, hr.shape)
