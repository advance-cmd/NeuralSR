"""检查第二天的 model.py（和 train.py 里的 evaluate）。

用法：python tools/check_day2.py           只查 model.py
      python tools/check_day2.py eval      再查 train.py 的 evaluate（约 10 秒）
"""
import copy
import os
import re
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = [a for a in sys.argv[1:] if a != "eval"]
sys.path.insert(0, args[0] if args else ROOT)
from model import SRNet  # noqa: E402

failed = 0


def check(ok, msg, hint=""):
    global failed
    print(("  通过  " if ok else "  没过  ") + msg + ("" if ok else f"\n        → {hint}"))
    failed += not ok
    return ok


def hint_for(err):
    if "except in dimension" in err:
        return "维度从 0 开始数：(B, C, H, W) 里 B 是第 0 维、C 第 1 维、H 第 2 维、W 第 3 维；按通道拼要沿 C 那一维"
    if "channels, but got" in err:
        return "第一层 Conv2d 的输入通道数，和实际拼出来的通道数对不上：颜色 3 + G-Buffer 9×4"
    if "permute" in err:
        return "forward 收到的 lr、gb 已经是 (B, C, H, W) 的 float 张量了，不要再过 to_tensor"
    if "pixel_shuffle" in err:
        return "PixelShuffle(2) 要 4 的倍数个通道：最后一层要输出 3×4 = 12 个"
    m = re.search(r"tensor a \((\d+)\) must match the size of tensor b \((\d+)\)", err)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a == 2 * b:
            return "detail 还是低分辨率：body 的输出要再过 shuffle 才变成 2 倍大"
        if 0 < a - b <= 16:
            return "卷积把图变小了，补的圈数少了：窗口中心到窗口边缘有几格，就补几圈（3×3 补 1 圈）"
        if 0 < b - a <= 16:
            return "卷积把图变大了，补的圈数多了：窗口中心到窗口边缘有几格，就补几圈（3×3 补 1 圈）"
    return "形状对不上，看报错里的数字"


def run(m, lr, gb):
    try:
        return m(lr, gb), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"


def main():
    torch.manual_seed(0)
    lr = torch.rand(2, 3, 48, 80)
    gb = torch.rand(2, 9, 96, 160)
    up = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
    for use_gb, name, want in ((False, "A", 24396), (True, "B", 53196)):
        print(f"[模型 {name}]")
        m = SRNet(use_gb)
        n = sum(p.numel() for p in m.parameters())
        first, last = m.body[0], m.body[-1]
        check(n == want, f"参数 {n}",
              f"应该是 {want}。第一层 Conv2d 的输入通道是 {first.in_channels}（应为 {39 if use_gb else 3}），"
              f"最后一层的输出通道是 {last.out_channels}（应为 12）")
        out, err = run(m, lr, gb)
        if not check(err is None, "前向能跑通", f"{err}\n        → {hint_for(err or '')}"):
            continue
        check(tuple(out.shape) == (2, 3, 96, 160), f"输出形状 {tuple(out.shape)}",
              "输入 (2, 3, 48, 80) 应该输出 (2, 3, 96, 160)。窗口中心到窗口边缘有几格，就补几圈")
        z = copy.deepcopy(m)
        with torch.no_grad():
            for p in z.parameters():
                p.zero_()
        zo, _ = run(z, lr, gb)
        check(zo is not None and torch.allclose(zo, up, atol=1e-6), "权重全为 0 时输出 = 双三次放大（残差）",
              "返回值应该是 双三次放大 + 补充细节；权重全 0 时细节是 0")
        out2, _ = run(m, lr, gb + 0.5)
        changed = out2 is not None and not torch.allclose(out, out2)
        if use_gb:
            check(changed, "改了 G-Buffer，输出跟着变", "模型 B 要把 G-Buffer 拼进输入")
        else:
            check(not changed, "改了 G-Buffer，输出不变", "模型 A 只用颜色，不该用到 gb")

    if "eval" in sys.argv[1:]:
        print("[train.py 的 evaluate]")
        from train import evaluate
        m = SRNet(True).cuda()
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
        s = evaluate(m, torch.device("cuda"))
        check(abs(s - 31.366) < 0.01, f"权重全 0 的模型 B：{s:.3f} dB（应该和双三次的 31.37 一样）",
              "比 31.37 低一点点：多半是忘了 clamp(0, 1)" if 31.2 < s < 31.366 else "看看 sr 是不是 model(lr, gb) 再截到 0..1")
    print("全部通过" if failed == 0 else f"{failed} 项没过")


if __name__ == "__main__":
    main()
