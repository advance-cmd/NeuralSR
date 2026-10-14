# 数据集：G-Buffer 引导的 2× 超分

从自己的 DX12 渲染器（LearnDX12，提交 9a46769：延迟渲染 + SSAO + Bloom + ACES 色调映射）导出。
同一个机位渲两次：1280×720 一次（输入），2560×1440 一次（目标）。两次的帧序号相同，水面波浪也一样。

## 划分

| | 机位数 | 说明 |
|---|---|---|
| `train/` | 200 | 随机机位 |
| `test/` | 40 | 33 个随机机位 + 7 个 demo 参考机位（id 233–239，`cams.csv` 里名字以 p1–p7 开头） |

测试集的随机机位都在训练没见过的方向：以场景中心 (8, -25) 为圆心，眼睛的方位角在 0°–60° 的归测试集。
机位由 `tools/gen_cams.py` 按固定种子生成，目标点按物体加权抽（铁丝网、木箱、镜子、地砖、测试球、山坡、水面），
视线被山挡住的机位丢掉。

## 每个样本的文件（`{split}/{id:04d}_*`）

| 文件 | 尺寸 | 类型 | 内容 |
|---|---|---|---|
| `lr.png` | 1280×720 | RGB uint8 | **输入**：720p 渲染的成品画面（屏幕上看到的，sRGB 编码） |
| `hr.png` | 2560×1440 | RGB uint8 | **目标**：2K 渲染的成品画面 |
| `albedo.png` | 2560×1440 | RGB uint8 | G-Buffer0 的基础色（已乘贴图，sRGB 编码）；天空 = 0 |
| `normal.png` | 2560×1440 | RGB uint8 | 世界空间法线，存的是 `n * 0.5 + 0.5`；天空 = 0 |
| `mr.png` | 2560×1440 | RGB uint8 | R = 金属度，G = 粗糙度，B = 0；天空 = 0 |
| `depth.npy` | 2560×1440 | float16 | 观察空间深度（世界单位，近平面 1、远平面 1000）；天空 = 0 |

G-Buffer 和深度都是 2K 的：在 2K 下 G-Buffer Pass 只要约 0.25 ms，
贵的是着色（光照、SSAO 等），所以只有着色降到 720p。

像素对齐：720p 的像素 (x, y) 对应 2K 的 (2x, 2y)–(2x+1, 2y+1) 这 2×2 块。
`PixelShuffle(2)` 正好是这个对应关系。

## 训练用的 `.npy`

`tools/pack_npy.py` 把上面的 PNG 另存了一份不压缩的 `.npy`，放在 PNG 旁边（约 9 秒，多占约 10 GB）：

- `lr.npy`、`hr.npy`、`albedo.npy`、`normal.npy`：和 PNG 一样，(H, W, 3) uint8；
- `mr.npy`：只留两个通道，(H, W, 2) uint8，依次是金属度、粗糙度。

`np.load(路径, mmap_mode="r")` 打开时不读盘，切片后 `np.array(...)` 只读切出来的那一块。
实测：解一个样本的 5 张 PNG 约 200 ms；从 `.npy` 裁一个 128×128 的 HR 块（6 个文件）约 5 ms。

## 要注意的地方

- **G-Buffer 里只有不透明物体。** 水面、镜子里的倒影、天空都是前向画的，不在 G-Buffer 里。
  水面区域的 G-Buffer 是水底的地形，深度也是到水底的，和成品画面对不上。
  评估时可以按 `depth == 0`（天空）分开统计，水面区域也值得单独看。
- 只有一个场景。测试集换了方向，但物体都是训练里见过的。
- 成品画面已经过色调映射和 Bloom，是 0–255 的 sRGB 值；训练时一般除以 255 变成 0–1。

## 读一个样本

```python
import numpy as np
from PIL import Image

base = "data/train/0000_"
lr = np.asarray(Image.open(base + "lr.png"))        # (720, 1280, 3) uint8
hr = np.asarray(Image.open(base + "hr.png"))        # (1440, 2560, 3) uint8
depth = np.load(base + "depth.npy")                 # (1440, 2560) float16，天空 = 0
```

`preview.jpg` 是几组样本的预览（`python tools/preview.py train:0 test:205` 可以换别的样本）。

## 重新导出

导出用的是 LearnDX12 的临时副本，加了导数据补丁（`tools/export/`）：
每帧按 `cams.txt` 换一个机位，读回后台缓冲、两张 G-Buffer 和深度，写成原始 `.npy`；
然后 `python tools/convert_raw.py <2K 原始目录> <720p 原始目录>` 转成这里的格式。
两轮共 240 × 2 帧，导出约 45 秒，转换约 50 秒；调试层 0 错误 0 警告。
