# Troubleshooting Notes

本文档记录本项目在环境配置、数据读取、nnU-Net 预处理和训练过程中遇到的主要问题及解决方法。  
这些问题是医学影像分割项目中非常常见的工程问题，记录下来可以方便后续复现，也能帮助解释项目推进过程。

---

## 1. nibabel 读取 `._la_xxx.nii.gz` 报错

### 问题现象

在检查 MSD Heart 数据集时，运行：

```bash
python scripts/check_heart_dataset.py
```

出现类似报错：

```text
nibabel.filebasedimages.ImageFileError:
File data/raw/Task02_Heart/imagesTr/._la_029.nii.gz is not a gzip file
```

### 问题原因

数据目录中混入了 macOS 生成的 AppleDouble 隐藏文件，例如：

```text
._la_029.nii.gz
.DS_Store
```

这些文件虽然看起来像 `.nii.gz`，但它们不是合法的 NIfTI 医学影像文件。`nibabel` 在读取这些文件时，会把它们当成压缩 NIfTI 文件处理，因此报：

```text
is not a gzip file
```

### 解决方法

删除所有 `._*` 隐藏文件。

PowerShell：

```powershell
Get-ChildItem data\raw\Task02_Heart -Recurse -Filter "._*" | Remove-Item -Force
```

Command Prompt：

```cmd
del /s /q data\raw\Task02_Heart\._*
```

同时，在代码中读取 `.nii.gz` 文件时，要过滤掉 `._` 开头的隐藏文件：

```python
files = [
    f for f in os.listdir(folder)
    if f.endswith(".nii.gz") and not f.startswith("._")
]
```

### 经验总结

医学影像项目读取数据前，一定要先检查文件列表。不要只用：

```python
f.endswith(".nii.gz")
```

更稳妥的写法是：

```python
f.endswith(".nii.gz") and not f.startswith("._")
```

---

## 2. `nibabel` 和 `SimpleITK` 未安装

### 问题现象

运行环境检查脚本：

```bash
python scripts/check_env.py
```

出现：

```text
nibabel: import failed -> No module named 'nibabel'
SimpleITK: import failed -> No module named 'SimpleITK'
```

### 问题原因

环境中尚未安装医学影像读取库。`.nii.gz` 文件通常可用 `nibabel` 或 `SimpleITK` 读取。

### 解决方法

在当前 conda 环境中安装：

```bash
pip install nibabel SimpleITK
```

然后重新检查：

```bash
python scripts/check_env.py
```

### 经验总结

医学影像项目常用基础库包括：

```bash
pip install numpy scipy pandas matplotlib scikit-image scikit-learn tqdm nibabel SimpleITK
```

其中：

- `nibabel`：常用于读取 NIfTI 文件；
- `SimpleITK`：常用于医学影像读取、重采样和格式转换；
- `scipy`：用于计算 HD95 等距离指标；
- `matplotlib`：用于可视化 MRI、mask 和 error map。

---

## 3. PyTorch 没有调用到 GPU

### 问题现象

运行：

```bash
python scripts/check_env.py
```

如果输出：

```text
CUDA available: False
GPU: CPU only
```

说明 PyTorch 没有调用到 NVIDIA GPU。

### 问题原因

可能原因包括：

1. 安装的是 CPU 版本 PyTorch；
2. NVIDIA 驱动未正确安装；
3. CUDA 版本和 PyTorch 安装版本不匹配；
4. 当前环境不是项目使用的 conda 环境。

### 解决方法

先检查显卡和驱动：

```bash
nvidia-smi
```

如果能看到显卡型号和 CUDA Version，再安装 GPU 版 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

安装后检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

正常应看到：

```text
cuda available: True
NVIDIA GeForce RTX 4060 Laptop GPU
```

### 经验总结

医学影像 3D 分割非常吃显存和计算资源。如果 PyTorch 没有正确调用 GPU，后续 3D U-Net 训练和 nnU-Net 训练会非常慢甚至不可行。

---

## 4. 3D U-Net 输入输出维度错误

### 问题现象

自实现 3D U-Net 前向传播时，如果输入维度不正确，可能出现类似报错：

```text
Expected 5D input to conv3d
```

或者 skip connection 拼接时报：

```text
Sizes of tensors must match except in dimension 1
```

### 问题原因

3D U-Net 的输入必须是 5 维：

```text
[B, C, D, H, W]
```

例如：

```text
[1, 1, 64, 96, 96]
```

其中：

- `B`：batch size；
- `C`：通道数；
- `D`：深度方向；
- `H`：高度；
- `W`：宽度。

而原始医学影像从 nibabel 读取出来通常是：

```text
[H, W, D]
```

需要转换为：

```text
[D, H, W]
```

再加 channel 维度：

```text
[1, D, H, W]
```

最后 dataloader 自动加 batch 维度：

```text
[B, 1, D, H, W]
```

### 解决方法

在 Dataset 中使用：

```python
image = np.transpose(image, (2, 0, 1))
mask = np.transpose(mask, (2, 0, 1))

image_patch = np.expand_dims(image_patch, axis=0)
```

并在模型测试中确认：

```python
x = torch.randn(1, 1, 64, 64, 64).to(device)
y = model(x)

print(x.shape)
print(y.shape)
```

预期输出：

```text
Input shape: torch.Size([1, 1, 64, 64, 64])
Output shape: torch.Size([1, 2, 64, 64, 64])
```

### 经验总结

医学图像处理中最容易出错的是维度顺序。每一步都应该明确当前数据是：

```text
HWD
DHW
CDHW
BCDHW
```

---

## 5. `F.one_hot()` 参数拼写错误

### 问题现象

运行 Dice Loss 测试时出现：

```text
TypeError: one_hot() got an unexpected keyword argument 'num_classess'
```

### 问题原因

参数名拼写错误。错误写法：

```python
F.one_hot(targets.long(), num_classess=num_classes)
```

正确参数名是：

```python
num_classes
```

### 解决方法

改成：

```python
targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
```

### 经验总结

这种错误是代码实现过程中很常见的拼写错误。遇到 `unexpected keyword argument` 时，优先检查参数名是否拼错。

---

## 6. Patch-level Dice 很高，但 whole-volume Dice 很低

### 问题现象

自实现 3D U-Net 在 patch-level 验证中取得较高 Dice，例如：

```text
Patch-level Dice: 0.9573
```

但 whole-volume 推理时，初始结果较低：

```text
Whole-volume Raw Dice: 0.7369
Mean Raw Dice: 0.7329
```

甚至早期脚本中一度出现 Dice 为 0 的情况。

### 问题原因

Patch-level 验证和 whole-volume 推理不是同一个难度。

Patch-level 验证中，patch 通常围绕左心房采样：

```python
positive_ratio = 1.0
```

模型看到的是已经裁剪到目标附近的图像，因此任务较简单。

Whole-volume 推理中，模型需要在完整 MRI 中识别左心房位置，包含大量背景区域和相似组织结构，容易出现远处假阳性或漏分割。

### 解决方法

1. 实现 whole-volume sliding window inference；
2. 保存完整预测 mask；
3. 计算完整病例 Dice / IoU / HD95；
4. 通过 probability map、bbox 和 threshold sweep 诊断错误；
5. 使用最大连通区域后处理去除远处假阳性。

后处理前后结果：

```text
Mean Raw Dice: 0.7329
Mean Raw HD95: 162.87 mm

Mean LCC Dice: 0.8276
Mean LCC HD95: 11.85 mm
```

### 经验总结

医学影像分割不能只看 patch-level 指标。真正接近临床应用的是 whole-volume 推理结果。

---

## 7. Raw prediction 中存在远处假阳性

### 问题现象

whole-volume raw prediction 的 HD95 非常高：

```text
Mean Raw HD95: 162.87 mm
```

某些病例中 raw prediction 的 bbox 远大于真实 bbox，例如：

```text
GT bbox:
z: 45-110, y: 115-159, x: 151-209

Pred bbox:
z: 1-129, y: 89-223, x: 1-311
```

### 问题原因

模型在完整 MRI 背景区域中产生了远处假阳性。这些小块假阳性对 Dice 的影响可能有限，但对 HD95 影响极大，因为 HD95 衡量的是边界距离。

### 解决方法

使用最大连通区域后处理，只保留最大的 3D 连通区域：

```python
from scipy import ndimage

def keep_largest_connected_component(mask):
    mask = mask.astype(np.uint8)
    labeled, num_features = ndimage.label(mask)

    if num_features == 0:
        return mask

    component_sizes = ndimage.sum(
        mask,
        labeled,
        index=range(1, num_features + 1)
    )

    largest_component_id = int(np.argmax(component_sizes)) + 1
    largest_mask = (labeled == largest_component_id).astype(np.uint8)

    return largest_mask
```

后处理后：

```text
Mean Raw Dice: 0.7329 → Mean LCC Dice: 0.8276
Mean Raw HD95: 162.87 mm → Mean LCC HD95: 11.85 mm
```

### 经验总结

对于单个连续器官结构，最大连通区域后处理是非常有效的医学图像分割后处理方法。它不能替代模型能力，但能明显减少远处假阳性。

---

## 8. nnU-Net 预处理 blosc2 报错

### 问题现象

运行 nnU-Net 预处理：

```bash
nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
```

报错：

```text
RuntimeError: Could not build empty array
```

堆栈中出现：

```text
blosc2.asarray
Could not build empty array
```

### 问题原因

nnU-Net v2 在预处理阶段使用 `blosc2` 保存 `.b2nd` 文件。在 Windows 环境下，长路径、中文路径、多进程和 blosc2 组合可能导致写入失败。

原路径包含中文且较长：

```text
D:\医学AI_保研\项目\基于U-Net的心脏MRI分割实验和可视化分析\...
```

### 解决方法

将 nnU-Net workspace 移动到英文短路径：

```text
D:/nnunet_work/
```

创建目录：

```powershell
D:
mkdir D:\nnunet_work
mkdir D:\nnunet_work\nnUNet_raw
mkdir D:\nnunet_work\nnUNet_preprocessed
mkdir D:\nnunet_work\nnUNet_results
```

复制数据集：

```powershell
Copy-Item -Recurse -Force .\nnUNet_workspace\nnUNet_raw\Dataset002_Heart D:\nnunet_work\nnUNet_raw\Dataset002_Heart
```

设置环境变量：

```powershell
$Env:nnUNet_raw = "D:/nnunet_work/nnUNet_raw"
$Env:nnUNet_preprocessed = "D:/nnunet_work/nnUNet_preprocessed"
$Env:nnUNet_results = "D:/nnunet_work/nnUNet_results"
```

重新运行：

```bash
nnUNetv2_plan_and_preprocess -d 2 -c 3d_fullres --verify_dataset_integrity
```

### 经验总结

在 Windows 上跑 nnU-Net，建议优先使用英文短路径，避免中文路径、空格和过深目录结构。

推荐路径：

```text
D:/nnunet_work/
```

不推荐路径：

```text
D:/医学AI_保研/项目/很长的中文目录/...
```

---

## 9. nnU-Net 训练时 `WinError 1450`

### 问题现象

运行：

```bash
nnUNetv2_train 2 3d_fullres 0
```

报错：

```text
RuntimeError: One or more background workers are no longer alive.
OSError: [WinError 1450] 系统资源不足，无法完成请求的服务。
```

### 问题原因

nnU-Net 默认会使用多个后台 worker 进行数据增强和数据加载。在 Windows 环境下，多进程数据增强容易导致系统资源不足，尤其是在 3D 医学影像任务中。

### 解决方法

降低 nnU-Net 数据增强 worker 数量：

```powershell
$Env:nnUNet_n_proc_DA = "1"
$Env:nnUNet_def_n_proc = "1"
```

然后重新训练：

```bash
nnUNetv2_train 2 3d_fullres 0
```

如果仍然出错，可以进一步调低：

```powershell
$Env:nnUNet_n_proc_DA = "0"
$Env:nnUNet_def_n_proc = "1"
```

但 `nnUNet_n_proc_DA=0` 更适合 debug，不建议长期作为正式训练设置。

### 经验总结

Windows 下跑 nnU-Net 时，如果 dataloader worker 崩溃，优先尝试调低：

```powershell
$Env:nnUNet_n_proc_DA
$Env:nnUNet_def_n_proc
```

---

## 10. nnU-Net 没有内置 `nnUNetTrainer_50epochs`

### 问题现象

尝试导入 50 epoch trainer：

```bash
python -c "from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_50epochs import nnUNetTrainer_50epochs; print('OK')"
```

报错：

```text
ModuleNotFoundError:
No module named 'nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_50epochs'
```

### 问题原因

当前安装的 nnU-Net v2.6.0 中没有内置该 trainer 变体。

### 解决方法

自定义一个 50 epoch trainer。

新建文件：

```text
D:\conda_envs\medai\Lib\site-packages\nnunetv2\training\nnUNetTrainer\nnUNetTrainer_50epochs.py
```

写入：

```python
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_50epochs(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        self.num_epochs = 50
```

测试导入：

```bash
python -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_50epochs import nnUNetTrainer_50epochs; print(nnUNetTrainer_50epochs); print('OK')"
```

正常输出：

```text
<class 'nnunetv2.training.nnUNetTrainer.nnUNetTrainer_50epochs.nnUNetTrainer_50epochs'>
OK
```

然后训练：

```bash
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_50epochs
```

### 经验总结

nnU-Net 支持自定义 trainer。当默认训练轮数过长时，可以自定义短训 trainer，用于项目验证和对照实验。

---

## 11. 自定义 `nnUNetTrainer_50epochs` 使用 `*args, **kwargs` 报错

### 问题现象

最开始自定义 trainer 时使用了：

```python
class nnUNetTrainer_50epochs(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 50
```

训练时报错：

```text
KeyError: 'args'
```

### 问题原因

nnU-Net 内部会记录 trainer 的 `__init__` 参数。如果使用 `*args, **kwargs`，nnU-Net 在读取初始化参数时可能找不到期望的参数名，从而报：

```text
KeyError: 'args'
```

### 解决方法

不要使用 `*args, **kwargs`。必须按照当前版本的 `nnUNetTrainer.__init__()` 参数显式定义。

先查看当前版本接口：

```bash
python -c "import inspect; from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer; print(inspect.signature(nnUNetTrainer.__init__))"
```

然后按接口定义：

```python
class nnUNetTrainer_50epochs(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        self.num_epochs = 50
```

### 经验总结

自定义 nnU-Net trainer 时，不要随便用 `*args, **kwargs`。不同版本 nnU-Net 的 trainer 初始化参数可能不同，要先用 `inspect.signature()` 查看真实接口。

---

## 12. 自定义 trainer 中包含不存在的 `unpack_dataset` 参数

### 问题现象

修改 trainer 后再次训练，报错：

```text
TypeError: nnUNetTrainer.__init__() got an unexpected keyword argument 'unpack_dataset'
```

### 问题原因

不同 nnU-Net 版本的 `nnUNetTrainer.__init__()` 参数不同。当前使用的 nnU-Net v2.6.0 中没有 `unpack_dataset` 参数。

### 解决方法

删除 `unpack_dataset` 参数，使用当前版本真实接口：

```python
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_50epochs(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        self.num_epochs = 50
```

### 经验总结

不要照搬其他版本 nnU-Net 的 trainer 参数。先检查本地版本：

```bash
python -c "import inspect; from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer; print(inspect.signature(nnUNetTrainer.__init__))"
```

---

## 13. nnU-Net 提示 `No module named 'hiddenlayer'`

### 问题现象

训练时出现：

```text
Unable to plot network architecture:
No module named 'hiddenlayer'
```

### 问题原因

nnU-Net 尝试绘制网络结构图，但当前环境没有安装 `hiddenlayer`。

### 解决方法

这个提示不影响训练，可以忽略。

如果想消除提示，可以安装：

```bash
pip install hiddenlayer
```

### 经验总结

并不是所有 warning 都需要立刻处理。如果不影响训练、推理和指标计算，可以记录下来，后续再处理。

---

## 14. nnU-Net 验证集和自实现 3D U-Net 验证集不一致

### 问题现象

自实现 3D U-Net 的验证病例为：

```text
la_014
la_016
la_003
la_007
```

nnU-Net fold 0 的验证病例为：

```text
la_007
la_016
la_021
la_024
```

两者不完全一致。

### 问题原因

自实现 3D U-Net 使用的是自己随机划分的训练/验证集；nnU-Net 使用自己的 5-fold split 文件：

```text
splits_final.json
```

因此两者验证病例不完全相同。

### 解决方法

在当前 V2.0 项目中，明确说明该对照是“方法级参考对照”，不是严格公平比较。

更严谨的 V3.0 方案：

1. 读取 nnU-Net 的 `splits_final.json`；
2. 使用相同 fold 0 训练/验证划分重新训练自实现 3D U-Net；
3. 在同一验证病例上计算 Dice / IoU / HD95；
4. 再和 nnU-Net 做公平对比。

### 经验总结

实验对比必须关注数据划分是否一致。如果验证集不同，不能直接声称某方法严格优于另一方法，只能作为参考对照。

---

## 15. GitHub 上传时误传大文件

### 问题现象

医学影像项目中容易误传：

```text
data/raw/
checkpoints/
results/predictions/
*.nii.gz
*.pth
*.b2nd
nnUNet_preprocessed/
nnUNet_results/
```

这些文件通常很大，不适合直接上传 GitHub。

### 解决方法

在 `.gitignore` 中加入：

```gitignore
# Medical image data
data/raw/
*.nii
*.nii.gz
*.mha
*.mhd
*.nrrd
*.dcm

# Model checkpoints
checkpoints/
*.pth
*.pt
*.ckpt

# Prediction outputs
results/predictions/
*.npy
*.npz

# nnU-Net workspace
nnUNet_workspace/
nnunet_work/
nnUNet_raw/
nnUNet_preprocessed/
nnUNet_results/
*.b2nd
```

如果已经 `git add` 了大文件，先取消追踪：

```bash
git rm -r --cached data/raw
git rm -r --cached checkpoints
git rm -r --cached results/predictions
git rm -r --cached nnUNet_workspace
```

然后重新提交。

### 经验总结

医学影像项目上传 GitHub 时，只上传：

```text
代码
README
docs
轻量结果 csv/txt
可视化 png
```

不要上传：

```text
原始数据
模型权重
预测 nii.gz
nnU-Net 预处理文件
```

---

## 16. 项目工程经验总结

本项目遇到的问题可以归纳为四类。

### 16.1 数据问题

包括隐藏文件、图像和标签路径不一致、维度顺序错误等。

解决策略：

- 先写数据检查脚本；
- 打印 shape、spacing、unique label；
- 做原图和 mask overlay 可视化；
- 过滤隐藏文件。

### 16.2 模型和训练问题

包括输入维度错误、loss 实现错误、patch-level 和 whole-volume 差异等。

解决策略：

- 先用合成数据验证训练闭环；
- 再接真实数据；
- 区分 patch-level validation 和 whole-volume inference；
- 保存训练日志和曲线。

### 16.3 后处理和指标问题

包括远处假阳性、HD95 异常高、边界误差等。

解决策略：

- 做 error map；
- 统计 TP / FN / FP；
- 计算 Dice / IoU / HD95；
- 使用最大连通区域后处理；
- 做 best / worst / most improved case 分析。

### 16.4 工程环境问题

包括 Windows 多进程、中文长路径、nnU-Net 版本差异和 GitHub 大文件管理。

解决策略：

- nnU-Net workspace 使用英文短路径；
- 降低 dataloader worker 数量；
- 使用 `inspect.signature()` 检查当前版本接口；
- 用 `.gitignore` 管理大文件；
- 用 tag 标记稳定版本。

---

## 17. 后续建议

后续 V3.0 可以继续解决以下问题：

1. 统一自实现 3D U-Net 与 nnU-Net 的 fold 0 划分；
2. 用 MONAI 实现标准医学影像 pipeline；
3. 加入更系统的数据增强；
4. 增加困难背景 patch 采样；
5. 尝试边界损失函数；
6. 分析更多失败案例；
7. 输出正式项目报告和 5 分钟汇报 PPT。
