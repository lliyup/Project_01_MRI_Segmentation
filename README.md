# Project 01: 基于自实现 3D U-Net 的心脏 MRI 左心房分割实验与误差分析

## 1. 项目简介

本项目面向三维医学影像分割任务，基于 **MSD Task02 Heart** 数据集，完成了一个从零实现的心脏 MRI 左心房分割 baseline。

项目使用 PyTorch 自实现轻量级 **3D U-Net**，并构建了从医学影像数据读取、MRI 强度归一化、三维 patch 采样、模型训练、whole-volume 滑窗推理、Dice/IoU/HD95 指标计算、最大连通区域后处理到 best/worst case 可视化的完整流程。

本项目的重点不是单纯追求最高指标，而是完整复现一个医学图像分割项目的基础工程闭环，便于后续扩展到 nnU-Net、MONAI、多模态医学影像分割和医学影像 AI 研究。

---

## 2. 项目任务

* **任务类型**：三维医学影像分割
* **数据模态**：心脏 MRI
* **分割目标**：左心房 Left Atrium
* **输入**：3D cardiac MRI volume
* **输出**：左心房二值分割 mask
* **类别设置**：

  * `0`：背景
  * `1`：左心房
* **核心模型**：自实现轻量级 3D U-Net
* **评价指标**：Dice、IoU、HD95

---

## 3. 当前项目结果

本项目当前包含两个版本：

```text
Version 1.0: Self-implemented 3D U-Net baseline
Version 2.0: Self-implemented 3D U-Net + nnU-Net v2 comparison
```

---

### 3.1 V1.0：自实现 3D U-Net baseline

#### Patch-level validation result

| Metric        |  Value |
| ------------- | -----: |
| Best Epoch    |     19 |
| Best Val Loss | 0.1117 |
| Best Val Dice | 0.9412 |
| Best Val IoU  | 0.8892 |

#### Whole-volume evaluation result

| Setting             | Mean Dice | Mean IoU | Mean HD95 |
| ------------------- | --------: | -------: | --------: |
| Raw prediction      |    0.7329 |   0.5882 | 162.87 mm |
| LCC post-processing |    0.8276 |   0.7076 |  11.85 mm |

说明：

* Patch-level 结果反映模型在裁剪到左心房附近的三维 patch 上表现较好；
* Whole-volume 结果反映模型对完整病例进行滑窗推理后的真实病例级表现；
* Raw prediction 存在远处假阳性，导致 HD95 偏高；
* 最大连通区域后处理 Largest Connected Component, LCC 能够有效去除远处假阳性，使 Dice 提升、HD95 大幅下降。

#### Case-level analysis

| Case            | Role                  | Result                                    |
| --------------- | --------------------- | ----------------------------------------- |
| `la_016.nii.gz` | Best LCC case         | LCC Dice = 0.8898, LCC HD95 = 9.34 mm     |
| `la_003.nii.gz` | Worst LCC case        | LCC Dice = 0.7976, LCC HD95 = 10.08 mm    |
| `la_007.nii.gz` | Most improved by Dice | Raw Dice = 0.5760 → LCC Dice = 0.8189     |
| `la_016.nii.gz` | Most improved by HD95 | Raw HD95 = 217.90 mm → LCC HD95 = 9.34 mm |

---

### 3.2 V2.0：nnU-Net v2 strong baseline comparison

在 V2.0 中，本项目进一步引入 **nnU-Net v2 3d_fullres** 作为医学影像分割强基线，并完成 50 epoch short-run 对照实验。

| Method                          | Setting                                    | Mean Dice | Mean IoU | Mean HD95 |
| ------------------------------- | ------------------------------------------ | --------: | -------: | --------: |
| Self-implemented 3D U-Net Raw   | whole-volume sliding window                |    0.7329 |   0.5882 | 162.87 mm |
| Self-implemented 3D U-Net + LCC | whole-volume + largest connected component |    0.8276 |   0.7076 |  11.85 mm |
| nnU-Net v2 3d_fullres           | fold 0, 50 epochs short-run                |    0.9325 |   0.8736 |   3.04 mm |

V2.0 结果说明：

* 自实现 3D U-Net 成功完成了医学影像分割的完整 baseline 流程；
* LCC 后处理能显著减少远处假阳性，使 whole-volume Dice 和 HD95 明显改善；
* nnU-Net v2 作为医学图像分割强基线，在自动规划、预处理、数据增强、推理和后处理方面明显优于手写 baseline；
* 当前对照实验中，自实现 3D U-Net 与 nnU-Net 的验证病例划分不完全一致，因此该表应视为方法级参考对照，而不是严格同一验证集下的公平比较。

---

## 4. 项目结构

```text
Project_01_MRI_Segmentation/
├── checkpoints/
│   └── best_heart_unet3d_epoch19.pth
│
├── data/
│   └── raw/
│       └── Task02_Heart/
│           ├── imagesTr/
│           ├── labelsTr/
│           └── imagesTs/
│
├── datasets/
│   ├── __init__.py
│   └── heart_patch_dataset.py
│
├── figures/
│   ├── heart_case_visualization.png
│   ├── heart_loss_curve.png
│   ├── heart_dice_curve.png
│   ├── heart_iou_curve.png
│   ├── la_016.nii.gz_best_volume_visualization.png
│   ├── la_003.nii.gz_worst_volume_visualization.png
│   └── la_007.nii.gz_most_improved_visualization.png
│
├── losses/
│   ├── __init__.py
│   └── dice_loss.py
│
├── models/
│   ├── __init__.py
│   └── unet3d.py
│
├── results/
│   ├── heart_training_log.csv
│   ├── heart_volume_metrics.csv
│   ├── heart_volume_metrics_with_hd95.csv
│   ├── final_experiment_summary.csv
│   ├── final_experiment_summary.txt
│   ├── nnunet_50epoch_validation_metrics.csv
│   ├── v2_comparison_summary.csv
│   ├── v2_comparison_summary.txt
│   └── predictions/
│       └── volume_eval/
│           ├── la_003_raw_pred.nii.gz
│           ├── la_003_lcc_pred.nii.gz
│           ├── la_003_prob.nii.gz
│           ├── la_007_raw_pred.nii.gz
│           ├── la_007_lcc_pred.nii.gz
│           ├── la_007_prob.nii.gz
│           ├── la_014_raw_pred.nii.gz
│           ├── la_014_lcc_pred.nii.gz
│           ├── la_014_prob.nii.gz
│           ├── la_016_raw_pred.nii.gz
│           ├── la_016_lcc_pred.nii.gz
│           └── la_016_prob.nii.gz
│
├── scripts/
│   ├── check_env.py
│   ├── check_heart_dataset.py
│   ├── visualize_heart_case.py
│   ├── train_synthetic_3dunet.py
│   ├── visualize_synthetic_prediction.py
│   ├── visualize_synthetic_error.py
│   ├── train_heart_3dunet.py
│   ├── visualize_heart_prediction.py
│   ├── create_heart_training_curves.py
│   ├── infer_heart_volume.py
│   ├── debug_heart_volume_inference.py
│   ├── postprocess_heart_volume.py
│   ├── evaluate_heart_volumes.py
│   ├── compute_heart_volume_hd95.py
│   ├── visualize_volume_cases.py
│   └── generate_final_summary.py
│
├── utils/
│   ├── __init__.py
│   └── metrics.py
│
└── README.md
```

---

## 5. 环境配置

### 5.1 推荐环境

本项目当前在以下环境中完成测试：

```text
OS: Windows
Python: 3.11
PyTorch: 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

### 5.2 创建 Conda 环境

```bash
conda create -n medseg python=3.11 -y
conda activate medseg
```

### 5.3 安装 PyTorch GPU 版本

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 5.4 安装医学影像与科学计算依赖

```bash
pip install numpy scipy pandas matplotlib scikit-image scikit-learn tqdm nibabel SimpleITK ipykernel
```

### 5.5 检查环境

```bash
python scripts/check_env.py
```

正常情况下应看到：

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
nibabel: OK
SimpleITK: OK
```

---

## 6. 数据准备

本项目使用 **Medical Segmentation Decathlon Task02 Heart** 数据集。

请将数据解压到：

```text
data/raw/Task02_Heart/
```

期望目录结构如下：

```text
data/raw/Task02_Heart/
├── imagesTr/
├── labelsTr/
├── imagesTs/
└── dataset.json
```

其中：

* `imagesTr/`：训练图像；
* `labelsTr/`：训练标签；
* `imagesTs/`：测试图像；
* `dataset.json`：数据集说明文件，可选。

注意：如果数据中存在 macOS 产生的隐藏文件，如：

```text
._la_003.nii.gz
.DS_Store
```

需要删除，否则 `nibabel` 可能读取失败。

Windows CMD 删除方式：

```bash
del /s /q data\raw\Task02_Heart\._*
```

PowerShell 删除方式：

```powershell
Get-ChildItem data\raw\Task02_Heart -Recurse -Filter "._*" | Remove-Item -Force
```

---

## 7. 数据检查与可视化

### 7.1 检查数据集

```bash
python scripts/check_heart_dataset.py
```

示例输出：

```text
Number of training images: 20
Number of training labels: 20
Image shape: (320, 320, 130)
Label shape: (320, 320, 130)
Label unique values: [0. 1.]
Voxel spacing: (1.25, 1.25, 1.37)
```

### 7.2 可视化真实心脏 MRI 和 mask

```bash
python scripts/visualize_heart_case.py
```

输出图像：

```text
figures/heart_case_visualization.png
```

该图展示：

* 原始 MRI 切片；
* 左心房 ground truth mask；
* MRI 与 mask 的叠加图。

---

## 8. 模型结构

模型文件：

```text
models/unet3d.py
```

本项目实现了一个轻量级 3D U-Net：

* 使用 `Conv3d` 进行三维特征提取；
* 使用 `MaxPool3d` 进行下采样；
* 使用 `ConvTranspose3d` 进行上采样；
* 使用 skip connection 融合浅层定位信息和深层语义信息；
* 使用 `InstanceNorm3d` 适应医学图像小 batch 训练；
* 输出类别数为 2，即背景和左心房。

测试模型前向传播：

```bash
python models/unet3d.py
```

预期输出：

```text
Input shape: torch.Size([1, 1, 64, 64, 64])
Output shape: torch.Size([1, 2, 64, 64, 64])
```

---

## 9. Loss 与指标

### 9.1 Dice Loss

文件：

```text
losses/dice_loss.py
```

运行测试：

```bash
python losses/dice_loss.py
```

### 9.2 Dice / IoU

文件：

```text
utils/metrics.py
```

运行测试：

```bash
python utils/metrics.py
```

---

## 10. 合成数据训练验证

在接入真实医学影像前，本项目先使用三维球体合成数据验证训练闭环。

目的：

* 检查模型结构是否正确；
* 检查 loss 是否能反向传播；
* 检查 Dice/IoU 指标是否正常；
* 检查 GPU 训练和模型保存是否正常。

训练合成数据：

```bash
python scripts/train_synthetic_3dunet.py
```

预测可视化：

```bash
python scripts/visualize_synthetic_prediction.py
```

误差图可视化：

```bash
python scripts/visualize_synthetic_error.py
```

---

## 11. 真实 Heart MRI patch-level 训练

### 11.1 3D patch Dataset

文件：

```text
datasets/heart_patch_dataset.py
```

功能：

* 读取 `.nii.gz` 图像和标签；
* MRI 非零区域 z-score 归一化；
* 原始 H × W × D 转换为 D × H × W；
* 围绕左心房区域采样三维 patch；
* 返回 PyTorch tensor。

测试 Dataset：

```bash
python datasets/heart_patch_dataset.py
```

示例输出：

```text
Image patch shape: torch.Size([1, 64, 96, 96])
Mask patch shape: torch.Size([64, 96, 96])
Mask unique values: tensor([0, 1])
```

### 11.2 训练真实 Heart MRI patch

```bash
python scripts/train_heart_3dunet.py
```

训练设置：

| Setting          |                         Value |
| ---------------- | ----------------------------: |
| patch size       |                  64 × 96 × 96 |
| batch size       |                             1 |
| base channels    |                             8 |
| optimizer        |                          Adam |
| learning rate    |                          1e-3 |
| loss             | Dice Loss + CrossEntropy Loss |
| train cases      |                            16 |
| validation cases |                             4 |
| epochs           |                            20 |

训练完成后保存：

```text
checkpoints/best_heart_unet3d_epoch19.pth
results/heart_training_log.csv
```

### 11.3 绘制训练曲线

```bash
python scripts/create_heart_training_curves.py
```

输出：

```text
figures/heart_loss_curve.png
figures/heart_dice_curve.png
figures/heart_iou_curve.png
```

---

## 12. Patch-level 预测可视化

运行：

```bash
python scripts/visualize_heart_prediction.py
```

输出示例：

```text
Dice: 0.9573
IoU: 0.9180
TP voxels: 39896
FN voxels: 1925
FP voxels: 1637
```

输出图像：

```text
figures/la_003.nii.gz_prediction_visualization.png
```

该图展示：

* MRI patch；
* ground truth mask；
* predicted mask；
* error map。

---

## 13. Whole-volume 滑窗推理

Patch-level 验证结果不能完全代表完整病例分割能力，因此本项目进一步实现 whole-volume sliding window inference。

运行：

```bash
python scripts/infer_heart_volume.py
```

滑窗设置：

```text
patch size = 64 × 96 × 96
stride = 32 × 48 × 48
threshold = 0.5
```

输出：

```text
results/predictions/*.nii.gz
```

---

## 14. Whole-volume 诊断与后处理

### 14.1 Whole-volume 失败诊断

运行：

```bash
python scripts/debug_heart_volume_inference.py
```

该脚本用于分析：

* 真实 mask bbox；
* 预测 mask bbox；
* GT 区域概率分布；
* 背景区域概率分布；
* 不同阈值下的 Dice/IoU；
* 概率图切片可视化。

### 14.2 最大连通区域后处理

运行：

```bash
python scripts/postprocess_heart_volume.py
```

目的：

* 去除远处假阳性；
* 保留最大连通左心房区域；
* 改善 Dice、IoU 和 HD95。

单病例示例结果：

```text
Raw Dice: 0.7369
Raw IoU: 0.5834

After LCC:
Dice: 0.7976
IoU: 0.6633
```

---

## 15. 多病例 whole-volume 评估

运行：

```bash
python scripts/evaluate_heart_volumes.py
```

输出：

```text
results/heart_volume_metrics.csv
results/predictions/volume_eval/
```

验证病例：

```text
la_014.nii.gz
la_016.nii.gz
la_003.nii.gz
la_007.nii.gz
```

结果：

```text
Mean Raw Dice: 0.7329
Mean Raw IoU: 0.5882
Mean LCC Dice: 0.8276
Mean LCC IoU: 0.7076
```

---

## 16. 计算 HD95

运行：

```bash
python scripts/compute_heart_volume_hd95.py
```

输出：

```text
results/heart_volume_metrics_with_hd95.csv
```

结果：

```text
Mean Raw HD95: 162.87 mm
Mean LCC HD95: 11.85 mm
```

解释：

* Raw prediction 中存在远处假阳性，导致 HD95 极高；
* LCC 后处理删除远处假阳性，使 HD95 大幅下降。

---

## 17. Best / Worst / Improved case 可视化

运行：

```bash
python scripts/visualize_volume_cases.py
```

输出：

```text
figures/la_016.nii.gz_best_volume_visualization.png
figures/la_003.nii.gz_worst_volume_visualization.png
figures/la_007.nii.gz_most_improved_visualization.png
```

说明：

* `la_016.nii.gz`：LCC Dice 最高，是 best case；
* `la_003.nii.gz`：LCC Dice 最低，是 worst case；
* `la_007.nii.gz`：LCC 后 Dice 提升最大，是 most improved case。

---

## 18. 生成最终实验 summary

运行：

```bash
python scripts/generate_final_summary.py
```

输出：

```text
results/final_experiment_summary.csv
results/final_experiment_summary.txt
```

该 summary 文件汇总：

* Patch-level 最佳结果；
* Whole-volume raw 平均结果；
* Whole-volume LCC 平均结果；
* Best case；
* Worst case；
* Most improved case；
* 项目主要结论。

---

## 19. V2.0：nnU-Net v2 对照实验

### 19.1 转换 MSD Heart 为 nnU-Net v2 格式

运行：

```bash
python scripts/prepare_nnunet_heart.py
```

目标目录：

```text
D:/nnunet_work/nnUNet_raw/Dataset002_Heart/
```

nnU-Net v2 数据格式要求：

```text
Dataset002_Heart/
├── dataset.json
├── imagesTr/
│   ├── la_003_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── la_003.nii.gz
│   └── ...
└── imagesTs/
```

### 19.2 配置 nnU-Net 环境变量

PowerShell 示例：

```powershell
$Env:nnUNet_raw = "D:/nnunet_work/nnUNet_raw"
$Env:nnUNet_preprocessed = "D:/nnunet_work/nnUNet_preprocessed"
$Env:nnUNet_results = "D:/nnunet_work/nnUNet_results"
$Env:nnUNet_n_proc_DA = "1"
$Env:nnUNet_def_n_proc = "1"
```

### 19.3 nnU-Net 预处理

```bash
nnUNetv2_plan_and_preprocess -d 2 -c 3d_fullres --verify_dataset_integrity
```

### 19.4 自定义 50 epoch trainer

本项目使用自定义短训 trainer：

```text
nnUNetTrainer_50epochs
```

其作用是将 nnU-Net 默认训练轮数缩短为 50 epoch，用于保研项目中的强基线对照实验。

### 19.5 训练 nnU-Net 3d_fullres fold 0

```bash
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_50epochs
```

训练完成后，nnU-Net validation 输出：

```text
Mean Validation Dice: 0.9325
```

### 19.6 计算 nnU-Net Dice / IoU / HD95

运行：

```bash
python scripts/compute_nnunet_metrics.py
```

输出：

```text
results/nnunet_50epoch_validation_metrics.csv
```

结果：

```text
Mean Dice: 0.9325
Mean IoU: 0.8736
Mean HD95: 3.04 mm
```

### 19.7 生成 V2.0 对照 summary

运行：

```bash
python scripts/generate_v2_comparison_summary.py
```

输出：

```text
results/v2_comparison_summary.csv
results/v2_comparison_summary.txt
```

---

## 20. 一键复现实验流程

如果环境和数据已经准备好，可以按照以下顺序复现实验：

```bash
# 1. 检查环境
python scripts/check_env.py

# 2. 检查数据
python scripts/check_heart_dataset.py

# 3. 可视化原始 MRI 与 mask
python scripts/visualize_heart_case.py

# 4. 测试模型、loss 和指标
python models/unet3d.py
python losses/dice_loss.py
python utils/metrics.py

# 5. 测试 Heart patch dataset
python datasets/heart_patch_dataset.py

# 6. 训练真实 Heart MRI patch-level baseline
python scripts/train_heart_3dunet.py

# 7. 绘制训练曲线
python scripts/create_heart_training_curves.py

# 8. patch-level 预测可视化
python scripts/visualize_heart_prediction.py

# 9. whole-volume 推理与诊断
python scripts/debug_heart_volume_inference.py
python scripts/postprocess_heart_volume.py

# 10. 多病例 whole-volume 评估
python scripts/evaluate_heart_volumes.py

# 11. 计算 HD95
python scripts/compute_heart_volume_hd95.py

# 12. 可视化 best/worst/improved case
python scripts/visualize_volume_cases.py

# 13. 生成最终 summary
python scripts/generate_final_summary.py

# 14. nnU-Net v2 对照实验相关步骤
python scripts/prepare_nnunet_heart.py
nnUNetv2_plan_and_preprocess -d 2 -c 3d_fullres --verify_dataset_integrity
nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_50epochs
python scripts/compute_nnunet_metrics.py
python scripts/generate_v2_comparison_summary.py
```

---

## 21. 当前结论

本项目表明，自实现轻量级 3D U-Net 能够在心脏 MRI 左心房分割任务中学习到有效的三维结构特征。

在 patch-level 验证中，模型取得：

```text
Best Val Dice = 0.9412
Best Val IoU = 0.8892
```

但 patch-level 指标偏乐观，不能完全代表完整病例分割能力。在 whole-volume 滑窗推理中，raw prediction 存在远处假阳性，导致：

```text
Mean Raw Dice = 0.7329
Mean Raw HD95 = 162.87 mm
```

通过最大连通区域后处理，可以显著减少远处假阳性，使结果提升至：

```text
Mean LCC Dice = 0.8276
Mean LCC IoU = 0.7076
Mean LCC HD95 = 11.85 mm
```

这说明模型能够分割左心房主体结构，但仍存在边界误差、局部漏分割和假阳性问题。

---

## 22. 项目局限性

当前版本仍存在以下局限：

1. 训练病例数量较少，仅使用 20 个带标签病例中的 16 个训练、4 个验证；
2. 当前模型为轻量级 3D U-Net，尚未与 nnU-Net/MONAI 等强基线进行对比；
3. patch-level validation 与 whole-volume inference 存在分布差异；
4. 训练时使用 patch 采样，尚未使用完整体数据上下文；
5. 后处理目前只使用最大连通区域，尚未引入更精细的解剖先验；
6. 尚未系统加入数据增强、边界损失或多尺度结构。

---

## 23. 后续改进方向

后续可从以下方向继续扩展：

1. **nnU-Net v2 对照实验**
   使用 nnU-Net v2 在同一数据集上训练和评估，作为医学分割强基线。

2. **MONAI pipeline 复现**
   使用 MONAI 构建更规范的数据增强、滑窗推理和指标计算流程。

3. **更强数据增强**
   加入随机旋转、缩放、翻转、强度扰动、随机裁剪等医学图像增强。

4. **背景 patch 采样优化**
   增加纯背景和困难负样本 patch，减少 whole-volume 远处假阳性。

5. **边界优化**
   尝试 Dice + CE + Boundary Loss，改善边界漏分割和 HD95。

6. **完整病例级训练策略**
   进一步优化 sliding window inference 和 patch aggregation，使推理结果更稳定。

7. **失败案例分析**
   系统分析 best case、worst case 和 most improved case，总结错误类型。

---

## 24. 项目定位

本项目可以作为医学影像 AI 方向的入门级完整 baseline，用于展示以下能力：

* 能够处理真实医学影像 NIfTI 数据；
* 理解三维医学图像分割任务；
* 能够自实现 3D U-Net、Dice Loss、Dice/IoU/HD95 指标；
* 能够完成 patch-level 训练和 whole-volume 推理；
* 能够进行医学图像分割误差分析；
* 能够从实验结果中总结局限性和改进方向。

---

## 25. 备注

本项目当前已经完成两个版本：

```text
Version 1.0: Self-implemented 3D U-Net baseline
Version 2.0: Self-implemented 3D U-Net + nnU-Net v2 comparison
```

下一版本建议扩展为：

```text
Version 3.0: Unified split comparison + MONAI pipeline + stronger error analysis
```

V3.0 可进一步实现：

* 统一自实现 3D U-Net 与 nnU-Net 的训练/验证划分；
* 使用 MONAI 复现数据增强、滑窗推理和指标计算流程；
* 增加更多失败案例分析；
* 尝试背景困难负样本采样与边界损失函数；
