\# Project Summary



\## 1. 项目目标



本项目面向三维医学影像分割任务，以 MSD Task02 Heart 数据集为对象，完成心脏 MRI 左心房分割实验。项目目标是构建一个从数据读取、预处理、模型训练、完整体推理、指标评估到误差分析的完整医学图像分割流程。



\## 2. 数据集



使用 Medical Segmentation Decathlon Task02 Heart 数据集。该任务为单模态 MRI 左心房分割，标签为二分类 mask：



\- 0：背景

\- 1：左心房



实验中使用 20 个带标签训练病例。



\## 3. 自实现 3D U-Net baseline



本项目首先使用 PyTorch 自实现轻量级 3D U-Net，包括：



\- NIfTI 医学影像读取；

\- MRI 非零区域 z-score 归一化；

\- 3D patch 采样；

\- Dice Loss + CrossEntropy Loss；

\- Dice / IoU / HD95 指标；

\- whole-volume sliding window inference；

\- largest connected component 后处理；

\- best / worst / most improved case 可视化。



自实现 3D U-Net 的结果：



| Setting | Mean Dice | Mean IoU | Mean HD95 |

|---|---:|---:|---:|

| Raw prediction | 0.7329 | 0.5882 | 162.87 mm |

| LCC post-processing | 0.8276 | 0.7076 | 11.85 mm |



\## 4. nnU-Net v2 strong baseline



在 V2.0 中，进一步使用 nnU-Net v2 3d\_fullres 作为医学影像分割强基线，并完成 fold 0 的 50 epoch short-run 实验。



nnU-Net v2 结果：



| Method | Mean Dice | Mean IoU | Mean HD95 |

|---|---:|---:|---:|

| nnU-Net v2 3d\_fullres, 50 epochs | 0.9325 | 0.8736 | 3.04 mm |



\## 5. 关键发现



1\. 自实现 3D U-Net 可以完成完整医学影像分割流程，但 whole-volume 推理中存在远处假阳性。

2\. 最大连通区域后处理能显著减少远处假阳性，使 Mean Dice 从 0.7329 提升到 0.8276，并使 Mean HD95 从 162.87 mm 降至 11.85 mm。

3\. nnU-Net v2 在自动规划、预处理、数据增强、推理和后处理方面明显优于手写 baseline，Mean Dice 达到 0.9325，Mean HD95 降至 3.04 mm。

4\. Patch-level validation 指标偏乐观，不能直接代表完整病例分割能力，必须进行 whole-volume inference。



\## 6. 当前局限



当前自实现 3D U-Net 与 nnU-Net 的验证病例划分不完全一致，因此 V2.0 对照结果属于方法级参考对照，不是严格同一验证集下的公平比较。



\## 7. 下一步计划



V3.0 计划包括：



\- 统一自实现 3D U-Net 与 nnU-Net 的训练/验证划分；

\- 使用 MONAI 复现标准医学图像训练 pipeline；

\- 增加更强数据增强；

\- 增加边界损失函数；

\- 系统整理失败案例分析；

