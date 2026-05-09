# 3D U-Net for Cardiac MRI Segmentation

基于 3D U-Net 的心脏 MRI 分割实验与可视化分析。

## 项目简介

本项目使用 3D U-Net 深度学习模型对心脏 MRI 图像进行自动分割，基于 [Medical Segmentation Decathlon Task02_Heart](http://medicaldecathlon.com/) 数据集。

## 目录结构

```
├── model/          # 3D U-Net 模型定义
├── datasets/       # 数据加载与预处理
├── losses/         # 损失函数 (Dice Loss)
├── utils/          # 评估指标等工具
├── scripts/        # 训练、推理、可视化脚本
├── checkpoints/    # 训练好的模型权重 (Git LFS)
├── data/           # 原始数据集 (Git LFS)
├── results/        # 训练日志与结果
├── figures/        # 可视化图表
├── reports/        # 实验报告
└── slides/         # 演示文稿
```

## 数据集

[Medical Segmentation Decathlon - Task02 Heart](http://medicaldecathlon.com/)，包含 20 例训练集和 10 例测试集的 3D 心脏 MRI 图像及左心房分割标注。

## 模型

3D U-Net 架构，使用 Dice Loss 进行训练。

## 使用方式

### 训练

```bash
python scripts/train_heart_unet3d.py
```

### 推理

```bash
python scripts/infer_heart_volume.py
```

### 可视化

```bash
python scripts/visualize_heart_case.py
python scripts/visualize_heart_prediction.py
```
