# 报告：MIT-BIH 单导联分段感知模型实现概述

## 模型结构
- **输入**：`(batch, 1, 360)` 的单导联 beat 序列。
- **分段卷积编码器**：P/QRS/T/Global 四组 `Conv1d(1, 4, kernel_size=4, stride=1, padding=0)`，输出通道数均为 4。P/QRS/T 输出长度约 117，Global 输出长度约 357。
- **多 token 池化**：
  - P 段：AvgPool1d 2 块（kernel/stride≈58），得到 2 个 4 维 token。
  - QRS 段：AvgPool1d 3 块（kernel/stride≈39），得到 3 个 4 维 token。
  - T 段：同 P 段，2 个 token。
  - Global 段：时间维全局平均，1 个 token。
  - 最终 8 个 token 堆叠为 `H0`，形状 `(batch, 8, 4)`。
- **Photonic MLP**：`num_mlp_layers`(默认 2，最少 2) 层 `Linear(4,4)+ReLU/tanh`，逐 token 独立映射，保持 `(batch, 8, 4)`。
- **全局汇聚与分类**：token 维平均得到 `h_pool (batch,4)` → 可选 Dropout → `Linear(4,2)` 分类头输出 logits。

## 知识蒸馏
- **Teacher**：一维 ResNet18，输出 `logits_T` 与中间 embedding `feat_T (embedding_dim，可配置)`。
- **Student 输出**：`logits_S` 与 `h_pool` 作为全局特征。
- **蒸馏损失**：
  - 分类交叉熵：`CE(logits_S, y)`。
  - Logit 蒸馏：`KL(softmax(z_T/T), softmax(z_S/T))`，`T=kd_temperature`。
  - 特征蒸馏：`proj_T(feat_T)`、`proj_S(h_pool)` 映射到 `d_kd` 维，L2 归一化后用 MSE 对齐。
  - 总损失：`alpha*CE + beta*KD_logits + gamma*KD_feat`，三者均可配置。
- **训练细节**：Teacher 仅前向、`eval()` 模式，`requires_grad=False`；梯度裁剪 `max_norm=1.0`；Adam 优化器、ReduceLROnPlateau 调度；验证集早停监控组合指标 `F1 - miss_rate - FPR`，`patience=15`、`min_epochs=20`，避免过早停止。

## 数值归一化与约束
- **输入预处理**：每条 beat 减均值、按最大绝对值缩放到 `[-1,1]`，必要时裁剪。
- **受限权重**：`ConstrainedConv1d/ConstrainedLinear` 使用可训练无界参数经 `tanh` 映射到 `[-scale, scale]`，优化器更新原始参数，前向使用映射后的权重/偏置。
- **激活约束**：
  - 若启用 `--use_tanh_activations`，MLP/卷积输出用 tanh，自然落在 `[-1,1]`；
  - 否则在进入受限线性层前调用 `scale_to_unit` 对当前 batch 归一化到 `[-1,1]`。
- **分类头**：保持未约束 logits，直接用于 `CrossEntropyLoss`。

## 数据集与划分
- 训练集记录：`100,101,102,103,104,105,107,108,109,111,112,113,115,117,121,122,123,200,201,202,203,207,209,210,212,213,219,222,223,230,231,232`。
- 泛化集记录：`106,114,116,118,119,124,205,208,214,215,220,221,228,233,234`。
- 数据加载使用 `wfdb` 读取注释，按中心对齐截取 360 点窗口，P/QRS/T 段按 120 点切片。
- `split_dataset` 将训练数据按 80/20 划分训练/验证，batch size 默认 128，可调。

## 超参数与可配置项
- 训练：`--batch_size`，`--lr=1e-3`，`--weight_decay=1e-4`，`--max_epochs`(默认 90)，`--patience=15`，`--min_epochs=20`，`--scheduler_patience=3`。
- 类别不平衡：`compute_class_weights` 提供异常类权重提升（默认 1.3x）。
- 模型：`--num_mlp_layers`(≥2)、`--dropout_rate`、`--use_value_constraint`、`--use_tanh_activations`、`--constraint_scale`。
- 蒸馏：`--use_kd`（默认开启，可用 `--no-use-kd` 关闭）、`--teacher_checkpoint`（若未提供且启用 KD，将自动轻量预训练一个 ResNet18 teacher 后用于蒸馏）、`--teacher_embedding_dim`、`--kd_temperature`、`--kd_d`、`--alpha/beta/gamma`。

## 代码接口与输出
- 前向接口：Student/Teacher 输入 `(batch,1,360)`，Student 输出 `(logits, h_pool)`，Teacher 输出 `(logits_T, feat_T)`。
- 训练脚本保存学生模型到 `saved_models/student_model.pth`，包含 CLI 配置；训练曲线、ROC（验证/泛化）与混淆矩阵 PNG 自动写入 `./artifacts`。

## 需求符合性检查
- 分段卷积 + 8 token 池化 + photonic MLP + 均值池化分类流水线已实现并可配置（`models/student.py`）。
- Adam 优化、梯度裁剪、可调 batch_size/epochs/early stopping/LR 调度均在 `train.py` 实现。
- 类别权重、CrossEntropyLoss、KD 的 logit/feature 蒸馏、可切换 KD 流程均实现。
- 数值约束通过 `tanh` 重参数化和输入缩放；输出 logits 未强制约束，符合需求。
