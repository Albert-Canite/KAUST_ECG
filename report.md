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
- **训练细节**：Teacher 仅前向、`eval()` 模式，`requires_grad=False`；梯度裁剪 `max_norm=1.0`；Adam 优化器、ReduceLROnPlateau 调度；验证集早停监控组合指标 `F1 - miss_rate - FPR`，`patience=25`、`min_epochs=25`，避免过早停止。

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
- 训练：`--batch_size`，`--lr=1e-3`，`--weight_decay=1e-4`，`--max_epochs`(默认 90)，`--patience=25`，`--min_epochs=25`，`--scheduler_patience=3`。
- 类别不平衡：
  - 类别权重：默认关闭，若开启（`--use_class_weights`）异常类默认 1.0x，可调且受 `--max_class_weight_ratio`（默认 2.0）限制。
  - 采样：默认关闭加权采样 (`--use_weighted_sampler/--no-use-weighted-sampler`)，异常类采样增强倍率 `--sampler_abnormal_boost`（默认 1.0，可调 >1）。
- 模型：`--num_mlp_layers`(≥2)、`--dropout_rate`、`--use_value_constraint`、`--use_tanh_activations`、`--constraint_scale`。
- 蒸馏：`--use_kd`（默认开启，可用 `--no-use-kd` 关闭）、`--teacher_checkpoint`（若未提供且启用 KD，将自动预训练 ResNet18 teacher）`--teacher_auto_train_epochs`（默认 15）、`--teacher_embedding_dim`、`--kd_temperature`、`--kd_d`、`--alpha/beta/gamma`、教师质量阈值 `--teacher_min_f1`、`--teacher_min_sensitivity`，若教师低于阈值会自动禁用 KD 以保护召回率。

## 代码接口与输出
- 前向接口：Student/Teacher 输入 `(batch,1,360)`，Student 输出 `(logits, h_pool)`，Teacher 输出 `(logits_T, feat_T)`。
- 训练脚本保存学生模型到 `saved_models/student_model.pth`，包含 CLI 配置；训练曲线、ROC（验证/泛化）与混淆矩阵 PNG 自动写入 `./artifacts`。

## v1_debug 结果分析与改进
- **现象（用户日志）**：`Val F1≈0.57，miss≈29%`，泛化 miss≈45%，FPR 约 4%。miss 居高，说明正类召回不足。
- **原因分析**：
  1. **类别不平衡**：当前训练集未做重采样，异常类占比低，单靠较小权重提升（原默认 1.3x）不足以提升正类召回。
  2. **教师质量不足**：自动教师仅预训练 5 个 epoch，可能精度/召回偏低，蒸馏信号会“拉低”学生的正类预测倾向，导致 miss 居高。
  3. **采样分布**：无加权采样时，小批次中异常样本稀疏，梯度对正类的强化不足。
- **代码修正**：
  - 提升异常类权重默认值至 2.5，默认启用加权采样并提供 `--sampler_abnormal_boost`（默认 2.0）以放大异常样本出现频次。
  - 自动教师预训练轮数提升至 15，并新增教师质量守卫（F1/TPR 低于阈值则禁用 KD），避免弱教师蒸馏拖低学生召回。
  - 保持早停最低训练轮数与组合指标，配合上述平衡手段，优先降低 miss rate。

## 需求符合性检查
- 分段卷积 + 8 token 池化 + photonic MLP + 均值池化分类流水线已实现并可配置（`models/student.py`）。
- Adam 优化、梯度裁剪、可调 batch_size/epochs/early stopping/LR 调度均在 `train.py` 实现。
- 类别权重、CrossEntropyLoss、KD 的 logit/feature 蒸馏、可切换 KD 流程均实现。
- 数值约束通过 `tanh` 重参数化和输入缩放；输出 logits 未强制约束，符合需求。

## v2_debug 问题分析（基于最新日志）
- **现象**：`Val F1≈0.11，miss≈0%，FPR≈100%`，即模型几乎把所有样本预测为异常；教师自动预训练 `ValLoss≈0.073` 表现正常。
- **数据规模**：Train 42232 / Val 10558 / Generalization 29781，与原始划分一致。
- **原因排查**：
  1. **双重过度放大异常类**：默认同时开启加权采样（异常倍率 2.0）和类别权重（异常 2.5x），在异常占比本就较低的情况下导致正类损失权重/采样概率远大于正常类。为了降低 FN，模型倾向于“全预测异常”，造成 FPR 100%。
  2. **教师正常，KD 信号不是主因**：教师验证损失低、质量守卫通过，学生崩塌主要来自损失权重与采样策略的偏置。
  3. **早停仍保留了崩塌解**：组合指标 `F1 - miss - FPR` 对 FPR=100% 会给出极低得分，但由于训练曲线未明显改善且 patience 已耗尽，提前在劣质解上停止。
- **改进措施（已代码化）**：
  - 下调默认异常类权重至 1.0，并新增 `--max_class_weight_ratio`（默认 2.0）对权重做比值裁剪，防止极端不平衡导致“全异常”预测。
  - 默认关闭采样与类别权重，需手动开启且倍率可控（采样默认 1.0），降低叠加偏置风险。
  - 新增 `--use_class_weights/--no-use-class-weights` 与 `--use_weighted_sampler/--no-use-weighted-sampler` 便于快速排查与调优。
  - 继续保留 F1/miss/FPR 组合早停，并提高 patience/min_epochs，避免短期波动导致的提前停止。

## v3_debug 问题分析（最新日志：FPR=100%，miss=0%，F1≈0.11）
- **现象**：学生依旧“全预测异常”，验证/泛化 TN≈0；教师预训练正常（ValLoss≈0.073）。
- **定位结果**：
  1. **正类偏置叠加仍在**：上轮虽降低权重/采样倍数，但默认依然开启，累积偏置仍将损失重心推向正类，模型在早期就进入“全异常”解。
  2. **缺少在线纠偏**：崩塌出现后，训练循环缺乏自动降权/换采样的防护，导致后续梯度继续固化该解。
- **本轮修复**：
  - 将重加权与重采样默认关闭（需显式开启），并将异常权重默认设为 1.0、权重比值上限 2.0、采样默认倍率 1.0，降低初始偏置。
  - 在训练循环中新增“正类崩塌”自检：若验证集满足 `FPR>95%` 且 `miss<5%`，自动切换为无权重、无采样的 DataLoader，并移除 CE 类别权重，重置早停计数，强制模型重新学习正常类。
  - 延长 patience/min_epochs 至 25，给自检后的再训练留出迭代窗口。
- **后续建议**：
  - 如需提升召回，可先单独调高 `class_weight_abnormal` 或 `sampler_abnormal_boost`，避免权重与采样叠加；观察 FPR，一旦上升可降低倍率或依赖自检。
  - 若仍出现崩塌，可暂时关闭 KD，或降低蒸馏温度，待学生稳定后再恢复 KD。

### v3_debug 补充（最新日志：miss≈50%，FPR≈1%）
- **现象**：最新运行中 FPR 已压到 1% 左右，但 miss 达到 50%+（大量异常被判为正常），Val AUC 较低，召回严重不足。
- **根因排查**：
  1. **失衡反向偏置**：为避免“全异常”崩塌，上一轮默认关闭了类别权重和重采样，结合 MIT-BIH 异常占比偏低，导致模型在 CE 训练中主要优化正常类 → 异常召回大幅下降。
  2. **缺少召回侧自适应**：训练循环只有正类崩塌（FPR>95%）的纠偏逻辑，没有在高 miss、低 FPR 场景下自动提高异常权重或采样，导致模型长期停留在“全正常”侧。
  3. **KD 信号不足以拉回**：教师质量守卫仍可用，但学生在不平衡 CE 下过早收敛为正常预测，蒸馏信号（尤其特征对齐）难以扭转类别偏置。
- **本轮修复**：
  - **温和默认重平衡**：将默认 `class_weight_abnormal` 上调至 1.2，并保持比值上限 2.0；`sampler_abnormal_boost` 设为 1.2，仍默认关闭重采样，但为召回救火预留。
  - **召回自适应**：新增 `--enable_adaptive_reweight`（默认开）。若验证集出现 `miss>30%` 且 `FPR<20%`，自动：
    1) 提升异常类权重（乘以 1.25，并受比值上限约束）；
    2) 若未启用采样，则开启加权采样，并将异常采样倍率提升至 ≥1.5。
    这样在保证 FPR 可控的前提下主动拉升召回。
  - **记录与可控性**：保留崩塌检测（FPR>95%、miss<5%），并允许通过 `--no-enable-adaptive_reweight` 禁用该策略，便于做对照实验。
  - **训练日志可视化**：继续在 `artifacts/` 输出 loss/metrics 曲线、ROC、混淆矩阵，便于观察召回修复效果。
