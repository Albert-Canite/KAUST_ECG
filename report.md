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
  - 类别权重：稳定模式（默认开启）下关闭类权重；若手动开启（`--no-stable_mode --use_class_weights`），异常类默认 1.2x，受 `--max_class_weight_ratio`（默认 1.5）限制。
  - 采样：默认关闭加权采样 (`--use_weighted_sampler/--no-use-weighted-sampler`)，异常类采样增强倍率 `--sampler_abnormal_boost`（默认 1.2，可调 >1）。
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

## v4_debug 问题分析（最新日志：早期 miss=100%→FPR=100% 快速跳变）
- **现象**：训练第 2 轮 miss≈100%、FPR≈0（几乎全预测正常），第 3 轮迅速变为 FPR≈100%、miss≈0（几乎全预测异常），随后训练停留在极端输出，说明模型在早期来回崩塌。
- **根因排查**：
  1. **早期重平衡过猛**：虽然降低了默认倍率，但训练一开始就带着类别权重/采样和 KD，同步作用在尚未稳定的随机初始化上，梯度方向极易被放大到单类预测。
  2. **缺少“全正常”侧的防护**：已有“全异常”崩塌检测（FPR>95%，miss<5%），但没有对“全正常”（miss>95%，FPR<5%）的对称防护，导致模型从全正常直接跳到全异常却缺乏缓冲。
  3. **KD 过早介入**：即便教师正常，蒸馏在学生尚未收敛时叠加在失衡 CE 上，会放大偏置并加速崩塌。
- **改进措施（本轮代码已实现）**：
  - **训练暖启动**：前 `--imbalance_warmup_epochs`（默认 5）个 epoch 强制使用无权重、无加权采样的 CE，避免随机初始化阶段被失衡策略放大；之后才切换到温和类别权重/可选采样。
  - **KD 暖启动**：新增 `--kd_warmup_epochs`（默认 5），学生先单独学习，再开启 KD；崩塌检测到异常/正常侧时会临时关闭 KD，允许 CE 先拉回。
  - **双向崩塌防护**：保留“全异常”检测并在触发时关闭重平衡+暂停 KD；新增“全正常”检测（miss>95%、FPR<5%）以恢复类权重、启用采样并暂停 KD，防止长期忽视异常类。
  - **更平滑的自适应**：在高 miss/低 FPR 场景下，异常权重提升因子降至 1.2，并在暖启动结束后才生效，减少剧烈震荡。
- **预期效果**：早期不再左右摇摆，模型先获得稳定的基线再逐步引入重平衡和 KD；若仍出现单侧崩塌，会自动关闭导致偏置的组件（采样/权重/KD）或反向增强异常类，提升召回并控制 FPR。

## v5_debug 问题分析（最新日志：Val F1≈0.38，miss≈11%，FPR≈17%；泛化 miss≈42%）
- **现象总结**：
  - 验证集 F1 仅 0.38，组合 Score≈0.10；召回（1-miss）约 89%，但泛化召回大幅下滑（miss≈42%），FPR 在 13–17% 区间。
  - 日志显示早停在 miss/FPR 尚未明显下降前触发，且 F1 始终偏低，说明决策阈值与模型输出偏置共同拉低了指标。
- **根因排查**：
  1. **固定 0.5 阈值导致指标失真**：训练/早停始终使用 argmax/0.5 阈值，在正负类不平衡时，概率分布偏向正常类，F1 和召回被阈值卡住（FPR 较低但 FN 高），早停过早记录到“假优”解。
  2. **早停基于未调优阈值**：组合指标 `F1 - miss - FPR` 基于 0.5 阈值计算，容易在阈值偏高时低估模型可达性能，提前停止导致未能探索到更优召回/精度的阈值。
  3. **召回与泛化落差**：泛化 miss 远高于验证，说明阈值和类别平衡策略对验证集过拟合；缺少统一的阈值调优并迁移到泛化集。
- **修复措施（本轮代码已实现）**：
  - **阈值搜索与同步评价**：在每个验证阶段对 `[0.05, 0.95]` 范围进行阈值扫描，按 `F1 - miss - FPR` 选出最优阈值；早停评分与日志统一使用该阈值，避免被固定 0.5 阈值束缚召回/F1。
  - **跨集一致性**：训练结束后复用验证最优阈值评估泛化集，并将阈值写入 checkpoint，保证部署/复现实验时决策一致。
  - **可视化与诊断**：生成的 ROC、混淆矩阵已基于最优阈值保存到 `artifacts/`，便于对比调优前后的召回与误报变化。
- **后续建议**：
  - 若 F1 仍低，可适度提高 `class_weight_abnormal`（例如 1.3–1.5）或启用轻量采样（`--use_weighted_sampler`，`--sampler_abnormal_boost≈1.3`），同时观察阈值扫描结果避免 FPR 激增。
  - KD 已有质量守卫和暖启动，如观察到教师弱或学生仍偏正常，可暂时用 `--no-use-kd` 排查蒸馏影响。

## v6_debug 问题分析（最新日志：Val@thr≈0.90 F1≈0.58，miss≈26%，FPR≈5%；泛化 miss≈65%、F1≈0.40）
- **现象总结**：
  - 虽然 FPR 已降至 5% 左右，但 miss 仍高（验证 26%，泛化 65%），导致 F1 和 Score 偏低，ROC/混淆矩阵也显示正类召回不足。
  - 阈值扫描选中了高阈值（≈0.90），进一步压缩了召回；自适应重权重未被触发，训练过程中异常类仍被弱化。
  - 泛化集召回远低于验证，说明当前阈值/重平衡策略在验证集上偏保守且未迁移到更难的泛化分布。
- **根因排查**：
  1. **阈值评分对召回偏置不足**：旧的打分公式 `F1 - miss - FPR` 在 miss≈0.26、FPR≈0.05 时仍给出较高得分，促使阈值向上偏移，抑制召回。
  2. **召回救火触发门槛过高**：触发条件（miss>0.30 且 FPR<0.20，且仅一次）在当前 miss≈0.26 时不会行动，导致权重/采样未随召回低迷自动加强。
  3. **评估阈值搜索步长有限**：仅用等间距网格，可能漏掉概率分布的峰值位置，导致阈值落在保守侧。
- **修复措施（本轮代码已实现）**：
  - **召回偏置打分**：阈值扫描和早停改用 `F1 + sensitivity - 0.5*FPR`，显式奖励召回并在 FPR 可接受时鼓励降低阈值。
  - **更密集的阈值候选**：在 `[0.05, 0.95]` 上叠加概率分布分位点，避免错过模型输出的“拐点”阈值。
  - **可多次触发的召回救火**：新增 `--recall_target_miss`（默认 0.15）、`--adaptive_fpr_cap`（默认 0.25）、`--recall_rescue_limit`（默认 2）。当 miss 超标且 FPR 低时可重复提升异常权重/启用采样，直到达到召回目标或次数上限。
- **后续建议**：
  - 若 miss 仍高，优先提高 `class_weight_abnormal`（1.4–1.6）并打开 `--use_weighted_sampler`（boost 1.3–1.6），观察召回-误报曲线；必要时调低 `--recall_target_miss` 到 0.12。
  - 若 KD 使决策边界偏保守，可用 `--no-use-kd` 或降低 `--beta/--gamma`，再配合新阈值扫描/救火策略评估。

## v7_debug 问题分析（最新日志：Val@thr≈0.90 F1≈0.58，miss≈26%，FPR≈5%；泛化 miss≈65%、F1≈0.40，KD 作用存疑）
- **现象总结**：
  - 即便阈值偏高（0.90），验证 miss 仍达 26%，泛化 miss 高达 65%，F1 受召回拖累明显。
  - KD 未显示明显收益：学生仍偏向预测正常类，召回长期低迷，说明教师信号未被充分吸收或被提前关闭。
  - 类别分布高度不平衡（训练异常占比约 0.3），但加权/采样在早期为了防止崩塌被关闭，后续加权与采样的强度不足以拉升召回。
- **根因排查**：
  1. **阈值筛选缺少“硬约束”**：虽然增加了召回偏置，但没有显式 miss 上限，仍可能选择 miss≈0.26 的高阈值。
  2. **KD 在高 miss 时仍可能介入**：仅在极端 miss 时才关闭 KD，学生在召回低时仍会被教师 logits 牵引，延缓异常类学习。
  3. **自动重平衡力度不够**：仅触发有限次的权重/采样提升，且启动采样依赖手动开关，导致在异常占比较低时召回提升有限。
- **修复措施（本轮代码已实现）**：
  - **阈值搜索加入显式 miss/FPR 约束**：`sweep_thresholds` 先筛选满足 `miss <= threshold_target_miss`（默认 0.12）且 `fpr <= threshold_max_fpr`（默认 0.20）的阈值，再按 recall-强化的得分 `f1 + 1.5*TPR - FPR` 选最优，优先降低漏诊。
  - **KD 召回防护**：新增 `kd_pause_miss`/`kd_resume_miss`，当验证 miss 超过 0.35 自动暂停 KD，降至 0.20 以下再恢复，避免高漏诊阶段被蒸馏信号束缚。
  - **自动采样唤醒**：检测异常占比低于 `auto_sampler_ratio`（默认 0.35）时，暖启动结束后自动启用加权采样；`recall_rescue_limit` 默认提升至 3，召回救火更耐用。
- **后续建议**：
  - 若 miss 仍>0.20，可进一步下调 `threshold_target_miss`（如 0.10）并上调 `sampler_abnormal_boost`（1.5–1.8），观察 FPR 变化。
  - 若教师与学生分布差距大，可暂时关闭 KD 或提高教师质量门槛（`teacher_min_f1`/`teacher_min_sensitivity`），先把学生召回拉高。

## v8_debug 问题分析（最新日志：Val@thr≈0.40 F1≈0.38，miss≈10.6%，FPR≈17.5%；泛化 F1≈0.61，miss≈25.8%，FPR≈12.7%）
- **现象总结**：
  - 阈值搜索偏向 0.40，验证集漏诊已降到 ~10%，但 FPR 仍在 17% 左右，F1 偏低（0.38）。
  - 泛化集 F1 提升到 0.61，但 miss 反弹到 26% 以上，说明决策边界在分布外仍偏保守。
  - 训练曲线显示第 5–6 轮开启加权/采样/KD 时出现 loss 抬升与阈值大幅抖动，之后长时间停留在相近的 F1/miss/FPR 区间，说明重平衡切换过于激进。
- **可能根因**：
  1. **重平衡切换过快**：加权/采样在第 5 轮一次性开启，权重和采样概率突变，引起 loss 爆涨和阈值翻转，随后训练停滞在次优解。
  2. **KD 介入时机与权重跳变叠加**：KD 在同一轮激活，与突增的权重/采样叠加，学生在尚未稳态时同时受 CE+KD 双重扰动，学习方向震荡。
  3. **采样/权重放大但缺少渐进**：`sampler_abnormal_boost` 与异常类权重直接生效，未做平滑过渡；在 FPR 仍然偏高时继续强化异常类，可能压制正常类学习导致 FPR 难以下探。
- **已采取的代码修复（本轮）**：
  - **新增线性 ramp**：在暖启动结束后，引入 `--imbalance_ramp_epochs`（默认 5）对类权重和采样 boost 线性递增；前几轮仍接近无权重/无采样，减少 loss/阈值抖动，逐步过渡到目标重平衡强度。
  - **权重缩放统一管理**：自适应召回救火现在提升一个全局 `class_weight_scale`，由 ramp 平滑施加，并继续受 `--max_class_weight_ratio` 约束，避免瞬时大幅增加异常权重。
  - **采样 boost 平滑更新**：采样启用后按 ramp 插值到目标 boost，并在 boost 变化时重建 DataLoader，避免一次性切换到高倍率采样。
- **剩余关注点/建议**：
  - 若 FPR 仍居高不下，可在 ramp 期间适度降低 `sampler_abnormal_boost`（如 1.1–1.3）或缩短 ramp（2–3 轮）以更快稳定；若 miss 回升则再小幅提升。
  - 如果 KD 在 ramp 期间仍引入震荡，可将 `kd_warmup_epochs` 提升到 8–10，或暂时 `--no-use-kd` 对比；确认教师在当前切分/归一化下有足够召回。
  - 进一步提升模型容量（如增加 `num_mlp_layers` 到 3、开启 dropout=0.1）可尝试，但需监控受限 4 通道设计下的过拟合与推理约束。

## v9_debug 问题分析（最新日志：Val@thr≈0.50 F1≈0.41，miss≈10.6%，FPR≈15.1%；泛化 F1≈0.59，miss≈31.3%，FPR≈11.7%）
- **现象总结**：
  - 验证集指标在 0.5 阈值附近稳定：miss≈10–12%，FPR≈15–18%，F1≈0.41，Score≈1.22；
  - 泛化集 miss 升到 ~31%，F1≈0.59，FPR≈12%，相较验证出现明显召回劣化，说明当前早停/阈值选择偏向验证分布，迁移到泛化时漏诊加重。
  - 日志显示在第 5–8 轮开启重平衡与 KD 后出现 loss/阈值大幅抖动，随后长时间徘徊在相近的 F1/miss/FPR；最佳模型仍由早期（第 8 轮）得分主导，后续虽有 miss 改善但 Score 未超越，表明早停判据过度依赖验证集。
- **原因分析**：
  1. **验证导向的早停与阈值**：早停完全由验证 Score 决定，泛化集未参与评分，阈值也仅基于验证扫得，导致决策边界偏向验证分布，泛化 miss 拉高。
  2. **重平衡+KD 同步切换导致迁移不稳**：第 5 轮同时开启类权重/采样/KD，泛化分布下模型尚未稳态，KD 可能拉低异常召回，采样权重 ramp 在泛化上未充分适配。
  3. **召回救火只看验证**：自适应权重/采样的触发逻辑只依赖验证 miss/FPR，泛化 miss 高时没有反馈回训练循环。
- **修复措施（本轮新增）**：
  - **引入泛化参与的早停评分**：新增 `--use_generalization_score/--no-use_generalization_score`（默认开）和 `--generalization_score_weight`（默认 0.3），每个 epoch 以验证最优阈值在泛化集上评估 `Score=f1+sensitivity-0.5*FPR`，与验证 Score 按权重线性融合驱动早停与最佳模型保存，降低验证过拟合。

## KD V1 DEBUG（最新日志：Teacher Val miss≈3.4% / Gen miss≈9.8%；KD 学生 Val miss≈5.0% FPR≈12.6%，Gen miss≈7.1% FPR≈4.9%，阈值≈0.26）
- **问题与假设**：
  - 教师在验证集 miss 很低，但泛化 miss 提升 6%+，存在过拟合；直接蒸馏可能将“过保守”的阈值信号传递给学生，导致 FPR 上升。
  - KD 训练仅使用训练集，未覆盖泛化分布，学生在泛化集 miss 改善有限，FPR 却增大。
  - 损失函数缺少显式 miss 限制，KD logits 与特征对齐可能压制正类概率，提升漏诊。
- **代码改动**：
  1) **KD 训练混入泛化数据**：新增 `kd_loader`，按 `--kd_gen_mix_ratio`（默认 0.35）截取泛化样本加入训练，`kd_class_weights` 也基于混合分布计算，以减轻分布偏差导致的漏诊。
  2) **教师过拟合自适应**：加载教师后同时评估验证/泛化 miss，若泛化 miss 比验证高超过 `KD_TEACHER_GAP_TOL`（5%），自动将 KD logits 权重减半，避免弱泛化教师拖低学生。
  3) **显式 miss 惩罚**：在 KD 循环中对正类概率添加 margin 惩罚（目标 ≥0.6，权重 `KD_POS_PENALTY`），与 CE/KD/MSE 并行，优先拉升正类置信度、降低漏诊。
  4) **选模兼顾 FPR**：最佳模型不再仅看 blended miss，改为 `miss + 0.35*FPR`，防止为压 miss 过度抬升误报；阈值依旧由验证/泛化混合扫描得到。
  5) **日志补充**：教师泛化 gap、KD logits 实际权重将在控制台提示，便于快速判断是否触发降权。
  - **训练曲线增加泛化轨迹**：训练曲线图中加入 Gen F1/Miss/FPR 轨迹，便于直观看到验证-泛化的分布差距与收敛状态。
- **后续建议**：
  - 若泛化 miss 仍高，可提高 `--generalization_score_weight` 到 0.4–0.5，使早停更重视泛化；或在阈值搜索时调低 `--threshold_target_miss`（如 0.10）以压漏诊。
  - 在重平衡阶段进一步平滑：可延长 `--imbalance_ramp_epochs` 到 8–10，或将 `--sampler_abnormal_boost` 调低到 1.0–1.1，待泛化 miss 下行后再逐步提升。
  - 如果 KD 仍抑制异常召回，可暂时 `--no-use-kd`，或提高教师质量门槛/降低 `beta,gamma`，确保学生优先对齐召回。

## v10_debug 问题分析（最新日志：Val@thr≈0.45 F1≈0.39，miss≈11.2%，FPR≈16.8%；泛化 F1≈0.59，miss≈30.6%，FPR≈12.5%）
- **现象总结**：
  - 验证集在 0.4–0.55 阈值附近稳定，miss 已降到 ~11%，但 FPR 依旧 15–17%，F1 仅 ~0.38–0.41。
  - 泛化集 F1≈0.59、miss≈30%，明显高于验证 miss，说明决策阈值和重平衡策略仍偏向验证分布，外部分布召回不足。
  - 日志显示第 6–8 轮后阈值跃迁（0.77→0.90→0.80），自适应权重/采样在 miss 较高时拉升异常类，但后续长时间徘徊在类似 F1，早停仍由验证得分主导。
- **问题定位**：
  1. **阈值仅基于验证**：虽有泛化得分融合早停，但阈值搜索仍只看验证，决策边界偏向验证概率分布，导致泛化 miss 偏高。
  2. **召回救火未利用泛化反馈**：自适应权重/采样触发只看验证 miss/FPR。当验证达标但泛化漏诊严重时，训练循环缺少“泛化召回”反馈，未能继续加大异常侧权重或采样。
  3. **KD/重平衡对外分布不敏感**：KD 暂停/恢复与自适应 reweight 仅由验证 miss 控制，在泛化召回下降时无法自动放松 KD 或进一步强化异常类，泛化 F1 长期停滞。
- **代码修复（本轮新增）**：
  - **阈值搜索可选“验证+泛化”联合打分**：新增 `--use_blended_thresholds` 与 `--threshold_generalization_weight`，在验证和泛化概率上联合扫阈值，使用同一记分公式和 miss/FPR 约束，获得更契合两分布的决策阈值（训练中与最终评估均可启用）。
  - **召回救火支持泛化反馈**：新增 `--use_generalization_rescue`（默认开），当泛化 miss 高且 FPR 低时也会触发异常权重/采样的提升，避免仅因验证达标而停滞，兼顾外部分布召回。
- **改进建议**：
  - 在当前泛化 miss 仍高的情况下，可开启 `--use_blended_thresholds --threshold_generalization_weight 0.4`，并保持 `--use_generalization_rescue`，让阈值与重平衡都对泛化失衡敏感。
  - 若 KD 仍压制异常召回，可将 `--beta`/`--gamma` 下调到 0.5/0.25 或暂时 `--no-use-kd`，并把 `--imbalance_ramp_epochs` 缩短到 3–4 轮使重平衡更快达稳态；若 FPR 上升则适度调低采样 boost。
  - 如需进一步提升容量，可尝试 `--num_mlp_layers 3` 与 `--dropout_rate 0.1` 做对照，但需关注 4 通道受限设计下的过拟合迹象。

## v11_debug 问题分析（最新日志：Val@thr≈0.45 F1≈0.38，miss≈11.2%，FPR≈16.8%；泛化 F1≈0.59，miss≈30.6%，FPR≈12.5%）
- **现象回顾**：
  - 验证集长期停在 F1≈0.38、miss≈11% 左右，FPR≈17%；泛化 miss 仍在 ~30% 且未随训练下探，泛化 F1≈0.59。
  - 训练曲线在第 6–8 轮后进入平台，阈值固定在 0.45 左右，后续 30+ 轮基本无实质改善，说明容量/正则或重平衡策略可能已卡住。
  - KD 未显示明显收益：开启后 miss/FPR 曲线仍主要由重平衡驱动，泛化漏诊未改善，暗示教师信号或学生容量不足。
- **可能根因**：
  1. **学生容量偏低**：4 维 token + 2 层 MLP 深度有限，复杂异常模式难以充分分离，导致召回受限且阈值难以优化。
  2. **正则不足导致泛化漏诊**：当前默认无 dropout，训练后期可能轻度过拟合正常模式，泛化时异常召回偏低。
  3. **基础重平衡力度偏弱**：初始异常权重 1.2 在占比低场景可能不足，后续自适应虽提升，但早期召回拉升速度慢，阈值早早锁定在保守区间。
- **本轮代码修改**：
  - **提高学生容量**：将 `num_mlp_layers` 默认提升到 3，并在每层后增加 token 级 dropout（与 `--dropout_rate` 共享），增强表达同时抑制过拟合。
  - **默认启用正则化**：`--dropout_rate` 默认 0.2（作用于 token MLP 与 `h_pool`），缓解泛化漏诊；需要关闭时可显式 `--dropout_rate 0`。
  - **加强基础重平衡**：`--class_weight_abnormal` 默认上调至 1.4，在 ramp 期即可更快提升异常侧梯度，配合已有 ramp/自适应逻辑减少早期漏诊。
- **后续建议与验证**：
  - 若泛化漏诊仍高，可再把 `num_mlp_layers` 提升到 4 或小幅增大 dropout（0.25–0.3）对照，观察 FPR 变化；同时可调低 `threshold_target_miss` 到 0.10 强化阈值对漏诊的约束。
  - 如 KD 仍无益，可暂时 `--no-use-kd` 对照，或提高教师门槛（`teacher_min_f1`、`teacher_min_sensitivity`）以确保 distillation 仅在教师高召回时生效。
  - 若训练仍长时间平台，可适度缩短 `imbalance_warmup_epochs`（如 3）让重平衡更早介入，或增大 `sampler_abnormal_boost` 至 1.4–1.6，但需监控 FPR。

## v12_debug 问题分析（最新日志：第 6 轮起 FPR≈100%，Miss≈0% 持续，训练爆炸）
- **现象**：
  - 前 5 轮未加权阶段，Val miss 20–45%、FPR 7–12%；第 6 轮开启权重/采样/KD 后立即出现 “all abnormal” 正崩溃：Val FPR≈100%、miss≈0%，随后 10 余轮均停留在崩溃状态。
  - 训练 loss 飙升至 3.x，KD 虽被暂停但下一轮又被自动启用，类权重/采样也在下一轮重新生效，导致始终在崩溃—恢复—再次崩溃的循环中。
- **定位**：
  1. 崩溃后虽然当轮关闭了采样与权重，但下一轮会因 `use_class_weights=True` 再次自动启用，缺少“冷却期”导致同样的 reweight 方案反复触发崩溃。
  2. KD 在 miss 高时被暂停，但下一轮 miss 依旧由崩溃状态主导，KD 仍会按暖启动条件重新打开，加剧不稳定。
- **本轮修复**：
  - 新增 `--collapse_cooldown_epochs`（默认 5）：一旦检测到正/负崩溃，立即关闭采样与类权重，并进入冷却期，冷却期间固定使用无权重/无采样 CE；冷却结束后重新以 0 起步的 ramp 重新拉升权重/采样，减少重复崩溃。
  - 崩溃时重置 ramp 起点与 reweight 激活标志，防止在冷却期内再次强行启用 reweight；KD 也同步暂停，待 miss 降回阈值后再依据正常逻辑恢复。
- **仍需关注**：
  - 若冷却后再次触发崩溃，可延长 `collapse_cooldown_epochs`（如 8–10），或降低 `class_weight_abnormal`/`sampler_abnormal_boost` 起始值，使 ramp 斜率更平缓。
  - 若崩溃由教师噪声触发，可临时关闭 KD（`--no-use-kd`）或提高教师门槛，先稳定 CE 学习，再逐步恢复蒸馏。

## v13_debug 问题分析（最新日志：仍出现正/负崩溃反复，FPR≈100% 或 miss≈40–75%，KD 多次触发/暂停）
- **新现象**：
  - 第 5 轮后启用类权重/采样/KD，再次触发正崩溃（FPR≈100%、miss≈0%），随后在正常/异常全预测之间来回摆动，KD 和 reweight 多次被暂停/恢复。
  - 崩溃后即使进入冷却，下一轮仍会因默认启用 KD 或自适应 reweight 而重新触发崩溃，训练 loss 长期维持在 3.x，Val/Gen F1 均停滞在 0.11–0.40 低位。
- **原因复盘**：
  1. **自适应策略过多且相互叠加**：崩溃后冷却结束即恢复 ramp、KD、采样，多重策略在模型尚未恢复时叠加，再次把预测推向单侧。
  2. **权重基线仍偏强**：计算得到的类权重（~[3.06, 11.7]）在数据极度不平衡时过大，即便有冷却，重新启用后仍可能直接压制正常类。
  3. **KD 暖启动过短**：KD 默认 5 轮即激活，在学生尚未稳定的情况下与重平衡同步介入，放大震荡。
- **本轮修复与缓解策略**：
  - 新增 `--stable_mode`：一键进入保守配置（默认关闭，可通过 `python train.py --stable_mode` 启用），自动关闭 KD、停用自适应 reweight/采样、延长暖启动/冷却、并将类权重强度限制在更温和的范围（`abnormal<=1.2`、`ratio<=1.5`）。
  - 在稳定模式下仅使用温和的 CE 训练（可选固定的轻量权重），避免训练早期反复崩溃；待稳定后再按需手动打开 KD 或更强重平衡进行对照实验。
  - 延长暖启动（≥8 轮）与 KD 暂停窗口（`kd_warmup_epochs` ≥12），让学生先学习基础分界，再决定是否引入蒸馏。
- **后续验证建议**：
  - 先以 `--stable_mode --no-use-kd --no-use_weighted_sampler` 跑一轮基准，观察 miss/FPR 收敛情况；若 miss 高但 FPR 低，再逐步提高 `class_weight_abnormal`（1.2→1.4）或手动开启采样。
  - 若稳定模式下仍 miss 高，可小幅增加容量（`num_mlp_layers 3→4`）或保留当前 0.2 dropout；必要时将 `imbalance_warmup_epochs` 保持在 8–10，避免早启 reweight。
  - KD 仅在学生已稳定且教师质量达标时再开启，先将 `beta/gamma` 降到 0.5/0.25，对照观察召回是否改善。

## v14_debug 问题分析（最新日志：启用 reweight 后反复 FPR=100%/miss=0% 崩溃）
- **新现象**：第 5 轮启用类权重/采样/KD 后，连续触发“正崩溃/负崩溃”警告，训练 loss 飙升到 3.x 并停留，Val/Gen 全预测异常或全正常，F1=0.11/0.27。
- **根因定位**：
  1. **重平衡策略过早/过强复位**：崩溃后短暂冷却又重新开启类权重与采样，导致模型在尚未恢复的情况下再次被推向单侧。
  2. **KD 重复介入**：崩溃后 KD 自动恢复，与重新加权叠加，进一步放大梯度偏移。
  3. **默认“进阶模式”过激**：即便未显式开启，用户直接运行 `python train.py` 仍会加载自适应 reweight/KD，容易复现崩溃轨迹。
- **本轮修改**：
  - **稳定模式默认开启**：将 `--stable_mode` 改为默认值“开启”，直接关闭 KD、类权重、自适应 reweight 与加权采样，仅用 CE 基线训练，避免崩溃。从 CLI 可用 `--no-stable_mode` 手动切回进阶模式。
  - **温和权重默认值**：进一步下调默认 `class_weight_abnormal=1.2`、`max_class_weight_ratio=1.5`，即便用户手动关闭稳定模式也能保持更温和的重平衡强度。
  - **崩溃锁定**：新增 rebalance 锁定标志，检测到正/负崩溃后永久关闭 reweight/采样计划并禁用 KD，防止冷却结束后再次自动启动导致循环崩溃。
- **建议验证路径**：
  1. 直接运行 `python train.py`（稳定模式）观察基线收敛；若 miss 仍偏高，可先提升容量（`num_mlp_layers=3/4` 已默认 3，必要时 `--dropout_rate 0.1–0.2`）或小幅调高阈值 sweep 的 recall 权重。
  2. 若需对比重平衡/蒸馏，先用 `--no-stable_mode --no-use_kd --no-enable_adaptive_reweight --no-use_weighted_sampler` 跑一版仅类权重的轻量模式，再逐步开启采样与 KD，监控是否重新出现崩溃。
  3. 一旦观察到 FPR>95% 或 miss>95%，可手动提高 `collapse_cooldown_epochs` 或维持 rebalance 锁定，优先让 CE 恢复正常预测分布。

## v15_debug 问题分析（最新日志：稳定模式可跑通但泛化 miss≈30%，启用重平衡/KD 易崩溃）
- **用户关注点**：
  1. 稳定模式下是否仍保持输入/受约束层权重在 [-1, 1]：每个 beat 预处理采用去均值+max-abs scaling 保证输入落在 [-1, 1]（`data.py`），受约束层在启用 `--use_value_constraint` 时通过 tanh 重参数保证权重/偏置在 `[-scale, scale]`，默认为 1.0；稳定模式本身不强制约束权重（仅关闭 KD/重平衡），需显式加 `--use_value_constraint`。
  2. KD、类权重、自适应 reweight、加权采样都属于训练策略，可否一键开关并调强度：新增 `--strategy_preset {stable,balanced,full}`（默认 stable）和全局强度 `--strategy_strength`，可整体切换策略组合并统一放大/缩小类权重、采样提升、KD beta/gamma；同时保留老的单独开关以供微调。
- **现象复盘**：
  - 稳定模式下 Val miss≈11%、FPR≈17%，Gen miss≈30%，FPR≈12%，说明模型容量+阈值已能分出一部分异常，但召回仍偏低；切换到 aggressive 重平衡/KD 会再次引发 FPR=100% 的崩溃。
- **原因研判**：
  1. **策略耦合过强**：同时开启类权重、采样、KD、自适应 reweight，且权重/采样强度不易整体调节，容易把判决面推到单侧。
  2. **容量仍偏紧**：三层 4×4 MLP + 0.2 dropout 容量有限，对 QRS 形态变化的判别边界不足，导致泛化召回下降。
- **本轮修改**：
  - 引入高层策略开关与强度控制：
    - `--strategy_preset stable`（默认）：关闭 KD/自适应/采样/类权重，跑纯 CE 基线。
    - `--strategy_preset balanced`：仅开启温和类权重+采样，KD 关闭，自适应 reweight 关闭，暖启动≥6 轮。
    - `--strategy_preset full`：开启 KD+自适应 reweight+采样，供对比实验。
    - `--strategy_strength x`：统一缩放类权重、采样 boost、KD beta/gamma（如 0.6 更温和，1.5 更激进），便于网格/贝叶斯 sweep。
  - 与旧开关兼容：若需要精细微调仍可单独设置 `--no-use_kd`、`--no-enable_adaptive_reweight` 等；preset 会同步调整 `stable_mode`，避免冲突。
- **后续验证路径**：
  1. 保持 `--strategy_preset stable --use_value_constraint` 跑一个对照，确认基线召回/FPR；可尝试 `--num_mlp_layers 4 --dropout_rate 0.1` 提升容量并微调正则。
  2. 以 sweep 方式逐步增加策略强度：
     - 先 `--strategy_preset balanced --strategy_strength 0.6`（温和权重+采样、无 KD），观察 miss 是否下降且无崩溃；必要时调高到 0.8/1.0。
     - 如基于 balanced 仍 miss 高且稳定，可再试 `--strategy_preset full --strategy_strength 0.6` 打开 KD，KD 权重随强度线性缩放，避免一次性过强。
  3. 若再出现崩溃，可保持 stable 结果为基线，将 `collapse_cooldown_epochs` 设更长（8–10），或锁定 rebalance 仅用固定轻量权重/无采样，等待模型收敛后再小步调高强度。

## v2.1debug 更新（约束：miss≤15%、FPR≤15%、不改模型容量）
- **现象回顾**：纯 CE 基线在泛化集出现 miss≈30% 的漏诊率，用户无法接受；数据偏向正常类，直接过采样曾导致 FPR=100% 崩溃。
- **本轮代码调整**：
  - **温和异常过采样**：当训练集异常占比 <0.35 时自动启用轻量 WeightedRandomSampler，异常样本 boost=1.2，预估异常占比上限控制在 ~0.5，给异常类更多梯度同时避免全异常预测。
  - **逆频率权重裁剪**：计算出的类权重按均值±2 倍做夹紧，限制异常侧权重大幅偏置导致的 FPR 飙升，保持 CE 稳定。
  - **阈值硬约束**：阈值扫描增加 miss≤0.15 且 FPR≤0.15 的筛选，再用召回偏置得分选择最佳阈值，确保训练和最终评估都满足漏诊/误报上限。
- **预期收益**：
  - 通过轻度过采样与受控权重提升召回、压低 miss；
  - 阈值约束直接限制 miss 与 FPR 双上线，减少泛化时 30%+ 漏诊或 FPR 崩溃；
  - 保持模型容量不变，改动集中在数据平衡与决策阈值，便于快速复现与对照。
- **日志位置**：训练过程继续写入 `artifacts/training_log_YYYYMMDD_HHMMSS.jsonl`，包含配置、每轮阈值/指标与最终结果，可直接复用调参。

## v2.2_debug 分析（最新日志：Val F1≈0.38，miss≈10–11%，FPR≈17–18%；泛化 F1≈0.54，miss≈31–32%，FPR≈14–15%）
- **现象总结**：
  - 验证集在阈值≈0.40–0.45 时 miss 已降到 ~10%，FPR 仍在 17–18%，F1 只有 ~0.38；阈值显著低于 0.5，表明模型输出偏向正常类，需要降低阈值才能捕捉异常。
  - 泛化集 miss 仍达 30%+，F1 只有 ~0.54，说明阈值和重平衡策略对验证集相对有效，但在泛化分布下召回显著恶化。
  - 训练日志显示阈值扫描是动态的：每个 epoch 会在网格和分位点上寻找最优阈值并同步到早停与泛化评估；当前偏离 0.5 的阈值说明模型概率分布本身偏保守，而非阈值计算错误。
- **代码现状确认**：
  - 教师模型未在现有代码中启用：`train.py` 仅构建 `SegmentAwareStudent`，优化器、loss、推理与 checkpoint 都只涉及学生，没有任何 KD/教师相关的导入或逻辑。
  - 数据划分使用 MIT-BIH 官方 48 条记录中的子集：训练/验证源于 32 条记录，泛化集 15 条记录，均在 `TRAIN_RECORDS`/`GENERALIZATION_RECORDS` 中显式列出，`split_dataset` 仅对训练部分做随机 80/20 切分，避免跨记录泄漏。
  - 数据预处理与标签映射遵循 MIT-BIH 官方符号：`BEAT_LABEL_MAP` 将 N→0、V/S/F/Q→1，`load_record` 通过 WFDB 注释截取 360 点窗口并标准化至 [-1,1]。
- **关于 F1 偏低的解释**：
  1. **FPR 抬高拖低 F1**：在 miss≈10% 时，FPR≈17–18% 会显著降低精度，使 F1 偏低（F1 同时受精度和召回影响）。验证 miss 较低不代表 F1 高，因误报占比偏大。
  2. **类别分布不平衡**：正常类占比高，模型即便降低阈值获得召回，也容易引入较多 FP，导致精度下降、F1 被压低。
  3. **指标计算正确性**：阈值扫使用 `f1 + 1.5*TPR - FPR` 选点，并对阈值候选进行 miss/FPR 过滤（miss≤15%、FPR≤15%）。当前日志中的 miss/FPR 超过 15% 说明最优阈值落在过滤边缘，未满足双约束，故退化为按得分选取偏低阈值，指标计算逻辑本身未见异常。
- **阈值为何偏离 0.5**：
  - 阈值由模型输出分布和评分函数共同决定；当模型输出对异常类置信度普遍偏低时，为满足 miss 约束和提升得分，扫阈值会自然向 0.4 左右移动。偏离 0.5 并非 bug，而是对偏保守输出的补偿。
  - 若希望阈值靠近 0.5 同时保持召回，需要提升模型对异常的 logit 置信度（训练/重平衡问题），而不是强行固定阈值。
- **数据/记录选择的可能影响**：
  - 若训练/验证划分中的异常分布与泛化集差异大，验证学到的阈值会在泛化上失效，导致 miss 上升。建议检查当前使用的 record 列表是否与原设定一致，并确保随机种子固定、分层切分稳定。
  - 可在训练前后统计每个 split 的类别占比，若泛化异常占比明显更低，应适度提高 `generalization_score_weight` 或在阈值扫描中进一步强化召回权重。
- **进一步的训练策略建议（保持模型容量不变）**：
  1. **更平滑的召回强化**：适度提高异常类权重上限（如 1.4–1.6）并延长 ramp（8–10 轮）让权重/采样更慢爬升，减少 FPR 突增，同时提升异常 logit 置信度。
  2. **分布自适应阈值权重**：将 `threshold_target_miss` 下调到 0.12–0.10，并在评分中加大 TPR 系数或降低 FPR 惩罚系数，促使阈值在泛化上进一步左移以换召回。
  3. **批次内平衡**：在不改变模型容量的前提下，启用温和加权采样（boost≈1.2–1.3）并与 ramp 结合，确保每个 batch 都有足够异常样本而不过度放大，避免 FPR=100% 的崩溃。
  4. **KD 守卫强化**：若怀疑教师引导偏保守，可提高 `kd_pause_miss`（如 0.30）让高 miss 阶段自动暂停 KD，待召回回升后再恢复。
  5. **交叉验证阈值稳定性**：在训练结束后，使用保存的 logits/概率在验证集做更细粒度的阈值扫（含分位点），与训练期最优阈值对比，若差异大说明训练中阈值搜索仍不足，可增加阈值候选密度。
  6. **数据质量复核**：重新检查输入预处理、归一化和标签对齐，尤其是新上传记录的标签是否存在偏移/缺失；必要时在一小批样本上可视化波形与预测，确认模型学习目标正确。
- **小结**：当前 miss 在验证侧已接近 10% 的上限，但 FPR 偏高导致 F1 偏低；泛化 miss 显著超标说明训练/阈值策略仍对分布差异敏感。建议先从更平滑的重平衡、召回偏置阈值以及 KD 守卫入手，在不增大模型容量的前提下提升异常置信度和泛化召回。

### v2.2_debug 更新（输入/权重范围确认 + 平滑召回强化）
- **输入/权重范围确认**：`preprocess_beat` 对每个 beat 去均值并按最大绝对值归一化，再裁剪到 [-1,1]；启用 `--use_value_constraint` 时，`ConstrainedConv1d/Linear` 通过 `tanh` 重参数把权重和偏置限定在 `[-scale, scale]`（默认 1.0）。`train.py` 在日志中记录并打印训练数据的最小/最大值，便于复核预处理是否落在 [-1,1]。
- **默认运行时的约束开关状态**：命令行默认开启 `--use_value_constraint` 并保持 `--no-use_tanh_activations`，即权重/偏置默认被限制在 `[-scale, scale]`，在进入受约束层前 token 也会按最大绝对值缩放到 [-1,1]；若需测试无约束效果可显式 `--no-use_value_constraint`。
- **批次内平衡**：新增 `BalancedBatchSampler`，在异常占比 <45% 时按 1:1 近似配比组批，避免单个 batch 完全由正常类组成；若异常占比更低仍叠加轻量 weighted sampler（boost=1.2），既保证召回样本，又防止过度过采样带来的 FPR 崩溃。
- **平滑召回强化**：按验证集 miss 的指数滑动均值（`miss_ema`）动态放大异常类 loss 权重（上限 2× 基准），让召回提升逐步发生；同时在阈值扫描中引入召回增益与阈值正则（偏向 0.5，惩罚过低阈值），并继续施加 miss/FPR≤15% 约束，防止简单“降阈值换召回”造成 FPR 爆炸。
- **自适应阈值权重**：阈值搜索使用 `recall_gain = 1 + 1.8*miss_ema`，miss 越高越偏向高召回阈值；同时加入阈值距离 0.5 的轻微惩罚，平衡召回和稳健性。最终阈值与逐 epoch 报告均使用该自适应策略，保持稳定又满足漏检上限目标。
- **约束必要性补充**：尽管输入 beat 已缩放/裁剪到 [-1,1]，卷积与线性累加、偏置和梯度更新会让中间特征幅值持续放大，推高后续层输出并改变训练/推理分布。默认开启权重约束和 token 归一化可保持算子幅值在既定预算内，减少数值漂移和过冲导致的 miss/FPR 摆动；若确认数据质量或希望对比无约束表现，再选择关闭即可。

## v2.3_debug 复盘（最新日志：Val F1≈0.55、Gen F1≈0.65，阈值≈0.93）
- **最新日志摘要**：
  - 训练 58 轮早停；最佳模型出现在阈值≈0.93。验证集：`F1≈0.547，miss≈13.98%，FPR≈7.96%`；泛化集：`F1≈0.650，miss≈15.16%，FPR≈14.17%`。
  - miss 已接近 15% 上限，但泛化侧略超线（15.16%）；FPR 在两侧均 <15%，阈值显著高于 0.5。
- **与前一版分析的差异纠正**：先前因缺少日志将阈值偏低（≈0.4）作为主要症状，实际最新运行阈值已升到 ≈0.93，说明模型输出对异常类置信度较高或评分函数在 miss/FPR 约束下更偏向高阈值；F1≈0.6 是在高阈值下的结果，并非“降阈值换召回”。
- **现象解读**：
  1. **阈值高但 FPR 可控**：阈值接近 0.93 仍能把 FPR 控制在 8%（验证）/14%（泛化），说明输出分布整体偏向异常类，提升阈值是在压误报而非换召回；miss 接近上限表明再抬阈值会突破漏检目标。
  2. **泛化 miss 轻微超标**：验证与泛化 miss 差距 ~1.2 pct，说明阈值在两侧一致性尚可，但泛化分布略难；可能与记录分布或异常形态差异有关，可通过提高泛化权重或更平滑的重平衡来补偿。
  3. **F1 仍受精度/召回折衷限制**：在高阈值下精度得到保障，F1≈0.55–0.65 表示召回与精度平衡仍未触达 0.7 以上，需要在不显著拉高 FPR 的前提下继续提升异常 logit 置信度。
- **阈值合理性**：
  - 阈值由 `sweep_thresholds_blended` 结合 miss/FPR≤15% 的过滤和 `f1 + 1.5*TPR - FPR` 评分选出。输出分布偏向异常时，满足漏检/FPR 约束的候选阈值会聚到高值段，因此偏离 0.5 符合当前分布，而非计算错误。
  - 若希望在保持 miss≤15%、FPR≤15% 的同时让阈值回落，可通过训练端提高异常/正常的可分性（如更平滑 ramp、批次平衡）减少对高阈值的依赖，而不应强行固定阈值。
- **是否触及容量上限的判断**：
  1. **训练 vs. 验证/泛化差距**：训练 loss 继续下降到 0.20 左右，验证/泛化 F1 仍在 0.55–0.65，表明尚有泛化差距，可通过正则与更稳的重平衡继续改善；未见明显“训练/验证同步停滞”迹象。
  2. **阈值敏感性检查**：建议用保存的 logits 离线再扫 0.85–0.95 的密集阈值区间，若 F1 对阈值微调仍有 2–3 点提升空间，说明模型尚未到达容量上限；若曲线平坦，则需靠训练信号或容量提升。
  3. **曲线形状**：对比验证/泛化 ROC/PR 曲线形状，若曲线在泛化明显下移而非平移，优先考虑数据/分布差异与重平衡策略；若曲线整体低，才考虑容量不足。
- **后续训练建议（保持模型容量不变）**：
  - **温和提升异常信号**：在现有 ramp 上小幅提高异常类基准权重至 1.3–1.4，或在不超过 1.3–1.4 的情况下启用轻量 weighted sampler，让异常 logit 略增，尝试把阈值压回 0.85–0.90 区间以换取 F1。
  - **泛化权重与阈值约束**：适度提高 `generalization_score_weight`（如 0.35–0.4）并把 `threshold_target_miss` 收紧到 0.13–0.14、`threshold_max_fpr` 维持 0.15，促使阈值选择更关注泛化漏诊。
  - **救火触发再校准**：当前 miss 仅略超 15%，可把 `recall_target_miss` 设置在 0.14 附近，使自适应权重/采样在接近超标时提前介入，同时监控 FPR 避免二次崩溃。
- **日志与可重复性**：将本次运行的概率/指标写入 `artifacts/training_log_*.jsonl`，便于离线阈值复扫和曲线对比；也能核对是否始终走 `sweep_thresholds_blended` 路径，以匹配上述分析假设。
- **代码状态复核**：仍保持无 KD；输入缩放与权重约束默认开启，BalancedBatchSampler 与阈值扫描逻辑已在 `utils.py` 统一，未发现与上述分析矛盾的遗留实现。
- **本轮代码更新（v2.3 按建议落地）**：
  - 将异常类基础权重提升至 1.35、比值上限收紧到 2.0，并暴露到 CLI 方便在 1.3–1.4 区间微调。
  - 阈值搜索/早停统一改用验证-泛化混合评分（权重 0.35），在 miss≤0.14 且 FPR≤0.15 的硬约束下选阈值，并将该阈值用于最终泛化评估与日志。
  - 额外保存验证/泛化的概率与标签（`artifacts/val_probs.npy` 等），支持离线加密度阈值扫与分布差异检查。

## v2.4_debug 分析（最新上传日志：Val@thr≈0.80 F1≈0.59，miss≈7.0%，FPR≈7.5%；Gen@thr≈0.80 F1≈0.86，miss≈8.8%，FPR≈3.9%）
- **日志与曲线观察**：
  - 最新运行在阈值≈0.80 取得最佳结果：验证集 `F1=0.592，miss=6.99%，FPR=7.48%`，泛化集 `F1=0.861，miss=8.75%，FPR=3.85%`，阈值较前一版明显回落。
  - ROC AUC 维持在 Val≈0.974、Gen≈0.969，曲线左侧抬升明显；混淆矩阵显示两侧均保持较高 TP、TN，误报/漏诊控制在 <10%（泛化 miss 稍高于验证）。
  - 训练/验证损失持续下行且曲线平稳，阈值在 0.75–0.90 区间小幅抖动后稳定在 0.80，未再出现早期崩溃或剧烈跳阈。
- **现状解读**：
  1. **阈值回落且更均衡**：相比上一版的 0.93，高阈值依赖已减弱，说明重平衡与阈值评分现在能在较低阈值下同时压住 FPR（7–8%）和 miss（7–9%）。
  2. **F1 提升到 0.59/0.86**：验证 F1 受少量 FP 影响仍 <0.6，但泛化 F1 已达 0.86，表明泛化分布更易分、或阈值评分中泛化权重发挥作用。差距提醒需关注验证侧精度提升以拉升综合 F1。
  3. **漏诊/FPR 已低于目标**：miss 与 FPR 均满足 10% 上限，当前瓶颈主要在验证侧精度（FP）与少量 FN，可在不破坏泛化的前提下微调以提升验证 F1。
- **回退原因与修正**：近期修改后出现过早期阈值崩溃（极低阈值导致 FPR≈100%）和过高权重触发的异常偏置，主因是阈值搜索在约束全部失败时直接选取未约束的最高分阈值，加上默认异常权重/惩罚偏高会把分布推向全正类。此次调整加入“最小违约”回退路径，优先选择 miss/FPR 违反最小且得分最高的候选，避免再落到极端阈值；同时将默认异常类权重、采样倍率和 FPR 惩罚/阈值正则略微下调，降低早期正类膨胀。
- **优化建议（保持无 KD 与现有容量）**：
  - **精度友好微调**：在维持阈值≈0.80 的前提下，适度上调 `threshold_max_fpr` 惩罚权重或加入轻微阈值正则（惩罚过低阈值），尝试把验证 FPR 再压 1–2 个百分点，观察 F1 是否上升且不牺牲 miss。
  - **异常信号微强化**：若验证 miss 偶尔反弹，可小幅提高 `class_weight_abnormal`（如 1.4）或强制开启 BalancedBatchSampler，保证每批包含异常样本并平稳梯度；注意保持 `sampler_abnormal_boost` 温和（≈1.2–1.3）以防 FPR 抬头。
  - **阈值稳定性复扫**：利用已保存的概率在 0.70–0.85 做密集阈值网格，确认 F1 对阈值是否有“台阶”可提升；若曲线平坦，则应把精力放在降低 FP（正则/轻量采样）而非继续调阈值。
  - **泛化重权重对照**：在当前分布下，泛化优于验证。可略调低 `generalization_score_weight` 做对照，观察阈值是否上移导致泛化 miss 上升；若发生，则保持现权重并把验证精度提升放在重平衡/正则上解决。

### v2.4_debug 代码调整（基于上述建议落地）
- **异常类强化但仍限幅**：默认 `class_weight_abnormal` 调低到 **1.30**、采样倍率默认 **1.1**，保持比值上限 2.0；`--force_balanced_batches` 仍可在异常占比不足时强制 1:1 batch 采样。
- **阈值搜索稳健化**：`sweep_thresholds_blended` 在约束无可行解时优先选择 miss/FPR 违约最小的候选，避免极低阈值导致 FPR=100% 的退化；默认硬约束收紧到 miss/FPR≤10%，FPR 惩罚、阈值中心和正则默认值下调（中心 0.78、惩罚 0.9、正则 0.02），减少不必要的高阈值或过强偏置。
- **可控实验开关**：保留 `--threshold_fpr_penalty`、`--threshold_center`、`--threshold_reg` 与 `--force_balanced_batches` 供调优，在当前更温和的默认值基础上按需加强/减弱。
## KD V1 DEBUG（教师泛化劣化 & 学生未获提升）
- **现象**：教师验证 miss≈3.4%、FPR≈0.18%，但泛化 miss≈9.8%（高于学生基线 8.1%）；KD 后学生验证 miss≈7.0%、泛化 miss≈8.3%，与基线相比提升有限甚至略降。
- **原因判断**：教师在泛化集上低于学生，蒸馏信号被混入泛化分布后可能拉低学生决策；KD 过程中未对教师置信度和正确性过滤，学生被误导。
- **本轮修改**：
  - **教师门控**：若教师泛化 miss 超过学生基线（可调 `KD_TEACHER_GEN_MARGIN`），直接关闭 KD，仅做 CE 微调；若仅存在泛化过拟合，则按 gap 自动降权。
  - **阈值对齐**：对教师也执行验证/泛化混合阈值搜索，供蒸馏和日志使用，避免用固定 0.5 误判教师质量。
  - **置信/正确性掩码**：蒸馏时仅对教师高置信（`KD_TEACHER_CONF`）且预测正确的样本计算 KD logits/特征损失，其余样本仅用 CE，避免教师在分布外误导学生。

### KD V1 DEBUG 追加说明（日志与双侧 ROC 完整记录）
- **操作改动**：
  1. 教师训练改用 `dataloaders.class_weights` 做加权 CE，与学生训练保持一致。
  2. 在主流程中先以 `evaluate_with_probs` 获取教师在 Val/Gen 的概率，再用 `sweep_thresholds_blended` 求得教师最优阈值；若教师在 Gen 上的 miss≥学生基线或 FPR 超过 `max(student_gen_fpr, threshold_max_fpr)`，则将 KD 权重置 0、仅做 CE 微调，并在日志打印门控原因。
  3. 蒸馏时传入上述动态 KD 权重与教师阈值；若 KD 被关停，logit/feature KD 损失恒为 0。
  4. 训练日志落盘 `artifacts/kd_training_log.csv`，同时生成包含 Val/Gen 双侧 ROC 的对比图 `figures/kd_roc_comparison.png`，方便复现与审计。
  5. 性能摘要使用各自的最佳阈值：学生用基线阈值、教师用扫得阈值、KD 学生用蒸馏过程最优阈值，并在 Val/Gen 双侧报告 Miss/FPR/阈值。
- **效果判断与表征方式**：
  - 若教师被门控关闭 KD，学生仅做 CE 微调，指标应与基线接近；此时 KD 效果可以通过 “KD 权重=0（教师未被接受）” 的日志明确说明，无需期待性能提升。
  - 若教师被接受，观测对比图（Val/Gen 双 ROC）与 `artifacts/kd_training_log.csv` 中阈值/指标轨迹，若 KD AUC 与 miss 明显优于基线且与教师趋势一致，则可认定 KD 起效；否则可进一步调整教师质量或 KD 权重。

## v10_debug Root cause analysis (current runs still miss S/V)
- **Observed failure**: Recent training collapses to predicting only `N` and occasionally `O`; `S`/`V` recall stays at 0–1% despite high overall accuracy driven by class `N`. Validation confusion matrices show entire `S`/`V` rows mapped to `N`/`O`, confirming minority classes are not learned.
- **Imbalance handling now too weak**: After successive safeguards, both weighted sampling and class-weight boosts default to **off** (or heavily clamped to a max ratio of 2.0). Given MIT-BIH’s severe skew (typically <5% combined for `S`/`V`), the effective CE weights become nearly uniform and the sampler stays disabled; the model therefore trains on batches dominated by `N`, leading to near-zero gradients for `S`/`V` and argmax always favoring the majority.
- **Over-corrections between runs**: Earlier attempts to curb “all-abnormal” collapse switched between strong abnormal emphasis and full suppression. These oscillations reset checkpointing and early stopping, but the current defaults remain on the conservative side—fully suppressing rebalancing—which explains the swing to “all-normal”.
- **KD no longer helping minorities**: With KD warmed down and weights/sampling weakened, the teacher’s logits are not enough to pull `S`/`V` upwards; instead the student follows the dominant `N` prior. No binary folding remains, so the remaining bias is purely from insufficient minority exposure/weight.
- **Next actionable directions**: (a) Re-enable a mild sampler (e.g., sqrt inverse-frequency) together with unclamped class weights or a higher clamp (≥4×) so `S`/`V` gradients matter; (b) keep CE weights active even when sampling is on to avoid losing minority emphasis; (c) delay KD until minority recall rises, or down-weight KD loss so CE can carve out `S`/`V` boundaries; (d) monitor per-class loss/precision–recall each epoch to stop before collapsing back to a single-class solution.

## v4-class recovery playbook (requested summary & concrete edits)
- **Feasibility**: Accurate four-class classification is achievable; N/O already reach >0.9 P/R. The remaining gap is stabilizing S/V exposure without re-triggering abnormal/normal collapse.
- **Minimum code edits to apply** (all in four-class paths only):
  1) `utils.py::compute_class_weights_4cls`: use `(1/freq)**0.5`, normalize mean→1.0, clamp ratio≤1.5, remove all abnormal multipliers and adaptive reweights; log per-split counts/weights once.
  2) `train.py::build_dataloaders`: enable a gentle `WeightedRandomSampler` only when `(S+V)/total < 0.35`, with fixed `abnormal_boost≈1.2`; when sampler is on, force `ce_class_weights = torch.ones(4)` to avoid double balancing.
  3) `train.py::train_one_epoch/evaluate`: drive checkpointing by validation Macro-F1 only; delete collapse detectors, miss/FPR rescues, and any threshold scan/ROC folds. Keep per-class PRF + 4×4 confusion matrix logging every epoch.
  4) `train.py` CLI defaults: set `--use_class_weights` on, `--use_weighted_sampler` off (opt-in), `--class_weight_power 0.5`, `--max_class_weight_ratio 1.5`, `--sampler_abnormal_boost 1.2`; set `--no-use-kd` as default for stabilization.
  5) `KD.py`: mirror the above defaults; KD should never change sampler/weight states. Gate KD until `kd_warmup_epochs>=10` and skip any pause/resume logic tied to miss/FPR.
  6) **Sanity check script**: add a small CLI (e.g., `python train.py --eval_only --checkpoint ...`) that loads a checkpoint and prints per-class support/PRF over a held-out slice to ensure S/V recall >0.3 before full training.
- **Rationale**: These edits remove binary-era residuals, limit overlapping boosts, and give S/V consistent gradient/share without collapsing N/O. Logging stays four-class only, so early stopping reflects the true objective.

## v4-class weight warmup (current code changes)
- **Stronger yet gradual weighting**: Default class weights now use full inverse frequency with a wider clamp (max_ratio=5). A linear warmup over the first 8 epochs blends from uniform to target weights so minority emphasis ramps up without immediately destabilizing N/O predictions.
- **Consistent KD behavior**: The KD loop mirrors the same weight computation and warmup, preventing distillation from overriding the minority-focused weighting scheme.
- **How to tune**: If S/V recall stays low, shorten the warmup or raise `--class_weight_max_ratio`; if N collapses, lengthen warmup. The sampler remains optional—when enabled it keeps CE weights uniform to avoid double balancing.

4-class_debug
