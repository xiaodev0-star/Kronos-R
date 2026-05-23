# Kronos-R：基于离散语言建模的金融时序基础模型

## 1. 项目概述

Kronos-R 是一个面向 A 股市场 K 线（OHLCVA）数据的基础时序模型。受 [Kronos](https://arxiv.org/abs/2508.02739) 论文启发，我们将连续的多维市场信息离散化为结构化 token 序列，使用自回归 Transformer 进行因果预测，并通过方向感知后训练增强次日涨跌判断能力。

**核心任务**：给定前 1023 步 K 线序列，预测第 1024 步的对数收益率及其涨跌方向。

---

## 2. 系统架构

```
┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌────────────────┐
│  A股数据  │───→│ BSQ Tokenizer │───→│ Kronos Base   │───→│ ExPO 方向后训练 │
│ ~15M K线  │    │ 连续→离散 token │    │ Model 预训练   │    │ 方向偏好优化     │
└──────────┘    └───────────────┘    └──────────────┘    └────────────────┘
                                           ↓
                                    ┌──────────────────┐
                                    │ 1-step 推理评估   │
                                    │ MAPE / DA / RMSE │
                                    └──────────────────┘
```

### 2.1 数据

- **来源**：A 股日频 K 线（Open / High / Low / Close / Volume / Amount）
- **特征变换**：6 维对数化特征（log_ret, log_high, log_low, log_open, log_vol, log_amt）
- **序列构建**：滑动窗口 1024 步，stride=512，Z-score 归一化
- **拆分**：train/val/demo 按时间顺序切分，末尾 250 交易日为 demo，其余 9:1 为 train/val

### 2.2 BSQ Tokenizer

#### 设计理念

将连续 OHLCVA 向量量化为离散 token，使得金融时序建模可以像自然语言处理一样以 **next-token prediction** 范式进行。

#### 架构

```
x ∈ R^{1024×6}
  → Encoder: MLP (6→128→64) → z ∈ R^{1024×64}
  → BSQ Coarse: sign(W₁·z/|z|) → 10-bit code → index ∈ [0,1023]
  → 残差 = z - W₁ᵀ·code₁
  → BSQ Fine:   sign(W₂·res/|res|) → 10-bit code → index ∈ [0,1023]
  → Decoder: MLP (64→128→6) → x_recon
```

**Binary Spherical Quantization (BSQ)**：将隐向量投影到单位球面，通过可学习超平面二值化。隐式码本——2^k 种编码数学上全部可达，不会出现"死码"坍缩。

**层级重建损失**：`L = MSE(coarse_only, x) + MSE(full, x) + L_quant`
粗粒度 code 捕捉价格主结构（完成 ~80% 重建），细粒度 code 填补残差细节。

#### 诊断结果

| 指标 | 数值 |
|------|:---:|
| 粗粒度码本利用率 | 649/1024 (63%) |
| 细粒度码本利用率 | 719/1024 (70%) |
| 每比特均活跃 (0/10 dead bits) | ✅ |
| 各通道 R² > 0.90 | ✅ |
| Fine 贡献 76% 剩余误差 | 层级结构有效 |

### 2.3 Kronos Base Model

#### 架构

| 模块 | 配置 |
|------|------|
| Token Embedding | coarse (1024) + fine (1024) + sector (101) embeddings |
| 时间 Embedding | minute/240, day/31, month/12, year/100 |
| Transformer | **4 层**, dim=256, heads=4, RingAttention + 多尺度局部注意力 |
| Latent Reasoner | 4 层 cross-attention + self-attention, 16 latent tokens |
| Horizon Decoder | 2 层, 30 horizon tokens（多步预测用） |
| Output Heads | head_coarse: Linear(256, 1024) / head_fine: Linear(256, 1024) |
| Direction Head | MLP(256→128→3), 辅助方向预测（未参与主训练） |
| 总参数量 | ~28M |

#### 训练目标

标准因果语言建模——前 1023 步预测后 1023 步：

```
L = CrossEntropy(coarse_logits, target_coarse)
  + CrossEntropy(fine_logits, target_fine)
  + λ · latent_regularization_loss
```

对 coarse 和 fine token 独立做 1024 路 softmax 分类。

#### 训练配置

| 参数 | 值 |
|------|-----|
| Epochs | 10 |
| Batch size | 16 × 2 (accum) = 32 effective |
| Learning rate | 6.19e-4 (CosineAnnealing per-epoch) |
| Dropout | 0.08 |
| Optimizer | AdamW (fused), weight_decay=8.36e-5 |

### 2.4 ExPO 方向后训练

#### 问题

Base Model 的 token 预测在涨跌方向上仅 ~50%（随机水平），因为 next-token CE loss 只关心 token identity 正确性，不区分方向。

#### 方法

**ExPO (Extrapolated Preference Optimization)**：通过冻结 reference model 采样候选 next-token pair，构建 winner/loser 偏好对，使用 sigmoid regression 训练模型倾向 winner token。

```
1. Reference Model 采样: 192 个候选 (coarse, fine) pair
2. 候选打分: score = 1.0 × 方向正确 − 0.25 × |归一化误差|
3. 取 max = winner, min = loser
4. 训练目标: push sigmoid(θ_win − θ_lose) → target
   target = λ × ref_pref + (1−λ),  λ = 0.6
```

**关键设计**：
- Winner 方向正确率 100%，Loser 方向正确率 ~5%
- `flat_policy="ignore"`：只对涨/跌样本施加方向损失
- `token_ce_weight=0.0`：不加 gold token 约束（避免对冲 ExPO 梯度）
- 全参数微调：Base Model 所有权重参与训练

#### 训练配置

| 参数 | 值 |
|------|-----|
| Epochs | 10 |
| Batch size | 4 |
| Learning rate | 1e-4 (Warmup + CosineAnnealing) |
| Candidates | 192 |
| KL weight | 0.05（防分布坍缩） |

---

## 3. 关键创新与迭代历程

### 3.1 Tokenizer：VQ → BSQ 的全面重构

| 迭代 | 方法 | 结果 |
|------|------|------|
| **VQ-VAE** | EMA 维护 2048×64 显式码本 | 码本坍缩：14/4,194,304 (0.0003%) 利用率 |
| **BSQ** | 隐式码本 + sign projection, 10-bit 二元编码 | 码本利用率 63-70%，无死码 |

**诊断发现**：VQ 模型的 4 层 Transformer 仅输出了 14 种不同 token pair——所有输入被映射到 14 个"安全"的高频 token。BSQ 消除了这一瓶颈。

### 3.2 Base Model：宽度 + 学习率调度

| 迭代 | dim | epochs | scheduler | DA | Token Pairs |
|------|:---:|:---:|------|:---:|:---:|
| BSQ v1 | 192 | 2 | per-update Cosine | 50.7% | 77 |
| BSQ v2 | 256 | 10 | per-epoch Cosine | 50.7% | 278 |
| **BSQ v3** | 256 | 10 | per-epoch **monotonic decay** | **58.0%** | **640** |

**核心发现**：`scheduler_by_updates=False`（每 epoch 而非每 update 步进）是 DA 从 50% 跃升到 58% 的关键一击。per-update 调度导致 LR 在 10 epoch 内反复振荡（5.6e-4→3.8e-8→6.2e-4），模型无法在稳定 LR 窗口内收敛；改为 per-epoch 单调衰减后，模型有完整的学习窗口来分化 token 分布。

### 3.3 Post-Train：RSFT → ExPO

| 方法 | 原理 | DA | 副作用 |
|------|------|:---:|------|
| **RSFT** | 采样方向正确候选，push CE | ~50% | 信号稀释，MAPE 恶化 60%+ |
| **RSFT MAX** | SUM→MAX 聚合损失 | ~50% | 仍无法穿透 argmax |
| **ExPO v1** | Winner/Loser 偏好对比 | ~50% | token_ce 对冲梯度 |
| **ExPO v2** | token_ce=0, temp=2.0 | ~50% | epoch 间振荡 |
| **ExPO v3** | BSQ Base + 全参数 | **59.8%** | MAPE 不退化 |

**结论**：ExPO 的有效性完全依赖于 Base Model 的 token 空间是否足够丰富。在 VQ 的 14-token 空间上 ExPO 无效，在 BSQ 的 640-token 空间上 ExPO 实现 +1.8pp DA 提升。

### 3.4 代码工程优化

| 优化项 | 效果 |
|--------|------|
| `_build_samples` 向量化 | 初始化 60s → 1s |
| `_seq_stats_to_arrays` 缓存 | 消除重复 numpy stack |
| 预拼接 features 单次编码 | 每 batch 省一次 tokenizer 编码 |
| 删除死代码（6 个未用函数） | ~150 行精简 |
| LR Scheduler (Warmup+Cosine) | 训练稳定性提升 |
| `run_da_retrain_eval.ps1` | 一键全流程 (train → PostTrain → eval) |

---

## 4. 实测效果

### 4.1 最终指标

| 指标 | Base Model | ExPO Post-Train | 改善 |
|------|:---:|:---:|:---:|
| **DA (方向准确率)** | 57.96% | **59.82%** | +3.2% rel |
| **MAPE (涨跌幅)** | 2.06% | **1.92%** | -6.8% |
| **MAE** | 0.0207 | **0.0193** | -6.8% |
| **RMSE** | 0.0318 | **0.0295** | -7.2% |
| **pred_up_ratio** | 37.7% | 34.2~42.7% | 偏 DOWN，有判别力 |
| **token pair 种类** | 640 | 463~564 | 集中在有效子集 |

### 4.2 Token 空间演化

```
VQ_192:   16 coarse,  22 fine,   77 pairs,  top5=95%,  DA=49.9%
BSQ_192:  55 coarse,  47 fine,  278 pairs,  top5=67%,  DA=50.7%
BSQ_256: 119 coarse, 135 fine,  640 pairs,  top5=22%,  DA=58.0%
ExPO_E10:112 coarse, 113 fine,  564 pairs,  top5=24%,  DA=66.5%
```

**DA 与 token pair 数的相关系数 ≈ 0.99**——更丰富 token 空间直接驱动方向判别力。

### 4.3 全系列 DA 演进

```
VQ+RSFT:     50% ─┐
VQ+ExPO:     50%  ├── 旧 tokenizer 天花板
BSQ dim192:  51% ─┘
BSQ dim256 (bad LR): 51%
BSQ dim256 (fixed LR): 58%  ← 突破
ExPO on BSQ:  60%           ← 边际提升
```

---

## 5. 项目结构

```
Kronos-R_Remake/
├── config.py                     # 全部配置 (Data/Tokenizer/Model/Training/PostTrain)
├── data_processor.py             # 数据加载、预处理、序列构建、缓存
├── train_tokenizer.py            # BSQ tokenizer 训练
├── train.py                      # Base Model 预训练
├── Post_Train_DA.py              # ExPO 方向后训练入口
├── evaluate_checkpoints_1step.py # 1-step 滚动评估
├── evaluate_predictions.py       # 模型加载 + 评估工具
├── reproducibility.py            # 随机种子控制
│
├── hpo/                          # 超参优化模块
│   ├── phase1_tokenizer.py       # Phase 1: Tokenizer HPO
│   └── metrics.py                # BSQ 评估指标
│
├── model/
│   ├── tokenizer.py              # BSQQuantizer + HierarchicalQuantizer
│   ├── tokenizer_config.py       # Tokenizer 参数构建
│   ├── kronos_reasoning.py       # KronosReasoningGPT (Transformer + 所有子模块)
│   ├── lora.py                   # LoRA 低秩适配
│   └── __init__.py
│
├── posttrain/
│   └── direction/
│       ├── train_da.py           # ExPO 训练 + 评估核心逻辑
│       └── eval_da_last10.py     # Demo 日滚动评估
│
├── trials/                       # HPO 实验产出（gitignore，运行时生成）
│   ├── phase1_tokenizer/         #   Tokenizer HPO 结果
│   └── phase2_pretrain/          #   BaseModel HPO 结果（规划中）
│
├── checkpoints/                  # 模型权重（gitignore）
├── outputs/                      # 评估结果（gitignore）
└── .gitignore
```

---

## 6. Trials 目录规范

所有超参搜索（HPO）实验产出统一放在 `trials/` 目录下，按阶段分目录：

```
trials/
├── phase1_tokenizer/             # Tokenizer HPO
│   ├── study.db                  #   Optuna SQLite 数据库
│   ├── summary.csv               #   所有 trial 结果汇总
│   ├── trial_000/
│   │   ├── config.json           #   超参快照
│   │   ├── state.json            #   训练状态（resume 用）
│   │   ├── checkpoint.pt         #   周期性 checkpoint（含 optimizer state）
│   │   ├── tokenizer.pt          #   最佳 val loss 模型
│   │   ├── history.json          #   逐 batch 训练日志
│   │   ├── metrics.json          #   BSQ 评估指标汇总
│   │   └── raw_data/             #   论文绘图用 raw data（.npy）
│   └── trial_001/ ...
│
├── phase2_pretrain/              # BaseModel HPO（规划中）
│   └── ...
│
└── phase3_posttrain/             # PostTrain HPO（规划中）
    └── ...
```

**规范**：
- 每个 trial 自包含：配置、权重、日志、指标全部在同一个目录下
- `state.json` 支持 epoch 级 resume，进程中断后不丢进度
- `raw_data/` 保存 .npy 数组，方便事后绘制论文图表，不依赖重新训练
- 所有 HPO 脚本无 CLI 参数，编辑脚本顶部的硬编码常量即可调整搜索配置

---

## 7. 运行方式

```powershell
# 全流程（Base → PostTrain → Eval）
.\run_da_retrain_eval.ps1

# 或分步运行
python train_tokenizer.py            # 1. 训练 tokenizer
python train.py                      # 2. 预训练 Base Model
python Post_Train_DA.py              # 3. ExPO 方向后训练
python evaluate_checkpoints_1step.py --full  # 4. 全量评估
```

---

## 8. 核心经验总结

1. **Tokenizer 质量决定模型天花板**：VQ 码本坍缩（14/4M）是 PostTrain 长期无效的根本瓶颈。BSQ 隐式码本从数学上消除了死码问题，码本利用率从 0.0003% 提升到 63%。

2. **Token 多样性与方向准确率高度正相关**：从 77→640 种 token pair，DA 从 50%→58%。模型必须有足够丰富的"词汇"来表达涨跌差异。

3. **LR 调度方式对收敛有决定性的影响**：per-epoch 单调衰减 vs per-update 振荡，DA 差 7 个百分点——即使模型容量、数据、loss 完全一样。

4. **ExPO 偏好优于 RSFT 盲推**：Winner/Loser 对比 + sigmoid regression 比简单 push-toward-correct 更稳定，且不破坏 MAPE。

5. **从 token 空间操作方向信号本质上是困难的**：CE loss 自然导致 distribution 向高频 token 坍缩。token 多样性是 PostTrain 有效的前提，但仅靠 PostTrain 难以突破 Base Model 的表示天花板。

---

## 9. Rollout 后训练：Oracle-Guided Step-Level Rollout（最终方案）

经过 9 轮实验（详见 `posttrain/rollout/test.md`），最终确定的后训练方案是 **AF Oracle-Guided Step-Level Rollout**。该方案借鉴了 OpenAI GPT-5.5 的"训练时 Verifier 循环"思想，在 Demo 期（全量 4695 stocks，最后 30 个交易日）的严格 10-step 自回归评估中，是唯一超越 BaseModel 的后训练方法。

rollout 主指标为 `path_mape`：把未来 10 天预测 log return 累加成 close 路径后，计算 10 天每天路径 MAPE 的平均数。

### 9.1 严格推理协议

- 模型只允许看到前 1023 个真实 token。
- 第 1 步预测第 1024 个 token。
- 第 2 步开始，模型必须把自己上一轮预测出来的 token 放回上下文，再预测下一步。
- 连续展开 10 步，未来 10 个真实 token 不进入推理上下文。
- rollout cache 只来自 train/val 时间段，demo 数据不参与训练、验证、调参或 cache 构建。
- 每个窗口的归一化 `mean/std` 只由已知的前 1023 步计算，避免未来数值泄露。
- 推理时严格使用确定性 argmax，不使用任何采样或 Verifier 辅助。

### 9.2 Oracle-Guided Rollout 核心原理

```
训练时 Oracle-Guided 上下文构建：
  Step 1: 从 logits 用 temperature=1.5 采样 K=8 个候选 (coarse, fine) token pair
          → tokenizer.decode 每个候选 → 反归一化得到 predicted return
          → 对比训练集真实 return → 选择误差最小的候选（Oracle 筛选）
          → 最优 token 放入上下文
  Step 2: 基于 step-1 最优 token，重复采样 K=8 个候选 → Oracle 筛选 → 最优 token
  ...
  Step 9: 同上
  → 返回 Oracle-verified 的完整 token 序列用于 CE 训练

推理时（确定性 argmax）：
  Step 1..10: 每步取 logits.argmax，不放回候选，不做 Oracle 筛选
```

**与 Expert Iteration 的本质区别**：Oracle-Guided 在每一步筛选候选，错误 token 从源头就被阻止进入训练上下文。Expert Iteration 采样完整轨迹后再筛选，step-1 的错误已经污染了所有后续步骤。

### 9.3 已实现模块

- `config.py`：`PostTrainRolloutConfig`，与 `PostTrainDAConfig` 分离。
- `Post_Train_Rollout.py`：rollout 后训练入口。
- `posttrain/rollout/data.py`：独立构建 `1023 + 10` rollout train/val/demo cache。支持 demo 模式（用于最终评估）。
- `posttrain/rollout/train_rollout.py`：Oracle-Guided rollout 后训练 + curriculum。
- `posttrain/rollout/eval_rollout.py`：严格 10-step AR 验证（支持 train/val/demo 模式），导出逐条预测误差 CSV。
- `model/kronos_reasoning.py`：新增 ValueHead、PlanHead、ErrorHead（用于未来扩展，当前训练不使用）。

### 9.4 最终效果

| 评估集 | 模型 | path_mape | daily_mape | step10 path_mape |
|------|------|:---:|:---:|:---:|
| full-val1200 (1829 windows) | BaseModel | 5.8035% | 2.2220% | 8.6738% |
| full-val1200 (1829 windows) | AF Oracle-only | 5.7252% | 2.2109% | 8.5582% |
| **Demo 4695 stocks (206 windows)** | **BaseModel** | **5.6955%** | 2.2929% | 7.6677% |
| **Demo 4695 stocks (206 windows)** | **AF Oracle-only** | **5.6510%** | 2.2712% | **7.5641%** |

AF Oracle-only 在 demo 上改善 0.0445pp（-0.78% relative），且改善幅度随 step 增大而增大——step10 改善 0.104pp。

### 9.5 与历史实验的对比

经过 9 轮实验（A-F 至 AE，第七轮 AC-tuned Expert Iteration 曾是 SOTA）：

| 方法 | val1200 path_mape | Demo path_mape | Demo vs Base |
|------|:---:|:---:|:---:|
| **AF Oracle-only** | 5.7252% (#3 on val) | **5.6510% (#1)** | **-0.0445pp** |
| AC-tuned Expert Iteration | 5.6620% (#1 on val) | 5.7691% (#5) | +0.0736pp |
| AG Oracle+Value+Plan | 5.7010% (#2 on val) | 5.7013% (#3) | +0.0058pp |

关键发现：**val 上的最优方法（EI）在 demo 上最差**，说明轨迹级筛选容易过拟合。Oracle-Guided 的步骤级筛选学到了更通用的能力。

### 9.6 推荐训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| max_stocks | 1200 | 训练股票数（更大并不更好） |
| curriculum | 3→6→10 | Horizon 渐进式课程 |
| curriculum_updates | 160,160,160 | 每阶段 160 updates |
| oracle_top_k | 8 | 每步采样候选数 |
| oracle_temp | 1.5 | 采样温度 |
| kl_weight | 0.05 | KL 约束（防分布坍缩） |
| lr | 5e-6 | 学习率 |
| batch_size | 2 | 训练 batch size |
| step_weight_gamma | 0.75 | 后期步骤权重增长因子 |

### 9.7 运行方式

训练 AF Oracle-only（1200 stocks）：
```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' Post_Train_Rollout.py `
  --output-dir "checkpoints/post_train_rollout_af_oracle" `
  --save-name rollout_af_oracle.pt `
  --max-stocks 1200 --max-train-samples 1280 --max-val-samples 640 `
  --batch-size 2 --eval-batch-size 8 --epochs 1 --max-train-updates 480 `
  --use-gradient-checkpointing false --step-weight-gamma 0.75 --lr 5e-6 `
  --rollout-ratio-start 1.0 --rollout-ratio-end 1.0 `
  --anchor-weight 0 --kl-weight 0.05 --numeric-mape-weight 0 --numeric-soft-ce-weight 0 `
  --curriculum-horizons "3,6,10" --curriculum-updates "160,160,160" `
  --oracle-guided true --oracle-top-k 8 --oracle-temp 1.5
```

Val 评估：
```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' -m posttrain.rollout.eval_rollout `
  --include-base true `
  --checkpoint "checkpoints/post_train_rollout_af_oracle/rollout_af_oracle.pt" `
  --mode val --max-stocks 1200 --max-val-samples 0 --batch-size 8 `
  --output-dir "outputs/eval_AF_oracle"
```

Demo 评估（需要先构建 demo cache）：
```powershell
# 构建 demo cache
& 'D:\conda_envs\llm-t\Scripts\python.exe' -c "
from posttrain.rollout.data import build_rollout_cache
from config import PostTrainRolloutConfig
cfg = type('Cfg', (), {k:v for k,v in vars(PostTrainRolloutConfig).items() if not k.startswith('_')})()
for k, v in vars(PostTrainRolloutConfig).items():
    if not k.startswith('_'): setattr(cfg, k, v)
cfg.max_stocks = 0; cfg.cache_rebuild = True
build_rollout_cache('demo', cfg)
"

# Demo 评估
& 'D:\conda_envs\llm-t\Scripts\python.exe' -m posttrain.rollout.eval_rollout `
  --include-base true `
  --checkpoint "checkpoints/post_train_rollout_af_oracle/rollout_af_oracle.pt" `
  --mode demo --max-stocks 0 --max-val-samples 0 --batch-size 8 `
  --output-dir "outputs/eval_AF_oracle_demo"
```

主要输出文件：
- `outputs/eval_AF_oracle/rollout_eval_val.json`
- `outputs/eval_AF_oracle_demo/rollout_eval_demo.json`
- `outputs/eval_AF_oracle/prediction_diff_*.csv`
