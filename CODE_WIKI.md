# Kronos-R Remake — Code Wiki

## 1. 项目概述

Kronos-R Remake 是一个面向 **A 股金融时间序列** 的深度学习推理与预测框架。核心思想是将金融 K 线数据通过 BSQ（Binary Spherical Quantization）分词器离散化为 token 序列，再由 KronosReasoningGPT 模型进行因果语言建模式的"下一个 token 预测"，从而实现对未来收益的预测。

项目包含完整的训练流水线：Tokenizer 训练 → Base Model 预训练 → Direction-Accuracy EXPO 后训练 → 评估与对比，并提供模型对比 Dashboard。

---

## 2. 项目架构总览

```
Kronos-R_Remake/
├── config.py                          # 全局配置（数据/Tokenizer/模型/训练/评估/LoRA/后训练）
├── reproducibility.py                 # 随机种子与可复现性工具
├── data_processor.py                  # 数据加载、预处理、Dataset/DataLoader、缓存与 Memmap
├── train_tokenizer.py                 # Stage A: BSQ Tokenizer 训练入口
├── train.py                           # Stage B1: Base Model 预训练入口
├── Post_Train_DA.py                   # Stage B2: Direction-Accuracy EXPO 后训练入口（委托调用）
├── train_lora.py                      # LoRA 微调入口（用户自定义 CSV）
├── evaluate_predictions.py            # 模型加载、滚动预测评估项构建
├── evaluate_checkpoints_1step.py      # 批量 Checkpoint 1-step 评估
├── diagnose_codebook_diversity.py     # Codebook 多样性诊断工具
├── run_da_retrain_eval.ps1            # PowerShell 一键训练+后训练+评估脚本
│
├── model/                             # 核心模型定义
│   ├── __init__.py                    # 模块导出
│   ├── tokenizer.py                   # BSQQuantizer + HierarchicalQuantizer
│   ├── tokenizer_config.py            # Tokenizer 构建参数工具
│   ├── kronos_reasoning.py            # KronosReasoningGPT 主模型
│   └── lora.py                        # LoRA 低秩适配器
│
├── posttrain/                         # 后训练模块
│   ├── __init__.py
│   └── direction/
│       ├── __init__.py
│       ├── train_da.py                # EXPO Direction-Accuracy 后训练核心逻辑
│       └── eval_da_last10.py          # 后训练评估
│
├── compare/                           # 模型对比模块
│   ├── __init__.py
│   ├── server.py                      # FastAPI 对比 Dashboard 服务
│   ├── run_inference.py               # 统一推理脚本（RW/TimeNet/XGBoost/Kronos-R）
│   └── models/
│       ├── rw/model.py                # Random Walk 基线
│       ├── timenet/model.py           # TimeNet (TCN) 基线
│       └── xgboost/                   # XGBoost 基线
│
├── requirements.txt                   # Python 依赖
└── dataset/                           # A 股 CSV 数据目录（需自行准备）
```

---

## 3. 核心模块详解

### 3.1 配置系统 — `config.py`

集中管理所有超参数，使用 Python 类作为命名空间。支持通过环境变量 `KRONOS_OVERRIDE_JSON` 指向 JSON 文件进行运行时覆盖。

| 配置类 | 职责 | 关键参数 |
|--------|------|----------|
| `DataConfig` | 数据集处理 | `data_dir`, `seq_len=1024`, `feature_cols`, `train_val_split=0.9`, `demo_days=30` |
| `TokenizerConfig` | BSQ Tokenizer | `input_dim=6`, `embedding_dim=64`, `bits_per_quantizer=10`, `num_quantizers=2`, `epochs=200` |
| `ModelConfig` | KronosReasoningGPT | `dim=256`, `depth=4`, `heads=4`, `num_latent_tokens=16`, `position_encoding="rope"`, `horizon_tokens=30` |
| `TrainingConfig` | 训练流程 | `epochs=10`, `batch_size=16`, `learning_rate`, `diversity_weight=0.6`, `use_gradient_checkpointing=True` |
| `EvaluationConfig` | 评估与预测 | `num_stocks=100`, `pred_steps=30`, `temperature=0.5` |
| `LoRAConfig` | LoRA 微调 | `rank=8`, `alpha=16`, `target_keywords` |
| `PostTrainDAConfig` | EXPO 后训练 | `expo_num_candidates=192`, `kl_weight=0.05`, `label_mode="rolling_vol"` |
| `PathConfig` | 路径管理 | `checkpoint_dir`, `output_dir`, `cache_file` |

**运行时覆盖机制**：`_apply_runtime_overrides()` 在模块加载时自动执行，读取 `KRONOS_OVERRIDE_JSON` 环境变量指向的 JSON 文件，按配置类名分组覆盖属性值。

### 3.2 数据处理 — `data_processor.py`

#### 3.2.1 AShareDataset

A 股序列数据集，负责从 CSV 文件加载、特征工程、序列切分、缓存管理。

**特征工程**：
- 6 维特征：`log_ret`, `log_high`, `log_low`, `log_open`, `log_vol`, `log_amt`
- 对每条序列做 z-score 归一化（保存 mean/std 用于反归一化）
- 时间特征：`minute`, `day`, `month`, `year`（相对 2010 年基准）

**数据切分**：
- `train`：最早 → `train_val_split` 比例处
- `val`：`train_val_split` → `1 - demo_ratio` 处
- `demo`：最近 `demo_days` 天

**缓存机制**：
- 首次构建后自动缓存为 `.pt` 文件（含 `_data_cache_signature` 签名验证）
- 支持 tokenizer 编码预计算缓存（`encoded_indices_coarse/fine`）
- 缓存文件命名格式：`dataset_cache_{mode}_seq{N}_stride{M}_demo{D}d_split{S}.pt`

#### 3.2.2 Memmap 后端

针对大数据集的零 RAM 加载方案：

| 类 | 职责 |
|----|------|
| `NpyMemmapBackend` | 单个 `.npy` 文件的 mmap 只读后端 |
| `MemmapCacheWriter` | 从 AShareDataset 导出分文件 mmap 格式 |
| `MemmapArrayDataset` | 基于 `.npy` mmap 文件的 Dataset |

`migrate_cache_to_memmap()` 可将现有 `.pt` 缓存迁移为 mmap 格式。

#### 3.2.3 关键函数

| 函数 | 说明 |
|------|------|
| `get_datasets(include_demo, use_memmap)` | 构建 train/val/demo 数据集 |
| `get_dataloaders(...)` | 构建 DataLoader，支持分布式、CUDA prefetch、自动资源调优 |
| `collate_fn(batch)` | 标准 collate，含预计算编码 |
| `collate_fn_v2(batch)` | 精简版 collate，仅搬运 encodings + time + sector |

### 3.3 Tokenizer — `model/tokenizer.py`

#### 3.3.1 BSQQuantizer

Binary Spherical Quantization 隐式码本量化器。

**核心流程**：
1. 输入向量归一化到单位球面
2. 通过可学习超平面投影得到 k-bit 二值码 `b ∈ {-1, 1}^k`
3. 使用 Straight-Through Estimator (STE) 实现梯度传播：`b = b_hard + b_soft - b_soft.detach()`
4. 词汇表大小 = `2^k`（隐式，所有码字可达）

**损失函数**：
- Commitment Loss：`MSE(logits, b_hard.detach())`
- Codebook Loss：`MSE(logits.detach(), b_soft)`
- Entropy Loss：负二值熵（鼓励码本均匀使用）

**关键方法**：
| 方法 | 说明 |
|------|------|
| `forward(z)` | 量化前向，返回 `(b, indices, quant_loss)` |
| `quantize(z)` | 纯量化（无梯度），返回 `indices` |
| `decode_ids(indices)` | 将索引解码回嵌入空间 |
| `vocab_size()` | 返回 `2^bits` |

#### 3.3.2 HierarchicalQuantizer

2 级 BSQ 层次化分词器，coarse→fine 残差量化。

**架构**：
```
Encoder: MLP (input_dim → hidden_dim → embedding_dim)
BSQ Coarse: k₁ bits → 捕获主结构
BSQ Fine:   k₂ bits → 编码残差细节
Decoder: MLP (embedding_dim → hidden_dim → input_dim)
```

**损失**：`L_coarse(仅粗粒度重建) + L_fine(完整重建) + quant_loss`

**关键方法**：
| 方法 | 说明 |
|------|------|
| `encode(x)` | 返回 `(idx_coarse, idx_fine)` |
| `decode(idx_coarse, idx_fine)` | 从索引重建特征 |
| `encode_all(x)` | 返回所有层级的索引栈 |

### 3.4 主模型 — `model/kronos_reasoning.py`

#### 3.4.1 KronosReasoningGPT

Kronos 推理模型，由四大组件构成：

```
A. History Encoder: LinearAttention blocks (因果线性注意力)
B. Latent Reasoner: 可学习 latent tokens (替代 GRU ThinkingLayer)
C. Horizon Decoder: 30 个 future query tokens (并行预测未来)
D. RevIN: 可逆实例归一化
```

**嵌入层**：
- `token_emb_coarse` + `token_emb_fine`：双粒度 token 嵌入
- `sector_emb`：行业板块嵌入（101 类）
- `time_emb_min/day/month/year`：时间特征嵌入
- `pos_emb`：可选的学习位置编码

**前向模式**：
| 方法 | 说明 |
|------|------|
| `forward(...)` | 标准 teacher-forced 前向，返回 `(logits_coarse, logits_fine, latent_states)` |
| `forward_direction(...)` | 额外输出方向预测 logits（3 分类：跌/平/涨） |
| `forward_horizon(...)` | 额外输出 horizon decoder 的未来预测 |
| `forward_with_cache(...)` | 带 KV-cache 的前向（用于推理加速） |
| `forward_incremental(...)` | 增量推理（单 token 步进） |

**输出头**：
- 粗粒度头：`head_coarse` → `vocab_size_coarse` 类
- 细粒度头：通过门控机制融合粗粒度信息后输出 `vocab_size_fine` 类
- 方向头：`direction_head` → 3 分类（跌/平/涨）
- Horizon 头：`horizon_head_coarse/fine` → 未来 30 天预测

#### 3.4.2 LinearAttention

支持分块长序列的因果线性注意力，融合全局线性注意力与多尺度局部注意力。

**位置编码支持**：
- `rope`：Rotary Position Embedding
- `alibi`：Attention with Linear Biases（指数衰减）
- `learned`：可学习位置嵌入

**多尺度局部注意力**：
- 支持多个窗口大小（默认 `[128, 512]`）
- 融合方式：`gated`（门控加权）、`weighted`（softmax 加权）、`concat`（拼接投影）

**关键方法**：
| 方法 | 说明 |
|------|------|
| `forward(x)` | 标准前向，自动选择短序列/长序列路径 |
| `forward_with_cache(x)` | 返回注意力输出 + KV 状态（用于缓存推理） |
| `forward_incremental(x_new, kv_state, k_state, ...)` | 增量单步推理 |
| `_causal_softmax_attention(q, k, v)` | 标准 softmax 注意力（用于可视化） |

#### 3.4.3 RingAttentionBlock

残差注意力块：`LayerNorm → LinearAttention → 残差 → LayerNorm → FFN → 残差`

- FFN：`Linear(dim, dim*4) → GELU → Dropout → Linear(dim*4, dim) → Dropout`
- 支持梯度检查点（`enable_gradient_checkpointing`）
- 支持 `forward_with_cache` 和 `forward_incremental`

#### 3.4.4 LatentReasoner

并行 Latent Reasoner，替代 GRU ThinkingLayer。

**架构**：
- 可学习 latent tokens（默认 16 个）
- 每层：`Cross-Attention(latent→history) → Self-Attention(latent→latent) → FFN`
- 门控融合：`gate(x, latent_mean) → output`

**正则化**：
- Diversity Loss：相邻层 latent 差异过小则惩罚
- Collapse Loss：latent 方差过小则惩罚

#### 3.4.5 HorizonDecoder

30 个 future query tokens 并行预测未来 30 天。

**架构**：
- Horizon Embedding + Day/Month Calendar Embedding
- 每层：`Cross-Attention(queries→history) → Causal Self-Attention(queries) → FFN`
- 双头输出：coarse logits + fine logits（门控融合）

#### 3.4.6 RevIN

Reversible Instance Normalization，时间序列预测的标准技巧。

- `norm`：沿时间维度归一化（保存 mean/std）
- `denorm`：反归一化恢复原始分布
- 可选仿射变换（`affine_weight`, `affine_bias`）

#### 3.4.7 KVCache

增量推理缓存数据结构：

| 字段 | 说明 |
|------|------|
| `linear_attn_states` | 每层的 `(kv_state, k_state, state_anchor)` |
| `prefix_hidden` | 已编码序列的隐藏状态 |
| `prefix_len` | 已编码序列长度 |
| `sector_emb_cache` | 行业嵌入缓存 |

### 3.5 LoRA — `model/lora.py`

| 组件 | 说明 |
|------|------|
| `LoRALinear` | 带低秩残差分支的 Linear 层：`output = base(x) + scaling * B(A(dropout(x)))` |
| `inject_lora(model, ...)` | 注入 LoRA 到匹配 `target_keywords` 的 Linear 层 |
| `save_lora_adapter(model, path, ...)` | 保存 LoRA 权重和配置 |
| `load_lora_adapter(model, path, ...)` | 加载 LoRA 适配器 |
| `mark_only_lora_trainable(model)` | 冻结基座，仅 LoRA 可训练 |

### 3.6 训练流程

#### 3.6.1 Stage A: Tokenizer 训练 — `train_tokenizer.py`

```
数据加载 → 构建 HierarchicalQuantizer → VQ-VAE 训练 → 验证 → 保存最佳 checkpoint
```

- 优化器：Adam（fused → foreach → 基础，自动降级）
- 调度器：CosineAnnealingLR
- 损失：`recon_loss + vq_loss`（重建 MSE + 量化损失）
- 支持 AMP（bfloat16 优先）、CUDA prefetch
- 输出：`checkpoints/tokenizer.pt`、训练曲线 PNG、损失 CSV

#### 3.6.2 Stage B1: Base Model 预训练 — `train.py`

```
加载 Tokenizer → 预计算编码 → 构建 KronosReasoningGPT → Teacher-Forced 训练 → 验证 → 保存
```

**核心训练逻辑**：
- `base_one_step_loss()`：全序列 teacher-forced next-token prediction
- 损失 = `pred_loss(coarse CE + fine CE) + latent_loss(diversity + collapse)`
- 支持梯度累积、梯度检查点、torch.compile（自动回退）
- 支持单机多卡 DDP（自动检测 GPU 数量）
- CUDA Prefetcher 异步数据搬运
- 每 epoch 保存 `basemode-{epoch}.pt`，验证改善时更新 `base_model.pt`

**关键函数**：
| 函数 | 说明 |
|------|------|
| `train_model(local_rank, world_size, distributed)` | 训练入口 |
| `load_pretrained_tokenizer(device)` | 加载冻结的 Tokenizer |
| `latent_regularization_loss(latent_states)` | Latent 正则化损失 |
| `_prepare_batch(features, sector_ids, time_features, tokenizer, ...)` | Batch 预处理 |

#### 3.6.3 Stage B2: EXPO 后训练 — `posttrain/direction/train_da.py`

Direction-Accuracy EXPO（Execution Preference Optimization）后训练：

**核心思路**：
1. 从冻结参考策略采样候选 next-token 对
2. 根据次日方向准确性和收益误差构建 winner/loser 偏好
3. 用回归 EXPO 微调 token 策略

**标签生成**：
- `label_mode="rolling_vol"`：基于滚动波动率自适应阈值划分涨/跌/平
- `flat_policy="ignore"`：平盘样本不参与训练

**损失组合**：
- Direction EXPO Loss（偏好优化）
- Token CE Loss（可选，保持 token 预测能力）
- KL Divergence（约束策略偏移）
- Latent Regularization（可选）

**入口**：`Post_Train_DA.py` → `posttrain.direction.train_da.main()`

### 3.7 评估模块

#### 3.7.1 evaluate_predictions.py

| 函数 | 说明 |
|------|------|
| `load_model(device, checkpoint_path, ...)` | 从 checkpoint 加载模型 + tokenizer |
| `build_rolling_1d_eval_items(demo_dataset, symbols)` | 构建滚动 1 天预测评估项 |

#### 3.7.2 evaluate_checkpoints_1step.py

批量评估所有 checkpoint 的 1-step 预测性能：

- 从缓存中采样序列，用前 n-1 步预测第 n 步
- 解码预测 token → 反归一化 → 计算指标
- 指标：MAPE、Return MAPE、DA（方向准确率）、MAE、RMSE
- 输出：JSON 指标、CSV 表格、MAPE/DA 柱状图 PNG

#### 3.7.3 diagnose_codebook_diversity.py

Codebook 多样性诊断工具（只读）：
- 对比目标 token 分布与模型 last-step logits
- 计算熵、有效码字数、top-k 质量占比
- 验证缓存编码与当前 tokenizer 的一致性

### 3.8 模型对比模块 — `compare/`

#### 3.8.1 对比基线模型

| 模型 | 文件 | 说明 |
|------|------|------|
| Random Walk | `compare/models/rw/model.py` | 预测 = 上一日收益，无需训练 |
| TimeNet | `compare/models/timenet/model.py` | TCN（膨胀因果卷积），8 层，全局平均池化 |
| XGBoost | `compare/models/xgboost/` | 梯度提升树，使用最近 60 天特征 |

#### 3.8.2 统一推理 — `compare/run_inference.py`

从 CSV 数据加载 → 各模型推理 → 计算指标 → 保存 JSON 结果。

支持模型：`rw`, `timenet`, `xgboost`, `kronos-r`（含 ExPO 变体）

#### 3.8.3 Dashboard 服务 — `compare/server.py`

FastAPI 服务，提供 REST API 和前端 Dashboard：

| 端点 | 说明 |
|------|------|
| `GET /` | Dashboard HTML 页面 |
| `GET /api/models` | 列出所有模型配置 |
| `GET /api/results/{model_key}` | 获取特定模型结果 |
| `GET /api/compare` | 跨模型对比数据 |
| `GET /api/models/{key}/distribution` | 逐样本 MAPE/DA 分布 |
| `POST /api/run-inference` | 触发推理运行 |

---

## 4. 关键类与函数索引

### 4.1 模型层

| 类 | 文件 | 说明 |
|----|------|------|
| `BSQQuantizer` | `model/tokenizer.py` | BSQ 隐式码本量化器 |
| `HierarchicalQuantizer` | `model/tokenizer.py` | 2 级 BSQ 层次化分词器 |
| `LinearAttention` | `model/kronos_reasoning.py` | 因果线性注意力（全局+多尺度局部） |
| `RingAttentionBlock` | `model/kronos_reasoning.py` | 残差注意力块 |
| `LatentReasoner` | `model/kronos_reasoning.py` | 并行 Latent Reasoner |
| `HorizonDecoder` | `model/kronos_reasoning.py` | 未来 30 天并行预测解码器 |
| `KronosReasoningGPT` | `model/kronos_reasoning.py` | 主模型（Encoder + Reasoner + Decoder） |
| `RevIN` | `model/kronos_reasoning.py` | 可逆实例归一化 |
| `KVCache` | `model/kronos_reasoning.py` | 增量推理缓存 |
| `LoRALinear` | `model/lora.py` | LoRA 低秩适配 Linear 层 |
| `TimeNet` | `compare/models/timenet/model.py` | TCN 基线模型 |
| `RandomWalkModel` | `compare/models/rw/model.py` | 随机游走基线 |

### 4.2 数据层

| 类/函数 | 文件 | 说明 |
|---------|------|------|
| `AShareDataset` | `data_processor.py` | A 股序列数据集 |
| `MemmapArrayDataset` | `data_processor.py` | 基于 mmap 的数据集 |
| `NpyMemmapBackend` | `data_processor.py` | .npy mmap 只读后端 |
| `MemmapCacheWriter` | `data_processor.py` | mmap 缓存写入器 |
| `get_datasets()` | `data_processor.py` | 构建 train/val/demo 数据集 |
| `get_dataloaders()` | `data_processor.py` | 构建 DataLoader |
| `collate_fn()` | `data_processor.py` | 标准 collate |
| `collate_fn_v2()` | `data_processor.py` | 精简版 collate（memmap 模式） |

### 4.3 训练层

| 函数 | 文件 | 说明 |
|------|------|------|
| `train_model()` | `train.py` | Base Model 训练入口 |
| `base_one_step_loss()` | `train.py` | Teacher-forced 损失计算 |
| `latent_regularization_loss()` | `train.py` | Latent 正则化损失 |
| `load_pretrained_tokenizer()` | `train.py` | 加载冻结 Tokenizer |
| `save_checkpoint()` | `train.py` | 保存完整 checkpoint |
| `train_tokenizer()` | `train_tokenizer.py` | Tokenizer 训练主循环 |
| `main()` | `posttrain/direction/train_da.py` | EXPO 后训练入口 |

### 4.4 评估层

| 函数 | 文件 | 说明 |
|------|------|------|
| `load_model()` | `evaluate_predictions.py` | 从 checkpoint 加载模型 |
| `build_rolling_1d_eval_items()` | `evaluate_predictions.py` | 构建滚动评估项 |
| `main()` | `evaluate_checkpoints_1step.py` | 批量 checkpoint 评估 |

---

## 5. 依赖关系

### 5.1 外部依赖

| 包 | 用途 |
|----|------|
| `torch` | 深度学习框架 |
| `numpy` | 数值计算 |
| `pandas` | 数据处理 |
| `tqdm` | 进度条 |
| `matplotlib` | 可视化 |
| `fastapi` | 对比 Dashboard API |
| `uvicorn` | ASGI 服务器 |
| `pydantic` | 数据校验 |
| `psutil` | 系统资源监控 |
| `optuna` | 超参优化 |
| `xgboost` | XGBoost 基线模型 |

### 5.2 模块依赖图

```
train_tokenizer.py ──→ config.py
                  ──→ data_processor.py ──→ config.py, reproducibility.py
                  ──→ model/tokenizer.py ──→ config.py
                  ──→ model/tokenizer_config.py ──→ config.py

train.py ──→ config.py
         ──→ data_processor.py
         ──→ model/kronos_reasoning.py ──→ config.py
         ──→ model/tokenizer.py
         ──→ model/tokenizer_config.py
         ──→ reproducibility.py

Post_Train_DA.py ──→ posttrain/direction/train_da.py
                 ──→ config.py, data_processor.py, evaluate_predictions.py
                 ──→ model/lora.py, model/kronos_reasoning.py, model/tokenizer.py

evaluate_checkpoints_1step.py ──→ evaluate_predictions.py ──→ model/*

compare/run_inference.py ──→ evaluate_predictions.py
                        ──→ compare/models/*
```

---

## 6. 数据流

```
CSV 文件 (dataset/)
    │
    ▼
AShareDataset._process_data()
    │  特征工程: log_ret, log_high, log_low, log_open, log_vol, log_amt
    │  z-score 归一化 + 时间特征提取
    │  序列切分 (seq_len=1024, stride=seq_len*0.5)
    ▼
Dataset Cache (.pt / memmap .npy)
    │
    ▼
Tokenizer.precompute_encodings()
    │  HierarchicalQuantizer.encode() → (idx_coarse, idx_fine)
    ▼
DataLoader → collate_fn
    │
    ▼
_prepare_batch()
    │  因果切分: input=[:−1], target=[1:]
    ▼
KronosReasoningGPT.forward()
    │  Embedding → LinearAttention Blocks → Latent Reasoner → Output Heads
    ▼
Loss: CE(coarse) + CE(fine) + Latent Regularization
    │
    ▼
预测时: argmax → Tokenizer.decode() → 反归一化 → 收益率
```

---

## 7. 项目运行方式

### 7.1 环境准备

```bash
pip install -r requirements.txt
```

需要 CUDA GPU（推荐 ≥ 10GB VRAM）。将 A 股 CSV 数据放入 `dataset/` 目录。

### 7.2 完整训练流水线

#### Stage A: 训练 Tokenizer

```bash
python train_tokenizer.py
```

输出：`checkpoints/tokenizer.pt`

#### Stage B1: 训练 Base Model

```bash
python train.py
```

输出：`checkpoints/base_model.pt`、`checkpoints/basemode-{epoch}.pt`

#### Stage B2: EXPO 后训练

```bash
python Post_Train_DA.py
```

输出：`checkpoints/post_train_da/direction_expo.pt`

#### 一键运行（PowerShell）

```powershell
.\run_da_retrain_eval.ps1
```

依次执行：train → Post_Train_DA → evaluate_checkpoints_1step

### 7.3 评估

```bash
# 批量评估所有 checkpoint
python evaluate_checkpoints_1step.py --full

# 评估特定 checkpoint
python evaluate_checkpoints_1step.py --checkpoint-glob "checkpoints/base_model.pt"

# Codebook 多样性诊断
python diagnose_codebook_diversity.py --checkpoint checkpoints/base_model.pt
```

### 7.4 模型对比

```bash
# 运行所有基线推理
python compare/run_inference.py --step 1

# 仅运行特定模型
python compare/run_inference.py --models rw,xgboost --step 1

# 启动 Dashboard
python compare/server.py --port 8080
```

### 7.5 LoRA 微调

```bash
python train_lora.py path/to/stocks.csv --adapter-name my_adapter --epochs 5
```

### 7.6 配置覆盖

```bash
# 通过环境变量覆盖配置
export KRONOS_OVERRIDE_JSON=/path/to/overrides.json
python train.py

# 通过环境变量指定路径
export KRONOS_CHECKPOINT_DIR=/path/to/checkpoints
export KRONOS_BASE_MODEL_PATH=/path/to/base_model.pt
export KRONOS_OUTPUT_DIR=/path/to/outputs
```

### 7.7 分布式训练

```bash
# 自动检测多 GPU 并启动 DDP
python train.py  # 自动模式，检测到多 GPU 时启动 DDP
```

---

## 8. 模型参数规模

| 组件 | 默认参数 |
|------|----------|
| Tokenizer (HierarchicalQuantizer) | ~50K |
| KronosReasoningGPT (dim=256, depth=4) | ~2M |
| LoRA (rank=8) | ~10K (仅新增参数) |

---

## 9. 关键设计决策

1. **BSQ 隐式码本**：相比传统 VQ-VAE 的显式码本，BSQ 通过超平面投影隐式定义 `2^k` 个码字，避免码本坍塌和死码问题
2. **层次化量化**：coarse 捕获主结构，fine 编码残差细节，类似残差 VQ
3. **线性注意力 + 多尺度局部注意力**：全局线性注意力保证 O(N) 复杂度，局部窗口捕获短期依赖
4. **Latent Reasoner**：用可学习 latent tokens 的 cross-attention 替代 GRU 递推，实现并行推理
5. **Horizon Decoder**：30 个 future query tokens 并行预测，避免自回归误差累积
6. **EXPO 后训练**：通过偏好优化直接优化方向准确率，而非仅依赖 token CE 代理损失
7. **RevIN**：可逆归一化消除时间序列非平稳性，提升泛化能力
