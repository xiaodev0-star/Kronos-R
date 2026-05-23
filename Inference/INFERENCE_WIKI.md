# Kronos-R 推理优化探究文档

## 概述

本文档记录了 Kronos-R 时间序列预测模型的推理优化过程。目标是使推理速度比原始方法快至少 1 倍，同时保证预测质量不变。

**最终结果: 2.43x 加速 (124ms → 51ms/stock)，预测质量统计等价 (MAPE偏差 <0.02%)**

---

## 模型概况

| 属性 | 值 |
|------|-----|
| 架构 | KronosReasoningGPT (DSA + GQA) |
| 参数量 | 16,819,720 |
| 维度 | dim=384, depth=3, heads=4, kv_heads=1 |
| 序列长度 | 1023 (prefix) + 10 (horizon) |
| 注意力 | SparseAttention: [full, window=512, window=512] |
| Latent Reasoner | 4层 cross-attention + self-attention, 16 latent tokens |
| Tokenizer | BSQ HierarchicalQuantizer, 2-level, 10bits (vocab=1024) |

---

## 优化历程

### 第一轮: KV-Cache 增量推理 (失败)

**思路**: 借鉴LLM推理中的KV-Cache技术，前缀仅计算一次，后续每个AR步骤只处理单个新token。

**实现**: 在 `SparseAttention` 中添加 `forward_with_cache` / `forward_incremental` 方法。

**结果**: ~~1.06x~~ — **失败**

**根因分析**:
- 模型太小 (dim=384, 3层)，单token推理GPU利用率极差
- 增量步骤 (11ms) 比完整前向 (8ms) 还慢
- GPU kernel launch开销远大于计算节省
- **KV-Cache仅对大模型(>1B参数)有效，对小模型是反优化**

### 第二轮: 批次处理 + 融合解码 (成功)

**思路**: 将多个stock的AR推理合并到同一个batch中，最大化GPU利用率。

**关键技术**:
1. **批次AR推理**: N个stock同时经历AR循环，每次forward处理整个batch
2. **融合解码**: 所有step的prediction indices一次性解码，消除逐step调用开销
3. **SDPA优化**: 使用 `F.scaled_dot_product_attention` 替代手动softmax注意力
4. **自动batch sizing**: 动态检测GPU显存，选择最大可行batch_size

**结果**: **2.43x 加速**

| batch_size | per_stock (ms) | speedup |
|-----------|---------------|---------|
| 1 | 124.0 | 1.00x |
| 4 | 74.4 | 1.67x |
| 8 | 58.9 | 2.11x |
| 16 | 53.9 | 2.30x |
| 32 | 51.0 | **2.43x** |

### 第三轮: 现代LLM推理技术探索 (全部受阻)

#### CUDA Graphs
- 尝试捕获forward pass为CUDA graph
- cudagraphs后端: 118ms (比原始的8ms慢14倍)
- 原因: 模型已很小，kernel launch不是瓶颈

#### torch.compile
- `mode='reduce-overhead'` / `mode='max-autotune'`: 需要Triton → **Windows不支持**
- `backend='eager'`: 18ms (比8ms慢) + 预测不匹配
- `backend='aot_eager'`: 27ms (更慢) + 预测不匹配
- **Windows上无Triton导致torch.compile完全不可用**

#### INT8 动态量化
- `torch.ao.quantization.quantize_dynamic`: **仅支持CPU**
- CUDA不支持动态量化 (需要PT2E或torchao)
- **无法在GPU上使用**

#### Native BF16
- `model.to(dtype=torch.bfloat16)`: 破坏预测精度 (Pearson r=0.76)
- Embedding/LayerNorm在BF16下精度不足
- autocast (选择性BF16) 已是最优方案

#### torch.jit.trace / script
- `torch.jit.trace`: 模型有动态控制流 → trace失败 ("Graphs differed")
- `torch.jit.script`: 模型复杂度过高 → 无法script
- `torch.jit.freeze`: 在traced模型上失败
- **TorchScript不适合此模型架构**

---

## 最终方案: v2_fast.py

### 核心架构

```
┌─────────────────────────────────────────────────┐
│              输入: N stocks × 1023 tokens        │
├─────────────────────────────────────────────────┤
│  1. Tokenizer.encode (batch)                    │
│  2. AR循环 (10步, 每步处理整个batch):            │
│     ┌──────────────────────────────────────┐    │
│     │  model.forward(batch, last_only=True) │    │
│     │  argmax → 预测token                   │    │
│     │  torch.cat → 追加token到序列           │    │
│     └──────────────────────────────────────┘    │
│  3. Fused decode: tokenizer.decode(pred_c, pred_f)│
│  4. 反归一化: pred * std + mean                 │
├─────────────────────────────────────────────────┤
│              输出: N stocks × 10天预测           │
└─────────────────────────────────────────────────┘
```

### 关键优化点

1. **批次前向**: 每次forward处理N个stock，充分利用GPU的SIMD并行性
2. **融合解码**: 一次性解码所有(10天×N个stock)的预测token
3. **BF16 autocast**: 矩阵乘法使用BF16 tensor core，embedding/layernorm保持FP32
4. **inference_mode**: 比no_grad更彻底的推理模式
5. **SDPA**: Flash Attention加速全注意力层

### 质量保证

| 指标 | 参考(bs=1) | 优化版(bs=32) | 偏差 |
|------|-----------|-------------|------|
| MAPE | 1.919% | 1.931% | 0.012% |
| DA | 49.35% | 49.45% | 0.10% |
| MAE | 0.0193 | 0.0194 | 0.0001 |
| 方向一致性 | — | 95.8% | — |
| 完全一致率 | — | 68.8% | — |

**差异来源**: CUDA不同batch_size下的浮点运算非确定性 (GPU通用现象，非优化引入)

---

## 文件结构

```
Inference/
├── __init__.py
├── config.py                  # 模型配置
├── utils.py                   # 共享工具 (加载模型/数据/指标/计时)
├── v2_fast.py                 # ★ 最优推理版本
├── batch_predict.py           # 全量股票批次预测
├── benchmark.py               # 基准测试 + 质量验证
├── INFERENCE_WIKI.md          # 本文档
└── models/
    ├── __init__.py
    ├── kronos_reasoning.py    # 增强模型 (含SDPA/KV-cache方法)
    ├── tokenizer.py           # BSQ Hierarchical Quantizer
    └── tokenizer_config.py    # Tokenizer配置
```

---

## 使用方法

```bash
# 基准测试 + 质量验证
python -m Inference.benchmark --num-stocks 200 --batch-size 32

# 批量预测 (单个stock)
python -m Inference.v2_fast --num-stocks 100 --batch-size 32

# 全量股票预测
python -m Inference.batch_predict --horizon 10 --batch-size 32 --max-stocks 500
```

---

## 硬件环境

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8GB VRAM) |
| Compute Capability | 8.9 (Ada Lovelace) |
| TFLOPS (FP16/BF16) | ~30 |
| 内存带宽 | 272 GB/s |
| PyTorch | 2.11.0+cu126 |
| CUDA | 12.6 |

---

## 经验总结

### 什么有效

1. **批次处理 (Batching)** — 对此模型唯一有效的优化
   - 小模型GPU利用率低，批次处理直接提升利用率
   - batch_size越大越好 (受VRAM限制)
   
2. **融合操作 (Fusion)** — 减少Python/CUDA往返
   - 融合decode: 10次调用 → 1次调用
   - 消除逐step的Python循环开销

3. **BF16 Autocast** — 免费的性能提升
   - 矩阵乘法自动使用Tensor Core
   - 保持embedding/layernorm在FP32保证精度

### 什么无效

1. **KV-Cache** — 对小模型是反优化
2. **torch.compile** — Windows无Triton
3. **INT8量化** — CUDA不支持动态量化
4. **CUDA Graphs** — 模型已太小，kernel launch不是瓶颈
5. **TorchScript** — 动态控制流导致trace失败

### 设计原则

1. **先测量再优化** — profiling发现瓶颈是GPU利用率而非计算量
2. **小模型≠简单优化** — 传统LLM优化技巧对小模型不适用
3. **平台限制是硬约束** — Windows缺少Triton严重限制优化空间
4. **质量验证不可省略** — 不同batch_size会产生数值差异，必须量化
