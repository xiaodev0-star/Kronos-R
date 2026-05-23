# Rollout 后训练 —— 新一轮实验结果

运行时间：2026-05-05 18:09–18:30
运行环境：RTX 4060 Laptop 8GB, PyTorch 2.4.1+cu124
数据规模：120 stocks, train=128 windows, val=189 windows (full)

---

## 1. 总表

| 实验 | 方法 | path_mape | daily_mape | step10 | vs Base |
|:---:|------|:---:|:---:|:---:|:---:|
| — | **BaseModel**（未训练） | **4.9391%** | 2.0034% | 6.7428% | — |
| **N** | **Curriculum (2→5→7→10)** | **4.9181%** | 1.9968% | 6.7221% | **-0.021pp** |
| O | Path-Aware Loss | 4.9409% | 2.0056% | 6.7243% | +0.002pp |
| P | Beam Distill | 4.9589% | 2.0024% | 6.7921% | +0.020pp |
| S1 | Temp Anneal 1.5→0 | 5.5119% | 2.1483% | 7.5445% | +0.573pp |
| S2 | Temp Anneal 1.0→0.5 | 5.2735% | 2.0917% | 6.9611% | +0.334pp |
| R | Contrastive Trajectories | — | — | — | OOM |
| Q | GRPO with Path Reward | — | — | — | OOM |

## 2. 逐步 path_mape 对比

| Step | Base | N (Curriculum) | N-Base | O (PathAware) | P (BeamDistill) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.176 | 2.200 | +0.024 | 2.180 | 2.201 |
| 2 | 3.146 | **3.112** | -0.034 | 3.159 | 3.118 |
| 3 | 3.858 | **3.817** | -0.041 | 3.882 | 3.817 |
| 4 | 4.502 | 4.526 | +0.024 | 4.514 | 4.526 |
| 5 | 5.048 | **5.023** | -0.025 | 5.075 | 5.092 |
| 6 | 5.170 | **5.119** | -0.051 | 5.165 | 5.197 |
| 7 | 5.832 | **5.791** | -0.041 | 5.815 | 5.860 |
| 8 | 6.358 | 6.341 | -0.017 | 6.340 | 6.386 |
| 9 | 6.558 | 6.531 | -0.027 | 6.554 | 6.600 |
| 10 | 6.743 | **6.722** | -0.021 | 6.724 | 6.792 |

**N 在 10 步中有 8 步优于 Base**。最大改善在 Step 6（-0.051pp），最小在 Step 8（-0.017pp）。Step 1 和 Step 4 略差于 Base（+0.024pp）。

## 3. 逐实验分析

### N: Curriculum Horizon（课程学习）—— 唯一有效 ✅

**方法**：4 阶段渐进训练（horizon=2 → 5 → 7 → 10），每个阶段 12 updates，pure rollout + KL=0.05。

**结果**：path_mape 4.9181%，比 Base（4.9391%）改善 0.021pp。这是所有实验中唯一稳定超越 Base 的方法。

**为什么有效**：从短 horizon 开始让模型先学会在自反馈下做短链预测，逐步延长 horizon 让模型渐进适应更长的误差累积。相比直接从 horizon=10 开始训练（K 实验），curriculum 让模型有"学习阶梯"。

**与历史最优对比**：N（4.9181%）略差于 K-pure+KL+softCE（4.9142%，旧记录），但 N 的方法更简单（不需要 numeric soft CE），改善机制更清晰。

### O: Path-Aware Loss —— 无效 ❌

**方法**：在每步对 top-k token pair 做 soft 解码得到期望 return，累积 10 步得期望路径，最小化期望路径与真实路径的误差。

**结果**：4.9409%，与 Base（4.9391%）几乎相同。

**为什么失败**：soft 期望分布和 argmax 之间存在本质差距。训练优化的是"概率加权后的期望路径"，而推理时用的是 argmax token 路径。当 token 分布较分散时，期望值可能对应一个"不存在的 token pair"，梯度方向对 argmax 选择帮助有限。

### P: Beam Distill —— 恶化 ❌

**方法**：用 frozen BaseModel 做 best-of-4 sampling（每步采样 4 个候选、选数值误差最小的），以此做 teacher 蒸馏。

**结果**：4.9589%，比 Base 差 0.020pp。

**为什么失败**：Teacher（BaseModel 的 best-of-4 sampling）和 BaseModel 的 argmax 差距本身就很小，蒸馏无法提供足够的额外信号。且 best-of-4 的 teacher 质量也可能不足以引导模型改进——如果 BaseModel 本身在自反馈场景下的 4 个采样都不好，选最好的仍然不好。

### S1/S2: Temperature Annealing —— 恶化 ❌

**方法**：推理时前 5 步用温度采样（S1: 1.5→0, S2: 1.0→0.5），后续用 argmax。

**结果**：S1=5.5119%, S2=5.2735%，均显著差于 Base 的 argmax（4.9391%）。

**为什么失败**：温度采样引入的噪声在自回归场景下被放大。第 1 步的采样误差污染第 2 步的输入，即使后续用 argmax 也无法纠正。这证实在 10 步自回归场景下，确定性推理（argmax）优于随机采样。

### R: Contrastive —— 无法运行 ❌

采样 2 条完整轨迹进行对比学习，每条需要 10 步模型前向 → 训练时显存不够（8GB GPU 限制）。

### Q: GRPO —— 无法运行 ❌

采样多条轨迹做组内相对优势优化，需要额外 3×10 步前向 → 显存不够。

## 4. 结论与建议

### 4.1 核心结论

1. **Curriculum Horizon（N）是目前唯一验证有效的新方法**——path_mape 改善 0.021pp，且改善在 10 步中的 8 步上一致。

2. 收益绝对值很小（0.021pp），但考虑到：
   - BaseModel 的 path_mape 已经很低（4.94%），改善空间有限
   - 训练预算极小（48 updates, batch_size=2, 仅 1 epoch）
   - 改善一致性高（8/10 步优于 Base）
   - 这个结果是有信息量的

3. **Path-Aware Loss（O）、Beam Distill（P）、Temperature Annealing（S）在 10 步自回归场景下均无效或恶化**。它们分别受限于：期望-vs-argmax 差距、teacher 质量不足、噪声放大。

4. **R（对比学习）和 Q（GRPO）在当前硬件（8GB）上无法运行**。它们需要额外生成多条完整轨迹，显存需求超出预算。

### 4.2 建议的下一步

1. **验证 N 的泛化性**：在 300-stock 上复现 N，确认 curriculum 是否在更大数据范围上有效。
2. **消融 N**：测试"纯 horizon 课程 vs horizon 课程 + KL"、"不同 horizon 序列"、"每阶段 update 数分配"。
3. **N + K 组合**：将 curriculum 与 numeric soft CE（K 实验的最好配置）组合，可能获得叠加收益。
4. **增大训练预算**：48 updates 严重不足。在更大 GPU 上将 curriculum 训练扩展到 200+ updates。

### 4.3 R 和 Q 的运行建议

如果将来有 ≥16GB GPU 可用，在 `run_rq.py` 中设置 `--batch-size 1` 应该能跑通 R 和 Q。预期：
- R（轨迹对比学习）在理论上比 O 更合理（直接对比轨迹而非期望路径），可能有效
- Q（GRPO）直接优化 path_mape reward，是理论上最完善的方法，值得在更大 GPU 上验证
