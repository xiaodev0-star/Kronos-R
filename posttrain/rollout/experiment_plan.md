# Rollout 后训练 —— 新一轮实验计划

更新时间：2026-05-05
制定依据：test.md 中 A–L 全部实验结果及 path_mape 口径复算。

---

## 1. 问题定位

### 1.1 核心瓶颈

当前 rollout 后训练在 120-stock 上最好的 K（pure + KL + softCE）path_mape = 4.9142%，仅比 Base（4.9391%）好 0.025pp。在 300-stock 上没有任何 checkpoint 超越 Base。根因有三层：

**第一层：训练信号与评估指标不一致。** 训练用 per-step token CE 做损失，评估用累计路径 path_mape。token CE 最小化的是 token identity 误差，而不是"10 天后收盘价路径的接近程度"。一个 token pair 在 CE 意义下 100% 正确，在数值上仍可能有偏差；10 步累积后，token 级别的微小数值偏差被路径复利放大。

**第二层：自反馈上下文的分布偏移。** 模型用 teacher-forcing 预训练（前一步永远是真值），而 rollout 推理时前一步是模型自己的（可能错误的）预测。即使 pure rollout 训练试图弥合这个 gap，48 个 update 远不足以让模型在自反馈分布上重新收敛。

**第三层：错误累积无反馈机制。** 第 t 步的预测误差会在第 t+1 步的输入中引入扰动，但标准 CE 训练只在第 t 步产生梯度——它不知道这个误差会如何影响第 t+1 到第 10 步。模型没有被训练"为后续步骤着想"。

### 1.2 为什么已有方法效果有限

| 方法 | 问题 |
|------|------|
| Scheduled self-rollout（D/E） | 混合 TF/self 上下文仍是分布折中，低 lr 仅减缓漂移 |
| Pure rollout（I） | 无约束时 token 分布坍缩 |
| Pure + KL（J/K） | KL 保持 Token 分布，但也限制了模型在自反馈下调整 prediction |
| Numeric soft CE（K） | 软标签只在 top-k 候选内构造，top-k 候选里的数值近邻未必改变 argmax |
| Heads-only（G/H/L） | 表示空间不变，输出头的调整能力有限 |
| Train-only 校准 | 不改变 token 生成路径；且优化 daily MAPE 的校准器在 path_mape 上可能反向 |

---

## 2. 实验设计原则

1. **优化目标必须贴近 path_mape**：不是 token CE 的变体，而是把累计路径误差注入训练梯度。
2. **训练必须让模型"感受"到错误累积的后果**：第 t 步的选择要知道它对第 t+10 步的影响。
3. **控制训练成本**：单实验 ~48 updates × batch_size=2，在 RTX 4060 上约 10–20 分钟。每个方案先在 120-stock 小规模验证，有希望的再扩到 300。
4. **评估口径统一为 path_mape**：主指标是 val full path_mape 平均（及第 10 天 path_mape），辅指标是 daily MAPE 和逐 step path_mape 曲线。
5. **所有实验从 BaseModel 初始化**，对照 BaseModel raw 的 val full path_mape。

---

## 3. 实验矩阵

### 实验 N：Scheduled Horizon Curriculum（课程学习）

**动机**：当前所有实验都直接从 horizon=10 开始训练。模型在自反馈下同时应付 10 步压力过大。先学会走（短 horizon 自反馈），再学跑（长 horizon）。

**方法**：
- 训练分 4 个阶段，每个阶段 horizon 递增：2 → 5 → 7 → 10
- 每个阶段 pure rollout（ratio=1.0）+ KL=0.05
- 阶段之间不重置 optimizer state，lr 持续衰减
- 总 updates 保持 ~48（每个阶段约 12 updates）

**预期**：短 horizon 阶段让模型先适应自反馈输入分布，逐步延长 horizon 让模型渐进适应更长的误差累积链。可能比直接从 horizon=10 训练更稳定。

**新增参数**：
- `--curriculum-horizons "2,5,7,10"`：各阶段 horizon 序列
- `--curriculum-updates "12,12,12,12"`：各阶段 update 数

**对比基线**：K（pure+KL+softCE, horizon=10 全程），Base raw。

---

### 实验 O：Differentiable Path-Aware Loss（可微路径损失）

**动机**：token CE 不感知数值和路径。如果能在训练时通过 soft token 构造可微的"期望路径"，直接最小化期望路径与真实路径的差异，梯度就能告诉模型哪些 token 在路径意义上更好。

**方法**：
- 在每一步：取 top-k coarse × top-k fine 候选 token pair（含 gold pair）
- 用 softmax 得到 pair 概率分布
- 通过 frozen tokenizer decoder 将每个 pair 解码为数值（close-ratio）
- 将概率分布与数值做加权求和，得到"soft 期望 return"
- 累积 10 步 soft return 得到期望路径
- 与真实路径做 path_mape-like loss（L1 或 Huber）
- 通过可微路径反向传播到 token logits
- 搭配 KL 和轻量 token CE 防止分布坍缩

**与 numeric soft CE（K 实验）的关键区别**：
- K 的 numeric soft CE 只在单个 step 内构造软标签，不做跨步累积
- O 的 path-aware loss 跨 10 步累积后再算 loss，梯度通过整个路径链回传
- 这会鼓励模型在早期步骤选择"虽然不是单步最优但利于后续路径"的 token

**新增参数**：
- `--path-aware-weight 0.5`：路径损失权重
- `--path-aware-top-k 8`：每步候选 pair 数
- `--path-aware-temp 0.005`：构造 soft 概率的温度

**实现要点**：
- 需要把 tokenizer decoder 设为 eval mode 并确保其可微分（它是标准 MLP，天然可微）
- 累积 soft return 时用 `exp(cumsum(log(1+soft_ret)))` 避免数值问题
- 梯度只传到 logits，tokenizer 权重冻结

**风险**：
- 10 步 soft 累积可能导致梯度消失或数值不稳定
- soft 期望路径可能与 argmax 路径差距较大（分布期望 vs 众数）
- 需要在 token CE 和 path loss 之间小心平衡

---

### 实验 P：Beam Search Distillation（束搜索蒸馏）

**动机**：argmax rollout 是推理时最便宜的方案，但不是最好的。beam search（每步保留 top-b 条路径）在推理时能显著改善路径质量。如果能用 beam search 生成更优的 pseudo-label 轨迹，再通过蒸馏让模型学会用 argmax 逼近 beam search 的质量，就能在不增加推理成本的前提下提升效果。

**方法**：
- 训练时对每个 batch 样本：
  1. 冻结 reference（BaseModel），对 prefix 做 beam search rollout（beam width=4）
  2. 从 4 条 beam 路径中选择 path_mape 最低的作为 teacher trajectory
  3. 用 teacher trajectory 的 token 作为 pseudo-label，对训练模型做 CE loss
- 搭配 KL 保持分布稳定
- 训练模型用 argmax 但学习 beam search 的选择

**新增参数**：
- `--beam-width 4`：蒸馏用的 beam 宽度
- `--distill-weight 1.0`：蒸馏 CE 权重

**实现要点**：
- Beam search 需要修改 `_build_scheduled_inputs`，从每步保留 top-b 条路径
- 计算每条 beam 路径的累计 path error 来选 best beam
- 蒸馏阶段不需要 rollout ratio 参数（teacher 来自 beam search）
- Beam search 用 BaseModel（frozen）执行，不参与梯度

**风险**：
- Beam search 在 token 级别做，coarse/fine pair 组合数大，需要高效的 top-b 筛选
- BaseModel 的 beam search 可能本身也不够好，天花板有限
- 蒸馏可能只是让模型记忆 teacher 的选择，泛化有限

---

### 实验 Q：GRPO with Path Reward（群组相对策略优化）

**动机**：强化学习可以直接优化 path_mape。GRPO（Group Relative Policy Optimization）不需要 value network，只需要对同一个 prefix 采样多条 rollout 轨迹，用组内相对排名做 advantage，训练稳定且实现较简单。

**方法**：
- 对每个 prefix，从当前模型采样 G=4 条 rollout 轨迹（temperature=1.0）
- 计算每条轨迹的 path_mape，组内归一化得到 advantage（path_mape 越低 advantage 越高）
- 用 GRPO 目标更新策略：`L = -E[ratio * advantage - beta * KL]`
- 同时保留轻量 token CE 防止 token 分布过度偏离
- Reference 模型用 BaseModel（frozen）

**新增参数**：
- `--grpo-group-size 4`：每组采样轨迹数
- `--grpo-clip-eps 0.2`：PPO clip 范围
- `--grpo-beta 0.04`：KL penalty 系数

**与 ExPO 的对比**：
- ExPO 优化的是单步 token pair 偏好（winner/loser pair），输入是 gold prefix
- GRPO 优化的是完整 10 步轨迹的 path_mape，输入包含自反馈
- GRPO 天然处理错误累积问题

**风险**：
- 采样 4 条轨迹 × batch_size=2 = 8 次完整前向，训练速度显著下降
- 小 batch 下 advantage 估计方差大
- RL 训练本身不稳定，可能需要较多调参

---

### 实验 R：Contrastive Trajectory Pairs（轨迹对比学习）

**动机**：与其用 token 级别的 CE，不如直接对比两条完整轨迹的优劣。采样两条轨迹，用 path_mape 判断好坏，用对比损失拉近好轨迹、推远坏轨迹。

**方法**：
- 对每个 prefix，用不同随机种子采样 2 条 rollout 轨迹
- 计算每条轨迹的 path_mape
- 用 margin-based contrastive loss：`L = max(0, margin + logP(good) - logP(bad))`
- 搭配 KL 约束
- 只在轨迹级别产生梯度，不需要每步 token label

**新增参数**：
- `--contrastive-margin 0.1`
- `--contrastive-temp 1.0`

**与 GRPO（Q）的对比**：
- R 更简单：只有 pair 比较，不涉及 importance sampling ratio 和 clip
- Q 在理论上更完整（multiple samples + ratio + clip），但实现更复杂
- 建议先跑 R 验证轨迹级别信号是否有效，再决定是否投入 Q

---

### 实验 S：Inference-Time Temperature Annealing（推理时温度退火）

**动机**：这不是训练方法，而是推理时的改进。早期 rollout 步的预测最不确定，用较高温度采样可能探索到更好的路径；后期步应该用低温/argmax 减少噪声。

**方法**：
- 修改 `predict_autoregressive_returns`：前 k 步用 temperature sampling（temp 从高到低），后续用 argmax
- 不涉及训练，直接在 BaseModel 和已有 checkpoint 上评估

**新增参数**（eval 侧）：
- `--anneal-start-temp 1.5`：第 1 步温度
- `--anneal-end-temp 0.0`：最后退火步温度（0.0 = argmax）
- `--anneal-steps 5`：退火持续的步数

**预期**：
- 这可能是"零成本"改进——不改权重，只改推理策略
- 如果有效，可以与任意训练方法叠加

---

## 4. 推荐执行顺序与优先级

建议按"投入产出比"从高到低执行：

| 优先级 | 实验 | 理由 |
|:---:|------|------|
| **P0** | **N: Curriculum** | 实现最简单，思路直接，训练成本不变。只改 horizon 调度，不改 loss。成功概率高。 |
| **P0** | **S: Temp Annealing** | 零训练成本，纯推理端改进。跑一次 eval 就能验证。如果有效可立即用于所有模型。 |
| **P1** | **O: Path-Aware Loss** | 首次把 path_mape 信号直接注入训练梯度。是"对的方向"。实现复杂度中等。 |
| **P1** | **P: Beam Distill** | 用更强的推理策略（beam search）当 teacher。训练成本增加（需 beam search），但思路清晰。 |
| **P2** | **R: Contrastive** | 轨迹级别对比，比 GRPO 简单。验证"轨迹级信号 > token 级信号"的假设。 |
| **P2** | **Q: GRPO** | 理论最完善（直接优化 path_mape），但实现最复杂，训练最慢。作为兜底方案。 |

## 5. 实现计划

### 5.1 实验 N（Curriculum）—— 预计改动量最小

修改 `train_rollout.py` 的 `train()` 函数：
- 新增 `curriculum_horizons` 和 `curriculum_updates` 参数
- 外层循环从 `for epoch` 改为 `for stage in curriculum`
- 每个 stage 使用不同的 `cfg.horizon`（需支持运行时覆盖）
- 每个 stage 的 rollout 数据需重建 DataLoader（不同 horizon 需要不同窗口长度）

**命令模板**：
```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' Post_Train_Rollout.py `
  --max-stocks 120 --max-train-samples 128 --max-val-samples 64 `
  --batch-size 2 --eval-batch-size 8 --epochs 1 --max-train-updates 48 `
  --output-dir checkpoints\post_train_rollout_exp_n_curriculum120 `
  --save-name rollout_exp_n.pt --use-gradient-checkpointing false `
  --rollout-ratio-start 1.0 --rollout-ratio-end 1.0 `
  --anchor-weight 0 --kl-weight 0.05 --numeric-mape-weight 0 `
  --numeric-soft-ce-weight 0 --step-weight-gamma 0.75 --lr 5e-6 `
  --curriculum-horizons "2,5,7,10" --curriculum-updates "12,12,12,12"
```

### 5.2 实验 O（Path-Aware Loss）—— 新增 loss 函数

在 `train_rollout.py` 中新增 `_differentiable_path_loss()`：
- 输入：每步 logits、tokenizer decoder、真实 returns、means/stds
- 流程：
  1. 每步取 top-k pair，softmax 概率
  2. 解码每个 pair → 数值 return
  3. 加权求和得 soft expected return
  4. 累积 10 步得期望路径
  5. 与真实路径算 smooth L1
- 反向传播自动通过 decoder（frozen）传到 logits

**命令模板**：
```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' Post_Train_Rollout.py `
  --max-stocks 120 --max-train-samples 128 --max-val-samples 64 `
  --batch-size 2 --eval-batch-size 8 --epochs 1 --max-train-updates 48 `
  --output-dir checkpoints\post_train_rollout_exp_o_pathaware120 `
  --save-name rollout_exp_o.pt --use-gradient-checkpointing false `
  --rollout-ratio-start 1.0 --rollout-ratio-end 1.0 `
  --anchor-weight 0 --kl-weight 0.05 --numeric-mape-weight 0 `
  --numeric-soft-ce-weight 0 --step-weight-gamma 0.75 --lr 5e-6 `
  --path-aware-weight 0.5 --path-aware-top-k 8 --path-aware-temp 0.005
```

### 5.3 实验 P（Beam Distill）—— 新增 beam search

在 `train_rollout.py` 中新增：
- `_beam_search_rollout()`：用 reference model 做 beam search，返回最佳轨迹的 token 序列
- 修改 `rollout_training_loss()`：增加 distill 模式，target 来自 beam search 而非 gold

### 5.4 实验 S（Temp Annealing）—— 仅改 eval

修改 `predict_autoregressive_returns()`（`eval_rollout.py` 中引用）：
- 新增 temperature schedule 参数
- 前 k 步用 multinomial sampling，后续用 argmax
- 不涉及训练

**命令模板**：
```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' -m posttrain.rollout.eval_rollout `
  --include-base true `
  --checkpoint checkpoints\post_train_rollout_exp_k_pure_softce120\rollout_exp_k.pt `
  --mode val --max-stocks 120 --max-val-samples 0 --batch-size 8 `
  --output-dir outputs\post_train_rollout_exp_s_anneal `
  --sample-temp-start 1.5 --sample-temp-end 0.0 --sample-steps 5
```

---

## 6. 成功标准

一轮实验的目标不是找到"大幅超越 Base"的银弹——在当前问题难度下，稳定的 +0.1–0.3pp path_mape 改善已经是进步。具体标准：

| 级别 | 标准 |
|:---:|------|
| **最低** | 任何一个新方法在 120-stock full-val 上 path_mape < Base（4.9391%），且差异 > 0.05pp |
| **中等** | 在 120-stock 上 path_mape 改善 > 0.1pp，且 300-stock 上不差于 Base |
| **理想** | 在 300-stock 上 path_mape 稳定优于 Base（> 0.1pp），且方法可推广 |

如果 N+O 组合（curriculum + path-aware loss）在 120 上拿到 path_mape < 4.85%，就值得投入更多资源做 300-stock 验证和消融实验。

---

## 7. 长期方向（本轮不实现，留作记录）

以下方向在当前训练预算下不太现实，但如果模型规模或训练预算扩大，值得考虑：

1. **RL + MCTS 轨迹搜索**：用 Monte Carlo Tree Search 在 token 空间搜索最优轨迹，类似 AlphaGo。需要大量的 rollout 模拟，当前单条 rollout ~1s 太慢。
2. **Meta-learning for self-correction**：训练一个轻量"修正网络"在每步后调整 hidden state，使其更接近"如果用了真值"的 hidden state 分布。
3. **Adversarial prefix augmentation**：在 prefix 中加入对抗扰动，提高模型对输入噪声的鲁棒性（因为自反馈产生的 token 等价于噪声输入）。
4. **Auxiliary inverse model**：训练一个反向预测头（从未来预测过去），用循环一致性约束正向 rollout 的路径合理性。
5. **Longer training with more data**：当前 48 updates 严重不足。如果有更大 GPU 和时间，pure rollout + KL 训练 500+ updates 可能自然收敛到更好的解。
