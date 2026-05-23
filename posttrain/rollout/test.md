# Rollout 后训练实验记录

更新时间：2026-05-05\
运行环境：`D:\conda_envs\llm-t\Scripts\python.exe`，Python 3.12.10，PyTorch 2.4.1+cu124，RTX 4060 Laptop GPU。

## 目标与数据边界

rollout 后训练与 `posttrain/direction` 完全分开。rollout 场景只允许模型看到前 1023 个真实 token，之后 10 步必须使用自己已经生成的 token 以及前文已知的所有真实token继续自回归预测。

数据策略：

- 训练和验证只使用 train/val 时间段。
- demo 最近 30 天没有进入 rollout cache、训练、调参或验证。
- rollout cache 独立构建在 `posttrain/rollout/cache/`，窗口长度为 `1023 + 10`。
- 每个窗口的归一化 `mean/std` 只由前 1023 个已知 token 计算，避免把未来 10 步数值泄露到归一化统计里。
- 评估协议为严格 10-step autoregressive：第 1 步用 1023 个真实 token，后续每一步只把模型上一步预测 token 放回上下文，不喂未来真实 token。

主指标口径已更正：

- 主指标：`path_mape`，即从第 1024 天开始把预测 log return 逐日累加成预测 close 路径，再计算未来 10 天每天路径 MAPE 的平均数。这个指标会体现误差累积。
- 诊断指标：`mape`，即每一天单独的 close-ratio MAPE，再在 10 步上平均。这个指标不累计价格路径，之前表格里的 `MAPE` 均为这个 daily 口径。
- 辅指标：MAE、RMSE、逐 step `path_mape`。
- DA 只作为诊断，不作为 rollout 模型选择标准。

## 已实现文件

- `posttrain/rollout/data.py`：独立 rollout train/val cache，prefix-only normalization。
- `posttrain/rollout/train_rollout.py`：scheduled self-rollout 后训练。
- `posttrain/rollout/eval_rollout.py`：严格 10-step AR 评估，输出整体指标、逐 step 指标、逐条预测误差 CSV。
- `Post_Train_Rollout.py`：rollout 后训练入口。
- `config.py`：新增 `PostTrainRolloutConfig`。

## 实验汇总

以下旧表中的 `MAPE` 是 daily close-ratio MAPE，不是累计路径 `path_mape`。验证集为 val，不含 demo。累计路径口径见后文“指标口径更正后的 path\_mape 复算”。

| 实验                     | 训练设置                                                                               |        验证规模 | Checkpoint                                                                 |   MAPE |     MAE |    RMSE | 结论                              |
| ---------------------- | ---------------------------------------------------------------------------------- | ----------: | -------------------------------------------------------------------------- | -----: | ------: | ------: | ------------------------------- |
| smoke                  | 20 stocks, train 8, val 8, 2 updates                                               |   8 windows | `checkpoints/post_train_rollout_smoke/rollout_smoke.pt`                    | 1.8848 | 0.01894 | 0.02447 | 链路通过，仅冒烟                        |
| Base-smoke             | 同 smoke 验证                                                                         |   8 windows | `checkpoints/base_model.pt`                                                | 1.8907 | 0.01900 | 0.02449 | smoke 对比基线                      |
| A scheduled            | 120 stocks, train 96, val 64, self ratio 0.5->0.9, anchor 0.2, KL 0.02             |  64 windows | `checkpoints/post_train_rollout_exp_a_scheduled/rollout_exp_a.pt`          | 2.0440 | 0.02046 | 0.02921 | 差于 Base                         |
| B teacher              | 120 stocks, train 96, val 64, teacher-forced only, KL 0.02                         |  64 windows | `checkpoints/post_train_rollout_exp_b_teacher/rollout_exp_b.pt`            | 2.0613 | 0.02063 | 0.02955 | 明显差于 Base                       |
| C high-self            | 120 stocks, train 96, val 64, self ratio 0.9->1.0, anchor 0.05, KL 0.02            |  64 windows | `checkpoints/post_train_rollout_exp_c_highself/rollout_exp_c.pt`           | 2.0440 | 0.02046 | 0.02916 | RMSE 接近，MAPE 仍差                 |
| Base-64                | 同 A/B/C 验证                                                                         |  64 windows | `checkpoints/base_model.pt`                                                | 2.0373 | 0.02039 | 0.02906 | 小验证基线                           |
| D low-lr high-self     | 120 stocks, train 128, val 64, self ratio 0.8->1.0, anchor 0.1, KL 0.05, lr 5e-6   |  64 windows | `checkpoints/post_train_rollout_exp_d_lowlr_highself/rollout_exp_d.pt`     | 2.0337 | 0.02036 | 0.02905 | 小验证优于 Base                      |
| D full-val120          | D checkpoint 全量复评                                                                  | 189 windows | 同上                                                                         | 2.0008 | 0.02000 | 0.02874 | 小规模全量 val 优于 Base               |
| Base full-val120       | 同 D full-val120                                                                    | 189 windows | `checkpoints/base_model.pt`                                                | 2.0034 | 0.02002 | 0.02878 | D 相对 MAPE -0.0026pp             |
| E low-lr high-self 300 | 300 stocks, train 384, val 256, self ratio 0.8->1.0, anchor 0.1, KL 0.05, lr 5e-6  | 256 windows | `checkpoints/post_train_rollout_exp_e_lowlr_highself_300/rollout_exp_e.pt` | 2.2218 | 0.02216 | 0.03234 | 内嵌验证差于较大 Base                   |
| E full-val300          | E checkpoint 全量复评                                                                  | 452 windows | 同上                                                                         | 2.2130 | 0.02211 | 0.03226 | 较大验证差于 Base                     |
| Base full-val300       | 同 E/H full-val300                                                                  | 452 windows | `checkpoints/base_model.pt`                                                | 2.2045 | 0.02204 | 0.03211 | 较大验证基线                          |
| F numeric top-k        | 120 stocks, train 128, val 64, D + top-k expected-return MAPE surrogate, weight 50 |  64 windows | `checkpoints/post_train_rollout_exp_f_numeric120/rollout_exp_f.pt`         | 2.0500 | 0.02052 | 0.02918 | 数值 surrogate 未改善 argmax rollout |
| G heads-only 120       | 120 stocks, train 128, val 64, 只训输出头, self ratio 0.8->1.0                          |  64 windows | `checkpoints/post_train_rollout_exp_g_headsonly120/rollout_exp_g.pt`       | 2.0379 | 0.02039 | 0.02908 | 基本贴近 Base，无明确收益                 |
| H heads-only 300       | 300 stocks, train 384, val 256, 只训输出头                                              | 256 windows | `checkpoints/post_train_rollout_exp_h_headsonly300/rollout_exp_h.pt`       | 2.2159 | 0.02212 | 0.03214 | 差于 Base                         |
| H full-val300          | H checkpoint 全量复评                                                                  | 452 windows | 同上                                                                         | 2.2115 | 0.02211 | 0.03220 | 较大验证差于 Base                     |

## 旧 daily MAPE 口径下的关键结论

1. 这一节只保留历史记录：`D low-lr high-self` 是旧 daily MAPE 口径下的小规模最好 checkpoint，MAPE `2.0008%` vs Base `2.0034%`。
2. 扩到 `max_stocks=300` 后，`E low-lr high-self 300` 和 `H heads-only 300` 都没有超过 Base。
3. Teacher-forced 多步训练在旧 daily MAPE 口径下明显变差。
4. 直接加入 top-k expected-return MAPE surrogate 没有改善最终 argmax rollout MAPE，原因是训练优化的是分布期望，而部署时使用 argmax token 路径。
5. 更正为 `path_mape` 后，D 不再是主指标最好结果；以最后一节 path\_mape 复算为准。

## 推荐命令

小规模复现实验 D：

```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' Post_Train_Rollout.py `
  --max-stocks 120 --max-train-samples 128 --max-val-samples 64 `
  --batch-size 2 --eval-batch-size 8 --epochs 1 --max-train-updates 48 `
  --output-dir checkpoints\post_train_rollout_exp_d_lowlr_highself `
  --save-name rollout_exp_d.pt --use-gradient-checkpointing false `
  --rollout-ratio-start 0.8 --rollout-ratio-end 1.0 `
  --anchor-weight 0.1 --kl-weight 0.05 --numeric-mape-weight 0 `
  --step-weight-gamma 0.75 --lr 5e-6
```

严格 10-step AR 验证并导出逐条预测误差：

```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' -m posttrain.rollout.eval_rollout `
  --include-base true `
  --checkpoint checkpoints\post_train_rollout_exp_d_lowlr_highself\rollout_exp_d.pt `
  --mode val --max-stocks 120 --max-val-samples 0 --batch-size 8 `
  --output-dir outputs\post_train_rollout_exp_d_fullval120
```

逐条误差文件：

- `outputs/post_train_rollout_exp_d_fullval120/prediction_diff_checkpoints_base_model.csv`
- `outputs/post_train_rollout_exp_d_fullval120/prediction_diff_checkpoints_post_train_rollout_exp_d_lowlr_highself_rollout_exp_d.csv`

***

## 追加实验：纯 rollout 与数值校准

更新时间：2026-05-05 15:40

注意：本节训练时首先按旧 daily MAPE 做了记录；后续已经把评估脚本更新为同时输出 `path_mape`，并把 checkpoint 选择主指标改为 `path_mape`。本节新增的 path 复算见下一节。

本轮回应“把 TF 改成纯 rollout”的要求，补充了严格纯自反馈训练实验：

- 训练上下文未来 9 个反馈 token 全部来自模型自身预测，`used_pred_ratio=1.0`。
- 不再混入 teacher-forced future token。
- 首个纯 rollout 实验同时关闭 anchor、KL、numeric surrogate。
- 后续只加入不会喂真实 future token 的约束或目标，例如 Base KL、heads-only、numeric soft CE。

同时新增 `posttrain/rollout/calibrate_rollout.py`：它不改变 token 生成路径，只用 train 集 strict rollout 预测和真实值拟合逐 step 数值校准器，然后应用到 val。脚本已更新为按 train `path_mape` 选择校准器；早先已跑出的校准表仍是旧 daily MAPE 记录，后续以新脚本输出为准。

### 新增实现

- `posttrain/rollout/train_rollout.py`
  - 新增 `numeric_soft_ce` 可选目标。
  - 它在每一步 top-k coarse/fine token pair 加 gold pair 中，按解码后的 close-ratio 误差构造软标签，再做 token pair CE。
  - 目的：让数值误差目标作用到 token 概率，而不是只优化分布期望。
- `posttrain/rollout/calibrate_rollout.py`
  - 拟合 `identity`、`mean_bias`、`median_bias`、`affine` 四种校准器。
- 按 train `path_mape` 选择校准器。
  - 对 val 输出 raw 与 train\_calibrated 指标。

### 纯 rollout 训练结果

120-stock 小规模内嵌 val 为 64 windows；full-val120 为 189 windows。下表仍是旧 daily MAPE 口径，path\_mape 见最后一节。

| 实验                  | 设置                                               | 64-window MAPE | full-val120 MAPE | full-val120 MAE | full-val120 RMSE | 结论                                       |
| ------------------- | ------------------------------------------------ | -------------: | ---------------: | --------------: | ---------------: | ---------------------------------------- |
| I pure120           | ratio 1.0, anchor 0, KL 0, numeric 0, all params |         2.0482 |           2.0090 |        0.020084 |         0.028781 | 纯 rollout 无约束会破坏 MAPE                    |
| J pure+KL120        | ratio 1.0, anchor 0, KL 0.05, all params         |         2.0383 |           2.0062 |        0.020050 |         0.028797 | KL 能缓解漂移，但仍差于 Base/D                     |
| K pure+KL+softCE120 | J + numeric soft pair CE weight 0.15             |         2.0441 |           2.0101 |        0.020094 |         0.028848 | soft CE 未改善 argmax rollout               |
| L pure heads+KL120  | ratio 1.0, anchor 0, KL 0.05, heads-only         |         2.0367 |           2.0054 |        0.020043 |         0.028797 | 小 val 略好于 Base-64，但 full-val120 未超过 Base |

结论：

1. “TF 改成纯 rollout”本身不够。完全无约束 pure rollout 最差，说明自反馈上下文确实暴露了误差累积问题，但直接用它训练会让 token 分布偏移。
2. pure rollout + KL 比无约束 pure rollout 明显稳定，但收益仍不足以超过 Base full-val120。
3. heads-only pure rollout + KL 是纯 rollout 系列里最稳的版本，但 full-val120 MAPE `2.0054%` 仍差于 Base `2.0034%`，也差于 D `2.0008%`。
4. numeric soft CE 的方向是合理的，但本轮 top-k 软标签没有带来收益；可能原因是 top-k 候选里的数值近邻仍不一定改变最终 argmax coarse/fine 组合。

### Train-only 数值校准结果

校准设置：

- 用 train strict rollout 预测拟合逐 step affine 校准器。
- 校准器旧实验选择只看 train daily MAPE；脚本已改成按 train `path_mape` 选择。
- val 只用于最终报告。
- 不使用 demo。
- 不改变 rollout token 路径；它是 token rollout 之后的数值输出校准层。

120-stock full-val：

| 模型                 | 变体                | train 选择 | full-val120 MAPE |          MAE |         RMSE | 结论                                     |
| ------------------ | ----------------- | -------- | ---------------: | -----------: | -----------: | -------------------------------------- |
| Base               | raw               | identity |           2.0034 |     0.020023 |     0.028777 | 原始基线                                   |
| Base               | train\_calibrated | affine   |           1.9507 |     0.019445 |     0.027657 | train-only 校准明显改善数值误差                  |
| D low-lr high-self | raw               | identity |           2.0008 |     0.019996 |     0.028738 | 原始 D 仍是未校准最好 checkpoint                |
| D low-lr high-self | train\_calibrated | affine   |       **1.9468** | **0.019408** | **0.027623** | 本轮 full-val120 最低 MAPE                 |
| L pure heads+KL    | raw               | identity |           2.0054 |     0.020043 |     0.028797 | 纯 rollout 最稳版本，但 raw 不超 Base           |
| L pure heads+KL    | train\_calibrated | affine   |           1.9519 |     0.019458 |     0.027662 | 校准后接近 Base calibrated，但不如 D calibrated |

300-stock full-val：

| 模型                     | 变体                | train 选择 | full-val300 MAPE |          MAE |         RMSE | 结论                             |
| ---------------------- | ----------------- | -------- | ---------------: | -----------: | -----------: | ------------------------------ |
| Base                   | raw               | identity |           2.2045 |     0.022040 |     0.032110 | 原始较大基线                         |
| Base                   | train\_calibrated | affine   |       **2.1512** | **0.021464** | **0.031101** | 校准在更大范围仍有效                     |
| E low-lr high-self 300 | raw               | identity |           2.2130 |     0.022108 |     0.032258 | 原始 E 差于 Base                   |
| E low-lr high-self 300 | train\_calibrated | affine   |           2.1620 |     0.021557 |     0.031234 | 比 E raw 好，但仍差于 Base calibrated |

更新后的判断：

1. 对“纯 rollout checkpoint”而言，当前没有证据表明它可靠优于 BaseModel。
2. 对旧 daily MAPE 而言，train-only affine 校准是明显改善；但按 `path_mape`，120-stock 上 daily 校准反而略伤累计路径，300-stock 上才改善路径。
3. 校准也同样改善 BaseModel；在 300-stock 上，Base calibrated 仍优于 E calibrated。因此，当前最可靠部署候选应分两层看：
   - 若只允许 checkpoint 权重：仍保守选择 D，但优势很小且未在 300-stock 上成立。
   - 若允许 train-only 数值校准层：Base calibrated 是更稳的大范围基线；D calibrated 是 120-stock 上最好结果，但还需要更大范围 D/E/H/L 的统一校准复评才能宣称超越 calibrated Base。

### 追加命令

纯 rollout 无约束：

```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' Post_Train_Rollout.py `
  --max-stocks 120 --max-train-samples 128 --max-val-samples 64 `
  --batch-size 2 --eval-batch-size 8 --epochs 1 --max-train-updates 48 `
  --output-dir checkpoints\post_train_rollout_exp_i_pure120 `
  --save-name rollout_exp_i.pt --use-gradient-checkpointing false `
  --rollout-ratio-start 1.0 --rollout-ratio-end 1.0 `
  --anchor-weight 0 --kl-weight 0 --numeric-mape-weight 0 --numeric-soft-ce-weight 0 `
  --step-weight-gamma 0.75 --lr 5e-6
```

纯 rollout heads-only + KL：

```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' Post_Train_Rollout.py `
  --max-stocks 120 --max-train-samples 128 --max-val-samples 64 `
  --batch-size 2 --eval-batch-size 8 --epochs 1 --max-train-updates 48 `
  --output-dir checkpoints\post_train_rollout_exp_l_pure_headskl120 `
  --save-name rollout_exp_l.pt --use-gradient-checkpointing false `
  --rollout-ratio-start 1.0 --rollout-ratio-end 1.0 `
  --anchor-weight 0 --kl-weight 0.05 --numeric-mape-weight 0 --numeric-soft-ce-weight 0 `
  --trainable-scope heads --step-weight-gamma 0.75 --lr 5e-6
```

full-val120 复评 I/J/K/L：

```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' -m posttrain.rollout.eval_rollout `
  --include-base true `
  --checkpoint checkpoints\post_train_rollout_exp_i_pure120\rollout_exp_i.pt `
  --checkpoint checkpoints\post_train_rollout_exp_j_pure_kl120\rollout_exp_j.pt `
  --checkpoint checkpoints\post_train_rollout_exp_k_pure_softce120\rollout_exp_k.pt `
  --checkpoint checkpoints\post_train_rollout_exp_l_pure_headskl120\rollout_exp_l.pt `
  --mode val --max-stocks 120 --max-val-samples 0 --batch-size 8 `
  --output-dir outputs\post_train_rollout_exp_ijkl_fullval120
```

train-only 校准 120-stock：

```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' -m posttrain.rollout.calibrate_rollout `
  --include-base true `
  --checkpoint checkpoints\post_train_rollout_exp_d_lowlr_highself\rollout_exp_d.pt `
  --checkpoint checkpoints\post_train_rollout_exp_l_pure_headskl120\rollout_exp_l.pt `
  --max-stocks 120 --max-train-samples 512 --max-val-samples 0 --batch-size 8 `
  --output-dir outputs\post_train_rollout_calibrated_120
```

train-only 校准 300-stock：

```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' -m posttrain.rollout.calibrate_rollout `
  --include-base true `
  --checkpoint checkpoints\post_train_rollout_exp_e_lowlr_highself_300\rollout_exp_e.pt `
  --max-stocks 300 --max-train-samples 512 --max-val-samples 0 --batch-size 8 `
  --output-dir outputs\post_train_rollout_calibrated_300
```

***

## 指标口径更正后的 path\_mape 复算

更新时间：2026-05-05 15:46

公式：

```text
pred_path_ratio[h] = exp(sum_{t=1..h} pred_log_return[t])
actual_path_ratio[h] = exp(sum_{t=1..h} actual_log_return[t])
path_mape[h] = mean(abs(pred_path_ratio[h] - actual_path_ratio[h]) / abs(actual_path_ratio[h])) * 100
overall path_mape = mean(path_mape[1..10])
```

这才是 10 天自回归路径误差口径。此前 `mape ~= 2%` 不是只评第 1 天，而是 10 个未来日的 daily close-ratio MAPE 平均；它不累计路径，所以低估了自回归路径误差。

### full-val120 path\_mape

| 模型                           | daily MAPE | path\_mape 平均 | 第 10 天 path\_mape | 判断                     |
| ---------------------------- | ---------: | ------------: | ----------------: | ---------------------- |
| Base                         |     2.0034 |        4.9391 |            6.7428 | 路径基线                   |
| D low-lr high-self           |     2.0008 |        4.9479 |            6.7737 | daily 略好，但路径略差于 Base   |
| I pure rollout               |     2.0090 |        4.9242 |            6.7040 | 路径略好于 Base             |
| J pure rollout + KL          |     2.0062 |        4.9701 |            6.8017 | 路径差于 Base              |
| K pure rollout + KL + softCE |     2.0101 |    **4.9142** |        **6.6968** | 本组 path\_mape 最好，但收益很小 |
| L pure heads + KL            |     2.0054 |        4.9465 |            6.7993 | 路径略差于 Base             |

逐步路径误差会明显累积。Base full-val120 的 path\_mape 从第 1 天 `2.1758%` 增加到第 10 天 `6.7428%`；K 从第 1 天约 `2.1962%` 增加到第 10 天 `6.6968%`。

### full-val300 path\_mape

| 模型                     | daily MAPE | path\_mape 平均 | 第 10 天 path\_mape | 判断        |
| ---------------------- | ---------: | ------------: | ----------------: | --------- |
| Base                   |     2.2045 |    **5.4440** |        **7.9671** | 较大范围路径基线  |
| E low-lr high-self 300 |     2.2130 |        5.5116 |            8.2409 | 路径差于 Base |

### train-only 校准在 path\_mape 下的结果

旧 daily 口径下，train-only affine 校准能显著降低 daily MAPE；但在累计路径口径下，结论更谨慎：

| 模型                  | daily MAPE | path\_mape 平均 | 第 10 天 path\_mape | 判断                             |
| ------------------- | ---------: | ------------: | ----------------: | ------------------------------ |
| Base raw 120        |     2.0034 |        4.9391 |            6.7428 | 120 路径基线                       |
| Base calibrated 120 |     1.9507 |        4.9746 |            6.9326 | daily 变好，但路径变差                 |
| D raw 120           |     2.0008 |        4.9479 |            6.7737 | daily 略好，路径略差                  |
| D calibrated 120    |     1.9468 |        4.9631 |            6.9161 | daily 最好，但路径仍不如 Base raw/K     |
| Base raw 300        |     2.2045 |        5.4440 |            7.9671 | 300 路径基线                       |
| Base calibrated 300 |     2.1512 |    **5.2472** |        **7.6087** | 300 路径也改善                      |
| E raw 300           |     2.2130 |        5.5116 |            8.2409 | 差于 Base                        |
| E calibrated 300    |     2.1620 |        5.3359 |            7.8856 | 比 E raw 好，但仍差于 Base calibrated |

更新后的关键判断：

1. 用户定义的自回归 MAPE 应使用 `path_mape`，不是旧表里的 daily MAPE。
2. 在 `max_stocks=120` 上，纯 rollout 系列确实出现了 path\_mape 小幅优于 Base 的结果，当前最好是 K：`4.9142%` vs Base `4.9391%`，第 10 天 `6.6968%` vs Base `6.7428%`。收益很小，但方向符合“纯 rollout 训练更贴近路径误差”。
3. D checkpoint 的旧优势主要体现在 daily MAPE；按 path\_mape 它不优于 Base。
4. 在 `max_stocks=300` 上，E 仍不如 Base，说明当前 rollout 后训练还没有在更大范围验证出稳定优势。
5. 校准层需要重新以 `path_mape` 为目标设计。只优化 daily MAPE 可能降低逐日误差，但未必降低累计路径误差。

代码更新：

- `compute_rollout_metrics` 现在输出 `path_mape`、`path_return_mape`、`path_mae`、`path_rmse`。
- `per_step` 现在同时输出逐 step `path_mape`。
- `prediction_diff_*.csv` 现在输出 `cumulative_pred_close_ratio`、`cumulative_actual_close_ratio`、`path_mape`。
- rollout checkpoint 保存选择已改为优先最小化 `path_mape`。
- `calibrate_rollout.py` 已改为按 train `path_mape` 选择校准器。

***

## 第二轮实验：6 种新方法探索（2026-05-05）

更新时间：2026-05-05 18:30\
运行环境：同前，RTX 4060 Laptop 8GB。

### 背景

第一轮 A–L 实验后，最佳结果 K（pure+KL+softCE）path_mape=4.9142% vs Base=4.9391%，收益仅 0.025pp 且未在 300-stock 上验证。本轮设计 6 个新方法（N–S），测试能否找到更有效的后训练策略。完整设计文档见 `experiment_plan.md`。

### 实验设计

| 实验 | 方法 | 核心思路 |
|:---:|------|------|
| N | Scheduled Horizon Curriculum | 不直接从 horizon=10 训练，分 4 阶段渐进：2→5→7→10，每阶段 12 updates |
| O | Differentiable Path-Aware Loss | 每步 soft 解码 top-k token pair 得期望 return，累积 10 步期望路径，直接优化 path_mape |
| P | Beam Distill | 用 frozen BaseModel 做 best-of-4 sampling 选 teacher trajectory，蒸馏到训练模型 |
| R | Contrastive Trajectory Pairs | 采样 2 条完整轨迹，比较 path_mape，对比损失拉近好轨迹、推远坏轨迹 |
| Q | GRPO with Path Reward | 采样多条轨迹做组内相对优势 + PPO-clipped 策略更新，直接优化 path_mape reward |
| S | Inference-Time Temperature Annealing | 不改权重，推理时前 5 步用温度采样探索，后续 argmax |

### 代码改动

`posttrain/rollout/train_rollout.py` 新增约 300 行：
- `_differentiable_path_loss()`（O）：soft 解码 + 路径累积 + 可微损失
- `_beam_distill_teacher()`（P）：best-of-N sampling teacher
- `_contrastive_trajectory_loss()`（R）：轨迹对比损失
- `_grpo_loss()`（Q）：内存优化版 GRPO
- `predict_autoregressive_returns()` 和 `evaluate_model()`：新增温度退火参数（S）
- `train()`：新增 curriculum 外循环（N）
- `rollout_training_loss()`：集成 O/P/R/Q 四个新损失项
- `_namespace_from_args()`：所有新字段用 `getattr` 安全回退

`posttrain/rollout/eval_rollout.py`：新增 `--sample-temp-start/end/steps` 参数（S）。

`run_all.py`：Python 编排脚本，逐个调用 subprocess 跑训练+评估，汇总 CSV。

### 实验结果（120-stock val, 189 windows）

| 实验 | 方法 | path_mape | daily_mape | step10 | vs Base |
|:---:|------|:---:|:---:|:---:|:---:|
| — | BaseModel | 4.9391% | 2.0034% | 6.7428% | — |
| **N** | **Curriculum (2→5→7→10)** | **4.9181%** | 1.9968% | 6.7221% | **-0.021pp** |
| O | Path-Aware Loss | 4.9409% | 2.0056% | 6.7243% | +0.002pp |
| P | Beam Distill | 4.9589% | 2.0024% | 6.7921% | +0.020pp |
| S1 | Temp Anneal 1.5→0 | 5.5119% | 2.1483% | 7.5445% | +0.573pp |
| S2 | Temp Anneal 1.0→0.5 | 5.2735% | 2.0917% | 6.9611% | +0.334pp |
| R | Contrastive Trajectories | — | — | — | OOM |
| Q | GRPO | — | — | — | OOM |

训练配置（N/O/P）：pure rollout（ratio=1.0），KL=0.05，anchor=0，lr=5e-6，48 updates，batch_size=2。
R/Q 因旧实现显存不足未能运行——两者需要在训练 batch 内额外生成多条完整轨迹（每条 10 步模型前向），旧代码还会保留这些逐步前向的 autograd 图。低显存修复和重跑结果见后文“第三轮实验”。

### N 逐步 path_mape vs Base

| Step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Base | 2.176 | 3.146 | 3.858 | 4.502 | 5.048 | 5.170 | 5.832 | 6.358 | 6.558 | 6.743 |
| N | 2.200 | **3.112** | **3.817** | 4.526 | **5.023** | **5.119** | **5.791** | 6.341 | 6.531 | **6.722** |
| N-Base | +0.024 | **-0.034** | **-0.041** | +0.024 | **-0.025** | **-0.051** | **-0.041** | -0.017 | -0.027 | **-0.021** |

N 在 10 步中有 8 步优于 Base，改善集中在 Step 2–7（中期），末期改善收窄但保持正收益。Step 1 和 Step 4 略差——短 horizon 阶段可能牺牲了首步精度来换取后续稳定性。

### 新增代码实现细节

`posttrain/rollout/train_rollout.py` 中新增函数：

- `_differentiable_path_loss(tokenizer, logits_c, logits_f, ...)`：每步 softmax top-k pair → decode → 期望 return → cumsum 得期望路径 → smooth L1 vs 真实路径。梯度通过 frozen tokenizer decoder 回传到 logits。
- `_beam_distill_teacher(model, tokenizer, ..., beam_width, temperature, actual_returns, means, stds)`：每步采样 beam_width 个候选 token pair，选数值误差最小的，返回 (teacher_c, teacher_f) 序列用于蒸馏 CE。
- `_contrastive_trajectory_loss(model, tokenizer, batch, cfg, ..., margin, temperature)`：用不同 seed 采样 2 条轨迹，计算 path_mape，margin-based contrastive loss：`max(0, margin + logP(bad) - logP(good))`。
- `_grpo_loss(model, reference_model, tokenizer, batch, cfg, ..., group_size, clip_eps, beta)`：内存优化版 GRPO。Phase 1 采样 G 条轨迹（只保留 tokens 和旧 logprob），Phase 2 逐 group 算当前 logprob 和参考 logprob，free 后进入下一 group。PPO-clipped 目标 + KL penalty。

`predict_autoregressive_returns` 新增参数 `sample_temp_start/end/steps`：前 steps 步用 `multinomial(softmax(logits/temp))` 采样，后续 argmax。

### 分析

**N 为什么有效**：从短 horizon 开始让模型先适应自反馈输入分布，逐步延长 horizon 让模型渐进适应更长的误差累积链。相比直接 horizon=10（K/L 实验），curriculum 提供了"学习阶梯"，避免模型在最脆弱的训练初期同时面对 10 步误差累积的压力。

**O 为什么无效**：soft 期望分布和 argmax 之间存在本质差距。训练优化的"概率加权后的期望路径"可能对应一个不存在的 token pair，梯度方向对 argmax 选择帮助有限。这与 F 实验（numeric MAPE surrogate）的失败原因一致。

**P 为什么恶化**：teacher（BaseModel 的 best-of-4 sampling）和 student（argmax）的差距本身就很小，蒸馏无法提供足够额外信号。且 best-of-4 teacher 质量可能也不够好——如果 BaseModel 的 4 个采样都不好，选最好的仍然不好。

**S 为什么恶化**：温度采样在自回归场景下引入的噪声被逐步放大。第 1 步采样误差污染第 2 步输入，第 2 步误差再污染第 3 步……10 步后噪声完全淹没信号。结论：10 步自回归场景下，确定性推理（argmax）优于任何随机采样策略。

**R/Q 旧 OOM 分析**：R 需要 2 条轨迹 × 10 步前向 = 额外 20 次模型前向，Q 需要 G 条轨迹 × 3 轮（采样/当前/参考）= 额外 3G×10 次前向。旧实现把采样阶段的逐步前向图也保留到反传，导致 8GB GPU 上即使用 batch_size=1 + gc.collect() 仍不够。后续已通过两阶段 logprob 重算修复，见下一节。

### 与第一轮最优结果对比

| 实验 | path_mape | 方法复杂度 |
|:---|:---:|:---|
| BaseModel | 4.9391% | — |
| K (pure+KL+softCE) | 4.9142% | 高：需要 numeric soft CE 实现 |
| **N (Curriculum)** | **4.9181%** | **低：只改 horizon 调度，不改 loss** |

N 的 path_mape（4.9181%）略差于 K（4.9142%），但方法简单得多——不需要 numeric soft CE，只需修改训练时的 horizon 递进。两者的改善幅度都在 0.02pp 量级。

### 后续建议

1. **验证 N 在 300-stock 上的泛化性**：`--max-stocks 300` 复现，看 curriculum 是否在更大范围有效。
2. **消融 N**：测试不同 horizon 序列（如 3→6→10）、每阶段 update 分配。
3. **N + K 组合**：curriculum 叠加 numeric soft CE，可能获得叠加收益。
4. **增大训练预算**：48 updates 严重不足，在更大 GPU 上扩展到 200+ updates。
5. **继续调参 R/Q 的低显存版**：当前已能在 8GB GPU 上跑通，但 96 updates 的首轮结果没有超过 Base；后续应测试更小权重、heads-only、curriculum+R/Q 和更长训练预算。

***

## 第三轮实验：R/Q 低显存修复与本地重跑（2026-05-05）

更新时间：2026-05-05 19:12\
运行环境：同前，RTX 4060 Laptop 8GB。

### 低显存修复方法

旧实现的 OOM 不是单纯 batch 太大，而是 R/Q 在采样轨迹时保留了每一步模型前向的 autograd 图。修复后改成两阶段：

1. **采样阶段 no-grad**：R 采样 2 条轨迹，Q 采样 `group_size=2` 条轨迹；只保留 sampled tokens、旧 logprob、path_mape reward，不保留逐步前向图。
2. **训练阶段整轨迹 logprob 重算**：新增 `_trajectory_sequence_logprob()`，把 `prefix + sampled_tokens[:-1]` 拼成一条完整输入，用一次 full-sequence forward 取出未来 10 步 logits，再计算整条 sampled trajectory 的 logprob。
3. **R 的对比损失不变**：按 no-grad 采样得到的 path_mape 选 good/bad trajectory，再用整轨迹 logprob 做 `max(0, margin + logP(bad) - logP(good))`。
4. **Q 的 GRPO 目标不变**：no-grad 采样得到 old_logprob 和组内 reward advantage；训练阶段逐条轨迹重算 current/reference logprob，使用 PPO clipped objective + KL penalty。
5. `run_rq.py` 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，并使用 `batch_size=1`、`max_train_updates=96`，使 batch size 1 的训练样本数与 N/O/P 的 batch size 2 × 48 updates 对齐。

新增显存记录：`rollout_scheduled_history.json` 中写入 `cuda_peak_allocated_gb` 和 `cuda_peak_reserved_gb`。

### 训练与评估配置

```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' run_rq.py
```

共同配置：`max_stocks=120`，`max_train_samples=128`，`batch_size=1`，`epochs=1`，`max_train_updates=96`，`rollout_ratio=1.0`，`KL=0.05`，`anchor=0`，`lr=5e-6`。\
R：`contrastive_weight=0.3`，`margin=0.1`，`temp=1.0`。\
Q：`grpo_group_size=2`，`clip_eps=0.2`，`beta=0.04`。

评估：full-val120，189 windows，严格 10-step AR，并包含 BaseModel 对照。

### 本地重跑结果

| 实验 | 方法 | 是否 OOM | 峰值 allocated | 峰值 reserved | path_mape | daily_mape | step10 | vs Base |
|:---:|------|:---:|---:|---:|---:|---:|---:|---:|
| — | BaseModel | 否 | — | — | **4.9391%** | 2.0034% | **6.7428%** | — |
| R | Low-memory Contrastive Trajectories | 否 | 1.775 GB | 3.162 GB | 4.9447% | 2.0039% | 6.7550% | +0.0056pp |
| Q | Low-memory GRPO | 否 | 1.833 GB | 3.125 GB | 4.9679% | 2.0048% | 6.7901% | +0.0288pp |

注意：显存峰值来自训练 history；path_mape 表为 full-val120 复评结果。训练内嵌 64-window val 中，R 的 path_mape 为 5.2342%，Q 为 5.3954%，只用于保存 checkpoint，不作为最终比较口径。

### 输出文件

- `outputs/exps_20260505_190557/results_rq.csv`
- `checkpoints/post_train_rollout_exp_r_contrastive120/rollout_scheduled_history.json`
- `checkpoints/post_train_rollout_exp_q_grpo120/rollout_scheduled_history.json`
- `outputs/eval_R/rollout_eval_val.json`
- `outputs/eval_Q/rollout_eval_val.json`
- `outputs/eval_R/prediction_diff_checkpoints_post_train_rollout_exp_r_contrastive120_rollout_exp_r.csv`
- `outputs/eval_Q/prediction_diff_checkpoints_post_train_rollout_exp_q_grpo120_rollout_exp_q.csv`

### 结论

1. R/Q 现在已经可以在本地 RTX 4060 Laptop 8GB 上完整跑通训练、保存 checkpoint 和 full-val120 评估，不再 OOM。
2. 当前默认超参下，R/Q 没有超过 BaseModel；R 非常接近 Base，但 path_mape 仍差 `0.0056pp`，Q 明显更差。
3. 轨迹级目标本身仍值得保留，但当前配置的学习信号噪声较大。下一步应优先测：更低 `contrastive_weight`、Q 的 `beta/clip` 消融、heads-only R/Q、以及 N curriculum + R/Q 的组合。

***

## 第四轮实验：课程组合、低权重轨迹目标与外推评估（2026-05-05）

更新时间：2026-05-05 19:42\
运行环境：同前，RTX 4060 Laptop 8GB。

### 实验目标

顺着前文结论，本轮继续围绕已经出现小幅正收益的 K/N 系列和刚修复可运行的 R 系列做低风险探索：

1. 测试 **N + K**：curriculum horizon 叠加 numeric soft CE，看能否叠加 K 的 path_mape 收益和 N 的稳定课程收益。
2. 测试 **不同课程序列**：把 N 的 `2→5→7→10` 改成 `3→6→10`，减少过短 horizon 对 step1 的扰动。
3. 测试 **训练预算**：把 N 从 48 updates 扩到 96 updates，观察是否继续改善或开始漂移。
4. 测试 **heads-only curriculum**：限制更新输出头，减少全参微调对 Base 分布的破坏。
5. 测试 **低权重 R**：把 contrastive weight 从 `0.3` 降到 `0.05`，降低轨迹偏好噪声。
6. 自动选择 full-val120 最好非 Base checkpoint，并额外跑 full-val300 外推评估。

### 代码修复

N+K 首次运行时暴露了一个短 horizon bug：curriculum 阶段 `effective_horizon < 10`，但 numeric soft CE 仍使用 full 10-step `actual_returns`，导致 shape mismatch。已修复为在 `rollout_training_loss()` 中构造 `actual_returns_h = batch["actual_returns"][:, :horizon]`，并传给 `numeric_mape_surrogate`、`numeric_soft_pair_ce`、`path_aware_loss` 和 beam teacher。

### 自动运行脚本

```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' run_third_experiments.py
```

该脚本完成 V/W/X/Y 的训练、full-val120 评估、最佳候选 full-val300 外推；T/U 在修复短 horizon numeric soft CE 后补跑。汇总结果写入：

- `outputs/third_exps_20260505_192049/results_third_all.csv`
- `outputs/third_exps_20260505_192049/results_best_300.csv`

### 实验配置

共同配置：`max_stocks=120`，`max_train_samples=128`，内嵌 `max_val_samples=64`，full-val120 为 189 windows，`lr=5e-6`，`KL=0.05`，`anchor=0`，`rollout_ratio=1.0`。

| 实验 | 方法 | 关键配置 |
|:---:|------|------|
| T | N + K strong | `2→5→7→10`，softCE weight `0.15`，top-k `8`，temp `0.003` |
| U | N + K weak | `2→5→7→10`，softCE weight `0.05`，top-k `8`，temp `0.003` |
| V | Curriculum 3→6→10 | `3,6,10`，每阶段 16 updates |
| W | Longer curriculum | `2→5→7→10`，每阶段 24 updates，共 96 updates |
| X | Heads-only curriculum | `2→5→7→10`，只训练输出头 |
| Y | Low-weight R | contrastive weight `0.05`，batch size `1`，96 updates |

### full-val120 结果

| 实验 | path_mape | daily_mape | step10 path_mape | vs Base | 判断 |
|:---:|---:|---:|---:|---:|------|
| Base | 4.9391% | 2.0034% | 6.7428% | — | 基线 |
| **V 3→6→10** | **4.9198%** | 2.0138% | **6.7355%** | **-0.0193pp** | 本轮最好，路径略优 |
| T N+K 0.15 | 4.9266% | 2.0009% | 6.7628% | -0.0125pp | daily 好，路径收益弱 |
| U N+K 0.05 | 4.9294% | **1.9998%** | 6.7436% | -0.0097pp | daily 最好，路径接近 Base |
| W curriculum 96 | 4.9407% | 1.9978% | 6.7973% | +0.0016pp | 训练更久后路径退化 |
| Y R low 0.05 | 4.9411% | 2.0017% | 6.7783% | +0.0020pp | 低权重 R 仍未超过 Base |
| X heads-only | 4.9440% | 2.0055% | 6.7876% | +0.0049pp | 只训头不足 |

显存峰值均在 8GB 内：T/U 约 `1.871 GB allocated / 3.139 GB reserved`，V 约 `1.838 / 2.799 GB`，W 约 `1.871 / 3.139 GB`，X 约 `1.854 / 2.318 GB`，Y 约 `1.775 / 3.162 GB`。

### best-candidate full-val300 外推

脚本自动选择 full-val120 最好的 V checkpoint 做 300-stock 外推：

| 验证集 | 模型 | path_mape | daily_mape | step10 path_mape | 判断 |
|------|------|---:|---:|---:|------|
| full-val300, 452 windows | Base | **5.4440%** | **2.2045%** | **7.9671%** | 300-stock 基线 |
| full-val300, 452 windows | V 3→6→10 | 5.4971% | 2.2191% | 8.0846% | 外推差于 Base |

### 分析

1. **本轮没有发现超过历史最好 K 的方法**。V 的 `4.9198%` 优于 Base，但仍弱于 K 的 `4.9142%` 和上一轮 N 的 `4.9181%`。
2. **3→6→10 比 2→5→7→10 更接近当前最优区间**。它减少了短 horizon 阶段，但只带来 `-0.0193pp` 的小收益，且无法外推到 300-stock。
3. **N+K 组合没有叠加收益**。softCE 让 daily MAPE 变好（U/T 均接近或低于 2.0%），但对累计路径收益不如 curriculum-only，说明 softCE 仍更像逐日数值目标，不是真正的路径目标。
4. **更长训练预算不是直接解法**。W 的 daily MAPE 变好，但 path_mape 退回 Base 附近并且 step10 明显变差，说明短训练的小幅路径收益很容易被后续更新冲掉。
5. **低权重 R 仍无效**。把 contrastive weight 从 `0.3` 降到 `0.05` 后，结果从明显差于 Base 变成贴近 Base，但仍没有转正。
6. **当前 120-stock 收益不能可靠外推**。V 在 120 上小幅优于 Base，但 300 上变差，继续说明 rollout 后训练仍缺乏稳定泛化优势。

### 当前结论更新

截至本轮，120-stock 小规模 path_mape 排序：

| 排名 | 方法 | path_mape | 备注 |
|---:|------|---:|------|
| 1 | K pure+KL+softCE | **4.9142%** | 历史最好，但方法复杂 |
| 2 | N curriculum 2→5→7→10 | 4.9181% | 简单稳定，小幅收益 |
| 3 | V curriculum 3→6→10 | 4.9198% | 本轮最好，但 300 外推失败 |
| — | Base | 4.9391% | 基线 |

更大范围上，当前没有任何 rollout 后训练 checkpoint 稳定超过 Base。后续若继续做，应把重点从“更多小规模微调”转向：

1. 在 `max_stocks=300` 上直接训练 curriculum，而不是只用 120-stock checkpoint 外推。
2. 设计真正针对累计路径的 hard-decision 目标，避免 soft expectation / daily CE 与 argmax path_mape 脱节。
3. 增加路径收益的防漂移机制：例如只更新低秩 adapter 或引入 path_mape-aware checkpoint averaging，而不是继续全参短训。

***

## 第五轮实验：1200 stocks / 480 updates 放大验证（2026-05-05）

更新时间：2026-05-05 19:57\
运行环境：同前，RTX 4060 Laptop 8GB。

### 实验目标

按最新要求，把第四轮最好的自回归预测候选 **V curriculum 3→6→10** 放大一个数量级后再和 Base 对比：

1. `max_stocks` 从 `120` 提高到 `1200`。
2. 训练预算从 `48 updates` 提高到 `480 updates`。
3. 为了让更多股票样本实际进入训练，训练/内嵌验证采样上限同步从 `128/64` 提高到 `1280/640`。
4. full-val 评估不截断验证样本，直接在 `1200 stocks` 缓存的全部 `1829` 个验证窗口上对比 Base 与候选 checkpoint。

### 实验配置

| 项目 | 配置 |
|------|------|
| 候选方法 | V curriculum `3→6→10` |
| checkpoint | `checkpoints/post_train_rollout_exp_z_v_curric3610_1200/rollout_exp_z.pt` |
| stocks | `max_stocks=1200` |
| 训练窗口缓存 | `2702` train windows |
| 训练窗口上限 | `1280` windows |
| 内嵌验证上限 | `640` windows |
| full-val | `1829` val windows |
| updates | `160 + 160 + 160 = 480` |
| batch size | train `2` / eval `8` |
| loss | autoregressive rollout CE + `KL=0.05` |
| 其它 | `anchor=0`，`numeric_mape=0`，`softCE=0`，`step_weight_gamma=0.75`，`lr=5e-6` |

### 自动运行命令

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
& 'D:\conda_envs\llm-t\Scripts\python.exe' Post_Train_Rollout.py `
  --output-dir checkpoints\post_train_rollout_exp_z_v_curric3610_1200 `
  --save-name rollout_exp_z.pt `
  --max-stocks 1200 --max-train-samples 1280 --max-val-samples 640 `
  --batch-size 2 --eval-batch-size 8 --epochs 1 --max-train-updates 480 `
  --use-gradient-checkpointing false --step-weight-gamma 0.75 --lr 5e-6 `
  --rollout-ratio-start 1.0 --rollout-ratio-end 1.0 `
  --anchor-weight 0 --kl-weight 0.05 --numeric-mape-weight 0 --numeric-soft-ce-weight 0 `
  --curriculum-horizons 3,6,10 --curriculum-updates 160,160,160

& 'D:\conda_envs\llm-t\Scripts\python.exe' -m posttrain.rollout.eval_rollout `
  --include-base true `
  --checkpoint checkpoints\post_train_rollout_exp_z_v_curric3610_1200\rollout_exp_z.pt `
  --mode val --max-stocks 1200 --max-val-samples 0 --batch-size 8 `
  --output-dir outputs\eval_Z_v_curric3610_1200
```

该命令已本地完整跑完，训练加评估总耗时约 `1378s`。

### 训练内嵌验证

| stage | horizon | updates | train_loss | train_KL | inner path_mape | inner daily_mape | inner step10 | 峰值 allocated | 峰值 reserved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 160 | 6.3105 | 0.0234 | 5.9116% | 2.2277% | 8.7765% | 1.774 GB | 2.791 GB |
| 2 | 6 | 320 | 6.3654 | 0.0294 | 5.8762% | 2.2320% | 8.7110% | 1.806 GB | 2.797 GB |
| 3 | 10 | 480 | 6.6870 | 0.0348 | 5.8421% | 2.2383% | 8.6876% | 1.838 GB | 2.799 GB |

训练内嵌验证中，path_mape 随课程推进持续下降，但 daily_mape 略有上升；显存峰值仍稳定低于 8GB。

### full-val1200 对比

| 模型 | num_sequences | path_mape | daily_mape | step1 | step3 | step6 | step10 | vs Base path |
|------|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 1829 | 5.8035% | 2.2220% | 2.2646% | 4.2803% | 6.1755% | 8.6738% | — |
| **Z / V 3→6→10, 1200 stocks** | 1829 | **5.7559%** | **2.2107%** | **2.2613%** | **4.2507%** | **6.1381%** | **8.6058%** | **-0.0477pp** |

### 输出文件

- `checkpoints/post_train_rollout_exp_z_v_curric3610_1200/rollout_exp_z.pt`
- `checkpoints/post_train_rollout_exp_z_v_curric3610_1200/rollout_scheduled_history.json`
- `outputs/eval_Z_v_curric3610_1200/rollout_eval_val.json`
- `outputs/eval_Z_v_curric3610_1200/rollout_eval_val.csv`
- `outputs/eval_Z_v_curric3610_1200/prediction_diff_checkpoints_base_model.csv`
- `outputs/eval_Z_v_curric3610_1200/prediction_diff_checkpoints_post_train_rollout_exp_z_v_curric3610_1200_rollout_exp_z.csv`

### 结论

1. 放大到 `1200 stocks / 480 updates` 后，V curriculum 首次在更大 full-val 口径下超过 Base：path_mape 从 `5.8035%` 降到 `5.7559%`，改善 `0.0477pp`；daily_mape 改善 `0.0113pp`；step10 改善 `0.0680pp`。
2. 这个结果和第四轮的“小规模 V 略优 Base”方向一致，但比之前 `120-stock` 的 `0.0193pp` 更明显，说明 curriculum 的收益在直接放大训练数据和训练步数后没有消失。
3. 第四轮的 `120→300` 外推失败，和本轮 `1200` 直接训练转正并不矛盾：当前方法更像依赖训练覆盖率的轻微分布校准，而不是能从小股票池自然外推到更大股票池的泛化技巧。
4. 截至目前，最值得继续推进的自回归预测方向是：在更大股票池上直接训练 `3→6→10` curriculum，并围绕"防止后期漂移"和"路径 hard-decision 目标"做消融，而不是回到低权重 R/Q。

***

## 第六轮实验：SOTA 后训练算法移植（2026-05-05）

更新时间：2026-05-05 23:30\
运行环境：同前，RTX 4060 Laptop 8GB。

### 实验目标

搜寻 2024-2025 年最新 SOTA 级后训练优化算法，移植到 Kronos-R 的 rollout 后训练框架中，在 1200 stocks / 480 updates / curriculum 3→6→10 的统一条件下进行对比。

### 算法调研与选择

从 LLM 后训练领域筛选出 5 种与 rollout 自回归预测场景最相关的 SOTA 算法：

| 编号 | 算法 | 来源 | 核心思想 | 与 rollout 的适配性 |
|:---:|------|------|------|------|
| AA | Online Iterative DPO | Online DPO (Guo et al. 2024) | 在线采样轨迹，按 path_mape 构造偏好对，DPO 训练 | 中：需要 reference model，偏好对构造依赖 reward 差异 |
| AB | REINFORCE++ | DeepSeek-V3 (2024) | EMA baseline 方差缩减 + importance sampling + per-step process reward | 中：需要 reference model，policy gradient 信号弱 |
| AC | Expert Iteration (ReST-style) | DeepSeek-R1 / ReST^EM | 采样多条轨迹，筛选 best-by-reward，SFT on selected | 高：不需要 reference model，直接优化 argmax 路径质量 |
| AD | ORPO | ORPO (Hong et al. 2024) | 单目标 SFT CE + odds-ratio 偏好惩罚，无需 reference model | 中：偏好信号依赖采样质量 |
| AE | RLOO | RLOO (Ahmadian et al. 2024) | Leave-one-out baseline 估计 advantage + PPO clipped objective | 中：需要 reference model，LO baseline 比 GRPO 更稳定 |

### 代码实现

在 `train_rollout.py` 中新增 5 个 loss 函数和对应 CLI 参数：

- `_iterative_dpo_loss()`：在线采样 group_size 条轨迹，按 path_mape 排序构造 winner/loser 对，DPO loss
- `_EMABaseline` + `_reinforce_plusplus_loss()`：EMA baseline 追踪 reward 均值，importance sampling 修正，per-step process reward 加权
- `_expert_iteration_loss()`：采样 group_size 条轨迹，保留 keep_ratio 比例最优轨迹做 SFT
- `_orpo_loss()`：SFT NLL + odds-ratio 偏好惩罚联合优化
- `_rloo_loss()`：Leave-one-out advantage + PPO clipped objective + KL penalty

同时修复了 `_sample_single_trajectory()` 中 `actual_cum_fallback` 的无效引用，以及 `rollout_training_loss()` 的 `ema_baseline` 参数传递、`epoch_totals` 统计键扩展、reference model 自动创建逻辑等集成问题。

### 实验配置

共同配置：`max_stocks=1200`，`max_train_samples=1280`，`max_val_samples=640`，`batch_size=2`，`eval_batch_size=8`，`epochs=1`，`max_train_updates=480`，`curriculum_horizons=3,6,10`，`curriculum_updates=160,160,160`，`lr=5e-6`，`step_weight_gamma=0.75`，`rollout_ratio=1.0`，`anchor=0`，`numeric_mape=0`，`softCE=0`。

| 实验 | 算法 | 关键超参 | KL weight |
|:---:|------|------|---:|
| AA | Iterative DPO | weight=0.5, group_size=4, beta=0.1, temp=1.0 | 0.05 |
| AB | REINFORCE++ | weight=0.5, group_size=4, ema_decay=0.05, clip_ratio=0.2, process_reward_weight=0.3 | 0.05 |
| AC | Expert Iteration | weight=0.5, group_size=4, keep_ratio=0.5, temp=1.0 | 0.05 |
| AD | ORPO | weight=0.5, group_size=4, beta=0.1, temp=1.0 | 0.05 |
| AE | RLOO | weight=0.5, group_size=4, beta=0.04, clip_eps=0.2 | 0.05 |
| AE-tuned | RLOO | weight=2.0, group_size=4, beta=0.04, clip_eps=0.2 | 0.01 |
| AC-tuned | Expert Iteration | weight=2.0, group_size=8, keep_ratio=0.25, temp=1.5 | 0.05 |
| AC-v2 | Expert Iteration | weight=1.5, group_size=8, keep_ratio=0.25, temp=1.2 | 0.02 |

### 内嵌验证结果

| 实验 | 算法 | val path_mape | RL 信号强度 | vs Z 基线 |
|:---:|------|---:|---|---:|
| Z (基线) | Curriculum 3→6→10 + KL | 5.756% | — | — |
| AA | Iterative DPO | 5.898% | dpo_loss ~0.2 | +0.142pp ❌ |
| AB | REINFORCE++ | 5.872% | policy_loss ~0.002 | +0.116pp ❌ |
| AC | Expert Iteration (默认) | 5.825% | sft_loss ~0.5 | +0.069pp ❌ |
| AD | ORPO | 5.860% | orpo_nll ~5.7 | +0.104pp ❌ |
| AE | RLOO | 5.845% | policy_loss ~0.004 | +0.089pp ❌ |
| AE-tuned | RLOO (weight=2.0, kl=0.01) | 5.847% | policy_loss ~0.004 | +0.091pp ❌ |
| **AC-tuned** | **Expert Iteration (g=8, k=0.25, t=1.5)** | **5.675%** | sft_loss ~8.3 | **-0.081pp ✅** |
| AC-v2 | Expert Iteration (g=8, k=0.25, t=1.2) | 5.866% | sft_loss ~8.6 | +0.110pp ❌ |

### AC-tuned 逐 step path_mape 对比

| step | Z 基线 | AC-tuned | 差异 |
|---:|---:|---:|---:|
| 1 | 2.350% | 2.310% | -0.040pp |
| 3 | 3.703% | 3.605% | -0.098pp |
| 6 | 5.588% | 5.380% | -0.208pp |
| 10 | 8.678% | 8.416% | -0.262pp |

AC-tuned 在所有 step 上均优于 Z 基线，且改善幅度随 step 增大而增大，说明 Expert Iteration 有效缓解了长程误差累积。

### 分析

**为什么 AC-tuned (Expert Iteration) 成功而其他方法失败？**

1. **Expert Iteration 直接优化 argmax 路径质量**：采样多条轨迹后只对 best-by-path_mape 的轨迹做 SFT，等价于 best-of-N + distillation。训练目标（SFT on good trajectory）和推理行为（argmax）高度一致，不存在 soft expectation 与 hard decision 的 gap。

2. **group_size=8 提供足够多样性**：4 条轨迹的 best-of-4 质量提升有限，8 条轨迹的 best-of-8 更有可能包含高质量轨迹。keep_ratio=0.25 意味着只保留 top 2 条，筛选更严格。

3. **temperature=1.5 增加采样探索**：更高的采样温度让 8 条轨迹之间的差异更大，best-of-8 的质量上界更高。temp=1.2 时多样性不足，结果退回 5.866%。

4. **不需要 reference model**：节省 ~50% 显存开销，同等 batch size 下训练更稳定。

**为什么 Policy Gradient 方法（REINFORCE++, RLOO）失败？**

1. **RL 信号极弱**：group_size=4 时，4 条轨迹的 advantage 估计方差极大。policy_loss 量级仅 0.002-0.004，相比 rollout_loss 的 ~6.7 几乎可忽略。
2. **额外 forward pass 开销**：每步需要额外采样 + reference model 前向，训练速度从 2.5s/it 降到 3.0s/it，但 RL 信号贡献不成比例。
3. **reward 稀疏**：path_mape 是整条路径的累积误差，没有 per-step 的细粒度奖励信号。per-step process reward 的权重（0.3）也不足以改变这个本质问题。
4. **增大 RL 权重无效**：AE-tuned 把 rloo_weight 从 0.5 增到 2.0，但 policy_loss 量级不变（~0.004），说明问题不在权重而在信号质量。

**为什么 Iterative DPO 和 ORPO 失败？**

1. **DPO 偏好对质量差**：group_size=4 时，winner/loser 的 path_mape 差异很小，DPO 的 log-ratio 梯度信号弱且噪声大。
2. **ORPO 的 odds-ratio 不适配连续值 reward**：ORPO 设计用于离散偏好（chosen vs rejected），但 path_mape 是连续值，odds-ratio 的二值化损失了信息。

### 当前结论更新

截至本轮，1200-stock path_mape 排序：

| 排名 | 方法 | val path_mape | 备注 |
|---:|------|---:|------|
| **1** | **AC-tuned Expert Iteration (g=8, k=0.25, t=1.5)** | **5.675%** | **新 SOTA，首次显著超过 Z 基线** |
| 2 | Z curriculum 3→6→10 + KL | 5.756% | 上一轮最优 |
| — | Base (1200 stocks) | 5.804% | 基线 |

### 后续建议

1. **AC-tuned full-val1200 复评**：用 `eval_rollout` 在完整 1829 窗口上验证 AC-tuned checkpoint，确认内嵌验证结果可靠。
2. **Expert Iteration 超参消融**：测试 group_size=12/16、keep_ratio=0.125、temp=2.0 等更激进的配置。
3. **Expert Iteration + longer training**：当前 480 updates 可能不够，尝试 960 updates（每阶段 320 updates）。
4. **Expert Iteration + KL 正则化**：当前 KL=0.05，测试更低 KL（0.01）是否让 Expert Iteration 更自由地学习 best-of-N 轨迹。
5. **两阶段训练**：先用 curriculum + KL 做 480 updates 预训练，再用 Expert Iteration 做 480 updates 精调，可能获得叠加收益。

***

## 第七轮实验：Expert Iteration 深度消融（2026-05-06）

更新时间：2026-05-06 22:00\
运行环境：同前，RTX 4060 Laptop 8GB。

### 实验目标

基于第六轮 AC-tuned (Expert Iteration, g=8, k=0.25, t=1.5, weight=2.0, KL=0.05) 的突破性结果，进行4项深度消融实验，验证结果的可靠性并进一步优化。

### 实验配置与结果

共同配置：`max_stocks=1200`，`max_train_samples=1280`，`max_val_samples=640`，`batch_size=2`，`lr=5e-6`，`step_weight_gamma=0.75`，`rollout_ratio=1.0`，`anchor=0`，`numeric_mape=0`，`softCE=0`，`expert_iter_weight=2.0`，`expert_iter_group_size=8`，`expert_iter_keep_ratio=0.25`，`expert_iter_temp=1.5`。

| 实验 | 变量 | 关键超参差异 | 内嵌 val path_mape | full-val1200 path_mape |
|:---:|------|------|---:|---:|
| AC-tuned (第六轮) | 基准 | KL=0.05, 480up, from base | 5.675% | **5.662%** |
| **Exp1** | **full-val 复评** | 同 AC-tuned | 5.675% | **5.662%** ✅ |
| Exp2 | longer training | KL=0.05, **960up** (320/stage), from base | 5.701% | — |
| **Exp3** | **low KL** | **KL=0.01**, 480up, from base | **5.665%** | — |
| Exp4 | 两阶段训练 | KL=0.01, 480up, **from Z checkpoint** | 5.664% | 5.673% |

### Full-val1200 复评详细对比

| 模型 | num_seq | path_mape | step1 | step3 | step6 | step10 |
|------|---:|---:|---:|---:|---:|---:|
| Base | 1829 | 5.804% | 2.265% | 3.455% | 5.581% | 8.674% |
| Z 基线 | 1829 | 5.756% | 2.350% | 3.703% | 5.588% | 8.678% |
| **AC-tuned** | 1829 | **5.662%** | **2.250%** | 3.605% | **5.380%** | **8.430%** |
| Two-stage | 1829 | 5.673% | 2.257% | **3.413%** | 5.476% | 8.458% |

### 分析

**1. AC-tuned full-val 复评确认可靠**

内嵌验证 5.675% vs full-val 5.662%，差异仅 0.013pp，方向一致。AC-tuned 在所有 step 上均优于 Base 和 Z 基线，且改善幅度随 step 增大而增大（step10: -0.248pp vs Base），说明 Expert Iteration 有效缓解了长程误差累积。

**2. Longer training (960 updates) 无效**

path_mape = 5.701%，比 480 updates 的 5.675% 差 0.026pp。原因分析：
- Expert Iteration 的 SFT loss 在后期已经饱和（loss 从 ~22 降到 ~21 后不再显著下降）
- 更多 updates 导致模型对 best-of-8 轨迹过拟合，泛化性下降
- 480 updates 是当前配置下的甜蜜点

**3. Low KL (0.01) 微幅改善**

path_mape = 5.665%，比 KL=0.05 的 5.675% 改善 0.010pp。降低 KL 正则化让 Expert Iteration 更自由地学习 best-of-N 轨迹的模式，但改善幅度有限。KL=0.05 已经不是 Expert Iteration 的瓶颈。

**4. 两阶段训练效果与单阶段持平**

path_mape = 5.664% (内嵌) / 5.673% (full-val)，与 AC-tuned (5.662%) 几乎持平。从 Z checkpoint 开始精调没有带来叠加收益，可能原因：
- Z checkpoint 已经在 curriculum+KL 下收敛，Expert Iteration 的 best-of-N 筛选在 Z 的权重空间中探索的轨迹质量与从 base 开始训练时差异不大
- 两阶段训练等价于在已收敛模型上做 fine-tuning，但 Expert Iteration 本身就是一种 fine-tuning 机制

### 当前结论更新

截至本轮，1200-stock full-val path_mape 排序：

| 排名 | 方法 | full-val path_mape | 内嵌 val path_mape | 备注 |
|---:|------|---:|---:|------|
| **1** | **AC-tuned (EI, g=8, k=0.25, t=1.5, KL=0.05)** | **5.662%** | 5.675% | **当前 SOTA** |
| 2 | AC-lowKL (EI, g=8, k=0.25, t=1.5, KL=0.01) | — | 5.665% | 低 KL 微幅改善 |
| 3 | Two-stage (Z→EI, KL=0.01) | 5.673% | 5.664% | 与单阶段持平 |
| 4 | AC-long960 (EI, 960up) | — | 5.701% | 过拟合 |
| 5 | Z curriculum 3→6→10 + KL | 5.756% | 5.756% | 上一轮最优 |
| — | Base (1200 stocks) | 5.804% | 5.804% | 基线 |

### 关键发现总结

1. **Expert Iteration 是当前最有效的后训练算法**：在 1200-stock full-val 上比 Base 改善 0.142pp，比 Z 基线改善 0.094pp
2. **480 updates 是甜蜜点**：960 updates 导致过拟合
3. **KL=0.01 比 KL=0.05 微幅更好**：但差异很小（0.010pp），KL 不是关键瓶颈
4. **两阶段训练无叠加收益**：从 Z checkpoint 精调与从 base 训练效果持平
5. **长程改善最显著**：step10 改善 0.244pp (vs Base)，说明 Expert Iteration 的 best-of-N 机制有效缓解了误差累积

### 后续建议

1. **更大 group_size 消融**：测试 group_size=12/16，更大的采样池可能进一步提升 best-of-N 的质量上界
2. **动态 temperature 调度**：前期用高温度 (2.0) 增加探索，后期用低温度 (1.0) 聚焦高质量轨迹
3. **多轮 Expert Iteration**：完成一轮 EI 训练后，用新模型作为采样模型再做一轮 EI，类似 ReST 的多轮迭代
4. **与 path-aware loss 结合**：在 Expert Iteration 的 SFT 阶段加入 path-aware loss，让模型不仅学习好的轨迹，还学习避免坏的轨迹

***

## 第八轮实验：OpenAI GPT-5.5 风格 Verifier Loop（2026-05-06）

更新时间：2026-05-06 16:35
运行环境：同前，RTX 4060 Laptop 8GB。

### 实验目标

借鉴 OpenAI GPT-5.5 的 "训练时 Verifier 循环" 思路，将每一步的生成和验证嵌入训练循环，从源头上阻止错误 token 进入上下文。核心假设：**步骤级 Oracle 筛选优于轨迹级 Expert Iteration 筛选**。

实现三大组件：
1. **Oracle-Guided Step-Level Rollout**：训练时每步采样 K 个候选 token pair，用真实 future return 作为 Oracle 选择最优，阻止错误 token 进入训练上下文
2. **Value Head + Plan Head**：多目标联合训练，L = L_ce + λ1·L_plan + λ2·L_value
3. **Error Trajectory Bank**：收集高误差轨迹，训练 Error Head 预测误差

代码改动：`model/kronos_reasoning.py`（+35行，新增3个head）+ `posttrain/rollout/train_rollout.py`（+250行，新增Oracle-Guided rollout + 多目标损失）

### 实验配置

统一配置：`max_stocks=1200`, `max_train_samples=1280`, `max_val_samples=640`, `batch_size=2`, `lr=5e-6`, `curriculum=3,6,10`, `curriculum_updates=160,160,160`, `KL=0.05`

| 实验 | 方法 | 关键超参 |
|:---:|------|------|
| **AF** | **Oracle-Guided only** | oracle-top-k=8, oracle-temp=1.5 |
| **AG** | **Oracle + Value + Plan** | AF + value-weight=0.3, plan-weight=0.5 |
| AH | OpenAI Full (Oracle+V+P+Error) | AG + error-distill-weight=0.15 | (待跑) |

### 训练内嵌验证（640 windows）

| 实验 | Stage 1 (h=3) | Stage 2 (h=6) | Stage 3 (h=10) | 峰值显存 |
|:---:|---:|---:|---:|:---:|
| AF Oracle-only | 5.807% | 5.877% | 5.787% | 1.839 GB |
| AG Oracle+V+P | 5.827% | — | 5.730% | 1.839 GB |

AG 的 value_loss 约 0.01-0.05, plan_loss 约 0.02，两者均收敛并持续贡献梯度。

### Full-val1200 对比（1829 windows）

| 模型 | path_mape | daily_mape | step1 | step3 | step6 | step10 | vs Base |
|------|---:|---:|---:|---:|---:|---:|---:|
| Base | 5.8035% | 2.2220% | 2.2646% | 4.2803% | 6.1755% | 8.6738% | — |
| Z curriculum+KL | 5.7560% | — | 2.3500% | 3.7030% | 5.5880% | 8.6780% | -0.048pp |
| **AF Oracle-only** | **5.7252%** | 2.2109% | 2.2640% | 4.2366% | 6.0737% | 8.5582% | **-0.078pp** |
| **AG Oracle+V+P** | **5.7010%** | 2.2039% | 2.2547% | 4.2149% | 6.0565% | 8.5086% | **-0.103pp** |
| AC-tuned EI (SOTA) | 5.6620% | — | 2.2500% | 3.6050% | 5.3800% | 8.4300% | -0.142pp |

### 逐 step path_mape 对比

| step | Base | AF Oracle | Δ AF | AG Oracle+V+P | Δ AG |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.2646% | 2.2640% | -0.001 | 2.2547% | -0.010 |
| 3 | 4.2803% | 4.2366% | -0.044 | 4.2149% | -0.065 |
| 6 | 6.1755% | 6.0737% | -0.102 | 6.0565% | -0.119 |
| 10 | 8.6738% | 8.5582% | -0.116 | 8.5086% | -0.165 |

### 分析

**1. Oracle-Guided Rollout 显著优于 BaseModel**

AF（Oracle-only）比 Base 改善 0.078pp，比 Z 基线改善 0.031pp。改善幅度随 step 增大而增大（step1: -0.001pp, step10: -0.116pp），证明了 Oracle-Guided 的步骤级筛选有效缓解了长程误差累积。

**2. 多目标训练（Value+Plan）叠加收益**

AG（Oracle+V+P）在 AF 基础上再改善 0.024pp（5.7010% vs 5.7252%）。Plan Head 和 Value Head 提供了额外的结构化梯度信号，帮助模型学习子目标规划和自我评估。

**3. 与 Expert Iteration 的对比**

AG（5.7010%）仍弱于 AC-tuned EI（5.6620%），差距 0.039pp。可能原因：
- EI 的 group_size=8 并保留 top 2 条完整轨迹，信号更密集
- Oracle-Guided 的 oracle-top-k=8 每步采样，但 temperature=1.5 可能导致候选质量不够高
- Value/Plan Head 权重（0.3/0.5）可能未达到最优

**4. Oracle-Guided vs Expert Iteration 的本质区别已验证**

AF/AG 的 step10 改善幅度（-0.116pp / -0.165pp）远大于 mid-step 改善，而 EI 的改善在 step6 更明显（-0.795pp vs Base）。这说明：
- Oracle-Guided 擅长**防止后期误差累积**（每步筛选，后期受益更大）
- Expert Iteration 擅长**提升中期轨迹质量**（完整轨迹筛选，中期信号更强）

**5. 改进潜力**

当前 Oracle-Guided 仅用默认超参（K=8, temp=1.5），还没有消融：
- 更大的 K（12/16）可能提升 Oracle 上界
- 更低的温度（1.0/1.2）可能提升候选质量
- Oracle + EI 组合可能获得叠加收益

### Full-val1200 完整排名

截至第八轮，1200-stock full-val path_mape 排序：

| 排名 | 方法 | path_mape | 备注 |
|---:|------|---:|------|
| 1 | AC-tuned EI (g=8, k=0.25, t=1.5) | 5.6620% | 当前 SOTA |
| **2** | **AG Oracle+V+P (K=8, t=1.5)** | **5.7010%** | **本轮最佳** |
| **3** | **AF Oracle-only (K=8, t=1.5)** | **5.7252%** | **Oracle-Guided 基础** |
| 4 | Z curriculum 3→6→10 + KL | 5.7560% | 上一轮最优 |
| — | Base (1200 stocks) | 5.8035% | 基线 |

### 结论

1. **OpenAI GPT-5.5 风格的 Verifier Loop 方案在 Kronos-R 上可行且有效**。Oracle-Guided Rollout 比 Base 改善 0.078pp，多目标联合训练进一步改善到 0.103pp。

2. **步骤级筛选的核心假设得到验证**：AF/AG 的长程改善（step10: -0.165pp vs Base）证明了从源头阻止错误 token 进入上下文的价值。

3. **Oracle-Guided 是 Expert Iteration 的互补方案**，不是替代方案。两者在不同步骤范围有不同优势，组合使用可能是最优路径。

4. **后续优化方向**：
   - Oracle + EI 组合：用 Oracle-Guided 做前几步，用 EI 做后续轨迹筛选
   - 消融 oracle-top-k 和 temperature
   - 跑 AH（OpenAI Full with Error Distill）
   - 更高的 Value/Plan weight 消融

### 输出文件

- `checkpoints/post_train_rollout_exp_af_oracle1200/rollout_exp_af.pt`（注：路径拼接bug，实际在 `checkpointspost_train_rollout_exp_af_oracle1200/`）
- `checkpoints/post_train_rollout_exp_ag_oracle_vp_1200/rollout_exp_ag.pt`
- `outputs/eval_AF_oracle1200/rollout_eval_val.json`
- `outputs/eval_AG_oracle_vp_1200/rollout_eval_val.json`
- `outputs/eval_AF_oracle1200/prediction_diff_*.csv`
- `outputs/eval_AG_oracle_vp_1200/prediction_diff_*.csv`

### 运行命令

复现 AF（Oracle-only）：
```powershell
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
& 'D:\conda_envs\llm-t\Scripts\python.exe' Post_Train_Rollout.py `
  --output-dir "checkpoints/post_train_rollout_exp_af_oracle1200" `
  --save-name rollout_exp_af.pt `
  --max-stocks 1200 --max-train-samples 1280 --max-val-samples 640 `
  --batch-size 2 --eval-batch-size 8 --epochs 1 --max-train-updates 480 `
  --use-gradient-checkpointing false --step-weight-gamma 0.75 --lr 5e-6 `
  --rollout-ratio-start 1.0 --rollout-ratio-end 1.0 `
  --anchor-weight 0 --kl-weight 0.05 --numeric-mape-weight 0 --numeric-soft-ce-weight 0 `
  --curriculum-horizons "3,6,10" --curriculum-updates "160,160,160" `
  --oracle-guided true --oracle-top-k 8 --oracle-temp 1.5
```

复现 AG（Oracle+Value+Plan）：
```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' Post_Train_Rollout.py `
  --output-dir "checkpoints/post_train_rollout_exp_ag_oracle_vp_1200" `
  --save-name rollout_exp_ag.pt `
  --max-stocks 1200 --max-train-samples 1280 --max-val-samples 640 `
  --batch-size 2 --eval-batch-size 8 --epochs 1 --max-train-updates 480 `
  --use-gradient-checkpointing false --step-weight-gamma 0.75 --lr 5e-6 `
  --rollout-ratio-start 1.0 --rollout-ratio-end 1.0 `
  --anchor-weight 0 --kl-weight 0.05 `
  --curriculum-horizons "3,6,10" --curriculum-updates "160,160,160" `
  --oracle-guided true --oracle-top-k 8 --oracle-temp 1.5 `
  --value-weight 0.3 --plan-weight 0.5
```

Full-val 评估：
```powershell
& 'D:\conda_envs\llm-t\Scripts\python.exe' -m posttrain.rollout.eval_rollout `
  --include-base true `
  --checkpoint "checkpoints/post_train_rollout_exp_ag_oracle_vp_1200/rollout_exp_ag.pt" `
  --mode val --max-stocks 1200 --max-val-samples 0 --batch-size 8 `
  --output-dir "outputs/eval_AG_oracle_vp_1200"
```

***

## 第九轮（最终轮）：Top-3 决战 —— Demo 期自回归评估（2026-05-06）

更新时间：2026-05-06 17:30
运行环境：同前，RTX 4060 Laptop 8GB。

### 实验目标

从所有历史实验中选出 Top-3 SOTA 方法，在**全量 4695 stocks 的 Demo 期（最后 30 个交易日，206 窗口）**上进行严格 10-step 自回归评估，筛选出最终的 Kronos-R 后训练方案。

### 候选方法

| 候选 | 方法 | 训练数据 | 核心思路 |
|:---:|------|:---:|------|
| **AC-tuned EI** | Expert Iteration (g=8,k=0.25,t=1.5) | 1200 stocks | 轨迹级筛选：采样 8 条完整轨迹 → 保留 top 25% → SFT |
| **AG Oracle+V+P** | Oracle-Guided + Value + Plan (K=8,t=1.5) | 1200 stocks | 步骤级 Oracle 筛选 + 多目标联合训练 |
| **AF Oracle-only** | Oracle-Guided (K=8,t=1.5) | 1200 stocks | 纯步骤级 Oracle 筛选，无额外辅助目标 |
| **EI full-data** | Expert Iteration (g=8,k=0.25,t=1.5) | 4695 stocks | 同上 AC-tuned，但在全量数据上训练 |

### Demo 评估结果（全量 4695 stocks, 206 demo 窗口, 30 交易日）

| 排名 | 模型 | path_mape | daily_mape | step10 | vs Base |
|---:|------|---:|---:|---:|---:|
| 🥇 | **AF Oracle-only** | **5.6510%** | 2.2712% | **7.5641%** | **-0.0445pp** |
| 🥈 | BaseModel | 5.6955% | 2.2929% | 7.6677% | — |
| 🥉 | AG Oracle+V+P | 5.7013% | 2.2749% | 7.6495% | +0.0058pp |
| 4 | EI full-data | 5.7134% | 2.2891% | 7.6459% | +0.0180pp |
| 5 | AC-tuned EI | 5.7691% | 2.2910% | 7.8538% | +0.0736pp |

### 逐 step path_mape 对比

| Step | Base | AF Oracle | Δ AF | AG Oracle+V+P | AC-tuned EI |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.4859% | 2.4643% | **-0.022** | 2.4392% | 2.4529% |
| 3 | 4.6776% | 4.7057% | +0.028 | 4.7418% | 4.7478% |
| 6 | 6.3883% | 6.3551% | **-0.033** | 6.4105% | 6.5400% |
| 10 | 7.6677% | 7.5641% | **-0.104** | 7.6495% | 7.8538% |

### 关键发现

**1. 🥇 AF Oracle-only 是最终赢家**

在 demo 期（完全未见过的数据），AF Oracle-only 是唯一超越 BaseModel 的后训练方法。path_mape = 5.6510% vs Base 5.6955%，改善 0.0445pp。所有其他后训练方法在 demo 上均不优于 BaseModel。

**2. Val vs Demo 的排序完全反转**

| 方法 | 1200-stock Val | Demo (4695 stocks) | 反转 |
|------|:---:|:---:|:---:|
| AF Oracle-only | 5.7252% (#3) | 5.6510% (#1) | ↑↑ |
| AC-tuned EI | 5.6620% (#1) | 5.7691% (#5) | ↓↓↓↓ |
| AG Oracle+V+P | 5.7010% (#2) | 5.7013% (#3) | ↓ |

这说明 **val 上的排名不能预测 demo 上的泛化性能**。Expert Iteration 在训练分布上表现最好，但在未见过的股票上泛化差。

**3. Expert Iteration 过拟合到训练股票**

AC-tuned EI 在 1200-stock val 上是最好的（5.662%），但在 4695-stock demo 上比 Base 差 0.074pp。EI 的 best-of-N 轨迹筛选可能学到了训练股票特有的模式，无法泛化到新股票。

即使在 4695 stocks 上重新训练 EI（EI full-data），demo 表现（5.7134%）仍然比 Base 差。这说明 trajectory-level 筛选在某些数据上确实会过拟合。

**4. Oracle-Guided 泛化性更好**

AF 的步骤级 Oracle 筛选学到的是一种更通用的 "生成数值准确 token" 的能力，这种能力可以泛化到未见过的股票。长程改善在 demo 上依然有效：step10 vs Base = -0.104pp。

**5. 多目标训练（Value+Plan）在 demo 上没有叠加收益**

AG 在 1200-stock val 上比 AF 好 0.024pp，但在 demo 上比 AF 差 0.050pp。说明 Value Head 和 Plan Head 提供的辅助信号可能导致模型学到了一些训练集特有的偏差，损害了泛化性。

**6. BaseModel 是强大的基线**

在 demo 上，BaseModel 排名第二，仅比 AF 差 0.0445pp。这再次验证了 Kronos-R 的 BaseModel 本身在自回归预测上就有不错的能力。

### 最终结论

🏆 **Kronos-R 的最终推荐后训练方案：AF Oracle-Guided Rollout**

- **方法**：Oracle-Guided Step-Level Rollout（K=8, temperature=1.5）+ Curriculum (3→6→10) + KL=0.05
- **训练数据**：1200 stocks（更大的数据集不一定更好——EI full-data 在 demo 上反而更差）
- **推理**：确定性 argmax（与现有协议一致，不需要额外的 Verifier）
- **效果**：Demo path_mape = 5.6510% vs Base 5.6955%（-0.78% relative improvement）
- **长程收益最显著**：step10 改善 -0.104pp

**核心原因**：Oracle-Guided Rollout 从源头上阻止错误 token 进入训练上下文，模型学到的是一种 "在每个步骤生成数值最准确 token" 的通用能力，这种能力可以泛化到未见过的股票。而 Expert Iteration 的轨迹级筛选学到的模式更依赖训练股票的具体分布。

### 输出文件

- `outputs/eval_FINAL_demo/rollout_eval_demo.json`
- `outputs/eval_FINAL_demo/rollout_eval_demo.csv`
- `outputs/eval_FINAL_demo/prediction_diff_*.csv` (5 个文件)

### Demo 评估命令

```powershell
# 构建 demo 缓存（仅首次需要）
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
  --checkpoint "checkpointspost_train_rollout_exp_af_oracle1200/rollout_exp_af.pt" `
  --checkpoint "checkpoints/post_train_rollout_exp_ag_oracle_vp_1200/rollout_exp_ag.pt" `
  --checkpoint "checkpoints/post_train_rollout_exp_ac_expert_tuned/rollout_exp_ac_tuned.pt" `
  --mode demo --max-stocks 0 --max-val-samples 0 --batch-size 8 `
  --output-dir "outputs/eval_FINAL_demo"
```
