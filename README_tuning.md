# README_tuning

## 1. 这份文档的用途

这份文档是当前项目调参与研究工作的正式交接文档。

后续如果开启新对话，新的助手应当先完整读完这份文档，再继续推进工作。
这里记录的不是泛泛而谈的“调参建议”，而是本项目当前已经做过的工作、已经确认的判断、已经踩过的坑、已经改过的脚本，以及下一步最合理的推进路线。

## 2. 研究边界与硬约束

当前只关注以下范围，不要发散：

- 模型只关注 `BOD` 和 `LightGCN`
- 数据集只关注 `iFashion_UB`
- `iFashion_UB` 是从捆绑推荐论文 `MultiCBR` 的 `iFashion` 数据中抽出的 `u-b` 交互
- 当前研究目标不是做大而全复现，而是验证 `BOD` 的 generator 是否能学到 `user-bundle` 交互权重
- 希望把 `BOD` 和 `LightGCN` 都调到各自较优状态，并且让 `BOD` 明显超过 `LightGCN`
- 如果 `BOD` 能稳定明显优于 `LightGCN`，才更有把握说明 generator 学到的交互权重具有后续迁移到 `MultiCBR` 的价值

本地开发环境约束：
 
- 当前本地电脑没有可用显卡，也没有完整训练环境
- 在本地不要主动运行训练
- 只在本地改代码、查代码、写脚本、整理文档
- 用户会把代码同步到服务器后再执行训练

## 3. 当前项目里最相关的文件

后续讨论和改动应优先围绕这些文件：

- `conf/BOD.conf`
- `conf/LightGCN.conf`
- `model/graph/BOD.py`
- `model/graph/LightGCN.py`
- `scripts/auto_search.py`
- `scripts/search_specs/*.json`
- `results/experiment_results.txt`
- 本文档 `README_tuning.md`

## 4. 当前研究目标的准确表述

当前要回答的问题不是“BOD 能不能跑”，而是：

1. 在 `iFashion_UB` 上，`LightGCN` 的强 baseline 能做到什么水平
2. 在同一数据集上，`BOD` 能否稳定超过 `LightGCN`
3. `BOD` 的提升是否主要来自 generator 对 `u-b` 交互权重的学习
4. 如果是，这些 learned weights 是否可以进一步服务于 `MultiCBR`

换句话说，当前研究工作分两层：

- 表层目标：把指标做上去
- 深层目标：把指标提升和 generator 的“权重识别能力”联系起来

## 5. 已经建立的关键理解

### 5.1 BOD 的训练结构

`BOD` 不是单层训练，而是双层：

- 内层：训练推荐器 backbone，这里当前使用 `LightGCN`
- 外层：训练 generator，让它为交互对输出权重，并通过梯度匹配去影响内层训练方向

相关代码位置：

- `model/graph/BOD.py`

### 5.2 当前 BOD 最核心的几个参数

- `GM_AU.-weight_uniformity`
  - 控制内层 `uniformity_loss` 的权重
  - 当前已被证明是强影响参数

- `GM_AU.-weight_bpr`
  - 控制加权 BPR 项的权重

- `GM_AU.-weight_alignment`
  - 控制加权 alignment 项的权重

- `GM_AU.-generator_lr`
  - generator 优化器学习率

- `GM_AU.-generator_reg`
  - 名字叫 generator regularization，但当前实现里并不是直接对 generator 参数做正则
  - 代码实际上是对 outer-loop 中的 `user_emb_ol` / `item_emb_ol` 做 `l2_reg_loss`

- `GM_AU.-outer_batch_size`
  - generator 外层更新时使用的 batch 大小

- `GM_AU.-outer_loop`
- `GM_AU.-inner_loop`
  - 控制每个 epoch 内层和外层的更新次数

### 5.3 一个非常重要的实现事实

当前代码里：

- `generator_lr` 确实直接作用于 `optimizer_generator`
- `generator_reg` 当前并没有真正正则到 generator 参数本身

对应代码：

- `model/graph/BOD.py`
  - `optimizer_generator = torch.optim.Adam(model_generator.parameters(), lr=self.generator_lr)`
  - `loss_reg = l2_reg_loss(self.generator_reg, user_emb_ol, item_emb_ol)`

- `util/loss_torch.py`
  - `l2_reg_loss(reg, *args)` 只是对传入 embedding 做 L2

这个事实对解释当前搜参结果非常重要。

## 6. 自动化脚本线已经做过的修改

### 6.1 `scripts/auto_search.py` 已经做过的关键修改

这部分是已经完成并在服务器上实际使用过的，不要丢。

#### 修改 1：加入项目根目录到 `sys.path`

原因：

- 服务器上用 `python scripts/auto_search.py ...` 运行时，子进程一开始报过：
  - `ModuleNotFoundError: No module named 'SELFRec'`

解决：

- 在 `scripts/auto_search.py` 开头加入 repo root 注入
- 让脚本模式下也能正常导入 `SELFRec`

#### 修改 2：失败自动重试

需求来源：

- 用户希望训练出错时不要立刻跳过，而是先重试
- 连续 3 次失败才跳到下一个 trial

当前行为：

- 单个 trial 失败时自动重试
- 默认最多 3 次
- 连续 3 次失败才记为失败
- 当前重试逻辑主要针对 trial 返回的 `failed`
- `timeout` 目前没有特殊重试策略

#### 修改 3：记录 attempt 信息

当前 `summary.jsonl` 中会记录：

- `attempt_count`
- `attempts`

方便回看哪些 trial 是重试后才成功的。

### 6.2 为什么必须保留子进程隔离

因为训练过程中实际出现过：

- `RuntimeError CUDA error: an illegal memory access was encountered`

这类 CUDA 错误一旦出现，同一 Python 进程中的 CUDA 上下文可能已经不干净。
所以 trial 级子进程隔离是有必要的，不建议改回同进程串行执行。

## 7. 当前配置基线

### 7.1 当前 `conf/BOD.conf`

当前本地配置文件已更新为：

- `dataset.name=iFashion_UB`
- `model.name=BOD`
- `trainmodel=LightGCN`
- `seed=3407`
- `embbedding.size=64`
- `num.max.epoch=30`
- `batch_size=256`
- `learnRate=0.001`
- `reg.lambda=0.0001`
- `LightGCN=-n_layer 1`
- `GM_AU=-generator_lr 0.001 -generator_reg 0.0001 -generator_emb_size 64 -outer_loop 1 -inner_loop 1 -outer_batch_size 128 -weight_bpr 1 -weight_alignment 1 -weight_uniformity 0.35`

重点：

- `weight_uniformity` 已经被正式定到 `0.35`
- `generator_lr` 基于 Stage 4 完整结果，更新为 `0.001`
- `generator_reg` 保留 `0.0001`
- `num.max.epoch` 已更新到 `30`

### 7.2 当前阶段默认固定住的参数

除非进入专门的新一轮搜索，否则当前默认固定：

- `embbedding.size = 64`
- `LightGCN.-n_layer = 1`
- `num.max.epoch = 30`
- `batch_size = 256`
- `learnRate = 0.001`
- `GM_AU.-outer_loop = 1`
- `GM_AU.-inner_loop = 1`
- `GM_AU.-outer_batch_size = 128`
- `GM_AU.-weight_bpr = 1`
- `GM_AU.-weight_alignment = 1`
- `GM_AU.-weight_uniformity = 0.35`
- `GM_AU.-generator_lr = 0.001`
- `GM_AU.-generator_reg = 0.0001`

## 8. 已完成的搜索阶段与结果

下面的结果是当前调参工作的核心历史。

### 8.1 Stage 1：扫描 `weight_uniformity = [0.0, 0.02, 0.05, 0.1]`

spec：

- `scripts/search_specs/bod_stage1_uniformity.json`

固定参数：

- `num.max.epoch = 20`
- `batch_size = 256`
- `learnRate = 0.001`
- `LightGCN.-n_layer = 1`
- `GM_AU.-outer_loop = 1`
- `GM_AU.-inner_loop = 1`

结果：

- `0.0`
  - `best_epoch = 17`
  - `Recall@20 = 0.0284423820`
  - `NDCG@20 = 0.0241358068`
  - 第一次因为 CUDA illegal memory access 失败，第二次重试成功

- `0.02`
  - `best_epoch = 17`
  - `Recall@20 = 0.0299849264`
  - `NDCG@20 = 0.0253468514`

- `0.05`
  - `best_epoch = 17`
  - `Recall@20 = 0.0360243853`
  - `NDCG@20 = 0.0307521621`

- `0.1`
  - `best_epoch = 20`
  - `Recall@20 = 0.0535159505`
  - `NDCG@20 = 0.0444552353`

结论：

- `weight_uniformity` 不是越小越好
- 至少在 `iFashion_UB` 上，`uniformity` 增大明显带来收益

### 8.2 Stage 2：继续向右扫描 `weight_uniformity = [0.1, 0.15, 0.2, 0.25]`

spec：

- `scripts/search_specs/bod_stage2_uniformity_right.json`

结果：

- `0.1`
  - `Recall@20 = 0.0535159505`
  - `NDCG@20 = 0.0444552354`
  - 重试一次后成功

- `0.15`
  - `Recall@20 = 0.0751615597`
  - `NDCG@20 = 0.0631558120`

- `0.2`
  - `Recall@20 = 0.0923676503`
  - `NDCG@20 = 0.0797284353`

- `0.25`
  - `Recall@20 = 0.0972465551`
  - `NDCG@20 = 0.0846317695`

结论：

- `weight_uniformity` 继续增大仍然持续显著变好
- 最优值尚未到头

### 8.3 Stage 3：`num.max.epoch = 30`，扫描 `weight_uniformity = [0.25, 0.3, 0.35, 0.4]`

spec：

- `scripts/search_specs/bod_stage3_uniformity_right.json`

结果：

- `0.25`
  - `best_epoch = 30`
  - `Recall@20 = 0.0987868349`
  - `NDCG@20 = 0.0863496586`

- `0.3`
  - `best_epoch = 27`
  - `Recall@20 = 0.0996268059`
  - `NDCG@20 = 0.0872392076`

- `0.35`
  - `best_epoch = 26`
  - `Recall@20 = 0.0998831146`
  - `NDCG@20 = 0.0874434840`

- `0.4`
  - 第一次报 `CUDA illegal memory access`
  - 第二次开始重跑时，用户决定不再继续等待

结论：

- 从 `0.25 -> 0.35` 仍在上涨，但涨幅明显变小
- 结合稳定性与收益，用户决定把当前最佳值定为 `0.35`
- 因此项目内默认配置已经更新为 `weight_uniformity = 0.35`

### 8.4 目前对 `weight_uniformity` 的理解

当前观察到的结果说明：

- 在 `iFashion_UB` 这个 `u-b` 图上，`uniformity` 是强影响参数
- 高 `uniformity` 并不一定异常，反而可能说明该场景下更需要把表示空间拉开
- 这与“generator 需要在更可分的表示空间里学习交互权重”是相容的

但必须注意：

- 现在只能说 `uniformity` 很重要
- 还不能说当前 `BOD` 的全部提升已经被证明来自 generator 的权重学习

## 9. 已完成的 Stage 4：generator 相关搜索

spec：

- `scripts/search_specs/bod_stage4_generator.json`

当前设置：

- 固定 `weight_uniformity = 0.35`
- 固定 `num.max.epoch = 30`
- 固定其他 backbone / outer settings
- 搜索：
  - `GM_AU.-generator_lr = [0.0001, 0.0005, 0.001]`
  - `GM_AU.-generator_reg = [0.00001, 0.0001]`

共 6 组。

### 9.1 全部 6 组结果

- `trial_0001`
  - `generator_lr = 0.0001`
  - `generator_reg = 1e-05`
  - `best_epoch = 26`
  - `Recall@20 = 0.0997639957`
  - `NDCG@20 = 0.0873932009`

- `trial_0002`
  - `generator_lr = 0.0001`
  - `generator_reg = 0.0001`
  - `best_epoch = 26`
  - `Recall@20 = 0.0997664990`
  - `NDCG@20 = 0.0873953261`
  - 第 1 次尝试因 `CUDA illegal memory access` 失败，第 2 次成功

- `trial_0003`
  - `generator_lr = 0.0005`
  - `generator_reg = 1e-05`
  - `best_epoch = 26`
  - `Recall@20 = 0.0998831146`
  - `NDCG@20 = 0.0874471235`
  - 第 1 次尝试因 `CUDA illegal memory access` 失败，第 2 次成功

- `trial_0004`
  - `generator_lr = 0.0005`
  - `generator_reg = 0.0001`
  - 连续 3 次都因 `CUDA illegal memory access` 失败

- `trial_0005`
  - `generator_lr = 0.001`
  - `generator_reg = 1e-05`
  - `best_epoch = 27`
  - `Recall@20 = 0.0998637286`
  - `NDCG@20 = 0.0875090974`
  - 第 1 次尝试因 `CUDA illegal memory access` 失败，第 2 次成功

- `trial_0006`
  - `generator_lr = 0.001`
  - `generator_reg = 0.0001`
  - `best_epoch = 27`
  - `Recall@20 = 0.0998743617`
  - `NDCG@20 = 0.0875125534`

### 9.2 对 Stage 4 结果的判断

从完整 6 组结果看：

- `generator_reg` 这一维几乎没有信息量
  - 在相同 `generator_lr` 下，`1e-05` 和 `0.0001` 的差距始终极小
  - 这和当前代码实现完全一致，因为它没有真正正则到 generator 参数

- `generator_lr` 有弱信号，但没有出现大幅提升
  - `0.0001` 两组略差
  - `0.0005` 和 `0.001` 更接近当前前沿
  - 完整成功结果里，`0.001 + 0.0001` 的 `NDCG@20` 最高

### 9.3 为什么变化整体不大

当前最重要的解释有两层：

#### 原因 1：`generator_reg` 当前实现基本没有真正调到 generator

当前代码中：

- `generator_lr` 直接作用到 `optimizer_generator`
- 但 `generator_reg` 加到的是 `l2_reg_loss(self.generator_reg, user_emb_ol, item_emb_ol)`
- 这意味着它正则的是 outer-loop 中抽出来的 embedding，而不是 `model_generator.parameters()`

所以 `generator_reg = 1e-5` 和 `generator_reg = 1e-4` 本来就不太可能拉出大差异。

#### 原因 2：当前 generator 学习强度仍然偏弱

- `outer_loop = 1`
- `outer_batch_size = 128`
- 总 epoch = `30`

在这套设置下，即便 `generator_lr` 从 `0.0001` 提到 `0.001`，也只表现为弱增益，而没有出现明显跳升。

### 9.4 当前阶段结论

Stage 4 完整跑完后，可以定下以下结论：

- `weight_uniformity = 0.35` 继续作为 BOD 当前正式工作点
- `generator_lr` 在完整成功结果里以 `0.001` 最优，因此本地默认配置更新为 `0.001`
- `generator_reg` 保留 `0.0001`，但当前不要再继续围绕它做搜索
- 仅凭 Stage 4 参数变化幅度很小，还不能推出“BOD 的提升没有来自 generator 学到的权重”
- 如果要判断 generator 权重机制是否真正有效，需要额外做 ablation：去掉 BOD 权重机制，直接与纯 `LightGCN` 比较，才能回答“有无 generator 权重机制是否真的造成差异”
- 当前阶段先不继续做 BOD 搜索，后续重点转向把 `LightGCN` 在 `iFashion_UB` 上调到更强 baseline

## 10. 当前最重要的研究判断

这是目前整个工作最关键的阶段性结论。

### 10.1 已经能确认的

- `BOD` 在 `iFashion_UB` 上对 `weight_uniformity` 非常敏感
- `weight_uniformity` 的有效区间明显不是小值，而是较高值
- 在 `0.35` 左右，指标已经稳定到接近当前阶段最好

### 10.2 还不能确认的

- 还不能确认当前 `BOD` 的提升有多大比例来自 generator 学到的交互权重
- Stage 4 只能说明：现有超参搜索没有把 generator 作用进一步明显放大
- 但这不等于“generator 权重机制没有贡献”；要证明这一点，必须做去掉该机制的 ablation，对照纯 `LightGCN`

### 10.3 当前最值得警惕的点

需要特别记住：

1. 当前 `generator_reg` 实现本身就基本不起作用
2. `generator_lr + outer_loop=1 + outer_batch_size=128` 这套设置下，generator 学习强度仍然偏弱

## 11. 当前搜索脚本与 spec 文件清单

已存在的 spec：

- `scripts/search_specs/bod_stage1_uniformity.json`
- `scripts/search_specs/bod_stage2_uniformity_right.json`
- `scripts/search_specs/bod_stage3_uniformity_right.json`
- `scripts/search_specs/bod_stage4_generator.json`

这些 spec 都是围绕当前研究主线创建的，不要误删。

## 12. 已踩过的实际坑

### 12.1 `bod_stage4_generator.json` 一度是空文件

已经修复。

如果服务器上再次出现：

- `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`

优先检查对应 spec 文件是否为空。

### 12.2 `SELFRec` 导入失败

历史问题：

- 服务器上直接运行 `python scripts/auto_search.py ...`
- 可能出现 `ModuleNotFoundError: No module named 'SELFRec'`

已经通过 `sys.path` 注入修复。

### 12.3 训练中偶发 `CUDA illegal memory access`

这是当前最常见的不稳定来源之一。

当前处理策略：

- trial 级子进程隔离
- 单个 trial 自动重试最多 3 次

## 13. 后续最合理的推进路线

### 路线 A：BOD 侧先收束，不再继续搜参

- BOD 当前工作点固定为：
  - `weight_uniformity = 0.35`
  - `generator_lr = 0.001`
  - `generator_reg = 0.0001`
- 当前不再继续做 BOD 参数搜索

### 路线 B：把 `LightGCN` 在 `iFashion_UB` 上调到尽量强

后续优先调：

- `learnRate`
- `reg.lambda`
- `batch_size`
- `LightGCN.-n_layer`
- 必要时再看 `num.max.epoch`

目标不是继续抠 BOD 内部微小差异，而是先把 `LightGCN` baseline 做强，再回头判断 BOD 的真实优势。

### 路线 C：如果未来还要重新验证 generator 机制

那时不要直接回到细抠现有超参，而应优先考虑：

1. 做“去掉 BOD 权重机制”的 ablation
2. 必要时再改 `generator_reg` 实现，使其真正正则到 generator 参数

## 14. 当前最推荐的工作原则

- 不在本地启动训练
- 本地只做代码修改、脚本修改、文档整理、逻辑分析
- 训练全部由用户在服务器上执行
- 每一轮搜索尽量只动一层变量
- 先看趋势，再看极小数点差异
- 早期更重视“哪个方向有效”，而不是过早抠极小差值

## 15. 给下一轮对话的直接交接

如果下一轮对话由新的助手接手，请直接按下面方式进入工作：

1. 先完整读完本文件
2. 确认当前 `conf/BOD.conf` 中 `weight_uniformity = 0.35`
3. 确认当前 `conf/BOD.conf` 中 `generator_lr = 0.001`
4. 确认 `scripts/auto_search.py` 已包含：
   - repo root 注入
   - trial 自动重试
5. 把 BOD 视为当前已收束的工作点，不再继续做参数搜索
6. 后续优先任务改为：把 `LightGCN` 在 `iFashion_UB` 上调到更强 baseline
7. 如果要讨论 generator 是否有效，优先考虑 ablation，而不是继续只看当前 Stage 4 的微小差值
8. 不要把讨论发散到其他模型或其他数据集

## 16. 当前一句话状态总结

当前已经确认：`BOD` 在 `iFashion_UB` 上对 `weight_uniformity` 极其敏感，当前工作点固定为 `weight_uniformity = 0.35`、`generator_lr = 0.001`、`generator_reg = 0.0001`；Stage 4 已结束，BOD 暂不继续搜参，下一阶段的重点转为把 `LightGCN` 在 `iFashion_UB` 上调到尽量强，并在需要时通过 ablation 再判断 generator 权重机制的真实贡献。
