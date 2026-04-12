# 调参与自动化搜索说明

这份文档记录当前项目后续调参的统一思路，主要面向两个模型：

- `BOD`
- `LightGCN`

当前约定如下：

- 只关注数据集 `iFashion_UB`
- 调参优先级：先 `BOD`，后 `LightGCN`
- 主评估指标：`NDCG@20`
- 辅助指标：`Recall@20`、`NDCG@10`
- `Recall@10` 只作为补充参考


## 1. 为什么先调 BOD

当前项目的研究目标不是“把所有模型都调一遍”，而是尽快回答下面两个问题：

1. `BOD` 在 `iFashion_UB` 上能否稳定带来增益
2. 如果能，它的增益主要来自哪里

之所以先调 `BOD`，原因有三点：

- `LightGCN` 已经能在 `iFashion_UB` 上正常收敛，可以作为可用 baseline
- `BOD` 的内层本质上还是 `LightGCN`，先调 `BOD`，实际上已经顺带探索了一部分 backbone 的有效区间
- `iFashion_UB` 是 `user-bundle` 数据，不是标准 `user-item` 交互，论文里的涨幅不一定原样复现，因此更需要优先验证 `BOD` 自身机制


## 2. 当前对数据和指标的判断

### 2.1 数据特点

`iFashion_UB` 与原始 `iFashion` 不同，主要差异是：

- 图更密
- 用户平均交互更多
- 测试集中每个用户的正样本更多
- `Top-50` 很容易让 `Recall` 接近饱和

所以当前我们已经统一成只看：

- `Top-10`
- `Top-20`

并且只保留：

- `Recall`
- `NDCG`


### 2.2 为什么主看 NDCG@20

`Recall` 更适合观察“有没有召回到”，但不够敏感，特别是在 `Top-K` 比较大、测试集正样本偏多时容易饱和。

`NDCG@20` 更适合做主指标，因为它同时考虑：

- 是否命中
- 命中的位置是否靠前

对 `BOD` 来说，这一点尤其重要。因为 `BOD` 不只是“多捞一点候选”，理论上更应该改善排序质量。


## 3. 调参总原则

### 3.1 先分层，不要一次全扫

参数要分两层看：

1. `GCN / backbone` 层
2. `BOD` 层

但执行顺序上，优先调 `BOD` 层，再回过头微调 `GCN` 层。


### 3.2 先粗搜，再细搜

建议总共分三步：

1. 先做 `BOD` 参数粗搜，找出有效区间
2. 在有效区间内细搜 `BOD`
3. 最后再补调 `LightGCN`，形成公平对比


### 3.3 不建议一开始就全量网格搜索

原因很现实：

- `BOD` 训练成本高
- GPU 稳定性历史上不是完全无风险
- 很多参数之间有强耦合

如果一开始就把所有参数都网格化，实验量会爆炸，而且很难读出结论。


## 4. BOD 调参计划

### 4.1 参数分层

#### 第一层：先固定住的参数

这部分先不要大动：

- `embbedding.size`
- `LightGCN.-n_layer`

当前建议先固定为：

- `embbedding.size = 64`
- `LightGCN.-n_layer = 1`


#### 第二层：优先调的 BOD 参数

这是最值得先扫的一层：

- `GM_AU.-weight_uniformity`
- `GM_AU.-generator_lr`
- `GM_AU.-generator_reg`
- `GM_AU.-outer_batch_size`
- `GM_AU.-weight_alignment`
- `GM_AU.-weight_bpr`


#### 第三层：桥接参数

这些参数虽然不完全属于 `BOD` 本身，但会明显影响训练动态：

- `learnRate`
- `batch_size`
- `num.max.epoch`


### 4.2 推荐的搜索顺序

#### 阶段一：先扫 uniformity

优先级最高。

推荐值：

- `0`
- `0.02`
- `0.05`
- `0.1`
- `0.2`

原因：

- `iFashion_UB` 更密
- `uniformity` 不一定越大越好
- 很有可能较小的 `uniformity` 会更合适


#### 阶段二：再扫 generator 学习强度

推荐先调：

- `GM_AU.-generator_lr`
- `GM_AU.-generator_reg`

建议区间：

- `generator_lr`: `1e-4`, `5e-4`, `1e-3`
- `generator_reg`: `1e-5`, `1e-4`, `5e-4`


#### 阶段三：再调外层 batch 和损失权重

推荐参数：

- `GM_AU.-outer_batch_size`
- `GM_AU.-weight_alignment`
- `GM_AU.-weight_bpr`

建议区间：

- `outer_batch_size`: `64`, `128`, `256`
- `weight_alignment`: `0.5`, `1`, `2`
- `weight_bpr`: `0.5`, `1`, `2`


#### 阶段四：最后微调 backbone 训练强度

只有前面 `BOD` 层面已经有明确增益时，才建议进入这一步。

建议范围：

- `learnRate`: `5e-4`, `1e-3`, `2e-3`
- `batch_size`: `128`, `256`


### 4.3 一个高性价比的 BOD 起始搜索空间

第一轮推荐只跑这些：

- 固定：
  - `LightGCN.-n_layer = 1`
  - `embbedding.size = 64`
  - `batch_size = 256`
  - `learnRate = 0.001`
  - `GM_AU.-outer_loop = 1`
  - `GM_AU.-inner_loop = 1`

- 先扫：
  - `GM_AU.-weight_uniformity = [0, 0.02, 0.05, 0.1]`

- 然后扫：
  - `GM_AU.-generator_lr = [1e-4, 5e-4, 1e-3]`
  - `GM_AU.-generator_reg = [1e-5, 1e-4]`

- 再扫：
  - `GM_AU.-outer_batch_size = [64, 128, 256]`
  - `GM_AU.-weight_alignment = [0.5, 1, 2]`
  - `GM_AU.-weight_bpr = [0.5, 1, 2]`


## 5. LightGCN 调参计划

`LightGCN` 放在 `BOD` 后面调，主要目标是拿到更强的 baseline，用于和最优 `BOD` 做公平比较。

推荐优先级：

- `learnRate`
- `reg.lambda`
- `batch_size`
- `LightGCN.-n_layer`

建议区间：

- `learnRate`: `0.001`, `0.002`, `0.003`
- `reg.lambda`: `1e-5`, `1e-4`, `5e-4`
- `batch_size`: `256`, `512`, `1024`
- `LightGCN.-n_layer`: `1`, `2`

建议流程：

1. 先固定 `n_layer=1`
2. 调 `learnRate` 和 `reg.lambda`
3. 再看 `batch_size`
4. 最后再决定要不要试 `n_layer=2`


## 6. 自动化脚本设计原则

自动化搜索脚本是：

- [scripts/auto_search.py](d:/PycharmProjects/My_BOD/scripts/auto_search.py:1)

设计原则如下：

- 不修改原始 `conf/*.conf`
- 每个 trial 都在独立子进程运行
- 一次 trial 失败不会影响后续 trial
- 支持 `grid` 和手写 `trials`
- 自动保存每次实验的临时配置、日志、结构化结果和总表

之所以必须用子进程隔离，是因为训练过程中曾出现过 `CUDA illegal memory access`。这类错误一旦发生，当前 Python 进程里的 CUDA 上下文可能已经不干净，不适合同进程继续跑下一组实验。


## 7. 自动化脚本支持什么

### 7.1 支持的功能

- 读取一个 JSON 搜索定义文件
- 基于 base config 生成多份临时 config
- 支持普通键覆盖
- 支持 `OptionConf` 风格参数覆盖
- 记录失败实验
- 支持超时控制
- 支持 `resume`
- 支持 `dry-run`


### 7.2 覆盖语法

#### 普通配置项

直接写原始 key：

```json
{
  "batch_size": 256,
  "learnRate": 0.001,
  "num.max.epoch": 20
}
```

#### `OptionConf` 参数

使用“主键.子参数”的写法：

```json
{
  "LightGCN.-n_layer": 1,
  "GM_AU.-generator_lr": 0.0005,
  "GM_AU.-weight_uniformity": 0.1
}
```

这会自动把：

- `LightGCN=-n_layer 1`
- `GM_AU=...`

这种字符串重新组装好。


## 8. 搜索定义文件格式

一个典型 spec 结构如下：

```json
{
  "search_name": "bod_stage1",
  "base_config": "conf/BOD.conf",
  "output_dir": "results/search_runs/bod_stage1",
  "resume": true,
  "timeout_sec": null,
  "fixed_overrides": {
    "num.max.epoch": 20,
    "batch_size": 256
  },
  "grid": {
    "GM_AU.-weight_uniformity": [0.0, 0.05, 0.1],
    "GM_AU.-generator_lr": [0.0001, 0.0005]
  },
  "trials": [
    {
      "name": "manual_alignment_boost",
      "overrides": {
        "GM_AU.-weight_alignment": 2,
        "GM_AU.-weight_bpr": 1
      }
    }
  ]
}
```


### 8.1 字段说明

- `search_name`
  - 搜索任务名称
- `base_config`
  - 基础配置文件
- `output_dir`
  - 结果输出目录
- `resume`
  - 是否跳过已完成 trial
- `timeout_sec`
  - 单次 trial 的超时时间，秒；`null` 表示不设
- `fixed_overrides`
  - 所有 trial 都共享的配置覆盖
- `grid`
  - 自动笛卡尔积展开
- `trials`
  - 手写 trial 列表


## 9. 自动化脚本的使用方式

### 9.1 打印示例 spec

```bash
python scripts/auto_search.py --print-example
```


### 9.2 先做 dry-run

推荐任何正式搜索前都先跑：

```bash
python scripts/auto_search.py --spec path/to/search.json --dry-run
```

`dry-run` 会做这些事：

- 解析 spec
- 展开所有 trial
- 生成临时配置文件
- 打印每个 trial 的签名和配置路径

但不会真正启动训练。


### 9.3 正式执行搜索

```bash
python scripts/auto_search.py --spec path/to/search.json
```


## 10. 输出结果会放到哪里

假设 `output_dir` 是：

```text
results/search_runs/bod_stage1
```

脚本会生成：

- `configs/`
  - 每个 trial 的临时配置文件
- `logs/`
  - 每个 trial 的完整 stdout/stderr
- `trial_results/`
  - 每个 trial 的结构化结果 JSON
- `summary.jsonl`
  - 机器可读总表，按 trial 逐条追加
- `summary.csv`
  - 适合直接看和筛选的表格版本


## 11. 失败和跳过的行为

### 11.1 单次 trial 失败

不会中断总控脚本。

脚本会：

- 记录失败状态
- 记录错误类型
- 记录错误信息
- 继续下一组 trial


### 11.2 超时

如果设置了 `timeout_sec`，超时 trial 会被标记为：

- `timeout`

也不会影响其他 trial。


### 11.3 resume

如果 `resume=true`，脚本会根据 trial 覆盖参数的签名跳过已经完成的实验。

适合这些场景：

- 上次搜索跑到一半中断
- 需要补跑后半段
- 搜索文件基本相同，只新增了少量 trial


## 12. 自动化搜索时的实际建议

### 12.1 先串行，不要并行

目前不建议一开始做 GPU 并行搜索，原因：

- `BOD` 本身较重
- 稀疏图训练对显存和 CUDA 稳定性更敏感
- 并行会放大干扰因素

先把串行搜索稳定跑通，再考虑更激进的方案。


### 12.2 每次只调一层

不建议一开始在同一轮搜索里同时混入：

- backbone 深度变化
- generator 学习率变化
- 多个损失权重大范围变化

更好的做法是：

1. 一轮只扫 `uniformity`
2. 下一轮固定最优值，扫 `generator_lr / generator_reg`
3. 再下一轮扫 `alignment / bpr / outer_batch_size`


### 12.3 先看趋势，再看最终最优

自动化搜索最重要的不是“第一次就找到最佳值”，而是尽快看出：

- 哪些参数方向有效
- 哪些参数明显无效
- 哪些参数会带来不稳定

所以早期实验更强调“筛方向”，不是“抠小数点”。


## 13. 一份建议的第一轮 BOD 搜索

如果现在就要开始做第一轮 `BOD` 搜索，我建议：

- base config：`conf/BOD.conf`
- 固定：
  - `num.max.epoch=20`
  - `batch_size=256`
  - `LightGCN.-n_layer=1`
  - `learnRate=0.001`
  - `GM_AU.-outer_loop=1`
  - `GM_AU.-inner_loop=1`
- 第一轮只扫：
  - `GM_AU.-weight_uniformity = [0, 0.02, 0.05, 0.1]`

如果第一轮里某个区间明显最好，再做第二轮：

- 固定最优 `uniformity`
- 扫：
  - `GM_AU.-generator_lr = [1e-4, 5e-4, 1e-3]`
  - `GM_AU.-generator_reg = [1e-5, 1e-4]`


## 14. 推荐工作流

后面建议按这个顺序推进：

1. 写一份 `BOD` 第一轮 spec
2. 先 `dry-run`
3. 正式跑自动化搜索
4. 看 `summary.csv`
5. 选出前几组
6. 再写第二轮 spec
7. 等 `BOD` 定住后，再做 `LightGCN` 搜索


## 15. 结论

当前最重要的不是“把所有参数一下子搜完”，而是：

- 先让搜索过程稳定、可追踪、可恢复
- 再按分层思路逐步缩小参数空间

所以目前最合理的策略是：

- 先用 `scripts/auto_search.py` 做阶段化的 `BOD` 搜索
- 主看 `NDCG@20`
- 从 `uniformity -> generator -> outer loss balance -> backbone strength` 这个顺序推进

后面如果开启新对话，就以这份文档作为统一上下文即可。
