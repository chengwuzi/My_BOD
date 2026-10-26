# LightGCN 预训练

在用户—bundle 图上训练 LightGCN，导出原始字符串 ID 到 64 维向量的 `.pt` 字典。已有数据集：iFashion_UB、Youshu、NetEase；新增数据集：MealRec+H、MealRec+L。

## MealRec 数据准备

在项目根目录运行：

```powershell
python scripts/prepare_multicbr_datasets.py MealRec+H MealRec+L --source-root Raw_Dataset --separate-validation
```

保留原始文件。在 `dataset/MealRec+H/` 和 `dataset/MealRec+L/` 下生成：

- `train.txt`：只来自原始 `user_bundle_train.txt`。
- `valid.txt`：只来自原始 `user_bundle_tune.txt`。
- `test.txt`：只来自原始 `user_bundle_test.txt`。
- `ratings.txt`：处理后三个集合的并集，仅作记录，不用于训练。
- `preparation.json`：各集合的原始交互数、保留数、过滤数和实体覆盖。

分别过滤验证、测试中未在训练出现的用户或 bundle；不合并 tune/test，不重新划分。集合如有交互重叠则报错。论文应注明过滤规则和过滤数量，特别是较稀疏的 MealRec+L 的实体覆盖限制。

已有数据集仍可使用转换脚本原来的默认模式（合并 tune/test）；新的 MealRec 调参流程应使用上述独立验证集模式。

## 自动分阶段调参

只查看计划，不训练：

```powershell
python scripts/auto_search.py --spec scripts/search_specs/mealrec_staged.json --dry-run
```

正式运行两个数据集（串行）：

```powershell
python scripts/auto_search.py --spec scripts/search_specs/mealrec_staged.json
```

每个数据集执行 13 次训练：

1. 初始配置训练一次。
2. `learnRate` 搜索六个值：初始值乘以 `[0.1, 0.25, 0.5, 2, 4, 10]`。
3. 比较初始实验和六个学习率实验，固定最佳学习率；`reg.lambda` 按其**原始初始值**乘同一组六个倍率搜索。
4. 比较六个正则实验与之前最佳实验，保留验证集表现最好的配置和对应最佳轮次的向量。
5. 直接加载选中的向量，对处理后的最终测试集评估一次，不重新训练、不使用测试指标选参数。

所有实验从头初始化，不沿用上一次的权重。维度、batch、层数、seed、最大 epoch 均固定为各自初始配置的值。

| 数据集 | 初始学习率 | 六个学习率候选 | 初始正则 | 六个正则候选 |
|---|---|---|---|---|
| MealRec+H | 0.002 | 0.0002, 0.0005, 0.001, 0.004, 0.008, 0.02 | 0.00005 | 0.000005, 0.0000125, 0.000025, 0.0001, 0.0002, 0.0005 |
| MealRec+L | 0.002 | 0.0002, 0.0005, 0.001, 0.004, 0.008, 0.02 | 0.0001 | 0.00001, 0.000025, 0.00005, 0.0002, 0.0004, 0.001 |

选优顺序与模型一致：NDCG@20、Recall@20、NDCG@10、Recall@10，按顺序比较；完全相同时保留之前的最佳实验。六个候选都变差时不会强制替换初始值。

MealRec 配置中的 `test.set` 兼容现有模型接口，实际指向 `valid.txt`；`evaluation.split=validation` 标明用途，`final.test.set` 是最终测试集。`main.py` 的选项 4/5 可以单独运行初始配置，仅用于验证集选优，不会执行完整搜索或最终测试。

## 结果与断点续跑

默认输出目录为 `results/search_runs/mealrec_staged/`，两个数据集分别位于 `MealRec_H/`、`MealRec_L/`：

- `00_initial/`、`01_learnRate/`、`02_reg_lambda/`：实验各自的配置、日志、向量和 JSONL/CSV 汇总；CSV 包含学习率、正则、最佳轮次及验证指标。
- `best.conf`：验证集选出的最佳配置，仍然使用验证集作为 `test.set`。
- `best_result.json`：最佳参数、最佳轮次、验证指标、最终测试指标及向量路径。
- `final_test.json`：选定向量在最终测试集上的结果。
- `best_embeddings/MealRec+H-user.pt`、`MealRec+H-bundle.pt`（L 同理）：选中实验的向量副本，供后续使用。

重复运行同一命令会复用成功且向量文件仍存在的实验。失败和超时最多尝试三次；任何候选失败会阻止该数据集进入下一阶段，修复后重新运行即可。签名包含完整有效配置、训练/验证数据内容和模型相关源码，改变这些内容会重新训练。相同选定向量、相同测试内容的最终评估也会复用。

13 次训练仅覆盖既定候选范围，不保证全局最优。最佳轮次落在 epoch 上限时，结果标记 `epoch_limit_reached`，应结合日志判断是否需要增加训练预算。

## 其他入口和验证

原来的单阶段 JSON 搜索规格仍可通过 `scripts/auto_search.py --spec ...` 使用。

依赖：Python、PyTorch、NumPy、SciPy。自动使用 CUDA（可用时），否则使用 CPU。

```powershell
python -m unittest discover -s tests -v
```

测试覆盖阶段间参数传递、验证/测试隔离、初始值保留、失败恢复、配置变化后的缓存失效，以及独立数据转换。
