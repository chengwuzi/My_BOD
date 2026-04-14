# BOD

This is the official PyTorch implementation for the paper:
> Zongwei Wang, Min Gao*, Wentao Li*, Junliang Yu, Linxin Guo, Hongzhi Yin. Efficient Bi-Level Optimization for Recommendation Denoising. KDD 2023.

<h2>Requirements</h2>
	
```
numba==0.53.1
numpy==1.20.3
scipy==1.6.2
torch>=1.7.0
```

<h2>Usage</h2>
<ol>
<li>Configure the xx.conf file in the directory named conf. (xx is the name of the model you want to run)</li>
<li>Run main.py and choose the model you want to run.</li>
</ol>

<h2>Acknowledgement</h2>

The implementation is based on the open-source recommendation library [SelfRec](https://github.com/Coder-Yu/SELFRec).

Please cite the following papers as the references if you use our codes.

<h2>当前说明</h2>

当前仓库正在围绕 `iFashion_UB` 场景做后续本地分析与实验整理。

近期调试过程中，我们发现当前 `BOD` 训练逻辑里可能存在一个 shape 广播风险：

- `GraphGenerator_VAE` 输出的权重 shape 为 `[batch_size, 1]`
- 这些权重会直接参与 `bpr_loss_weight` 和 `alignment_loss_weight_1` 中的一维逐样本 score / loss 计算
- 在 PyTorch 广播机制下，这一步有可能从预期的 `[batch_size]` 逐样本加权，扩展成 `[batch_size, batch_size]` 的二维广播计算

目前观察到的现象：

- 该问题不会每次都立刻触发崩溃
- 有些实验可以正常跑完，而且指标依然较高
- 也有一些实验会报 `RuntimeError: CUDA error: an illegal memory access was encountered`
- 报错出现的 epoch 不稳定：有时在前几个 epoch，有时在十几个 epoch 之后

当前阶段的处理决定：

- 为了保证复现连续性，服务器正在运行的实验仍先保持 BOD 作者原始实现不变
- 等原始版本实验跑完后，再视情况决定是否单独跑一个 bug 修复版本，作为对照实验 / ablation
- 因此，当前分支上的历史结果与正在产生的结果，应理解为“原始实现行为”的结果，而不是 bug 修复后的结果






LightGCN实验结果：


seed:2024

Recall@10:0.04149008537251189 | NDCG@10:0.046340750642915325 | Recall@20:0.06723353020189972 | NDCG@20:0.05715474652932544








































