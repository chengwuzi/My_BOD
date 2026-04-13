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






LightGCN首轮实验结果：


Top 10
Recall:0.03970707934454995
NDCG:0.04516291892099532


Top 20
Recall:0.06488826699608541
NDCG:0.055933278815941734


Top 50
Recall:0.11744872918425435
NDCG:0.07675686700180316








































