
# 二进制函数相似性

## 解决方案：

- InfoNCE loss + In-batch negatives
- Byte + DPCNN
- ASM + DPCNN
- Integer + Textcnn
- ACFG + GIN
- Adjacency matrix + resnet

特征：函数语义信息 + 控制流图

语义特征提取，分别提取函数的字节数据、汇编指令数据、整数数据，分别使用独立的编码器（DPCNN、TextCNN）进行文本表示编码，获取其Embedding表示。

结构特征提取，基于CFG和每个Block内的汇编指令，生成ACFG，使用图神经网络对ACFG进行编码，获取Embedding表示；此外，考虑到相似函数的控制流图的节点顺序也相似，将CFG的邻接矩阵作为输入，使用CNN获取其Embedding表示。

对比学习模型结构：InfoNCE loss + In-batch negatives


[Order Matters: Semantic-Aware Neural Networks for Binary Code Similarity Detection](https://keenlab.tencent.com/en/whitepapers/Ordermatters.pdf)

[Investigating Graph Embedding Methods for Cross-Platform Binary Code Similarity Detection](https://www.mhumbert.com/publications/eurosp22_2.pdf)

[SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf)

# 使用说明


# 实验环境
## 基础环境
python 3.7+

pytorch 1.12.1 cuda 11.3

pyg (避免安装出错，参考官方文档 https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) 

如 torch-1.12.0+cu113
```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```

## 性能
显存消耗 22G