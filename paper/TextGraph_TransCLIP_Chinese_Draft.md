# TextGraph-TransCLIP：文本引导边界保持图传导的遥感零样本场景分类

> **稿件状态说明**：本文为面向 ICASSP 2027 的中文研究草稿。作者、单位、基金信息和官方模板尚待补充。已有实验数值仅用于零样本与 RS-TransCLIP 复现基线；TextGraph-TransCLIP 的结果均保留为“待补充”，未写入未经验证的结论。图片采用 Markdown 路径占位符，替换路径后即可插入正式图片。

**作者1，作者2，作者3**  
单位，城市，国家  
email1@example.com，email2@example.com

---

## 摘要

遥感视觉语言模型能够利用类别文本直接完成零样本场景分类。RS-TransCLIP 进一步通过高斯混合建模和图拉普拉斯正则联合修正无标签测试集合中的预测，但其图结构仅由图像嵌入之间的余弦相似度确定。遥感场景具有显著的纹理复用和尺度混淆，不同语义类别可能在视觉空间中形成高相似邻域，例如密集住宅区与中等住宅区、港口与河流、工业区与商业区。固定视觉图会在这些类别边界上传播错误伪标签，产生图过平滑。本文提出 **TextGraph-TransCLIP**，在保留 RS-TransCLIP 高斯模型、文本锚和闭式更新的基础上，仅重构图传播项。方法首先使用零样本文本后验计算节点间的 Hellinger 语义亲和度，再根据节点预测置信度控制文本信息对视觉边权的抑制强度。置信度较高且语义后验明显不一致的视觉邻边被削弱；预测不确定时，图结构自动退化为原始视觉图，从而避免错误文本先验过度破坏局部结构。该方法不引入训练参数，不更新视觉或文本编码器，也不改变候选类别先验。实验将在十个遥感场景分类数据集和多种遥感视觉语言骨干上验证标准分类性能，并通过跨类别边比例、困难类别对、邻居数量敏感性和计算开销分析图边界保持机制。现有 GeoRSCLIP ViT-L/14 复现结果将作为统一基线，TextGraph-TransCLIP 的数值结果将在完成实验后补充。

**关键词：** 遥感场景分类；视觉语言模型；零样本分类；传导推理；图信号处理；边界保持

---

## 1 引言

视觉语言模型通过大规模图文对比学习获得开放词汇视觉表示，使类别名称和自然语言描述能够直接构造零样本分类器 [1]。RemoteCLIP、GeoRSCLIP 和 SkyCLIP 等方法进一步利用遥感图文数据进行领域预训练，在航空与卫星场景分类中表现出较强迁移能力 [2–4]。然而，常规零样本分类逐图像独立推理，没有利用无标签目标集合内部的邻域关系和聚类结构。

TransCLIP 将测试集预测写成传导优化问题，在目标特征分布和文本伪标签之间联合估计类别分配 [5]。RS-TransCLIP 将该框架用于遥感场景分类，其目标函数由高斯混合似然、图拉普拉斯正则和文本伪标签约束构成 [6]。该方法只需计算一次图像与文本嵌入，随后在嵌入空间中完成推理，因而具有较低的额外开销。

RS-TransCLIP 的关键假设之一是：图像嵌入的局部邻域具有类别同质性。其公开实现使用图像余弦相似度构建固定近邻图，并在图上鼓励相邻样本具有相似分配。该假设对遥感场景并不总是成立。遥感图像中的建筑密度、道路纹理、水体形态和植被覆盖可能跨类别重复出现；同一类别也可能因地理区域、拍摄高度和空间分辨率不同而产生较大类内变化。当视觉近邻跨越真实类别边界时，拉普拉斯平滑会把一个样本的错误伪标签传播给视觉相似但语义不同的样本。

一个直接方案是删除低置信样本或重新估计类别先验，但这些操作同时改变了文本原型、类别质量或整体求解过程，使性能变化难以归因。本文采用更聚焦的设计：保持 RS-TransCLIP 的高斯混合模型、均匀类别先验、106 个提示模板平均文本原型及交替更新不变，只使用文本后验修正视觉图的边导通率。这样能够直接回答一个问题：

> 文本模态能否作为图边界证据，减少视觉图在遥感相似场景之间的错误传播？

为此，本文提出 TextGraph-TransCLIP。对于视觉近邻 \((i,j)\)，方法计算两幅图像零样本类别后验之间的 Hellinger 亲和度。当两个节点均具有较高预测置信度且后验分布明显不一致时，对应视觉边被抑制；当任一节点预测不确定时，文本门控趋近于 1，保留原始视觉边。该设计具有三个特点：第一，只修改图权重，不引入新的类别先验或提示学习模块；第二，文本门控是连续的，不进行不可逆的硬边删除；第三，语义亲和度与视觉亲和度的乘性融合可解释为跨模态边导通率。

本文的贡献如下。

1. 针对遥感视觉嵌入中的跨类别近邻问题，提出文本引导的边界保持图，将零样本文本后验作为视觉图边的语义导通率。
2. 提出置信度控制的回退机制：文本预测可靠时抑制语义冲突边，文本预测不确定时自动恢复为原始视觉图。
3. 在不修改 RS-TransCLIP 其他变量和更新规则的前提下建立可归因实验，结合分类性能、图边纯度、困难类别对和超参数稳定性验证图结构改造的作用。
4. 保持训练自由和稀疏图复杂度，新增计算仅发生在已有视觉近邻边上。

<!-- FIGURE 1 PLACEHOLDER -->
![图1 TextGraph-TransCLIP总体框架](figures/fig1_textgraph_framework.pdf)

**图1绘制说明。** 左侧为冻结的图像编码器和文本编码器。图像编码器输出目标特征 \(F\)，106 个遥感提示模板经平均后得到类别文本原型 \(T\)，二者计算零样本后验 \(\hat Y\)。中间上方构建原始视觉 \(k\)-近邻图 \(W^{v}\)，中间下方根据 \(\hat Y\) 计算 Hellinger 语义亲和度与节点置信度。两部分融合为文本引导图 \(W^{tg}\)。右侧保持 RS-TransCLIP 的高斯似然、文本锚和交替更新不变。图中用红色虚线表示被削弱的跨语义视觉边，用绿色实线表示被保留的同语义边。

---

## 2 相关工作

### 2.1 遥感视觉语言模型

CLIP 通过图文对比学习获得可迁移的开放词汇视觉表示 [1]。RemoteCLIP 使用遥感图文数据进行领域预训练 [2]；RS5M/GeoRSCLIP 扩大遥感图文数据规模并强化地理语义对齐 [3]；SkyScript/SkyCLIP 强调遥感文本描述的规模与语义多样性 [4]。这些方法主要关注预训练数据和跨模态表示，而本文不修改编码器，仅研究冻结嵌入上的传导推理。

### 2.2 视觉语言模型的传导推理

TransCLIP 通过目标特征分布和文本先验联合更新无标签测试样本的预测 [5]。RS-TransCLIP 在遥感场景分类中加入图像亲和关系，并使用共享对角协方差的高斯混合模型 [6]。现实测试时适配研究进一步指出，传导方法可能依赖完整类别和有利批次组成，因此需要报告方法的适用条件 [7]。

图传播已被用于视觉语言模型零样本分类。ZLaP 将文本类别节点和无标签图像节点置于统一图中，并通过测地距离执行标签传播 [8]；后续工作通过动态图扩展和上下文特征重加权提高训练自由传播效率 [9]。COSMIC 则利用文本、CLIP 与 DINOv2 特征构造多语义图，并结合团结构完成测试时适配 [10]。这些方法说明“文本与图结合”本身并非新的研究命题。

TextGraph-TransCLIP 的差异在于，它不重新构造包含文本节点的多模态传播系统，也不增加缓存、提示学习或额外视觉编码器，而是把文本后验解释为 RS-TransCLIP 既有视觉边上的连续导通率。高斯混合模型、文本锚、候选类别先验和交替更新保持不变，因此实验能够直接检验图边界修正是否有效。

### 2.3 图上的边界保持传播

传统图半监督学习通常利用特征相似度构建图，并通过拉普拉斯平滑传播标签。固定二次平滑会在异质边上造成类别边界模糊。边界保持图方法通常根据局部标签分布、预测差异或自适应导通率抑制跨类边。视觉语言模型提供了独立于视觉近邻的文本后验，因此可用于判断一条高视觉相似边是否具有一致语义。本文使用 Hellinger 亲和度而非硬伪标签一致性，保留后验分布中的次优类别与不确定性；同时通过熵置信度保证低质量文本预测退化回原始视觉图。

---

## 3 方法

### 3.1 问题定义与 RS-TransCLIP 基线

设无标签目标集合包含 \(N\) 幅图像，冻结图像编码器得到归一化特征

\[
F=[f_1,\ldots,f_N]^\top\in\mathbb{R}^{N\times d},
\qquad \|f_i\|_2=1.
\tag{1}
\]

对 \(K\) 个候选类别，将 106 个遥感提示模板的文本嵌入平均并归一化，得到

\[
T=[t_1,\ldots,t_K]^\top\in\mathbb{R}^{K\times d}.
\tag{2}
\]

零样本文本后验为

\[
\hat y_i
=
\operatorname{softmax}
\left(
s f_iT^\top
\right),
\tag{3}
\]

其中 \(s\) 为视觉语言模型的相似度尺度。

RS-TransCLIP 使用软分配 \(Z=[z_{i,k}]\)、类别均值 \(\mu_k\) 和共享对角协方差 \(\Sigma\) 建模目标集合。其分配更新可写为

\[
z_i
\leftarrow
\operatorname{softmax}
\left[
\frac{\ell_i}{T_\ell}
+\eta_g(WZ)_i
+\eta_t\log\hat y_i
\right],
\tag{4}
\]

其中 \(\ell_i\) 为高斯类别对数似然，\(W\) 为图像特征构成的归一化亲和图。原始图权重仅依赖视觉余弦相似度：

\[
w^v_{ij}
=
\left[f_i^\top f_j\right]_+,
\qquad j\in\mathcal N_k(i).
\tag{5}
\]

### 3.2 文本后验语义亲和度

对于两个视觉近邻 \(i\) 和 \(j\)，本文使用零样本后验的 Hellinger 亲和度衡量其语义一致性：

\[
a_{ij}
=
\sum_{c=1}^{K}
\sqrt{\hat y_{i,c}\hat y_{j,c}}.
\tag{6}
\]

\(a_{ij}\in[0,1]\)。当两个后验分布接近时，\(a_{ij}\) 接近 1；当概率质量集中于不同类别时，\(a_{ij}\) 减小。与比较 \(\arg\max \hat y_i\) 是否相同相比，式（6）保留了次优类别和预测不确定性。

### 3.3 置信度控制的边导通率

文本后验可能受到类别描述不充分或视觉语言域差异影响。若直接用 \(a_{ij}\) 重构图，错误文本预测会删除有用视觉边。本文根据归一化熵定义节点置信度：

\[
q_i
=
1-
\frac{
-\sum_{c=1}^{K}\hat y_{i,c}\log\hat y_{i,c}
}{
\log K
}.
\tag{7}
\]

节点对置信度为

\[
q_{ij}
=
(q_iq_j)^{\gamma/2},
\tag{8}
\]

其中 \(\gamma\) 控制置信度响应。文本门控因子定义为

\[
g_{ij}
=
1-
\lambda q_{ij}
\left(
1-a_{ij}^{\beta}
\right),
\tag{9}
\]

其中 \(\lambda\in[0,1]\) 为语义门控强度，\(\beta\) 控制对语义分歧的敏感度。

式（9）具有两个边界性质：

- 当任一节点接近均匀预测时，\(q_{ij}\rightarrow0\)，从而 \(g_{ij}\rightarrow1\)，图退化为原始视觉图；
- 当两个节点均高置信但后验分布不一致时，\(q_{ij}\) 较大、\(a_{ij}\) 较小，对应边被显著削弱。

最终文本引导边权为

\[
w^{tg}_{ij}
=
w^v_{ij}g_{ij}.
\tag{10}
\]

本文只在视觉 \(k\)-近邻候选边上计算式（10），不会引入新的全连接边。图经对称化和度归一化后代入式（4）。

### 3.4 交替推理

TextGraph-TransCLIP 与 RS-TransCLIP 使用相同的初始化和高斯更新。类别均值与共享方差分别为

\[
\mu_k
=
\operatorname{norm}
\left(
\frac{\sum_i z_{i,k}f_i}
{\sum_i z_{i,k}}
\right),
\tag{11}
\]

\[
\operatorname{diag}(\Sigma)
=
\frac{
\sum_{i,k}z_{i,k}(f_i-\mu_k)^2
}{
\sum_{i,k}z_{i,k}
}.
\tag{12}
\]

算法最多执行 10 轮外层更新，每轮包含 5 次分配更新。除图 \(W^{tg}\) 外，其余配置与复现的 RS-TransCLIP 完全一致。该控制变量设计使性能变化能够直接归因于文本引导图。

<!-- FIGURE 2 PLACEHOLDER -->
![图2 文本引导边权计算](figures/fig2_edge_conductance.pdf)

**图2绘制说明。** 左侧展示两个视觉高度相似的遥感样本；中间分别绘制其零样本类别概率柱状图。上方示例为同语义节点，Hellinger 亲和度高，边权基本保留；下方示例为高置信语义冲突节点，边导通率降低。最右侧给出置信度较低时回退到视觉边的示意。

---

## 4 实验设计

### 4.1 数据集、模型与实现设置

实验使用 AID [11]、EuroSAT [12]、MLRSNet [13]、OPTIMAL31 [14]、PatternNet [15]、RESISC45 [16]、RSC11 [17]、RSICB128、RSICB256 [18] 和 WHURS19 [19]。第一阶段固定 GeoRSCLIP ViT-L/14，以复用现有图像与文本特征，并与当前复现结果直接比较。第二阶段扩展至 CLIP、RemoteCLIP、GeoRSCLIP 和 SkyCLIP50 的 ViT-L/14；只有在方法稳定后才运行全部 11 个模型—架构组合。

比较方法包括：

1. 等权提示零样本 VLM；
2. 复现的 RS-TransCLIP；
3. 仅将视觉图改为互近邻图的结构基线；
4. 视觉图乘以 Hellinger 语义亲和度但不使用置信度回退；
5. 完整 TextGraph-TransCLIP。

主要指标为 Top-1、Macro-F1、ECE、求解时间和峰值显存。机制分析额外报告无权图边纯度、加权图边纯度和跨类别边总权重。真实标签仅用于离线评价图边，不进入求解器。

默认设置与 RS-TransCLIP 保持一致：\(k=3\)，高斯均值由每类 8 个最高置信样本初始化，共享方差初始化为 \(1/d\)，最大迭代数为 10。TextGraph 参数初始设为 \(\lambda=1\)、\(\beta=1\)、\(\gamma=1\)，并使用统一参数完成所有数据集实验。

### 4.2 研究问题

实验围绕以下问题组织。

- **RQ1：** 文本引导图是否在标准十数据集上稳定提升 RS-TransCLIP？
- **RQ2：** 性能变化是否与跨类别视觉边的削弱相关？
- **RQ3：** 置信度回退是否能避免低质量文本后验对视觉图的破坏？
- **RQ4：** 方法在困难类别对和不同邻居数量下是否更稳健？
- **RQ5：** 文本引导边权是否保持原方法的低额外开销？

### 4.3 标准分类性能

表1保留已完成的 GeoRSCLIP ViT-L/14 基线。TextGraph-TransCLIP 的结果完成实验后填入。

**表1  GeoRSCLIP ViT-L/14 在十个完整类别数据集上的 Top-1 准确率（%）。**

| 方法 | AID | EuroSAT | MLRSNet | OPTIMAL31 | PatternNet | RESISC45 | RSC11 | RSICB128 | RSICB256 | WHURS19 | 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 零样本 VLM | 74.42 | 64.13 | 66.78 | 83.66 | 77.36 | 73.77 | 75.00 | 33.66 | 52.30 | 88.46 | 68.95 |
| RS-TransCLIP | 80.63 | 59.13 | 74.45 | 93.60 | 95.63 | 86.75 | 81.74 | 52.81 | 64.61 | 98.91 | 78.83 |
| TextGraph-TransCLIP | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

本文不以少数数据集上的最大提升作为主要结论。首要判据是十数据集平均性能高于 RS-TransCLIP，并且退化超过 1 个百分点的数据集数量有限。若平均增益不足 0.5 个百分点，应将工作定位为图机制分析而不是性能增强方法；若多数数据集没有改善，则停止扩展跨骨干实验。

### 4.4 图边界质量

为了验证方法是否确实减少跨类别传播，分别对视觉图和 TextGraph 计算：

\[
\operatorname{Purity}_{u}
=
\frac{
\sum_{(i,j)\in E}\mathbb I(y_i=y_j)
}{
|E|
},
\tag{13}
\]

\[
\operatorname{Purity}_{w}
=
\frac{
\sum_{(i,j)\in E}w_{ij}\mathbb I(y_i=y_j)
}{
\sum_{(i,j)\in E}w_{ij}
}.
\tag{14}
\]

式（13）衡量图拓扑中的同类边比例；由于 TextGraph 不改变候选边集合，无权纯度应与视觉图相同。式（14）衡量同类边获得的权重比例，是本文的主要结构指标。还需报告跨类别边权重降低比例及其与准确率增益之间的 Spearman 相关系数。

**表2  图边界诊断结果。**

| 数据集 | 视觉图无权纯度 | 视觉图加权纯度 | TextGraph 加权纯度 | 跨类边权下降 | Top-1 变化 |
|---|---:|---:|---:|---:|---:|
| AID | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| RESISC45 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| RSICB128 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 其余数据集 | 见附录 | 见附录 | 见附录 | 见附录 | 见附录 |

<!-- FIGURE 3 PLACEHOLDER -->
![图3 图边纯度变化与准确率增益](figures/fig3_purity_gain.pdf)

**图3绘制说明。** 横轴为 TextGraph 相对视觉图的加权边纯度提升，纵轴为 TextGraph-TransCLIP 相对 RS-TransCLIP 的 Top-1 变化。每个点对应一个数据集—骨干组合，标注主要异常点并给出 Spearman 相关系数。

### 4.5 困难类别对分析

从 RS-TransCLIP 的混淆矩阵中提取高频互混淆类别对，例如住宅密度类别、港口与河流、工业区与商业区。对每个类别对报告：

- RS-TransCLIP 类别对平均准确率；
- TextGraph-TransCLIP 类别对平均准确率；
- 两类之间的视觉边数；
- TextGraph 对跨类边总权重的削弱比例。

该实验用于证明改进来自边界保持，而不是高斯原型的偶然变化。类别对根据统一的 RS-TransCLIP 基线确定，并在报告中列出全部选择规则，避免只展示有利样例。

<!-- FIGURE 4 PLACEHOLDER -->
![图4 困难类别对与边权可视化](figures/fig4_confusing_pairs.pdf)

**图4绘制说明。** 每组包含两个类别的代表图像、二维 UMAP 嵌入、原始视觉图边和 TextGraph 边。跨类别边用红色表示，边宽对应归一化权重。建议同时展示一个明显改善的类别对和一个没有改善或退化的类别对。

### 4.6 消融与参数稳定性

**表3  TextGraph 组件消融。**

| 配置 | 视觉相似度 | Hellinger 亲和度 | 置信度回退 | Top-1 | 加权边纯度 |
|---|:---:|:---:|:---:|---:|---:|
| RS-TransCLIP | ✓ |  |  | 待补充 | 待补充 |
| 视觉 × 语义 | ✓ | ✓ |  | 待补充 | 待补充 |
| 完整 TextGraph | ✓ | ✓ | ✓ | 待补充 | 待补充 |
| 互近邻 TextGraph | ✓ | ✓ | ✓ | 待补充 | 待补充 |

邻居数量 \(k\in\{3,5,10,20\}\)，语义强度 \(\lambda\in\{0.25,0.5,0.75,1.0\}\)。参数不能按数据集单独选择。正文报告平均曲线和最差数据集变化，完整结果置于补充材料。

### 4.7 计算开销

TextGraph 只对视觉 \(k\)-近邻边计算语义亲和度，额外复杂度为 \(\mathcal O(|E|K)\)。由于 \(|E|\approx kN\) 且 \(k\) 很小，该项通常低于图像特征提取开销。

**表4  推理开销。**

| 方法 | 图构建时间 | 求解时间 | 峰值显存 | 图边数 |
|---|---:|---:|---:|---:|
| RS-TransCLIP | 待补充 | 待补充 | 待补充 | 待补充 |
| TextGraph-TransCLIP | 待补充 | 待补充 | 待补充 | 待补充 |

---

## 5 讨论

本文选择文本引导图，而不继续叠加类别先验、提示可靠性和双求解器门控，原因在于图传播是 RS-TransCLIP 相对于普通 TransCLIP 的主要遥感扩展，也是最容易产生跨类别错误传播的结构位置。保持其余变量不变能够形成明确的因果链：视觉图存在异质边，文本后验提供边界证据，置信度门控抑制高可靠语义冲突边，最终减少图过平滑。

该设计仍有明确限制。第一，当零样本文本后验在某些类别上系统性错误且置信度过高时，TextGraph 可能错误削弱真实同类边。第二，Hellinger 亲和度只利用类别后验，没有利用更细粒度的属性文本。第三，本文仍采用完整测试集合的传导设置，尚未覆盖在线流式场景。上述问题应通过低零样本准确率数据集分析、置信度回退消融和未来在线扩展进行讨论，而不应在当前版本中继续增加新模块。

现阶段已有的 RAP-TransCLIP 类别先验偏移实验不建议并入本文主线。它们可作为独立探索保留在旧实验目录，但不会用于支撑 TextGraph 的创新性。新的论文只回答图边界问题，以避免方法目标和实验协议相互分散。

---

## 6 结论

本文提出 TextGraph-TransCLIP，通过零样本文本后验修正 RS-TransCLIP 的视觉图边权。方法使用 Hellinger 亲和度衡量视觉近邻的语义一致性，并根据节点预测置信度决定文本门控强度。与此前包含多个适配模块的方案相比，TextGraph-TransCLIP 只改变图传播项，保留原始高斯模型、文本锚、均匀先验和交替更新，因此结构更简洁、实验更易归因。后续实验将重点验证标准十数据集性能、图加权纯度、困难类别边界、邻居数量稳定性和额外开销。只有当分类增益与跨类别边权下降之间形成稳定关系时，才能确认文本模态对遥感传导图具有实质性改进作用。

---

## 参考文献

[1] A. Radford et al., “Learning transferable visual models from natural language supervision,” in *Proc. ICML*, 2021.

[2] F. Liu et al., “RemoteCLIP: A vision language foundation model for remote sensing,” *IEEE Transactions on Geoscience and Remote Sensing*, 2024.

[3] Z. Zhang et al., “RS5M and GeoRSCLIP: A large-scale vision-language dataset and a large vision-language model for remote sensing,” *IEEE Transactions on Geoscience and Remote Sensing*, 2024.

[4] Z. Wang et al., “SkyScript: A large and semantically diverse vision-language dataset for remote sensing,” in *Proc. AAAI*, 2024.

[5] M. Zanella, B. Gérin, and I. Ben Ayed, “Boosting vision-language models with transduction,” in *Advances in Neural Information Processing Systems*, vol. 37, 2024.

[6] K. El Khoury et al., “Enhancing remote sensing vision-language models for zero-shot scene classification,” in *Proc. IEEE ICASSP*, 2025.

[7] M. Zanella et al., “Realistic test-time adaptation of vision-language models,” in *Proc. IEEE/CVF CVPR*, 2025.

[8] V. Stojnić, Y. Kalantidis, and G. Tolias, “Label propagation for zero-shot classification with vision-language models,” arXiv preprint arXiv:2404.04072, 2024.

[9] Y. Li, Y. Su, A. Goodge, K. Jia, and X. Xu, “Efficient and context-aware label propagation for zero-/few-shot training-free adaptation of vision-language models,” arXiv preprint arXiv:2412.18303, 2024.

[10] F. Huang et al., “COSMIC: Clique-oriented semantic multi-space integration for robust CLIP test-time adaptation,” in *Proc. IEEE/CVF CVPR*, 2025.

[11] G.-S. Xia et al., “AID: A benchmark data set for performance evaluation of aerial scene classification,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 55, no. 7, pp. 3965–3981, 2017.

[12] P. Helber et al., “Introducing EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification,” in *Proc. IGARSS*, 2018.

[13] X. Qi et al., “MLRSNet: A multi-label high spatial resolution remote sensing dataset for semantic scene understanding,” *ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 169, pp. 337–350, 2020.

[14] Q. Wang et al., “Scene classification with recurrent attention of VHR remote sensing images,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 57, no. 2, pp. 1155–1167, 2019.

[15] W. Zhou et al., “PatternNet: A benchmark dataset for performance evaluation of remote sensing image retrieval,” *ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 145, pp. 197–209, 2018.

[16] G. Cheng, J. Han, and X. Lu, “Remote sensing image scene classification: Benchmark and state of the art,” *Proceedings of the IEEE*, vol. 105, no. 10, pp. 1865–1883, 2017.

[17] L. Zhao et al., “Feature significance-based multi-bag-of-visual-words model for remote sensing image scene classification,” *Journal of Applied Remote Sensing*, vol. 10, 2016.

[18] H. Li et al., “RSI-CB: A large-scale remote sensing image classification benchmark using crowdsourced data,” *Sensors*, vol. 20, no. 6, 2020.

[19] G.-S. Xia et al., “Structural high-resolution satellite image indexing,” *International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, vol. 38, 2010.

---

## 作者侧实验决策清单（正式投稿前删除）

1. 先在 AID、RESISC45、RSICB128 上运行 GeoRSCLIP ViT-L/14 小规模验证。
2. 若三数据集平均不优于 RS-TransCLIP，不运行完整矩阵，先检查加权边纯度。
3. 完成十数据集主实验，并报告所有退化数据集。
4. 运行图边纯度分析，验证准确率变化与跨类别边权下降相关。
5. 完成困难类别对分析，至少包含一个失败案例。
6. 完成 \(\lambda\)、\(k\)、\(\beta\) 和置信度回退消融。
7. 扩展到四个 ViT-L/14 骨干；只有稳定后再运行全部 11 种配置。
8. 报告三次独立运行或说明算法完全确定性。
9. 将现有 RAP/SA-RAP 结果移出正文，避免与本文图边界主线混合。
10. 转入 ICASSP 2027 官方模板，并根据页数限制压缩相关工作和扩展实验。
