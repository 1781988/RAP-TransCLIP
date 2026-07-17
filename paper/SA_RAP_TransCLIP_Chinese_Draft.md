# SA-RAP-TransCLIP：面向类别先验偏移的自适应门控遥感视觉语言传导推理

> **稿件状态**：面向 ICASSP 2027 的中文研究稿。作者、单位、基金信息和官方模板尚待补充。本文保留了已完成实验的真实结果；新增的自适应门控实验以“待补充”标记，不包含虚构数值。

**作者1，作者2，作者3**  
单位，城市，国家  
email1@example.com，email2@example.com

---

## 摘要

遥感视觉语言模型能够通过类别文本直接完成零样本场景分类，而传导推理进一步利用无标签目标集合中的聚类结构修正逐样本预测。现有遥感传导方法通常在完整数据集上评测，隐含假设所有候选类别均出现在目标集合中，并且类别比例不会发生严重偏移。我们首先复现实验并发现：基于类别相关提示、活跃类别先验和可靠图传播的 RAP-TransCLIP 在严重类别先验偏移下具有明显优势，但在完整类别协议中平均性能低于 RS-TransCLIP。该现象表明，先验修正并非始终有益，关键问题是如何仅在目标集合确实发生显著类别先验偏移时启用适配。为此，本文提出 **Shift-Aware RAP-TransCLIP（SA-RAP-TransCLIP）**。该方法从无标签零样本预测中联合估计有效类别压缩、类别先验偏离和类别证据缺失，并生成连续的批次级偏移门控；门控用于在原始 RS-TransCLIP 与先验自适应 RAP-TransCLIP 之间进行保守选择或软融合。为降低提示可靠性计算的显存开销，本文进一步采用分块充分统计，避免同时存储全部提示—样本—类别概率。现有实验表明，RAP-TransCLIP 在仅有 25% 候选类别实际出现时相较 RS-TransCLIP 提升 6.49 个百分点，在 Dirichlet 浓度为 0.1 的强长尾条件下提升 5.46 个百分点，但在完整类别协议下下降 1.99 个百分点。SA-RAP-TransCLIP 的实验目标是在保留前述严重偏移收益的同时，将完整类别性能恢复到 RS-TransCLIP 水平。本文围绕偏移检测准确性、门控安全性、严重偏移收益、跨骨干泛化和计算效率设计系统评测。

**关键词：** 遥感场景分类；视觉语言模型；零样本学习；传导推理；类别先验偏移；测试时适配；自适应门控

---

## 1 引言

视觉语言模型通过大规模图文对比学习建立视觉内容与自然语言之间的共享表示空间，使类别名称和自然语言描述能够直接构造零样本分类器 [1]。RemoteCLIP、GeoRSCLIP 和 SkyCLIP 等工作进一步利用遥感图文数据进行领域化预训练，从而缓解俯视视角、尺度变化和地理语义差异带来的迁移困难 [2–4]。然而，标准零样本分类仍然逐图像独立推理，没有利用同一无标签目标集合内部的类别比例、局部邻域和聚类结构。

TransCLIP 将视觉语言模型的测试集预测写成传导优化问题，通过联合估计目标分配和类别原型改善零样本识别 [5]。RS-TransCLIP 将该思想用于遥感场景分类，在文本伪标签、高斯混合模型和图传播之间进行交替更新，并在十个遥感数据集上取得较大增益 [6]。这类方法通常以完整测试数据集作为一个传导集合，因此每个候选类别都具有一定数量的目标样本。实际遥感任务不一定满足该条件：区域巡检可能只包含港口、机场和工业区等少数类别，灾害评估批次可能呈现显著长尾，连续到达的数据流还可能在不同时间段具有不同的类别支持。

当真实活跃类别少于候选类别时，均匀类别先验会向缺失类别持续分配概率质量；当类别频率高度不均衡时，少数类也可能在全局分配和图传播中被高频类别覆盖。近期视觉语言测试时适配研究已经证明，完整类别和独立同分布假设会显著高估方法的部署稳健性 [7]。Bayesian Class Adaptation 进一步指出，类别条件似然与类别先验应当联合适配 [8]。另一方面，类别相关的无训练提示重加权表明，同一提示模板对不同类别的有效性并不一致 [9]。这些研究说明，类别先验、提示语义和目标结构均可能影响传导推理，但也意味着单独采用其中任一机制不足以形成明确的新颖性。

我们此前构建了 RAP-TransCLIP，将类别相关提示权重、活跃类别先验和可靠图传播耦合到 RS-TransCLIP 的嵌入空间求解器。已完成实验揭示了一个关键边界：RAP-TransCLIP 在强偏移条件下有效，但在完整类别条件下平均退化。具体而言，当仅有 25% 候选类别出现时，RAP-TransCLIP 比 RS-TransCLIP 提高 6.49 个百分点；在 Dirichlet \(\alpha=0.1\) 的强长尾协议下提高 5.46 个百分点；但在完整类别协议下从 78.83% 降至 76.84%。该结果表明，问题并不是继续增强先验修正强度，而是判断何时应该适配、何时应保留原始求解器。

为此，本文提出 SA-RAP-TransCLIP。该方法将 RS-TransCLIP 视为完整类别或弱偏移专家，将 RAP-TransCLIP 视为强先验偏移专家，并通过仅依赖无标签目标预测的偏移估计器生成门控系数。与始终启用类别先验修正不同，SA-RAP-TransCLIP 在估计偏移较小时直接退化为 RS-TransCLIP，在偏移较大时启用 RAP-TransCLIP，在不确定区域对两个专家的概率分配进行连续融合。

本文的主要贡献如下。

1. 提出一种面向遥感传导推理的无监督类别先验偏移估计方法，将有效类别压缩、先验偏离、类别证据缺失和整体置信度统一为批次级偏移分数。
2. 提出双专家自适应门控求解器，在不访问目标标签的条件下选择或融合 RS-TransCLIP 与 RAP-TransCLIP，直接针对“强偏移有效、完整类别退化”的实验矛盾。
3. 将提示可靠性计算改写为分块充分统计，避免显式保存 \(M\times N\times K\) 的全部提示概率张量，降低峰值显存。
4. 建立以严重偏移收益和完整类别安全性为双重目标的实验体系，并通过跨数据集、跨骨干、门控敏感性和偏移条件消融验证方法的适用范围。

<!-- FIGURE 1 PLACEHOLDER -->
![图1 SA-RAP-TransCLIP总体框架](figures/fig1_sa_rap_framework.pdf)

**图1绘制说明。** 图像建议分为三部分。左侧为冻结图像编码器和文本编码器，输出图像特征、106个提示模板的文本特征及等权文本原型。中间上支路为 RS-TransCLIP 专家，下支路为 RAP-TransCLIP 专家；RAP 支路包含类别相关提示、活跃类别先验和可靠互近邻图。中间中央放置无监督偏移估计器，其输入为等权零样本概率，输出偏移分数 \(\delta\) 和门控 \(\lambda\)。右侧显示三种行为：\(\lambda\) 很小时仅执行 RS，\(\lambda\) 很大时仅执行 RAP，中间区域进行概率软融合。图下方用类别频率柱状图示意完整类别、部分类别和长尾三种批次。

---

## 2 相关工作

### 2.1 遥感视觉语言模型

CLIP 通过开放规模图文对比学习获得可迁移的视觉语义表示 [1]。RemoteCLIP [2]、RS5M/GeoRSCLIP [3] 和 SkyScript/SkyCLIP [4] 分别从遥感图文数据构建、领域预训练和语义多样性等角度增强遥感零样本能力。常见零样本分类器将多个提示模板生成的文本嵌入等权平均，从而降低单一提示表达的偶然性。已有无标签提示加权方法表明，不同模板具有不同的全局可靠性 [10]；CARPRT 进一步建模提示与类别之间的相关性 [9]。因此，本文不将类别相关提示权重单独视为核心创新，而将其作为先验自适应专家中的语义修正组件。

### 2.2 传导推理与现实测试时适配

TransCLIP 通过目标特征分布和文本约束联合更新视觉语言模型的零样本预测 [5]。RS-TransCLIP 将图像亲和关系引入遥感传导分类，并使用共享对角协方差的高斯混合模型进行交替推理 [6]。StatA 系统研究了有效类别数量变化和非独立同分布批次，证明多种传导或测试时适配方法会牺牲原始零样本鲁棒性 [7]。BCA 从贝叶斯角度同时更新类别条件表示和类别先验 [8]。这些工作分别覆盖现实协议、先验更新和统计锚定，但没有直接处理遥感传导求解器在完整类别与强先验偏移之间的条件性选择。

### 2.3 选择性适配与安全门控

测试时适配的主要风险是错误伪标签引起的适配漂移。已有方法通过置信样本筛选、统计锚、动态缓存或不确定性融合降低该风险 [7,8,11]。本文采用不同的问题视角：不对每个样本独立决定是否适配，而是判断整个目标集合是否呈现足以启用类别先验修正的批次级证据。该设计与传导推理的集合建模方式一致，并允许在低偏移条件下完全保留原始 RS-TransCLIP。

---

## 3 方法

### 3.1 问题定义与双专家结构

设无标签目标集合包含 \(N\) 幅图像，冻结图像编码器产生归一化特征

\[
F=[f_1,\ldots,f_N]^\top\in\mathbb{R}^{N\times d},
\qquad \|f_i\|_2=1.
\tag{1}
\]

对于 \(M\) 个提示模板和 \(K\) 个候选类别，冻结文本编码器产生 \(t_{m,k}\in\mathbb{R}^{d}\)。RS-TransCLIP 对每个类别的提示嵌入等权平均，并在高斯似然、图传播和文本锚之间更新软分配 \(Z^{\mathrm{RS}}\)。RAP-TransCLIP 进一步估计类别相关提示权重、非均匀类别先验和可靠图，得到 \(Z^{\mathrm{RAP}}\)。

SA-RAP-TransCLIP 不再假设 RAP 更新始终优于原始求解器，而是通过无监督门控 \(\lambda\in[0,1]\) 决定两个专家的使用方式。

### 3.2 类别相关提示与活跃先验专家

第 \(m\) 个提示模板对图像 \(i\) 的零样本类别概率为

\[
p_i^{(m)}
=
\operatorname{softmax}
\left(
s f_iT_m^\top
\right),
\tag{2}
\]

其中 \(s=100\)。对提示—类别对 \((m,k)\)，选取该类别响应最高的

\[
L=\max\left(8,\left\lceil0.02N\right\rceil\right)
\]

个样本作为支持集，并根据平均置信度 \(C\)、归一化熵 \(H\)、类别间隔 \(D\) 和与当前传导分配的一致性 \(A\) 计算

\[
r_{m,k}
=
\beta_c C_{m,k}
-\beta_h H_{m,k}
+\beta_d D_{m,k}
+\beta_a A_{m,k}.
\tag{3}
\]

提示权重和类别文本原型为

\[
\alpha_{m,k}
=
\frac{\exp(r_{m,k}/\tau_p)}
{\sum_{q=1}^{M}\exp(r_{q,k}/\tau_p)},
\tag{4}
\]

\[
\bar t_k
=
\operatorname{norm}
\left(
\sum_{m=1}^{M}\alpha_{m,k}t_{m,k}
\right).
\tag{5}
\]

由加权文本原型得到初始分配 \(\hat Y\)。RAP 专家根据 \(\hat Y\) 估计样本可靠性，并建立可靠性加权互近邻图。当前软分配的类别计数经过 Dirichlet 平滑和文本先验锚定后形成活跃类别先验 \(\pi\)。其分配更新写为

\[
z_i
\leftarrow
\operatorname{softmax}
\left(
\frac{\ell_i}{T_\ell}
+\log\pi
+\eta_g(WZ)_i
+\eta_t\log\hat y_i
\right),
\tag{6}
\]

其中 \(\ell_i\) 为共享对角协方差高斯模型的类别对数似然。

### 3.3 无监督批次偏移估计

使用等权文本原型得到零样本概率 \(P\in\mathbb{R}^{N\times K}\)，并计算目标预测先验

\[
\pi^{(0)}
=
\frac{1}{N}\sum_{i=1}^{N}P_i.
\tag{7}
\]

仅使用 \(\pi^{(0)}\) 的不均衡程度难以区分类别缺失与模型整体不确定，因此本文联合使用三类互补统计量。

**有效类别压缩。** 由预测先验熵定义归一化有效类别比例

\[
r_{\mathrm{eff}}
=
\frac{\exp(H(\pi^{(0)}))}{K},
\qquad
s_{\mathrm{eff}}=1-r_{\mathrm{eff}}.
\tag{8}
\]

当概率质量集中于少量类别时，\(s_{\mathrm{eff}}\) 增大。

**先验偏离。** 计算预测先验与均匀候选先验 \(u_k=1/K\) 之间的归一化 KL 散度：

\[
s_{\mathrm{KL}}
=
\frac{D_{\mathrm{KL}}(\pi^{(0)}\|u)}
{\log K}.
\tag{9}
\]

**类别证据缺失。** 对每个类别同时考察平均概率质量和样本级峰值响应：

\[
a_k
=
\sigma
\left(
\frac{\max_i P_{i,k}-\theta_p}{T_e}
\right)
\sigma
\left(
\frac{\pi_k^{(0)}-\theta_m/K}{T_e}
\right),
\tag{10}
\]

\[
s_{\mathrm{miss}}
=
1-\frac{1}{K}\sum_{k=1}^{K}a_k.
\tag{11}
\]

该项用于区分“类别比例较小但仍存在证据”和“候选类别在目标集合中基本没有支持”。

为防止在零样本模型整体失效时错误触发适配，使用平均最大概率构造置信保护项

\[
q_c
=
\sigma
\left(
\frac{\frac{1}{N}\sum_i\max_k P_{i,k}-\theta_c}
{T_c}
\right).
\tag{12}
\]

最终偏移分数与门控为

\[
\delta
=
q_c
\left(
w_e s_{\mathrm{eff}}
+w_k s_{\mathrm{KL}}
+w_m s_{\mathrm{miss}}
\right),
\tag{13}
\]

\[
\lambda
=
\sigma
\left(
\frac{\delta-\tau_g}{T_g}
\right).
\tag{14}
\]

所有权重和阈值在全部数据集上固定，不使用目标标签进行数据集级调参。

### 3.4 选择性执行与软融合

当 \(\lambda\leq\epsilon_{\mathrm{low}}\) 时，目标集合被判断为弱偏移，算法仅执行 RS-TransCLIP；当 \(\lambda\geq\epsilon_{\mathrm{high}}\) 时，仅执行 RAP-TransCLIP；在中间区域同时执行两个专家，并进行凸组合：

\[
Z^{\ast}
=
(1-\lambda)Z^{\mathrm{RS}}
+
\lambda Z^{\mathrm{RAP}}.
\tag{15}
\]

最终对每行重新归一化。该设计具有两项直接作用：其一，低偏移批次能够严格退化为原始 RS-TransCLIP，避免无条件先验更新造成的完整类别退化；其二，高偏移批次保留 RAP-TransCLIP 已经验证的先验修正能力。

<!-- FIGURE 2 PLACEHOLDER -->
![图2 无监督偏移估计与双专家门控](figures/fig2_shift_gate.pdf)

**图2绘制说明。** 左侧展示零样本概率矩阵 \(P\)。中间分别计算有效类别比例、均匀先验 KL 散度、类别证据支持和平均置信度；四个统计量汇合得到 \(\delta\) 与 \(\lambda\)。右侧使用三段式示意：RS-only、soft fusion 和 RAP-only。建议在图下方绘制三种典型概率直方图，分别对应完整类别、部分缺失和强长尾。

### 3.5 分块提示可靠性计算

原实现一次性构造全部提示概率张量，空间复杂度为

\[
\mathcal{O}(MNK).
\]

当提示数、图像数和类别数同时较大时，该张量是 RAP-TransCLIP 峰值显存的主要来源。本文将提示模板划分为大小为 \(C\) 的块，仅保留当前块的概率，并累计提示可靠性所需的充分统计量。样本级提示分歧通过在线一阶矩与二阶矩计算，不再保存全部提示概率。峰值空间复杂度降低为

\[
\mathcal{O}(CNK+MK),
\qquad C\ll M.
\tag{16}
\]

该修改不改变提示可靠性定义，仅改变计算顺序。实验中默认 \(C=4\)。

---

## 4 实验设计

### 4.1 数据集、骨干与评价指标

实验使用 AID [12]、EuroSAT [13]、MLRSNet [14]、OPTIMAL31 [15]、PatternNet [16]、RESISC45 [17]、RSC11 [18]、RSICB128、RSICB256 [19] 和 WHURS19 [20]。主要偏移实验固定使用 GeoRSCLIP ViT-L/14，以便与已完成实验保持一致。跨骨干实验优先选择 CLIP ViT-L/14、RemoteCLIP ViT-L/14、GeoRSCLIP ViT-L/14 和 SkyCLIP50 ViT-L/14；完整 11 组架构矩阵作为扩展实验。

比较方法包括：

- 等权提示零样本 VLM；
- 复现的 RS-TransCLIP；
- RAP-TransCLIP；
- SA-RAP-TransCLIP；
- StatA [7]；
- BCA [8]；
- CARPRT [9] 与 RS-TransCLIP 的组合基线；
- 原始 TransCLIP [5]。

评价指标包括 Top-1、Macro-F1、ECE、求解时间和峰值显存。新增报告门控分数与真实偏移强度之间的 Spearman 相关系数、完整类别误触发率、严重偏移漏触发率，以及 SA-RAP 相对两个专家中较优者的后悔值。

所有偏移协议使用随机种子 1、2 和 3。主要比较采用数据集—协议—随机种子配对的 Wilcoxon 符号秩检验，并使用成对 bootstrap 计算 95% 置信区间。多组方法比较使用 Holm–Bonferroni 校正。最终论文同时报告均值、标准差、效应量和胜负次数。

### 4.2 研究问题

实验围绕以下问题展开。

- **RQ1：** SA-RAP 是否保留 RAP 在严重类别先验偏移下的收益？
- **RQ2：** SA-RAP 是否能够避免 RAP 在完整类别和轻度偏移下的退化？
- **RQ3：** 无监督偏移分数是否与真实类别支持变化和长尾强度单调相关？
- **RQ4：** 提示、先验、图和门控分别贡献多少？
- **RQ5：** 分块提示统计能否降低峰值显存而不改变预测结果？
- **RQ6：** 方法是否跨视觉语言骨干保持相同的条件性规律？

---

## 5 已完成结果与后续验证

### 5.1 严重类别先验偏移

表1保留已完成的 RS-TransCLIP 与 RAP-TransCLIP 结果，并为 SA-RAP 留出待补充列。现有结果已经证明先验自适应专家在严重偏移下具有使用价值，因此 SA-RAP 的首要目标不是进一步追求更大增益，而是以尽可能小的损失保留该优势。

**表1  GeoRSCLIP ViT-L/14 在十个数据集和三个随机种子下的类别先验偏移结果。**

| 目标协议 | RS-TransCLIP | RAP-TransCLIP | SA-RAP-TransCLIP | RAP-RS | SA-RAP-RS |
|---|---:|---:|---:|---:|---:|
| 部分类别，25% | 60.33 | **66.82** | 待补充 | +6.49 | 待补充 |
| 部分类别，50% | 67.63 | **70.90** | 待补充 | +3.27 | 待补充 |
| 部分类别，75% | 73.53 | 74.40 | 待补充 | +0.87 | 待补充 |
| 长尾，\(\alpha=0.1\) | 62.40 | **67.86** | 待补充 | +5.46 | 待补充 |
| 长尾，\(\alpha=0.5\) | 74.77 | 75.77 | 待补充 | +1.00 | 待补充 |
| 长尾，\(\alpha=1.0\) | **79.28** | 77.63 | 待补充 | -1.65 | 待补充 |

SA-RAP 的预期判据为：在 25% 部分类别和 \(\alpha=0.1\) 下，平均性能不低于 RAP-TransCLIP 1 个百分点以上；在 75% 部分类别和 \(\alpha=1.0\) 下，不低于 RS-TransCLIP。若无法同时满足这两个条件，需要重新设计门控统计或阈值，而不能仅通过选择有利协议形成结论。

<!-- FIGURE 3 PLACEHOLDER -->
![图3 性能增益与偏移强度的关系](figures/fig3_gain_vs_shift.pdf)

**图3绘制说明。** 横轴为部分类别比例或 Dirichlet 浓度，纵轴分别绘制 RAP-RS 和 SA-RAP-RS 的 Top-1 差值及 95% 置信区间。若门控有效，SA-RAP 曲线应在强偏移区接近 RAP，在弱偏移区接近零。

### 5.2 完整类别安全性

完整类别结果是门控设计是否成立的关键验证。现有 RAP-TransCLIP 平均准确率为 76.84%，低于 RS-TransCLIP 的 78.83%。SA-RAP 必须将该差距显著缩小，否则新增门控没有解决论文当前最主要的问题。

**表2  GeoRSCLIP ViT-L/14 在完整类别协议下的 Top-1 准确率（%）。**

| 方法 | AID | EuroSAT | MLRSNet | OPTIMAL31 | PatternNet | RESISC45 | RSC11 | RSICB128 | RSICB256 | WHURS19 | 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 零样本 | 74.42 | 64.13 | 66.78 | 83.66 | 77.36 | 73.77 | 75.00 | 33.66 | 52.30 | 88.46 | 68.95 |
| RS-TransCLIP | 80.63 | 59.13 | **74.45** | **93.60** | **95.63** | **86.75** | 81.74 | **52.81** | **64.61** | **98.91** | **78.83** |
| RAP-TransCLIP | **81.06** | 58.44 | 70.21 | 91.02 | 91.43 | 83.23 | **86.93** | 45.73 | 61.42 | **98.91** | 76.84 |
| SA-RAP-TransCLIP | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

除平均准确率外，还需要报告完整类别批次中 \(\lambda\) 的均值、标准差和高于 0.5 的比例。理想情况下，绝大多数完整类别数据集应处于 RS-only 或低权重融合区域。对仍然错误触发 RAP 的数据集，应分析其零样本类别证据、数据集天然不均衡程度和类别混淆结构。

### 5.3 偏移条件组件消融

此前消融仅在完整类别协议下进行，无法解释严重偏移增益的来源。新的主要消融应在 25% 部分类别和 \(\alpha=0.1\) 两个协议上完成。

**表3  严重偏移条件下的组件消融，结果为十数据集三随机种子均值。**

| 配置 | 类别提示 | 活跃先验 | 可靠图 | 偏移门控 | Partial-25 Top-1 | LT-0.1 Top-1 |
|---|:---:|:---:|:---:|:---:|---:|---:|
| RS-TransCLIP |  |  |  |  | 待补充 | 待补充 |
| 仅活跃先验 |  | ✓ |  |  | 待补充 | 待补充 |
| 仅类别提示 | ✓ |  |  |  | 待补充 | 待补充 |
| 仅可靠图 |  |  | ✓ |  | 待补充 | 待补充 |
| 提示 + 先验 | ✓ | ✓ |  |  | 待补充 | 待补充 |
| 先验 + 可靠图 |  | ✓ | ✓ |  | 待补充 | 待补充 |
| 完整 RAP | ✓ | ✓ | ✓ |  | 待补充 | 待补充 |
| SA-RAP | ✓ | ✓ | ✓ | ✓ | 待补充 | 待补充 |

该表决定论文能否证明三个自适应模块之间存在互补关系。若某一单独模块已经达到完整 RAP 的结果，应简化方法而不是保留不必要组件。

### 5.4 门控质量与敏感性

门控实验需要将模型内部偏移分数与协议真实强度分开报告。真实类别比例仅用于离线诊断，不参与门控计算。

**表4  无监督门控诊断。**

| 协议 | 平均偏移分数 \(\delta\) | 平均门控 \(\lambda\) | RS-only 比例 | RAP-only 比例 | 软融合比例 |
|---|---:|---:|---:|---:|---:|
| 完整类别 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 部分类别 75% | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 部分类别 50% | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 部分类别 25% | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 长尾 \(\alpha=1.0\) | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 长尾 \(\alpha=0.1\) | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

门控阈值 \(\tau_g\) 和温度 \(T_g\) 应进行二维敏感性分析。正文只保留一幅热力图，完整数值放入补充材料。阈值不能按数据集单独选择；推荐固定一个全局配置，并报告在 \(\tau_g\in\{0.10,0.15,0.20,0.25,0.30\}\)、\(T_g\in\{0.02,0.04,0.08\}\) 下的稳定区间。

<!-- FIGURE 4 PLACEHOLDER -->
![图4 门控阈值和温度敏感性](figures/fig4_gate_sensitivity.pdf)

**图4绘制说明。** 左图为完整类别平均准确率热力图，右图为两种严重偏移协议平均准确率热力图。另用等高线或标记显示“完整类别不低于 RS 0.5 个百分点、严重偏移不低于 RAP 1 个百分点”的可接受区域。

### 5.5 计算效率

现有完整矩阵中，RAP-TransCLIP 平均峰值显存为 2008.7 MB，RS-TransCLIP 为 688.7 MB。新的分块实现应首先比较预测一致性，再比较显存和时间。

**表5  提示可靠性实现和双专家门控的效率结果。**

| 方法 | 提示块大小 | 求解时间（s） | 峰值显存（MB） | 相对原 RAP 预测一致率 |
|---|---:|---:|---:|---:|
| 原始 RAP | 全部 | 35.95 | 2008.7 | 100% |
| 分块 RAP | 16 | 待补充 | 待补充 | 待补充 |
| 分块 RAP | 8 | 待补充 | 待补充 | 待补充 |
| 分块 RAP | 4 | 待补充 | 待补充 | 待补充 |
| SA-RAP | 4 | 待补充 | 待补充 | — |

由于 SA-RAP 在中间门控区域可能执行两个专家，其平均时间不一定低于 RAP。效率优势应主要来自两点：低偏移批次直接跳过 RAP，以及提示概率分块降低峰值显存。需要分别报告三种执行分支的时间，而不能只给整体平均值。

### 5.6 跨骨干泛化

跨骨干实验不应一开始重复全部 11 种配置。建议先在四个 ViT-L/14 骨干上完成完整类别、Partial-25 和 LT-0.1 三个协议。只有当 SA-RAP 在至少三个骨干上同时满足完整类别安全性和严重偏移收益后，再扩展到 RN50、ViT-B/32 和 ViT-H/14。该顺序能够减少资源消耗，并避免在门控尚未稳定时运行大规模低价值矩阵。

---

## 6 讨论

RAP-TransCLIP 的现有结果揭示了类别先验适配的双重作用。缺失类别和强长尾使均匀混合假设失效，先验更新能够释放被错误分配给缺失类别的概率质量；但当候选类别得到充分支持时，同一机制可能压低困难类别并放大早期伪标签偏差。因此，适配强度必须由目标集合本身决定。

SA-RAP-TransCLIP 的创新重点不在于再次组合提示、先验和图结构，而在于将“是否启用先验适配”建模为一个可观察、可诊断和可验证的无监督决策问题。门控分数由集合级统计量构成，并与传导推理的批次建模方式一致。若实验能够证明该门控在完整类别条件下保留 RS 性能、在严重偏移条件下接近 RAP 性能，那么论文的贡献将从条件性负结果分析提升为具有明确安全机制的自适应求解器。

该方法仍有三项限制。第一，门控基于初始零样本概率，当视觉语言模型对目标域整体失配时，偏移估计也可能失真。第二，双专家软融合在中间区域增加计算时间。第三，当前偏移协议由真实标签离线构造，虽然标签不参与推理，但仍需要进一步评估自然时间流和地理区域批次。后续可研究无需运行双专家的参数级门控、在线偏移累积以及面向开放集类别的扩展。

---

## 7 结论

本文重新审视了遥感视觉语言传导推理中的完整类别假设。已完成结果表明，RAP-TransCLIP 在严重类别先验偏移下相较 RS-TransCLIP 提升 5–6 个百分点，但在完整类别和轻度偏移下存在退化。基于这一实验边界，本文提出 SA-RAP-TransCLIP，通过无标签批次统计估计类别先验偏移，并在 RS-TransCLIP 与 RAP-TransCLIP 之间进行选择或软融合。与此同时，分块提示充分统计用于降低提示可靠性计算的峰值显存。后续实验将重点验证两个目标：严重偏移性能接近 RAP，以及完整类别性能接近 RS。只有同时满足二者，才能证明自适应门控真正解决了当前方法的核心矛盾。

---

## 参考文献

[1] A. Radford et al., “Learning transferable visual models from natural language supervision,” in *Proc. ICML*, 2021.

[2] F. Liu et al., “RemoteCLIP: A vision language foundation model for remote sensing,” *IEEE Transactions on Geoscience and Remote Sensing*, 2024.

[3] Z. Zhang et al., “RS5M and GeoRSCLIP: A large-scale vision-language dataset and a large vision-language model for remote sensing,” *IEEE Transactions on Geoscience and Remote Sensing*, 2024.

[4] Z. Wang et al., “SkyScript: A large and semantically diverse vision-language dataset for remote sensing,” in *Proc. AAAI*, 2024.

[5] M. Zanella, B. Gérin, and I. Ben Ayed, “Boosting vision-language models with transduction,” in *Advances in Neural Information Processing Systems*, vol. 37, 2024.

[6] K. El Khoury et al., “Enhancing remote sensing vision-language models for zero-shot scene classification,” in *Proc. IEEE ICASSP*, 2025.

[7] M. Zanella et al., “Realistic test-time adaptation of vision-language models,” in *Proc. IEEE/CVF CVPR*, 2025.

[8] L. Zhou et al., “Bayesian test-time adaptation for vision-language models,” in *Proc. IEEE/CVF CVPR*, 2025.

[9] R. Dong et al., “CARPRT: Class-aware zero-shot prompt reweighting for black-box vision-language models,” arXiv preprint arXiv:2607.14125, 2026.

[10] J. U. Allingham et al., “A simple zero-shot prompt weighting technique to improve prompt ensembling in text-image models,” arXiv preprint arXiv:2302.06235, 2023.

[11] L. Zhou et al., “Bayesian test-time adaptation for object recognition and detection with vision-language models,” arXiv preprint arXiv:2510.02750, 2025.

[12] G.-S. Xia et al., “AID: A benchmark data set for performance evaluation of aerial scene classification,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 55, no. 7, pp. 3965–3981, 2017.

[13] P. Helber et al., “Introducing EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification,” in *Proc. IGARSS*, 2018.

[14] X. Qi et al., “MLRSNet: A multi-label high spatial resolution remote sensing dataset for semantic scene understanding,” *ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 169, pp. 337–350, 2020.

[15] Q. Wang et al., “Scene classification with recurrent attention of VHR remote sensing images,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 57, no. 2, pp. 1155–1167, 2019.

[16] W. Zhou et al., “PatternNet: A benchmark dataset for performance evaluation of remote sensing image retrieval,” *ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 145, pp. 197–209, 2018.

[17] G. Cheng, J. Han, and X. Lu, “Remote sensing image scene classification: Benchmark and state of the art,” *Proceedings of the IEEE*, vol. 105, no. 10, pp. 1865–1883, 2017.

[18] L. Zhao et al., “Feature significance-based multi-bag-of-visual-words model for remote sensing image scene classification,” *Journal of Applied Remote Sensing*, vol. 10, 2016.

[19] H. Li et al., “RSI-CB: A large-scale remote sensing image classification benchmark using crowdsourced data,” *Sensors*, vol. 20, no. 6, 2020.

[20] G.-S. Xia et al., “Structural high-resolution satellite image indexing,” *International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, vol. 38, 2010.

---

## 作者侧实验清单（投稿前删除）

1. 运行 SA-RAP 的完整类别、六种偏移协议和三个随机种子实验。
2. 在 Partial-25 和 LT-0.1 下完成八组组件消融。
3. 运行门控阈值与温度敏感性实验，并固定全局参数。
4. 对比 StatA、BCA、CARPRT+RS-TransCLIP 和原始 TransCLIP。
5. 验证分块提示计算与原 RAP 的预测一致性，并报告块大小 16、8、4。
6. 在四个 ViT-L/14 骨干上验证完整类别安全性和严重偏移收益。
7. 使用 Wilcoxon 检验、成对 bootstrap 置信区间和 Holm 校正重新计算统计显著性。
8. 绘制门控分数分布、真实活跃类别比例相关性、提示权重和类别先验诊断图。
9. 确认 AID 与 Million-AID Level-2 的数据版本、类别列表和样本数量。
10. 将最终内容迁移到 ICASSP 2027 官方模板，并根据页数限制压缩正文。
