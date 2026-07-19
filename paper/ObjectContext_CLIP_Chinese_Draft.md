# ObjectContext-CLIP：上下文锚定的多线索局部目标残差遥感零样本场景分类

> **稿件状态**：面向 ICASSP 2027 的中文研究草稿。当前稿件只描述最终精简方法，不将此前内部探索版本作为对比方法。全部最终结果由 `outputs/results/object_context_core_v1/` 生成后填写。

**作者1，作者2，作者3**  
单位，城市，国家  
email1@example.com，email2@example.com

---

## 摘要

遥感场景通常同时由全局空间布局与局部判别目标定义。整图视觉语言推理能够表达地表结构、空间组织和周边环境，但可能弱化飞机、船舶、储油罐与运动设施等局部线索；简单多裁剪又容易将缺少完整上下文的局部区域直接解释为场景类别。本文提出 ObjectContext-CLIP，一种无需下游训练的上下文锚定局部目标残差推理方法。方法首先使用整图特征、类别名称和场景描述构造全局上下文分类器，再利用确定性局部裁剪和多个类别特定目标/结构短语获得局部目标分数。与固定替换式融合不同，本文始终保留全局上下文作为基础分数，仅将经过边界控制的局部目标残差添加到上下文分数。完成的诊断实验表明，上下文熵门控、候选集合限制和类别一致性融合没有提供稳定增益，因此均从最终方法中删除；最终框架只保留多视图局部聚合、多线索语义表示和上下文锚定残差。所有图像与文本编码器保持冻结，每幅图像独立推理。实验将在十个遥感场景数据集和四种 ViT-L/14 视觉语言骨干上，与整图类别名称、多裁剪、全局上下文、局部目标和固定融合进行比较，并通过残差方向、局部视图、局部线索、概念负对照、分辨率和统计显著性实验验证方法。

**关键词：** 遥感场景分类；视觉语言模型；零样本学习；局部目标；多线索语义；残差推理

---

## 1 引言

视觉语言模型将视觉内容和自然语言映射到共享语义空间，使类别名称或文本描述能够直接构造零样本分类器。RemoteCLIP、GeoRSCLIP 和 SkyCLIP 等遥感视觉语言模型进一步通过遥感图文数据增强领域语义，但常见推理方式仍主要依赖单一整图特征。

遥感场景分类并不等价于单主体识别。机场由跑道、航站区和飞机共同定义；港口同时包含水陆边界、码头、船舶与堆场；工业区、商业区和住宅区可能具有相似建筑纹理，但局部设施及其组织方式不同。整图表示能够保持场景范围，却可能弱化小目标；局部裁剪能够放大目标，却容易丢失完整场景语义。

直接对全局与局部分数进行固定融合存在明显风险。局部分支可能对共享纹理或非判别目标产生较高响应，从而破坏原本正确的全局排序。本文不将局部分支视为独立场景分类器，而将其定义为全局上下文分类器的局部证据补充。

本文围绕一个核心问题展开：**如何在保持全局上下文语义的前提下，将多视图、多线索局部目标证据以轻量残差方式注入零样本分类分数。** 本文只保留一个主要方法贡献：

1. 提出上下文锚定的多线索局部目标残差推理，以全局上下文为基础分类器，并将局部目标优势限制为有界残差修正。

多视图聚合、类别局部概念库和完整实验协议作为该方法的实现与验证组成部分，不拆分为额外创新点。

<!-- FIGURE 1 PLACEHOLDER -->
![图1 ObjectContext-CLIP 总体框架](figures/fig1_object_context_framework.pdf)

**图1绘制说明。** 左侧为整图和两个尺度的确定性局部裁剪；中间上方为类别名称与场景描述，下方为每个类别的多个目标/结构短语；右侧分别得到全局上下文分数和局部目标分数，并通过“Context Anchor + Bounded Local Residual”得到最终预测。图中不再出现不确定性门控、Top-M 候选限制或类别一致性融合。

---

## 2 相关工作

### 2.1 遥感视觉语言模型

CLIP 提供开放词汇视觉表示，RemoteCLIP、GeoRSCLIP 和 SkyCLIP 等工作进一步增强遥感图文对齐。本文不训练或修改预训练编码器，而研究冻结表示上的推理机制。

### 2.2 遥感场景的全局与局部建模

监督遥感分类方法长期使用多尺度卷积、局部注意和双路径网络联合建模全局与局部信息。本文关注无需下游标签和参数更新的视觉语言推理，并避免将局部区域直接解释为完整场景。

### 2.3 多裁剪与语言知识增强

多裁剪测试能够增加局部观察，但通常仍将每个裁剪与完整类别名称进行匹配。场景描述能够增强全局文本原型，却不能直接确认图像中是否出现类别相关局部目标。本文分别设置多裁剪类别名称、全局上下文、局部目标、固定融合和最终残差方法，以区分性能来源。

---

## 3 方法

### 3.1 问题定义

给定图像 \(x_i\) 和候选类别集合

\[
\mathcal C=\{c_1,\ldots,c_K\},
\]

冻结图像编码器 \(E_I\) 与文本编码器 \(E_T\)。每个类别具有类别名称文本、场景上下文描述和多个局部目标/结构短语。

### 3.2 全局上下文分支

整图特征为

\[
f_i^g=\operatorname{norm}(E_I(x_i)).
\tag{1}
\]

类别名称原型和上下文描述原型分别记为 \(t_k^n\) 与 \(t_k^c\)。上下文分数定义为

\[
s_{i,k}^{c}=\omega_n\langle f_i^g,t_k^n\rangle
+(1-\omega_n)\langle f_i^g,t_k^c\rangle.
\tag{2}
\]

本文固定 \(\omega_n=0.5\)。

### 3.3 多视图、多线索局部目标证据

对每幅图像生成确定性局部视图集合 \(\mathcal V_i\)。默认使用尺度 \(0.50\) 和 \(0.75\)，每个尺度包括中心与四角位置，共十个局部裁剪。局部特征为

\[
f_{i,v}^{l}=\operatorname{norm}(E_I(v)),
\qquad v\in\mathcal V_i.
\tag{3}
\]

对类别 \(k\) 的第 \(m\) 个局部短语 \(t_{k,m}^{o}\)，计算

\[
a_{i,v,k,m}=\langle f_{i,v}^{l},t_{k,m}^{o}\rangle.
\tag{4}
\]

为降低单一裁剪异常响应，对每个短语选取最高的 \(r_v\) 个视图求均值：

\[
\bar a_{i,k,m}
=
\frac{1}{r_v}
\sum_{v\in\operatorname{Top}r_v(a_{i,:,k,m})}
a_{i,v,k,m}.
\tag{5}
\]

随后对类别 \(k\) 选取最高的 \(r_o\) 个局部短语并求均值：

\[
s_{i,k}^{o}
=
\frac{1}{r_o}
\sum_{m\in\operatorname{Top}r_o(\bar a_{i,k,:})}
\bar a_{i,k,m}.
\tag{6}
\]

默认 \(r_v=r_o=2\)。

### 3.4 上下文锚定局部残差

对上下文分数和局部分数分别在类别维度标准化，得到 \(\hat s_{i,k}^{c}\) 与 \(\hat s_{i,k}^{o}\)。局部目标分数门控为

\[
g_{i,k}
=
\sigma\left(
\frac{\hat s_{i,k}^{o}-b}{\tau_o}
\right),
\tag{7}
\]

其中 \(b=0\)，\(	au_o=0.5\)。局部优势残差为

\[
r_{i,k}
=
\left[
\hat s_{i,k}^{o}-\hat s_{i,k}^{c}
\right]_+.
\tag{8}
\]

最终分类分数为

\[
s_{i,k}
=
\hat s_{i,k}^{c}
+
\lambda g_{i,k}r_{i,k},
\tag{9}
\]

其中 \(\lambda=0.5\)，单类别残差修正被限制在预设上界内。

式（9）的核心性质为：

- 全局上下文始终作为基础分类器；
- 局部目标证据只提供补充，不直接替换全局分数；
- 多视图和多个局部短语共同决定类别局部目标分数；
- 不依赖测试集联合优化、目标域标签或参数更新。

### 3.5 最终方法的精简原则

此前诊断实验已表明，上下文熵门控、Top-M 候选限制和类别一致性融合没有稳定提升。因此，最终方法删除这些组件，不在论文中将其包装为创新，也不将包含这些组件的内部版本作为对比方法。

### 3.6 推理复杂度

额外开销主要来自局部裁剪的图像编码。文本原型可预先缓存；局部相似度聚合、残差计算与门控均为轻量矩阵运算。方法不需要梯度更新，也不依赖同一测试批次中的其他样本。

---

## 4 实验设计

### 4.1 数据集

使用 AID、EuroSAT、MLRSNet、OPTIMAL31、PatternNet、RESISC45、RSC11、RSICB128、RSICB256 和 WHURS19。所有数据集按照统一索引完成零样本分类。

### 4.2 骨干

主实验使用 GeoRSCLIP ViT-L/14。跨骨干实验使用 CLIP、RemoteCLIP、GeoRSCLIP 和 SkyCLIP50 的 ViT-L/14。

### 4.3 对比方法

| 方法 | 整图语义 | 局部裁剪 | 类别局部短语 | 融合方式 |
|---|---:|---:|---:|---|
| Global-ClassName | 类别名称 | 否 | 否 | 无 |
| MultiCrop-ClassName | 类别名称 | 是 | 否 | 多裁剪融合 |
| Global-Context | 类别名称+场景描述 | 否 | 否 | 无 |
| Object-Only | 否 | 是 | 是 | 仅局部 |
| Fixed Object-Context | 是 | 是 | 是 | 固定替换融合 |
| ObjectContext-CLIP | 是 | 是 | 是 | 上下文锚定残差 |

此前内部探索版本不作为论文对比方法。

### 4.4 指标与统计检验

报告 Top-1、Macro-F1、推理时间和峰值显存。统计分析包括成对 bootstrap 95% 置信区间、精确 McNemar 检验以及跨数据集 Wilcoxon 符号秩检验。由于不同受控方法不共享统一校准的概率标尺，ECE 不作为论文主结论指标。

### 4.5 主实验

| 方法 | AID | EuroSAT | MLRSNet | OPTIMAL31 | PatternNet | RESISC45 | RSC11 | RSICB128 | RSICB256 | WHURS19 | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Global-ClassName | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| MultiCrop-ClassName | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Global-Context | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Object-Only | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Fixed Object-Context | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| **ObjectContext-CLIP** | **待填** | **待填** | **待填** | **待填** | **待填** | **待填** | **待填** | **待填** | **待填** | **待填** | **待填** |

### 4.6 聚焦消融

最终实验只保留与当前公式直接相关的四项消融：

| 配置 | 作用 |
|---|---|
| 完整方法 | 多视图、多线索、目标分数门控、正残差 |
| 去除目标分数门控 | 检验 \(g_{i,k}\) 是否必要 |
| 有符号残差 | 检验正残差约束是否必要 |
| 单局部视图 | 检验多视图聚合 |
| 单局部短语 | 检验多线索语义表示 |

消融结果由 `analysis/table_ablation.csv` 填写。与最终公式无关、且此前已经证明无效的组件不再重复占用论文篇幅。

### 4.7 概念负对照

比较正确类别—局部线索映射、随机打乱映射和无类别判别性的统一局部概念。只有正确映射稳定优于负对照时，才能将收益归因于局部语义信息。

### 4.8 分辨率实验

在 AID、PatternNet 和 RESISC45 上使用原始分辨率以及 2×、4×、8× 下采样。该实验用于分析局部目标证据随空间分辨率变化的行为，不预设“分辨率鲁棒”结论。

### 4.9 跨骨干实验

在 CLIP、RemoteCLIP、GeoRSCLIP 和 SkyCLIP50 的 ViT-L/14 上比较 Global-ClassName、MultiCrop-ClassName、Global-Context 和 ObjectContext-CLIP。

### 4.10 类别级 Rescue/Damage

以 Global-Context 为基础，统计：

- Rescue：上下文预测错误、最终预测正确；
- Damage：上下文预测正确、最终预测错误；
- Net Rescue：Rescue 与 Damage 的差值。

逐类别结果用于定位局部目标证据的适用类别和失效类别。

---

## 5 结果与分析

### 5.1 主结果

待全量实验完成后，从以下文件填写：

```text
outputs/results/object_context_core_v1/analysis/table_main_top1.csv
outputs/results/object_context_core_v1/analysis/table_main_macro_f1.csv
```

主要结论必须围绕 ObjectContext-CLIP 与 Global-Context 的差异展开，不能只报告相对较弱的整图类别名称或简单多裁剪增益。

### 5.2 聚焦消融

从 `table_ablation.csv` 判断目标分数门控、正残差、多视图和多线索是否得到支持。论文必须如实报告最终套件中直接相关的负消融；不能只选择正结果展示。

### 5.3 概念负对照

从 `table_concept_controls.csv` 判断正确局部概念是否优于随机和统一概念。

### 5.4 跨骨干与分辨率

从 `table_cross_backbone.csv` 和 `table_resolution.csv` 填写。跨骨干实验用于判断方法是否依赖单一遥感视觉语言模型；分辨率实验用于解释局部目标线索的可见性条件。

### 5.5 统计显著性

从以下文件填写：

```text
analysis/table_significance_per_dataset.csv
analysis/table_significance_across_datasets.csv
```

报告平均数据集差值、胜/平/负数量、bootstrap 区间、McNemar 检验和 Wilcoxon 检验。

### 5.6 失败案例

重点分析局部证据产生 Damage 的类别，区分以下原因：

- 局部目标在多个场景中共享；
- 类别名称或局部短语存在语义歧义；
- 低分辨率下目标不可见；
- 局部裁剪缺少场景边界和空间组织。

---

## 6 结论

本文提出一种上下文锚定的多线索局部目标残差推理方法，在冻结遥感视觉语言模型中使用局部目标证据补充整图上下文。最终结论将在十数据集全量实验完成后根据以下条件确定：相对 Global-Context 的平均增益、跨数据集胜率、最差退化、Rescue/Damage 和统计显著性。若最终结果未证明稳定优于 Global-Context，则不得将该方法描述为通用性能增强方法。

---

## 附录：结果文件

```text
outputs/results/object_context_core_v1/
├── main_comparison.csv
├── main_decision.csv
├── classwise_analysis.csv
├── semantic_group_analysis.csv
├── missing_paper_experiments.csv
└── analysis/
    ├── table_main_top1.csv
    ├── table_main_macro_f1.csv
    ├── table_ablation.csv
    ├── table_concept_controls.csv
    ├── table_resolution.csv
    ├── table_cross_backbone.csv
    ├── table_efficiency.csv
    ├── table_significance_per_dataset.csv
    ├── table_significance_across_datasets.csv
    └── paper_results_summary.md
```
