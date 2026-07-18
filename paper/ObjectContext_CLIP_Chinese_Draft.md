# ObjectContext-CLIP：面向遥感零样本场景分类的多尺度目标—上下文协同推理

> **稿件状态**：面向 ICASSP 2027 的中文研究草稿。AID、PatternNet 和 RESISC45 的开发集实验已经完成；其余七个数据集、分辨率、跨骨干和完整统计结果尚待运行。所有未完成数值均保留为“待补充”，不写入推测结果。

**作者1，作者2，作者3**  
单位，城市，国家  
email1@example.com，email2@example.com

---

## 摘要

遥感场景通常由全局环境布局与局部判别目标共同定义。现有遥感视觉语言模型多使用单个整图特征与类别名称进行零样本匹配，容易忽略飞机、船舶、储油罐和运动场等小目标；简单多裁剪虽然引入局部视图，却仍要求每个裁剪独立表达完整场景语义。本文提出 **ObjectContext-CLIP**，一种无需下游训练的多尺度目标—上下文协同推理框架。方法在文本侧将类别知识拆分为场景上下文描述和局部目标/结构线索，在视觉侧分别使用整图与十个确定性多尺度裁剪进行对应匹配。为降低单一裁剪偶然响应的影响，每个局部线索由多个高响应视图共同支持；同时为每个候选类别估计多视图一致性，并与上下文分支和局部分支的分类间隔共同决定自适应融合权重。所有视觉和文本编码器保持冻结，推理不依赖同一测试集合中的其他样本。开发集实验在 AID、PatternNet 和 RESISC45 上表明，ObjectContext-CLIP 平均 Top-1 为 76.41%，相较整图类别名称、多裁剪类别名称和固定融合分别提高 2.21、2.41 和 0.77 个百分点；正确类别—局部概念映射比随机错位映射提高 3.54 个百分点。本文进一步采用三个开发数据集与七个冻结验证数据集的协议，系统评估十数据集泛化、概念负对照、尺度与裁剪消融、分辨率退化、跨骨干稳定性、统计显著性和失败类别。

**关键词：** 遥感场景分类；视觉语言模型；零样本学习；多尺度推理；局部目标；场景上下文

---

## 1 引言

视觉语言模型通过图文对比学习建立共享语义空间，使类别名称或自然语言描述能够直接构造零样本分类器 [1]。RemoteCLIP、GeoRSCLIP 和 SkyCLIP 等遥感视觉语言模型进一步利用遥感图文数据进行领域化预训练，在航空和卫星影像上获得更强的迁移能力 [2–4]。这些模型的常见推理方式仍然是将整幅图像压缩为一个全局向量，再与类别文本原型计算余弦相似度。

遥感场景分类并不等价于自然图像中的单主体识别。一个场景类别往往同时依赖全局空间布局、地表纹理、周边环境和局部目标。例如，机场由跑道布局、航站区和飞机共同决定；港口同时包含水陆边界、码头、船舶和堆场；工业区与商业区具有相似建筑纹理，但局部设施和空间组织不同。全局特征能够表达场景范围和布局，却可能压缩小目标；局部裁剪能够放大细粒度目标，却可能失去完整场景语义。

简单多裁剪测试将多个局部视图仍与完整类别名称匹配，其隐含假设是每个局部裁剪都足以表达完整类别。这一假设对遥感场景并不成立。一块包含水体的裁剪不能单独区分河流、湖泊和港口；一块建筑屋顶也不能直接确定住宅区、工业区或商业区。局部视图更适合回答“是否出现支持某类场景的目标或结构”，而整图负责约束完整类别。

本文从视觉尺度和文本语义两个维度同时分解分类过程。整图与场景上下文描述匹配，多尺度局部裁剪与类别特定目标/结构短语匹配。局部目标证据不采用单一最大裁剪，而由多个高响应局部视图共同确认。每个候选类别还具有独立的多视图一致性，避免一个样本级可靠性标量同时作用于全部类别。最终融合由当前图像的无标签证据决定，不训练额外参数，也不使用目标数据标签。

本文的主要贡献如下。

1. 提出遥感类别的目标—上下文语义因子化，将完整场景描述与局部目标/结构线索分别对齐至整图和局部视觉尺度。
2. 提出多视图局部线索聚合与类别级一致性估计，降低单裁剪偶然高响应对局部分支的影响。
3. 提出基于分支间隔、局部一致性和类别证据的训练自由自适应融合，实现单图像独立推理。
4. 建立开发集—冻结验证集分离的实验协议，并从十数据集主结果、概念负对照、尺度消融、分辨率、跨骨干、统计显著性和失败类别多个角度验证方法。

<!-- FIGURE 1 PLACEHOLDER -->
![图1 ObjectContext-CLIP 总体框架](figures/fig1_object_context_framework.pdf)

**图1绘制说明。** 左侧为一幅遥感图像及其整图和两尺度十个局部裁剪；中间上方为类别名称和场景上下文描述，中间下方为类别特定局部目标/结构短语；右侧分别计算全局上下文分数、局部目标分数、多视图一致性、分支可靠性和类别级融合权重。图中明确标出图像编码器和文本编码器均冻结。

---

## 2 相关工作

### 2.1 遥感视觉语言模型

CLIP 通过开放规模图文对比学习获得可迁移表示 [1]。RemoteCLIP [2]、RS5M/GeoRSCLIP [3] 和 SkyScript/SkyCLIP [4] 分别从遥感图文数据、领域规模和文本语义多样性角度增强遥感零样本能力。本文不修改预训练编码器，而研究冻结表示上的推理机制。

### 2.2 遥感场景的全局—局部表示

高分辨率遥感影像具有显著类内尺度变化、类间纹理相似和小目标密集等特点。监督学习方法通常通过双路径网络、局部注意和多尺度特征融合联合建模全局与局部信息 [5–7]。这些方法需要目标数据训练，而本文研究无需下游训练的视觉语言推理。

### 2.3 CLIP 的局部推理和语言知识增强

已有工作使用 patch 特征、多裁剪、区域提示或属性描述扩展 CLIP 的局部和文本语义 [8–10]。与直接将局部裁剪视为完整类别不同，本文将局部视图仅用于验证类别相关目标和结构线索，再由整图上下文约束完整场景。为区分收益来源，实验同时设置多裁剪类别名称、上下文描述、局部目标、固定融合、随机错位概念和通用概念等控制变量。

---

## 3 方法

### 3.1 问题定义

给定遥感图像 \(x_i\) 和候选类别集合 \(\mathcal C=\{c_1,\ldots,c_K\}\)，冻结图像编码器 \(E_I\) 和文本编码器 \(E_T\)。每个类别 \(c_k\) 预先定义三类文本：

- 类别名称集合 \(\mathcal T_k^n\)；
- 场景上下文描述集合 \(\mathcal T_k^c\)；
- 局部目标或结构线索集合 \(\mathcal T_k^o=\{o_{k,1},\ldots,o_{k,M_k}\}\)。

所有概念文本在正式实验前固定，不根据验证集结果逐类修改。

### 3.2 多尺度视觉表示

整图特征为

\[
f_i^g=\operatorname{norm}(E_I(x_i)).
\tag{1}
\]

对每幅图像生成尺度集合 \(\mathcal S=\{0.50,0.75\}\) 与位置集合 \(\mathcal P=\{\text{center},\text{four corners}\}\) 的确定性裁剪，共十个局部视图。局部特征为

\[
f_{i,v}^l=\operatorname{norm}(E_I(v)),\qquad v\in\mathcal V_i.
\tag{2}
\]

确定性视图使结果可复现，并允许将局部证据追溯到原图位置。

### 3.3 目标—上下文文本因子化

类别名称与上下文原型分别为

\[
t_k^n=\operatorname{norm}\left(\frac{1}{|\mathcal T_k^n|}\sum_{t\in\mathcal T_k^n}E_T(t)\right),
\tag{3}
\]

\[
t_k^c=\operatorname{norm}\left(\frac{1}{|\mathcal T_k^c|}\sum_{t\in\mathcal T_k^c}E_T(t)\right).
\tag{4}
\]

局部线索保留为多个独立原型 \(t_{k,m}^o\)，不在编码前合并，以便区分类别中的不同对象和结构因素。

### 3.4 全局上下文分支

全局分支同时使用类别名称和上下文描述：

\[
s_{i,k}^{c}=\omega_n\langle f_i^g,t_k^n\rangle+(1-\omega_n)\langle f_i^g,t_k^c\rangle.
\tag{5}
\]

本文固定 \(\omega_n=0.5\)。该分支负责完整场景布局和环境关系。

### 3.5 多视图局部目标证据

对类别 \(k\) 的局部线索 \(m\)，计算全部局部视图响应

\[
a_{i,v,k,m}=\langle f_{i,v}^l,t_{k,m}^o\rangle.
\tag{6}
\]

为避免单一裁剪偶然高响应，选择最强的 \(r_v\) 个视图求平均：

\[
e_{i,k,m}=\frac{1}{r_v}\sum_{v\in\operatorname{Top}r_v(a_{i,:,k,m})}a_{i,v,k,m}.
\tag{7}
\]

默认 \(r_v=2\)。再从类别的有效局部线索中选取最高的 \(r_o\) 个求平均：

\[
s_{i,k}^{o}=\frac{1}{r_o}\sum_{m\in\operatorname{Top}r_o(e_{i,k,:})}e_{i,k,m},
\tag{8}
\]

默认 \(r_o=2\)。该两级聚合要求局部目标同时获得跨视图和跨线索支持。

### 3.6 类别级多视图一致性

对于每个局部视图，先在候选类别维度标准化局部分数并转换为软支持值。类别 \(k\) 的一致性定义为其最强若干视图支持度的平均：

\[
A_{i,k}=\frac{1}{r_a}\sum_{v\in\operatorname{Top}r_a}\sigma\left(\frac{\bar s_{i,v,k}^{o}-\delta_a}{\tau_a}\right).
\tag{9}
\]

与单个样本级一致性不同，\(A_{i,k}\) 能分别表示不同候选类别在多个裁剪中的稳定程度。

### 3.7 证据自适应融合

对全局和局部分数分别进行样本内标准化，得到 \(ar s_i^c\) 和 \(ar s_i^o\)。根据 Top-1 与 Top-2 间隔估计两个分支可靠性：

\[
r_i^c=\sigma\left(\frac{\Delta(\bar s_i^c)-\delta_c}{\tau_m}\right),
\tag{10}
\]

\[
r_i^o=\sigma\left(\frac{\Delta(\bar s_i^o)-\delta_o}{\tau_m}\right)A_{i,\hat k_i^o}^{\gamma},
\tag{11}
\]

其中 \(\hat k_i^o\) 为局部分支预测类别。样本级局部分支权重为

\[
\lambda_i=\operatorname{softmax}\left(\frac{[\log(r_i^c+\epsilon)+b_c,\log(r_i^o+\epsilon)]}{\tau_r}\right)_2.
\tag{12}
\]

类别级门控为

\[
g_{i,k}=\sigma\left(\frac{\bar s_{i,k}^{o}-\delta_g}{\tau_g}\right)A_{i,k}^{\gamma_a}.
\tag{13}
\]

最终分数为

\[
s_{i,k}=(1-\lambda_i g_{i,k})\bar s_{i,k}^{c}+\lambda_i g_{i,k}\bar s_{i,k}^{o}.
\tag{14}
\]

所有权重均由当前图像的无标签证据计算。

<!-- FIGURE 2 PLACEHOLDER -->
![图2 多视图线索聚合和类别级一致性](figures/fig2_multiview_evidence.pdf)

**图2绘制说明。** 对一个类别展示十个局部裁剪与多个目标短语的相似度矩阵。先在视图维度选择 Top-\(r_v\)，再在线索维度选择 Top-\(r_o\)，同时从每个类别的跨视图支持计算 \(A_{i,k}\)。用一个单裁剪偶然高响应的失败例说明 Top-2 聚合如何抑制噪声。

---

## 4 实验设计

### 4.1 数据集和冻结协议

使用 AID、EuroSAT、MLRSNet、OPTIMAL31、PatternNet、RESISC45、RSC11、RSICB128、RSICB256 和 WHURS19 十个数据集。

为避免在同一批数据集上反复调节方法，采用如下协议：

- **开发集**：AID、PatternNet、RESISC45，用于方法设计、组件消融和参数选择；
- **冻结验证集**：EuroSAT、MLRSNet、OPTIMAL31、RSC11、RSICB128、RSICB256、WHURS19；
- 所有方法参数、概念库构造规则和裁剪配置在运行七个验证集前冻结。

最终报告十数据集结果，但分别给出开发集和验证集平均值。

### 4.2 骨干、比较方法和指标

主实验固定 GeoRSCLIP ViT-L/14。跨骨干实验使用 CLIP、RemoteCLIP、GeoRSCLIP 和 SkyCLIP50 的 ViT-L/14。

比较方法包括：Global-ClassName、MultiCrop-ClassName、Global-Context、Object-Only、Fixed Object-Context 和 ObjectContext-CLIP。

评价指标包括 Top-1、Macro-F1、ECE、推理时间、峰值显存和特征缓存大小。统计分析包括：

- 每个数据集上的成对 bootstrap 95% 置信区间；
- ObjectContext 与各基线的精确 McNemar 检验；
- 十数据集差值的 Wilcoxon 符号秩检验；
- 胜、平、负次数；
- Rescue、Damage 和 Net Rescue。

### 4.3 完整实验矩阵

论文所需实验分为：

1. 十数据集六方法主实验；
2. 开发集上的视图 Top-k、线索 Top-k、一致性、单尺度和裁剪数量消融；
3. 十数据集正确、随机错位和通用局部概念负对照；
4. 开发集 1×、2×、4×、8× 分辨率退化；
5. 四个 ViT-L/14 骨干的跨骨干实验；
6. 逐类别、语义分组、Rescue/Damage 和失败案例；
7. 时间、显存、特征提取时间和缓存大小。

---

## 5 已完成开发集结果

### 5.1 三数据集主结果

**表1  GeoRSCLIP ViT-L/14 开发集 Top-1 准确率（%）。**

| 方法 | AID | PatternNet | RESISC45 | 平均 |
|---|---:|---:|---:|---:|
| Global-ClassName | 72.6400 | 76.5592 | 73.4095 | 74.2029 |
| MultiCrop-ClassName | 73.4800 | 74.6579 | 73.8730 | 74.0036 |
| Global-Context | 71.6300 | **79.0757** | 77.0825 | 75.9294 |
| Object-Only | 65.1000 | 60.1579 | 65.9651 | 63.7410 |
| Fixed Object-Context | 72.6600 | 76.1776 | 78.0825 | 75.6400 |
| **ObjectContext-CLIP** | **73.2500** | 77.4046 | **78.5746** | **76.4097** |

ObjectContext-CLIP 相较 Global-ClassName、MultiCrop-ClassName、Global-Context 和固定融合的平均增益分别为 +2.2068、+2.4061、+0.4803 和 +0.7697 个百分点。结果说明收益不能仅由增加裁剪数量解释，自适应融合也优于固定权重。但 PatternNet 上 Global-Context 仍高于完整方法 1.6711 个百分点，表明局部目标证据并非对所有数据集都有效。

### 5.2 组件和概念控制

**表2  开发集机制实验 Top-1（%）。**

| 配置 | AID | PatternNet | RESISC45 | 平均 | 相对 v2 |
|---|---:|---:|---:|---:|---:|
| 单视图硬最大 | 73.0300 | 77.1447 | 78.3524 | 76.1759 | -0.2338 |
| 无类别级一致性 | 73.1900 | 77.2928 | 78.4302 | 76.3045 | -0.1052 |
| **v2：Top-2 + 类别一致性** | **73.2500** | **77.4046** | **78.5746** | **76.4097** | 0.0000 |
| Top-3 视图 | 待从 CSV 填入 | 待从 CSV 填入 | 待从 CSV 填入 | 76.4031 | -0.0066 |
| Shuffled concepts | 68.7400 | 待从 CSV 填入 | 待从 CSV 填入 | 72.8733 | -3.5365 |
| Generic concepts | 71.6300 | 79.0757 | 77.0825 | 75.9294 | -0.4803 |

Top-2 相较单视图最大值提高 0.2338，类别级一致性额外提高 0.1052。两者属于稳定性组件而非主要性能来源。正确局部概念比随机错位概念高 3.5365，证明类别—目标语义对应关系具有实质作用；正确概念仅比通用概念高 0.4803，说明上下文描述仍承担较大比例的性能收益。

### 5.3 Rescue、Damage 与失败类别

三个开发数据集的 Rescue/Damage 统计为：

| 数据集 | Rescue | Damage | Net Rescue |
|---|---:|---:|---:|
| AID | 693 | 632 | +61 |
| PatternNet | 1934 | 1677 | +257 |
| RESISC45 | 3023 | 1396 | +1627 |

所有数据集的净修正为正，但类别级变化并不均匀。已观察到的主要失败包括：

- AID `viaduct`：66.67% 降至 1.90%；
- PatternNet `christmas tree farm`：86.13% 降至 41.00%；
- RESISC45 `sparse residential`：50.00% 降至 6.14%。

`viaduct` 在 Global-Context 中已经显著退化，说明部分错误来自类别描述与数据集标签语义不一致，而非局部裁剪本身。最终论文必须同时展示成功和失败类别，不以平均准确率掩盖语义错配。

<!-- FIGURE 3 PLACEHOLDER -->
![图3 Rescue、Damage 和类别级性能变化](figures/fig3_rescue_damage.pdf)

**图3绘制说明。** 左侧绘制十数据集 Rescue 和 Damage 柱状图；右侧绘制逐类别 ObjectContext 相对 Global 的差值分布，标注提升最大的类别和退化最严重的类别。

---

## 6 待完成的冻结验证和完整实验

### 6.1 十数据集主结果

**表3  GeoRSCLIP ViT-L/14 十数据集 Top-1（%）。**

| 方法 | AID | EuroSAT | MLRSNet | OPTIMAL31 | PatternNet | RESISC45 | RSC11 | RSICB128 | RSICB256 | WHURS19 | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Global-ClassName | 72.6400 | 待补充 | 待补充 | 待补充 | 76.5592 | 73.4095 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| MultiCrop-ClassName | 73.4800 | 待补充 | 待补充 | 待补充 | 74.6579 | 73.8730 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| Global-Context | 71.6300 | 待补充 | 待补充 | 待补充 | 79.0757 | 77.0825 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| **ObjectContext-CLIP** | **73.2500** | 待补充 | 待补充 | 待补充 | 77.4046 | **78.5746** | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

验证集主要判断标准预先固定为：相对 Global 平均增益不低于 +1.0，相对 MultiCrop 不低于 +0.5，相对 Global-Context 为正，至少 5/7 个验证集不低于 Global，且不存在超过 5 个百分点的数据集级退化。

### 6.2 完整消融

**表4  开发集消融结果。**

| 配置 | AID | PatternNet | RESISC45 | 平均 |
|---|---:|---:|---:|---:|
| ObjectContext v2 | 73.2500 | 77.4046 | 78.5746 | 76.4097 |
| View Top-1 | 待补充 | 待补充 | 待补充 | 76.1759 |
| View Top-3 | 待补充 | 待补充 | 待补充 | 76.4031 |
| Cue Top-1 | 待补充 | 待补充 | 待补充 | 待补充 |
| Cue Top-3 | 待补充 | 待补充 | 待补充 | 待补充 |
| 无类别一致性 | 待补充 | 待补充 | 待补充 | 76.3045 |
| 仅 0.50 尺度 | 待补充 | 待补充 | 待补充 | 待补充 |
| 仅 0.75 尺度 | 待补充 | 待补充 | 待补充 | 待补充 |
| 仅中心裁剪 | 待补充 | 待补充 | 待补充 | 待补充 |

### 6.3 概念负对照

**表5  十数据集概念控制。**

| 概念模式 | 开发集平均 | 验证集平均 | 十数据集平均 |
|---|---:|---:|---:|
| Correct | 76.4097 | 待补充 | 待补充 |
| Generic | 75.9294 | 待补充 | 待补充 |
| Shuffled | 72.8733 | 待补充 | 待补充 |

### 6.4 分辨率鲁棒性

**表6  开发集不同下采样倍率平均 Top-1。**

| 方法 | 1× | 2× | 4× | 8× |
|---|---:|---:|---:|---:|
| Global-ClassName | 待补充 | 待补充 | 待补充 | 待补充 |
| MultiCrop-ClassName | 待补充 | 待补充 | 待补充 | 待补充 |
| Global-Context | 待补充 | 待补充 | 待补充 | 待补充 |
| ObjectContext-CLIP | 待补充 | 待补充 | 待补充 | 待补充 |

预期分析不要求 ObjectContext 在所有退化强度下都提高。严重下采样会使小目标消失，此时应检查局部分支权重是否下降以及全局上下文是否承担主要判断。

### 6.5 跨骨干实验

**表7  四个 ViT-L/14 骨干的开发集平均 Top-1。**

| 骨干 | Global | MultiCrop | Context | ObjectContext |
|---|---:|---:|---:|---:|
| CLIP | 待补充 | 待补充 | 待补充 | 待补充 |
| RemoteCLIP | 待补充 | 待补充 | 待补充 | 待补充 |
| GeoRSCLIP | 74.2029 | 74.0036 | 75.9294 | 76.4097 |
| SkyCLIP50 | 待补充 | 待补充 | 待补充 | 待补充 |

### 6.6 效率和统计显著性

报告整图和十个局部视图的特征提取时间、缓存大小，以及各方法在缓存特征上的推理时间和峰值显存。对每个数据集报告 bootstrap 区间和 McNemar 检验，并对十数据集差值进行 Wilcoxon 检验。

---

## 7 讨论

开发集实验已经证明 ObjectContext-CLIP 具有明显高于整图和简单多裁剪的平均收益，但结果也揭示三个限制。第一，局部目标分支单独使用时性能较低，它必须由完整场景上下文约束。第二，正确概念相对通用概念的增益小于相对随机错位概念的增益，说明语言描述质量和局部概念都重要。第三，少数类别会因类别名称歧义或概念描述与数据集定义不一致而灾难性退化。

因此，本文不声称局部目标对所有类别均有效，也不将人工语义分组作为模型输入。目标型、上下文型和混合型分组仅用于离线分析。最终结论取决于七个冻结验证数据集：若验证集仍保持对 Global、MultiCrop 和 Global-Context 的正向平均结果，则可以支持通用的训练自由协同推理结论；若验证集只在部分数据集有效，则应将论文定位为条件性方法并明确适用边界。

---

## 8 结论

本文提出 ObjectContext-CLIP，通过文本语义因子化和多尺度视觉匹配联合利用遥感场景的全局上下文与局部目标证据。方法使用多视图局部线索聚合和类别级一致性抑制偶然裁剪响应，并通过无训练自适应融合形成最终预测。开发集实验显示，该方法相对整图类别名称和简单多裁剪分别提高 2.21 和 2.41 个百分点，且正确概念映射明显优于随机错位映射。后续冻结验证将决定该方法能否形成稳定的十数据集和跨骨干结论。

---

## 参考文献

[1] A. Radford et al., “Learning transferable visual models from natural language supervision,” in *Proc. ICML*, 2021.

[2] F. Liu et al., “RemoteCLIP: A vision language foundation model for remote sensing,” *IEEE Transactions on Geoscience and Remote Sensing*, 2024.

[3] Z. Zhang et al., “RS5M and GeoRSCLIP: A large-scale vision-language dataset and model for remote sensing,” *IEEE Transactions on Geoscience and Remote Sensing*, 2024.

[4] Z. Wang et al., “SkyScript: A large and semantically diverse vision-language dataset for remote sensing,” in *Proc. AAAI*, 2024.

[5] G.-S. Xia et al., “AID: A benchmark data set for performance evaluation of aerial scene classification,” *IEEE Transactions on Geoscience and Remote Sensing*, 2017.

[6] G. Cheng, J. Han, and X. Lu, “Remote sensing image scene classification: Benchmark and state of the art,” *Proceedings of the IEEE*, 2017.

[7] W. Zhou et al., “PatternNet: A benchmark dataset for performance evaluation of remote sensing image retrieval,” *ISPRS Journal of Photogrammetry and Remote Sensing*, 2018.

[8] A. work on CLIP local or patch-level inference, complete bibliographic information to be verified before submission.

[9] A. work on language-description-enhanced zero-shot classification, complete bibliographic information to be verified before submission.

[10] A. work on training-free multi-crop vision-language inference, complete bibliographic information to be verified before submission.

---

## 作者侧执行清单（投稿前删除）

1. 使用 `configs/paper.yaml` 冻结全部方法参数和开发/验证划分。
2. 运行 `scripts/run_paper_suite.py --stages all`。
3. 运行 `scripts/analyze_paper_results.py` 生成全部表格、置信区间和统计检验。
4. 仅从生成 CSV 中填写表3–表7，不手工估计结果。
5. 对验证集单独计算平均增益、胜负次数和最大退化。
6. 绘制成功与失败类别，不隐藏灾难性退化。
7. 补充并核验相关工作 [8]–[10] 的正式引用。
8. 将稿件迁移至 ICASSP 2027 官方模板并根据页数限制压缩。
