# RAP-TransCLIP: Reliability-Aware Prompt and Active-Prior Transduction for Zero-Shot Remote Sensing Scene Classification

**Working ICASSP 2027 paper draft — results and author information are not yet complete.**

> Formatting note: this Markdown draft is organized for the conventional ICASSP short-paper structure: four pages of technical content and an optional fifth page containing references only. The official ICASSP 2027 author kit should replace this note and determine the final LaTeX layout when released.

**Author 1, Author 2, Author 3**  
*Affiliation, City, Country*  
*email1@example.com, email2@example.com*

## Abstract

Remote-sensing vision-language models enable zero-shot scene recognition by matching image embeddings with textual class prototypes. RS-TransCLIP improves this paradigm through label-free transductive inference, but it uniformly averages a large prompt bank and assumes a balanced mixture over all candidate classes. These assumptions are fragile because prompt quality is class dependent and a target image collection may exhibit substantial class-prior mismatch. We introduce **RAP-TransCLIP**, a training-free framework that jointly refines category-wise prompt reliability, image assignments, visual class prototypes, and active class priors while keeping the vision-language backbone frozen. First, a category-wise reliability estimator scores each prompt using confidence, predictive entropy, class separation, and agreement with the current transductive assignments. Second, an anchored active-prior update estimates the effective target distribution without using labels. Third, a reliability-weighted mutual-nearest-neighbor graph suppresses propagation through uncertain samples. The resulting alternating optimization retains the low-cost embedding-space operation of RS-TransCLIP. We plan to evaluate RAP-TransCLIP on ten remote-sensing scene-classification benchmarks with CLIP, RemoteCLIP, GeoRSCLIP, and SkyCLIP. **Preliminary/main results: TBD.**

**Index Terms—** remote sensing, vision-language models, zero-shot classification, transductive inference, prompt reliability, class-prior estimation.

## 1. Introduction

Remote-sensing (RS) scene classification supports land-use analysis, environmental monitoring, urban planning, and disaster assessment. Recent vision-language models (VLMs), represented by CLIP [1], transfer semantic knowledge to downstream recognition tasks through natural-language class descriptions. Domain-adapted models such as RemoteCLIP [2], GeoRSCLIP [3], and SkyCLIP [4] further improve the alignment between overhead imagery and text. Their zero-shot predictions, however, remain sensitive to the wording of prompts and to the distribution of the target images.

RS-TransCLIP [5] addresses the second limitation partially by predicting a target collection jointly rather than classifying every image independently. It combines text-derived pseudo-labels, a Gaussian-mixture likelihood, and graph regularization over image embeddings. This yields substantial improvements without updating the VLM parameters. Nevertheless, two design choices limit its robustness. First, its 106 text templates are averaged uniformly into one prototype per category, even though phrases describing resolution, viewpoint, illumination, or quality are not equally informative for every RS class. Second, the Gaussian mixture uses an effectively uniform class prior, which favors a full and balanced target collection. These assumptions are particularly consequential in transductive inference because an unreliable prompt or an incorrect prior can influence many samples through graph propagation.

We propose **RAP-TransCLIP**, a reliability-aware extension that retains the training-free and backbone-frozen setting. The framework estimates a separate prompt distribution for each category, infers an anchored active-class prior from the unlabeled target collection, and constructs a mutual-nearest-neighbor graph whose edges are attenuated by sample reliability. All variables are updated in the pretrained embedding space.

Our contributions are threefold:

1. We replace uniform prompt averaging with a **category-wise prompt reliability posterior** based on confidence, entropy, separation, and transductive agreement.
2. We introduce an **anchored active-prior update** that relaxes the balanced-class assumption without using target labels.
3. We develop a **reliability-aware mutual graph** that reduces error amplification while preserving lightweight, parameter-free inference.

## 2. Related Work

### 2.1 Remote-Sensing Vision-Language Models

CLIP learns a transferable image-text embedding space from large-scale paired data [1]. RemoteCLIP [2], GeoRSCLIP [3], and SkyCLIP [4] adapt this paradigm to overhead imagery using remote-sensing image-text corpora. These models achieve strong zero-shot scene classification but generally rely on fixed prompt templates or a uniformly averaged prompt ensemble. Prompt ensembling improves robustness relative to a single phrase, yet it does not account for category-dependent prompt quality.

### 2.2 Transductive and Test-Time VLM Inference

Transductive inference exploits the geometry of an unlabeled target set. TransCLIP [6] integrates text prototypes into an embedding-space transductive objective, and RS-TransCLIP [5] applies this principle to RS scene classification. Recent work has also shown that favorable full-class target batches can overstate the robustness of test-time adaptation [7], while Bayesian class adaptation emphasizes explicit prior estimation [8]. RAP-TransCLIP differs by coupling prior estimation with category-wise prompt reliability and reliability-aware graph propagation for RS VLMs.

## 3. Proposed Method

### 3.1 Problem Formulation

Let

\[
F=[f_1,\ldots,f_N]^\top\in\mathbb{R}^{N\times d}
\]

be the normalized image embeddings of an unlabeled target collection, and let

\[
T=\{t_{m,k}\mid m=1,\ldots,M;\ k=1,\ldots,K\}
\]

be normalized text embeddings for \(M\) prompt templates and \(K\) candidate categories. The VLM encoders remain frozen. We estimate soft assignments

\[
Z=[z_1,\ldots,z_N]^\top\in\Delta^{N\times K},
\]

category prototypes \(\mu_k\), a shared diagonal covariance \(\Sigma\), category-wise prompt weights \(A=[\alpha_{m,k}]\), and a target class prior \(\pi\in\Delta^K\).

### 3.2 Category-Wise Prompt Reliability

For prompt \(m\), the zero-shot probability of image \(i\) is

\[
p^{(m)}_{i}=\operatorname{softmax}(sF_iT_m^\top),
\]

where \(s\) is the pretrained or fixed logit scale. For every prompt-category pair, we form a high-response support set \(\mathcal{S}_{m,k}\) from the top-ranked target images. Its reliability score is

\[
r_{m,k}=\beta_c C_{m,k}-\beta_hH_{m,k}
+\beta_dD_{m,k}+\beta_aA_{m,k},
\tag{1}
\]

where \(C_{m,k}\) is mean class confidence, \(H_{m,k}\) is normalized predictive entropy, \(D_{m,k}\) is the margin over the strongest competing category, and \(A_{m,k}\) measures agreement with the current assignment \(Z\). The category-specific prompt posterior is

\[
\alpha_{m,k}=
\frac{\exp(r_{m,k}/\tau_p)}
{\sum_{q=1}^{M}\exp(r_{q,k}/\tau_p)}.
\tag{2}
\]

The resulting text prototype is

\[
\bar t_k=\operatorname{norm}
\left(\sum_{m=1}^{M}\alpha_{m,k}t_{m,k}\right).
\tag{3}
\]

Unlike global prompt selection, (2) allows, for example, a scale-related prompt to be useful for an airport category but irrelevant for a forest category.

### 3.3 Sample Reliability and Graph Construction

The weighted prototypes produce anchored pseudo-labels

\[
\hat y_i=\operatorname{softmax}
\left(s[f_i^\top\bar t_1,\ldots,f_i^\top\bar t_K]\right).
\tag{4}
\]

We define sample reliability \(q_i\in[0,1]\) from low ensemble entropy and low disagreement among the \(M\) prompts. Given a mutual \(k\)-nearest-neighbor relation, the edge weight is

\[
w_{ij}=\mathbf{1}[i\in\mathcal{N}_k(j),j\in\mathcal{N}_k(i)]
q_iq_j
\exp\left(-\frac{1-f_i^\top f_j}{\sigma_i}\right),
\tag{5}
\]

where \(\sigma_i\) is a local scale. We symmetrically normalize \(W\) before propagation. Mutual neighborhoods and reliability factors remove weak one-way edges and attenuate uncertain nodes.

### 3.4 Active Class-Prior Estimation

RS-TransCLIP models the target embedding distribution with one Gaussian component per candidate category. We retain the efficient shared diagonal covariance but introduce an explicit prior:

\[
p_{i,k}\propto \pi_k\,
\mathcal{N}(f_i;\mu_k,\Sigma).
\tag{6}
\]

At iteration \(r\), soft counts are combined with a symmetric Dirichlet prior:

\[
\tilde\pi_k^{(r)}=
\frac{\sum_i z_{i,k}^{(r)}+\alpha_0}
{N+K\alpha_0}.
\tag{7}
\]

An active-class gate \(g_k\) is computed from the mean and maximum target response of category \(k\). To avoid premature removal of rare categories, the update is anchored to the initial text-derived prior \(\pi^{(0)}\):

\[
\pi^{(r+1)}=\operatorname{norm}
\left[(1-\rho)g\odot\tilde\pi^{(r)}
+\rho\pi^{(0)}+\epsilon\mathbf{1}\right].
\tag{8}
\]

### 3.5 Alternating Transductive Updates

With \(A,\pi,\mu,\Sigma\), and \(W\) fixed, assignments are updated by

\[
z_i\leftarrow\operatorname{softmax}
\left[
\eta_g(WZ)_i
+\eta_t\log\hat y_i
+\eta_l\log p_i
\right].
\tag{9}
\]

The Gaussian mean and covariance use weighted closed-form estimates:

\[
\mu_k=\operatorname{norm}
\left(\frac{\sum_i z_{i,k}f_i}{\sum_i z_{i,k}}\right),
\quad
\operatorname{diag}(\Sigma)=
\frac{\sum_{i,k}z_{i,k}(f_i-\mu_k)^2}{\sum_{i,k}z_{i,k}}.
\tag{10}
\]

RAP-TransCLIP alternates (1)–(10) until a fixed iteration budget or assignment convergence. The image and text encoders are evaluated once. With a sparse \(k\)-NN graph, each assignment update is linear in the number of graph edges; nearest-neighbor construction is accelerated with FAISS when available.

**Algorithm 1: RAP-TransCLIP inference**

```text
Input: image embeddings F, prompt embeddings T
1: Compute per-prompt probabilities and initialize category-wise prompt weights A
2: Construct weighted text prototypes and pseudo-labels Y_hat
3: Estimate sample reliability q and mutual-kNN graph W
4: Initialize Z = Y_hat, Gaussian parameters, and text-derived prior pi
5: repeat
6:     Update assignments Z using Gaussian, graph, text, and prior terms
7:     Update Gaussian means and shared diagonal covariance
8:     Update active class prior pi with a text anchor
9:     Re-estimate category-wise prompt weights A from current Z
10:    Refresh weighted text prototypes and Y_hat
11: until convergence or maximum iterations
Output: target assignments Z
```

## 4. Experiments

### 4.1 Experimental Setup

**Datasets.** We first follow the RS-TransCLIP evaluation protocol on ten scene-classification datasets: AID, EuroSAT, MLRSNet, OPTIMAL31, PatternNet, RESISC45, RSC11, RSICB128, RSICB256, and WHURS19. The exact source, class list, image count, and checksum of every processed dataset will be released with the code. Particular care is required to distinguish AID from Million-AID Level-2.

**Backbones.** We evaluate CLIP, RemoteCLIP, GeoRSCLIP, and SkyCLIP with the architectures supported by their public checkpoints. The first screening uses GeoRSCLIP ViT-L/14 on all ten datasets. The remaining backbones are added after the method configuration is frozen.

**Baselines.** We compare the inductive zero-shot VLM, the official or faithfully reproduced RS-TransCLIP baseline [5], and RAP-TransCLIP. Subject to implementation availability and page budget, the final comparison will include TransCLIP [6], StatA [7], and BCA [8].

**Protocol.** All methods use identical precomputed image and text embeddings. Ground-truth labels are used only after inference to compute metrics. Hyperparameters are fixed across datasets and are not selected using target labels. We report top-1 accuracy, macro-F1, expected calibration error (ECE), runtime, and peak memory.

### 4.2 Main Results

**Table 1: Top-1 accuracy (%) on the ten standard RS benchmarks using GeoRSCLIP ViT-L/14.**

| Method | AID | EuroSAT | MLRSNet | OPTIMAL31 | PatternNet | RESISC45 | RSC11 | RSICB128 | RSICB256 | WHURS19 | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot VLM | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RS-TransCLIP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RAP-TransCLIP | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

The intended analysis should answer four questions: (i) whether category-wise prompt weighting consistently improves the uniform ensemble, (ii) whether active-prior estimation remains stable on full datasets, (iii) which datasets benefit from reliability-aware graph propagation, and (iv) whether improvements are obtained without material inference overhead. Numerical claims must not be written before the complete result files are generated.

### 4.3 Ablation Study

**Table 2: Component ablation averaged over the ten datasets.**

| Configuration | Prompt reliability | Active prior | Reliable mutual graph | Top-1 | Macro-F1 | ECE |
|---|:---:|:---:|:---:|---:|---:|---:|
| RS-TransCLIP baseline |  |  |  | TBD | TBD | TBD |
| + category-wise prompts | ✓ |  |  | TBD | TBD | TBD |
| + active prior | ✓ | ✓ |  | TBD | TBD | TBD |
| RAP-TransCLIP | ✓ | ✓ | ✓ | **TBD** | **TBD** | **TBD** |

We will additionally remove each term in (1), compare global and category-wise prompt weights, and compare directed cosine kNN with the proposed mutual RBF graph.

### 4.4 Efficiency and Diagnostic Analysis

**Table 3: Additional inference cost over feature extraction.**

| Dataset | Images | RS-TransCLIP time | RAP-TransCLIP time | Peak memory | Iterations |
|---|---:|---:|---:|---:|---:|
| WHURS19 | TBD | TBD | TBD | TBD | TBD |
| AID | TBD | TBD | TBD | TBD | TBD |
| MLRSNet | TBD | TBD | TBD | TBD | TBD |

We will visualize (i) class-specific prompt weights, (ii) initial and final class priors, and (iii) graph neighborhoods corrected or corrupted by propagation. These diagnostics are necessary to show that the gains originate from the proposed reliability mechanisms rather than an additional tuned scalar.

## 5. Conclusion

We presented RAP-TransCLIP, a training-free extension of transductive RS VLM inference. The method replaces uniform prompt averaging with category-wise reliability posteriors, relaxes the balanced-mixture assumption through an anchored active-class prior, and suppresses uncertain graph propagation. The complete method operates only on frozen image and text embeddings. Experiments on ten RS benchmarks and multiple RS VLMs are in progress. The final manuscript will report verified accuracy, calibration, and efficiency results and will include realistic partial-class and long-tail protocols if the initial standard-benchmark screening supports the proposed design.

## References

[1] A. Radford *et al.*, “Learning transferable visual models from natural language supervision,” in *Proc. ICML*, 2021.

[2] F. Liu *et al.*, “RemoteCLIP: A vision language foundation model for remote sensing,” *IEEE Trans. Geosci. Remote Sens.*, 2024.

[3] Z. Zhang *et al.*, “RS5M and GeoRSCLIP: A large scale vision-language dataset and a large vision-language model for remote sensing,” 2024.

[4] Z. Wang *et al.*, “SkyScript: A large and semantically diverse vision-language dataset for remote sensing,” in *Proc. AAAI*, 2024.

[5] K. El Khoury *et al.*, “Enhancing remote sensing vision-language models for zero-shot scene classification,” in *Proc. IEEE ICASSP*, 2025.

[6] M. Zanella, B. Gérin, and I. Ben Ayed, “Boosting vision-language models with transduction,” in *Advances in Neural Information Processing Systems*, vol. 37, 2024.

[7] M. Zanella, C. Fuchs, C. De Vleeschouwer, and I. Ben Ayed, “Realistic test-time adaptation of vision-language models,” 2025.

[8] L. Zhou *et al.*, “Bayesian test-time adaptation for vision-language models,” 2025.

[9] I. Ziko *et al.*, “Laplacian regularized few-shot learning,” in *Proc. ICML*, 2020.

[10] X. Li *et al.*, “Vision-language models in remote sensing: Current progress and future trends,” *IEEE Geosci. Remote Sens. Mag.*, vol. 12, no. 2, pp. 32–66, 2024.

---

## Internal completion checklist — remove before submission

- [ ] Replace author and affiliation placeholders.
- [ ] Confirm the final method name after novelty search.
- [ ] Reproduce official RS-TransCLIP numbers with identical dataset versions.
- [ ] Complete Table 1 for GeoRSCLIP ViT-L/14 on ten datasets.
- [ ] Complete cross-backbone comparison.
- [ ] Complete component and prompt-score ablations.
- [ ] Add runtime and memory measurements.
- [ ] Add framework figure and prompt-reliability visualization.
- [ ] Add partial-class/long-tail experiments if standard results are positive.
- [ ] Verify every citation and replace preprints with published records where available.
- [ ] Transfer to the official ICASSP 2027 LaTeX template.
- [ ] Check page count, fonts, margins, PDF compliance, anonymity rules, and AI-use disclosure against the released 2027 author instructions.
