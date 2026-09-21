---
title: "SWA 有效感受野：O(W) 上界的假设与推导"
date: 2026-09-15
draft: false
visibility: public
categories: ["注意力算法"]
tags: ["SWA","数学推导"]
description: "SWA 有效感受野：O(W) 上界的假设与推导；保留技术细节、出处与适用边界。"
---

> 这是技术笔记的公开整理版，原始记录日期为 2026-09-15，本次编辑于 2026-09-21。保留原笔记的公式、代码位置与证据分级；本次仅整理内容，未重新运行 GPU 实验或逐项复核上游。标为【码】【核验】的内容指原记录的核查结果，不代表当前版本仍然如此。

> 起因：DeepSeek-V4.1 报告 2.2 节（CED）写：
> "prior work (**Chen et al., 2025**) has shown that the **actual effective receptive field of SWA is much smaller than the theoretical `n_win × L/2`**. Motivated by this observation, we introduce **Decoder SWA Bounded Replay**..."
>
> 本文回答两件事：**Chen et al. 2025 是谁**、**这个上界是怎么证出来的**。

---

## 0. 先给结论

**V4.1 的引用链条指向两篇不同的东西**，而**真正证明 `O(W)` 上界的那一份，不在 V4.1 的参考文献列表里**：

| 角色 | 出处 | 是否在 V4.1 参考文献中 |
|---|---|---|
| V4.1 实际引用的 "Chen et al., 2025" | **PowerAttention**，arXiv:2503.03588 | ✅ 在（第 1916-1918 行） |
| **真正给出 `O(W)` 上界证明的** | **《Why Stacking Sliding Windows Can't See Very Far》**，Guangxuan Xiao（MIT HAN Lab），2025-08-25 | ❌ **不在**（V4.1 列表里搜不到） |

**【核验】** 我在 V4.1 参考文献全文里搜过：没有任何条目指向 `guangxuanx.com`、`hanlab.mit.edu`，也没有 "stacking sliding window" 字样。

**这意味着**：V4.1 用一个**讲"如何设计更优稀疏模式"的论文**，去支撑一句关于**SWA 有效感受野上界**的论断——**语义上是错位的**。真正该引的是那篇博客。

> 不过要公平地说：**错位不等于错误**。V4.1 需要的是"SWA 实际感受野远小于理论上界"这个经验事实，而 PowerAttention 的实证部分（§4.6 probing）**确实间接支撑了这个方向**（见第 3 节）。只是**定量的上界证明来自博客，不来自 PowerAttention**。

---

## 1. PowerAttention 是什么

**元数据**【核验，来自 arXiv abs 页】：

| 项 | 内容 |
|---|---|
| 标题 | PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention |
| 作者 | Lida Chen\*, Dong Xu\*, Chenxin An, Xintao Wang, Yikai Zhang, Jiangjie Chen, Zujie Liang, Feng Wei, Jiaqing Liang, Yanghua Xiao, Wei Wang（\* 共同一作） |
| 提交 | **2025-03-05** |
| 分类 | cs.CL; cs.LG |
| 页数 | 15 |

**它解决的问题**：LLM 长上下文下注意力的二次复杂度。既有稀疏注意力方法**要么有效上下文不完整，要么实现复杂**。

**它的核心贡献**：把稀疏注意力**建模成 DAG 上的边集选择问题**，然后设计一个能让感受野**指数增长**的稀疏模式。

**核心论断**【原文，§3.3】：

> "Our edge set construction ensures that in a DAG, any node can reach all nodes within a distance of `n` in at most `log n` steps, while maintaining a maximum out-degree of `log n`. This is achieved by **connecting each node only to nodes whose index differences are powers of 2**... Under our pattern, we guarantee that the receptive field grows **exponentially with the maximum distance `d`**, while capturing information from all tokens within a distance of `2^d`."

**注意箭头方向**：PowerAttention 证明的是**它自己**能达到指数感受野，**不是**证明 SWA 有多差。SWA 在它文里是"被比较的基线"。

---

## 2. PowerAttention 的形式化证明（Appendix B，逐字）

这是它唯一的正式定理。

### 定理陈述

> **Theorem B.1.** For a directed acyclic graph (DAG) with vertices labeled from 1 to n, let the edge set be
> `E = {(i,j) | i − j = 2^k, k ∈ Z*}`
> Then the following properties hold:
> 1. For any vertex `i`, the out-degree of `i` is less than `log n`.
> 2. For any vertices `i` and `j` where `j < i`, the distance from `i` to `j` is at most `log n`.

### 证明（原文全文）

> **Proof.** We prove both of the properties:
>
> **(1)** For any edge `(i,j) ∈ E`, we have `i − j = 2^k` where `2^k < n`. Therefore, `k < log n`. Since each possible value of `k` corresponds to at most one outgoing edge from vertex `i`, the out-degree of any vertex is bounded by `log n`.
>
> **(2)** Consider any vertex pair `i` and `j` where `j < i`. Let `d = i − j` be the difference. Since `d < n`, the binary representation of `d` has at most `log n` bits, and consequently, at most `log n` ones.
> Let `k_1, k_2, ..., k_m` denote the positions of ones in the binary representation of `d`. Then we can say:
> `d = Σ_{t=1}^{m} 2^{k_t}`
> This decomposition naturally induces a path from `i` to `j`:
> `i → (i − 2^{k_1}) → (i − 2^{k_1} − 2^{k_2}) → ... → (i − Σ_{t=1}^{m−1} 2^{k_t}) → j`
> The length of this path equals the number of ones in the binary representation of `d`, which is at most `log n`. Therefore, the distance from `i` to `j` is at most `log n`.

### 这个证明的精妙处（为什么它是"优雅"的）

**关键洞察：把二进制表示当成路径。**

任意距离 `d` 都能唯一分解成 2 的幂之和。而每条边的定义就是"索引差是 2 的幂"。所以：

- **二进制里每一个 1 位 → 图上一条边**
- **距离 d 需要的跳数 = d 的二进制里 1 的个数（popcount）**
- **popcount(d) ≤ log₂ d**（最坏情况是全 1，如 `d = 2^k − 1`）

所以 `L` 层能覆盖的最大距离是 `2^L`（指数），而不是 SWA 的 `L × W`（线性）。

**两个性质的对偶关系**（这是设计的漂亮之处）：
- 性质 (1) 是**成本约束**：出度 ≤ log n → 计算量等价于 SWA
- 性质 (2) 是**能力保证**：任意距离 ≤ log n 跳 → 感受野指数增长

**即：在同样的稀疏度下，把"线性覆盖"换成了"指数覆盖"，代价是牺牲"局部连续性"。**

**【核验】** 我用 `d = i − j = 二进制分解` 手工验算了几个例子（`d=7 → 4+2+1 → 3 跳`，`d=8 → 8 → 1 跳`），与证明一致。

---

## 3. PowerAttention 里的实证部分（这才是 V4.1 真正想要的东西）

因为 PowerAttention 没有给出 SWA 上界的**理论证明**，V4.1 那句话只能靠它的**实证**（§4.6 Probing of Information Flow）来支撑。而那部分说的是：

**【原文，§4.6】**，实验设置：28 层 Qwen2-7B，16K 上下文，序列切成 64 个 block（每块 256 token），每层每个位置训一个 logistic 分类器，判断隐状态里是否还含有 passkey 信息（6 选 1，随机猜是 1/6）。

关键结论（Figure 5 的分类准确率，最后一层最后一个 block）：

| 注意力类型 | 末层末块准确率 |
|---|---|
| Full Attention | **1.00** |
| **Sliding Window** | **0.48** |
| PowerAttention（未训练） | 0.56 |
| PowerAttention（后训练） | **1.00** |

**【原文，§4.6 结论段】**：
> "In sparse attentions, this phenomenon is even more evident: **the receptive field of sliding window attention expands progressively across layers at a linear rate**, and the receptive field of POWER ATTENTION, in contrast, exhibits **phase transition-like jumps** across layers..."

**注意**：这里说的是"SWA 的感受野**逐层线性扩张**"——这是在描述**理论上的扩张方式**，**没有**给出"有效感受野 = O(W)"的定量上界。

**而且有一段话意外地帮了 SWA 说话**【原文，§4.6】：
> "This suggests that even though **full attention theoretically allows it to attend to any position in a single step, the attention heads still exhibit a degree of spatial locality**."

——full attention 的理论感受野是全体，但实测信息也是**局部**聚集的。这其实说明"**有效感受野 < 理论感受野**"是个**普遍现象**，不只 SWA 有。

**以及附录 C.2 的一个反直觉数据**【原文】：
> "Interestingly, the model's performance in information flow **degrades post-training**, with accuracy declining from **0.48 to 0.37**."

——SWA 后训练后信息流反而变差（作者归因于 overfitting）。

---

## 4. ★ 真正证明 `O(W)` 上界的那份：MIT HAN Lab 博客

**《Why Stacking Sliding Windows Can't See Very Far》**
作者 **Guangxuan Xiao**（MIT HAN Lab，StreamingLLM 一作），**2025-08-25**，约 20 分钟阅读。
原文：https://guangxuanx.com/blog/stacking-swa.html ｜ 实验室转载：https://hanlab.mit.edu/blog/stacking-swa

**TL;DR（原博客自己的话）**：
> "A mathematical explanation of why sliding window attention's **effective receptive field is O(W) rather than the theoretical O(LW)**, regardless of depth, due to **information dilution** and **exponential decay from residual connections**."

**这正是 V4.1 那句话的内容。** 而且博客里**明确引用了 PowerAttention 作为"理论线性扩张"那一方的图示来源**——也就是说：**博客站在 PowerAttention 的对立面，把它当作"理论乐观派"来反驳。**

### 4.1 推导框架

**定义**：记 `P_l(d)` 为"距离 `d` 的位置对当前位置、经 `l` 层后的影响权重"，满足 `Σ_d P_l(d) = 1`。

**核心假设**【原文】：假设注意力权重在窗口内**平均分布**（每个可见位置权重 `1/W`）。
> "This isn't exactly true for trained models... but it captures the **fundamental architectural bias**—the starting point that any learned pattern must work from."

### 4.2 情形一：纯 SWA，无残差连接

每层是一个均匀卷积：

```
P_1(d) = 1/W        if d < W
       = 0          if d ≥ W

P_l(d) = Σ_{j=0}^{W-1} P_{l-1}(d − j) · (1/W)
```

**关键论证：把"信息回溯距离"看成 L 个独立同分布跳的累加。**

每层"向后跳"的距离均匀分布于 `[0, W−1]`，单跳统计量：

```
μ₁ = (W−1)/2
σ₁² = (W²−1)/12
```

（`σ₁²` 的推导：`(1/W)Σj² − μ₁² = (W−1)(2W−1)/6 − ((W−1)/2)² = (W²−1)/12`）

**由中心极限定理**（重复卷积的类 CLT 行为，原文用 Galton 板作类比），L 层后：

```
μ_L  = L(W−1)/2  ≈ LW/2
σ_L² = L(W²−1)/12
σ_L  = sqrt(L(W²−1)/12) ≈ 0.29·W·√L
```

**有效感受野取高斯分布的宽度**：

```
D_eff^no-res ≈ 2σ_L ≈ 0.58·W·√L        ← 注意是 √L，不是 L！
```

**【核验】** 我用 `W=100, L=100` 算：`σ_L = 288.66`，`2σ_L = 577.32`，而 `0.58·W·√L = 580.00` —— **吻合**。

**【原文注】** 这个 `√L` 的结论不是他们首创，而是从 CNN 的 effective receptive field 研究搬过来的：**Luo et al. (2017)**，arXiv:1701.04128 —— "effective receptive fields follow Gaussian distributions and grow sublinearly with network depth"。

**物理意义：信息稀释**（information dilution）。信息在层间被反复平均，像"传话游戏"一样弥散。

### 4.3 情形二：加残差连接（真实 transformer）—— **这才是"深度无用"的原因**

真实模型是：

```
h_t^(l) = α·h_t^(l−1) + (1−α)·SWA(h^(l−1))_t
```

`α` 是残差路径的"概念强度"。**在真实模型里 `α` 不是显式参数，而是 LayerNorm 等的涌现属性，通常非常接近 1（0.9–0.99）** —— 即 **90–99% 的信息绕过注意力层原样传下去**。

**第一层后形成 "spike-and-slab" 分布**（尖峰 + 平板的术语借自贝叶斯统计）：

```
P_1(d) = α + (1−α)/W     if d = 0        ← 尖峰（当前位置）
       = (1−α)/W          if 1 ≤ d < W    ← 平板（窗口内其它位置）
       = 0                if d ≥ W
```

**数值直觉**【原文】：`α = 0.95, W = 100` 时，当前位置权重 `0.9505`，窗口内其它每个位置 `0.0005` —— **相差 1900 倍**。

**递推关系**：

```
P_l(d) = α·P_{l−1}(d) + ((1−α)/W)·Σ_{j=0}^{W−1} P_{l−1}(d−j)
```

**关键论证**：要传播到距离 `d > W`，信息**至少要走 `k = ⌈d/W⌉` 次"注意力跳"**（因为每次注意力跳最多桥接 `W` 个位置）。而**每走一次注意力路径（而不是残差路径），就被乘一次 `(1−α)`**。所以：

```
P_l(d) ≤ C · (1−α)^⌈d/W⌉
```

（原文说"上界在实践中也是极好的近似"）大 `d` 时写成指数衰减：

```
P_l(d) ≈ C · (1−α)^(d/W) = C · e^(−λd),   λ = −ln(1−α)/W
```

**指数直觉**【原文】：`α = 0.95` 时——1 个窗口宽度后剩 5%，2 个后剩 0.25%，3 个后剩 0.0125%。

### 4.4 有效视野公式（最终结果）

```
情形一（无残差，α = 0）：
    D_eff^no-res ≈ 2σ_L ≈ 0.58·W·√L
    → 随深度增长，但是 √L（次线性）

情形二（有残差，α > 0）：
    令 P_l(D_eff) = ε（取 ε = 0.01，即衰减到 1%）
    (1−α)^(D_eff/W) = ε
    ⇒ D_eff^res = W · ln(ε)/ln(1−α) = W · |ln ε| / |ln(1−α)|

    ε = 0.01 时（|ln 0.01| = 4.6052）：
    D_eff^res ≈ 4.6·W / |ln(1−α)|
    ★ 与 L 无关！
```

**【核验】** 我逐项算过这张表（`|ln ε| = 4.6052`）：

| α | 1−α | `\|ln(1−α)\|` | `4.6/\|ln(1−α)\|` | 倍数 × W |
|---|---|---|---|---|
| 0.90 | 10% | 2.3026 | 1.9978 | **≈ 2.0 × W** |
| 0.95 | 5% | 2.9957 | 1.5355 | **≈ 1.5 × W** |
| 0.98 | 2% | 3.9120 | 1.1759 | ≈ 1.2 × W |
| 0.99 | 1% | 4.6052 | 0.9989 | ≈ 1.0 × W |

与原文给的表格**完全一致**。

**过渡行为**【原文】：`α → 0` 时 `|ln(1−α)| ≈ α → 0`，于是 `D_eff^res → ∞`——指数壁垒消失，此时重新由 `O(W√L)` 的高斯扩散主导。

**"为什么加深层没用"的论证**【原文】：
> "Even if information *could* travel 100,000 tokens (which would require going through all 100 layers), its influence would be `(1−α)^100 = 0.05^100 ≈ 10^-130`. For perspective, that's an astronomically small number."
>
> "Depth does not extend the model's effective horizon. After about `D_eff/W` layers (**typically 2-3 layers**), you've already reached the maximum useful distance."

### 4.5 根本权衡（原文的重要结论）

```
需要稳定训练  →  高 α  →  局部性偏置（D_eff 小）
想要长上下文  →  低 α  →  训练不稳定
```

> "This isn't a bug—it's a fundamental trade-off baked into the architecture."

并以此解释**为什么混合架构（hybrid）重要**：把问题切块，块内用 SWA，块间用少量 full attention **打破指数壁垒**。

**【原文对线性注意力的展望】**：SWA 可视为最简单的线性注意力；Mamba / DeltaNet / Gated DeltaNet 等也压缩历史信息，"与残差网络结合后，远距离精确召回的实际能力可能面临类似的衰减挑战"——**但明确标注"may / could"，是推测不是结论**。

---

## 5. 这对 V4.1 意味着什么（把两边接起来）

V4.1 的推理链条是：

```
Chen et al., 2025  声称：SWA 实际感受野「远小于」理论上界 n_win × L/2
        ↓
所以：Decoder SWA Bounded Replay 只 replay 最近 n_win 个 token 是合理的
        ↓
于是：decoder 前向被限制在 n_win 内，prefill 计算量几乎砍半
```

用博客的公式来检验 V4.1 的 `L/2` 上界：

| 量 | V4.1 的 | 博客的 |
|---|---|---|
| 理论感受野 | `n_win × L/2`（`L/2` 是 decoder 层数） | `L × W` |
| **有效感受野** | 未给公式，只说"much smaller" | **`≈ 1.5 W`（α=0.95，与层数无关）** |

**【推断】** 代入典型值 `L/2 = 20` 层：理论上界 `20 × n_win`，而博客给出的有效视野是 `≈1.5 × n_win` —— **约 13 倍差距**，而且**差距随层数增加而拉大**（因为一边是线性、一边是常数）。

**这恰好为 Bounded Replay 提供了定量依据**：
- Bounded Replay 只重放 `n_win` 个 token，相当于把 decoder 的 SWA 视野**截断到 1.0 × W**
- 而博客说有效视野本来就只有 `≈1.5 × W`（`α=0.95`）
- **所以截断到 1.0 × W 损失的信息，落在那 0.5 × W 的"半衰区"里** —— 这与 V4.1 报告说的"approximate state"、"barely compromises response quality"在数量级上自洽

**【推断】** 这也解释了 V4.1 为什么强调 "for **multi-turn interactions with short prompts per turn**, this computational overhead becomes non-negligible" —— 短 prompt 多轮场景下，`n_win × L/2` 的 replay 成本相对于短 prompt 本身是巨大的，而收益（多看到 0.5W）却很小。

**⚠️ 但有个重要的不对称**：博客的推导假设**均匀注意力 + 单一 α**，是**架构先验**（architectural bias），不是训练后的实际行为。V4.1 面对的是**训练过的模型**，其 α 可能逐层不同、注意力也不是均匀的。所以博客公式应当看作**量级估计**，不是可代入的精确公式。**这一点原文自己也承认**（"This isn't exactly true for trained models"）。

---

## 6. 阅读顺序与验证实验

### 6.1 直接读什么

| 目的 | 读什么 |
|---|---|
| 理解 V4.1 引用的那篇 | 本目录 `2026-09-PowerAttention-2503.03588.pdf`（15 页，重点 §3、Appendix B） |
| **理解 `O(W)` 上界** | **MIT HAN Lab 那篇博客**（`guangxuanx.com/blog/stacking-swa.html`）—— 本文第 4 节已完整复述 |
| 上界的理论源头 | Luo et al. 2017, arXiv:1701.04128（CNN 的 effective receptive field） |
| 实证方法（probing） | PowerAttention §4.6 + Appendix C.1（线性分类器探针） |

### 6.2 可验证的小实验（Mac 上就能做）

博客的推导有两个**关键假设**，都可以用数值模拟检验：

1. **均匀注意力假设**：用纯 numpy 复现 `P_l(d)` 的卷积递推，验证
   - 无残差时是否真是高斯、`σ_L ≈ 0.29W√L`
   - 有残差时是否真是指数衰减、`D_eff ≈ 1.5W`（α=0.95）
2. **"深度无用"结论**：把 `L` 从 10 扫到 1000，看 `D_eff` 是否真的**不随 L 变化**（有残差时）

**【推断】** 这个模拟的价值在于：你可以**自己造一个反例**——比如用 Zipf 分布（模拟真实注意力的重尾特性）代替均匀分布，看 `O(W)` 结论是否还成立。如果成立，说明结论稳健；如果不成立，说明博客的结论**依赖于均匀假设**，这是一个可写的技术点。

### 6.3 与 V4.1 的"未解决问题"连接

V4.1 Section 6 明确列了未解决问题：
> "**approximate state reconstruction in SWA Bounded Replay** may still cause capability degradation in untested boundary cases... with particular attention to... **SWA state reconstruction at cache-resumption boundaries**"

**用本文第 4 节的公式，可以给这个"未刻画"的问题一个定量抓手**：Bounded Replay 的误差应该与 `(1−α)^(Δ/W)` 同阶，其中 `Δ` 是 replay 起点到真实依赖起点的距离。**这是一个可以设计实验验证的具体假设。**

---

## 7. 证据附录

### 7.1 V4.1 报告中的引用位置
- 正文引用：`v41_report.txt:366-367`（§2.2 CED）
- 参考文献条目：`v41_report.txt:1916-1918`
  > L. Chen, D. Xu, C. An, X. Wang, Y. Zhang, J. Chen, Z. Liang, F. Wei, J. Liang, Y. Xiao, et al. **Powerattention: exponentially scaling of receptive fields for effective sparse attention.** arXiv preprint **arXiv:2503.03588, 2025**.
- 另一处易混淆条目：`v41_report.txt:1919-1920`（`L. Chen ... Babyvision ... arXiv:2601.06521, 2026`）—— **不是同一篇**
- 还有一篇容易被误认的：`L. Chen, D. Xu, ... InfiniteHiP ... arXiv:2502.08910`（**【核验】** 我查过，是"3M token 单卡上下文"的推理框架，**与感受野上界无关**）

### 7.2 PowerAttention 关键位置
| 内容 | 位置 |
|---|---|
| 定义点（把稀疏注意力建模成 DAG 边集） | §3.1 Problem Formulation |
| 既有方法最短路径分析（SWA 需 `O(N)` 层） | §3.2 Limitations of Existing Sparse Attention |
| 指数感受野论断 | §3.3 PowerAttention |
| **形式化定理 + 证明** | **Appendix B（Theorem B.1）** |
| 实证 probing | §4.6 + Appendix C.1/C.2 |
| RULER 等基准数字 | 正文表格（`powerattention.txt:581-584`） |

### 7.3 MIT HAN Lab 博客关键位置
| 内容 | 位置 |
|---|---|
| TL;DR（O(W) vs O(LW)） | 开篇 |
| `P_l(d)` 定义与均匀假设 | Part I |
| 无残差 → 高斯 → `0.58W√L` | Part I "The Result: O(W√L) Growth" |
| spike-and-slab 分布 | Part II "The Spike-and-Slab Pattern" |
| 指数衰减与 `D_eff = W·ln ε/ln(1−α)` | Part II–III |
| 数值表（α=0.90/0.95/0.98/0.99） | Part III "Putting the Formula into Practice" |
| 根本权衡与混合架构 | "The Fundamental Dilemma" |

### 7.4 核验命令与结果
```
# 公式核验（ε=0.01, |ln ε|=4.6052）
α=0.90 → 4.6/2.3026 = 1.9978 ≈ 2.0×W
α=0.95 → 4.6/2.9957 = 1.5355 ≈ 1.5×W
α=0.98 → 4.6/3.9120 = 1.1759 ≈ 1.2×W
α=0.99 → 4.6/4.6052 = 0.9989 ≈ 1.0×W
→ 与原文表格完全一致

# σ_L 核验（W=100, L=100）
σ_L = sqrt(100·(100²−1)/12) = 288.66
2σ_L = 577.32  vs  0.58·W·√L = 580.00   → 吻合
```

---

## 8. 无法确认 / 需注意

1. **V4.1 是否知道那篇博客**：无法确认。可能是（a）引用错位，（b）他们内部有别的未公开依据，（c）博客内容在他们内部以其他形式流传。**我只确认了博客不在 V4.1 参考文献列表里。**
2. **`α` 在真实模型里的实际值**：博客说"typically 0.9–0.99"，但**未给出测量方法与具体模型**。这是一个**未经验证的关键参数**——`D_eff` 对 α 极度敏感（α 从 0.95→0.99，`D_eff` 从 1.5W 降到 1.0W）。
3. **均匀注意力假设的偏移量**：真实注意力是重尾的（attention sink、检索头等），博客承认这是"architectural bias"而非实际行为，**但未量化真实分布会让结论偏移多少**。
4. **V4.1 的 `n_win` 具体取值**：报告未给出 V4.1 的窗口大小，因此无法把 `1.5 × n_win` 换算成绝对 token 数。
5. **博客未经同行评审**：是研究博客（有 BibTeX 但非正式发表）。引用时宜标注为 blog post。


## 文献版本说明

本文所引 DeepSeek-V4.1-Flash 技术报告为原笔记记录的 2026-09-10、51 页版本（文件名 `DeepSeek_V41_Tech_Report.pdf`），页码对应这一版本。本次整理未获得可独立确认的公开下载链接，未将 PDF 打包进本站；涉及该报告的数值沿用原笔记，仍需对照原文复核。
