---
title: "V4.1 的局限、自改进机制与证据边界"
date: 2026-09-15
lastmod: 2026-09-21
draft: false
visibility: public
categories: ["模型架构"]
tags: ["DeepSeek","RSI"]
description: "V4.1 的局限、自改进机制与证据边界；保留技术细节、出处与适用边界。"
---

> 这是技术笔记的公开整理版，原始记录日期为 2026-09-15，本次编辑于 2026-09-21。保留原笔记的公式、代码位置与证据分级；本次仅整理内容，未重新运行 GPU 实验或逐项复核上游。标为【码】【核验】的内容指原记录的核查结果，不代表当前版本仍然如此。


---

## 零、三个问题的直接回答

| 问题 | 回答 |
|---|---|
| **下一步模型发展方向？** | 论文**没有单列 roadmap**，但从散落陈述可归纳出四条，且**主线不是架构，而是"数据与环境的工程"** |
| **承认了哪些不足？** | 分四类共 **12 条**（见 §2）。**最重的一条是"与闭源前沿仍有明显整体差距"** |
| **提到 RSI 吗？** | **没有。** 全文搜 `RSI` / `recursive self` / `self-improv` / `self-evolv` / `self-modif` / `automated research` → **全部零命中** |

**但第三问有个重要的补充**：论文虽然**从不使用 RSI 这个词**，却在 §5.1.1 描述了一套**结构上就是自改进循环**的机制。这是本文最值得读的一段，见 §3。

---

## 一、下一步发展方向（四条，按论文强调程度排序）

### 方向 1：数据与环境的工程 > 算法创新（**论文的首要论断**）

**【原文，§5.1】** 开篇就说：

> "In this release, we **refrain from introducing novel post-training algorithms**. The overall recipe follows the standard paradigm of SFT → RL → OPD, **without algorithmic modifications** beyond well-established practices. Instead, our efforts are concentrated almost entirely on **what the model is trained on rather than how it is optimized**."

接着给出**整个报告里最强的一句判断**：

> "We find that, under a fixed and unremarkable optimization procedure, systematic improvements in the **scale, diversity, and verifiability of synthesized data and environments** account for **essentially all of the observed gains**. This observation echoes a broader lesson: at the current stage, the **marginal return of engineering the data and environment pipeline substantially exceeds that of algorithmic novelty in post-training**."

**⭐ 这是一个战略级判断，不是技术细节。** 摘要里也重申："introduces **no algorithmic innovation**... All substantive changes lie instead in the data pipeline."

### 方向 2：可控推理努力（Controllable Reasoning Effort）

**【原文，§5.1.4】** 用一个标量 `b ∈ {1,...,100}` 作为显式条件信号，拼进 system prompt：

> "Reasoning Effort: {effort} (range 1–100; higher values request more thorough reasoning)"

机制细节：
- 每个 prompt `x` 在**每个 effort level** `b ∈ B` 采样 `M_b` 个回答
- **同一 `(x,b)` 组内做组相对优势（mean-centered）** ← 关键：**不同 effort level 的回答不直接比较**
- effort 相关的行为**通过让 reward 的长度惩罚项依赖 `b` 来诱导**：
  `r^len_{b,j} = −min{ C_max, k(b) · ℓ_{b,j} / L_norm }`
- 公开 API 暴露三档：`low/high/max` ↔ `b = 50/75/100`

**【推断】** 这个设计很聪明的地方：**不是训一个模型再调采样参数，而是把"该想多久"变成模型的一个条件能力**。它让同一个 checkpoint 覆盖 cost–quality 前沿，且 effort 是**训练出来的**而非外挂的。

### 方向 3：Model–Harness 协同设计（Co-design）

**【原文，结论章】**：
> "We will also actively **integrate model–harness co-design**, enabling the joint system to evolve and be optimized together."

论文对此已有实证投入（§5.3.4）：**跨 6 个 scaffold 家族、8 种配置**评估（Claude Code / Codex / OpenCode / Pi / mini-SWE / DeepSeek Harness 的 Minimal/Standard/PTC 三模式），且**联合训练多个 scaffold**。

**【原文，§5.3.4】** 给出的理由：
> "a model that **overfits to one particular harness may degrade substantially** when placed in another"


### 方向 4：协同 scaling（数据 + 模型容量 + RL）

**【原文，结论章】**：
> "further advances in model intelligence will depend on the **coordinated scaling of data, model capacity, and RL**. ... systematically address key challenges in **large-scale data synthesis and RL scaling**."

架构侧的"继续"信号只有一句（结论章开头）："DeepSeek-V4.1-Flash also serves as a **new starting point for our continued scaling efforts**"——**没有透露下一代架构**。

---

## 二、承认的不足（穷尽核对，共 12 条）

### A 类：能力差距（3 条）

| # | 不足 | 原文位置 | 原文 |
|---|---|---|---|
| **1** | **与闭源巨头仍有明显整体差距**（最重的一条） | §5.3.1 | "**we acknowledge that a distinct overall performance gap remains** when compared to giant closed-source systems." |
| **2** | 多模态与闭源前沿有可测量差距 | §5.3.2 | "**we acknowledge that a measurable gap still remains** when benchmarked against the leading closed-source alternatives." |
| **3** | 科学类 agentic 任务有差距 | §1 | "a **gap with giant models remains** on science-oriented agentic tasks, such as Terminal-Bench 4.0" |

配套（§5.3.1）："can already match closed-source frontier models on the **vast majority** of benchmarks, and is capable of completing **over 95%** of real-world tasks" —— 95% 这个数字本身也是"还有 5% 做不到"的承认。

### B 类：鲁棒性边界（4 条，**报告自己列为未解决问题**）

**【原文，结论章】**：
> "the newly introduced architectural changes also create **robustness boundaries that have yet to be fully characterized**. ... **no finite test suite can cover every extreme input and deployment condition**."

| # | 不足 | 报告承诺 |
|---|---|---|
| **4** | **CSA2 的 selection errors** | 扩大 stress-testing |
| **5** | **SWA Bounded Replay 的近似状态重建** | 同上 |
| **6** | **长上下文稀疏检索**（sparse retrieval over long contexts） | 重点关注 |
| **7** | **缓存恢复边界的 SWA 状态重建** | 重点关注 |

### C 类：方法论/自我评价（3 条）

| # | 不足 | 原文位置 | 原文 |
|---|---|---|---|
| **8** | **参数选择主要靠经验** | §2.6 | "our choices for those parameters are **mostly empirical**" |
| **9** | **基准测试已饱和**，分数接近不等于能力齐平 | 结论章 | "standard evaluation benchmarks have increasingly **reached saturation**... this parity **does not imply** that the model matches the frontier capabilities... on complex, high-difficulty reasoning and **edge cases**." |
| **10** | **主动接受近似**（设计上的自觉妥协） | §3.2.2 | "**accepting approximate states**"；bounded replay 后 "the global KV and SWA KV computed for the uncached suffix **depend on the cache-hit position** and are **not mathematically identical across positions**" |

### D 类：架构/系统的未竟之处（2 条）

| # | 不足 | 原文位置 | 说明 |
|---|---|---|---|
| **11** | **mHC 没做到理论下界** | §2.4.1 | 两趟实现是 `(3n+2)d`，**理论下界 `(2n+2)d`** —— 差的那一截没做到 |
| **12** | **SWA KV 精确恢复被证明不可行** | §3.2.1 | V4 提的 Zero SWA Caching "**proved prohibitive in production deployments**"（精确恢复要跑 `L × n_win` 个 token 的前向） |

另有几处"软性承认"：
- 优化器归一化策略的**通信开销未被研究**（§2.5，唯一用 "future direction" 字样的一处）：
  > "These normalization strategies may differ in optimization performance and **communication overhead**. **We leave more detailed investigation as a future direction.**"
- **前人工作没覆盖三个维度**（§2.3）：
  > "**none of these methods covers all three multiplicative dimensions**"
- **DSec 的 agent 作恶只能事后处理**（§5.1.3）：agent 崩溃时"we treat the crash as a **failed trajectory**"——即**检测到了，但只能当失败样本，不能阻止**
- **DSpark 是独立训练的**（§2.1）：backbone 预训练时省略 MTP 模块，DSpark 在 backbone 预训练**之后单独训练**

---

## 三、有没有 RSI？—— 没有这个词，但有这个机制

### 3.1 字面回答：【核验】零命中

```
grep -niE "\bRSI\b|recursive self|self-improv|self-evolv|self-play|self-reward|automated research|self-modif"
→ 唯一命中："re-bootstrapped it from Common Crawl"（数据管线重建，无关）
```

### 3.2 但 §5.1.1 描述的是一套**自改进循环**

**【原文，§5.1.1 Large-Scale Agent Task Synthesis】** —— 请逐句读：

> "Tasks serve as the fundamental fuel for agent learning. However, constructing high-quality training tasks has traditionally required substantial manual effort. **We observe that the model is already beginning to exhibit the ability to construct its own training tasks, though this capability remains far from perfect. Recognizing this potential, we have invested considerable effort in strengthening the model's task-construction and quality-verification abilities.**
>
> We formalize each task as a triplet **(problem, environment, verification system)** and evaluate its quality along two dimensions: **difficulty**—ensuring the task is non-trivial—and **correctness**—guaranteeing that no critical flaws exist among the three components. **Using difficulty and correctness as reward signals, we iteratively train the model to construct better tasks.**
>
> We also **monitor RL tasks across their full lifecycle**. Whenever a task is used in a new RL run, the resulting **trajectories provide fresh evidence for quality re-auditing**."

**把这段拆成循环，就是标准的自改进闭环**：

```
① 模型自己生成任务 (problem, environment, verification system)
        ↓
② 用 difficulty + correctness 当 reward
        ↓
③ 迭代训练模型"构造更好的任务"
        ↓
④ 任务被用于新一轮 RL
        ↓
⑤ 新轨迹成为"质量重新审计"的新证据  ──┐
        ↓                              │
        └──────────────────────────────┘  闭环
```

**而且还有一层"失败驱动的课程"**【原文，§5.1.1】：
> "we collect negative feedback and model failure cases submitted by internal employees **at scale**, and incorporate them into the pipeline to generate both single-turn and multi-turn agent environments grounded in real workflows. ... the pipeline enables **systematic replay of failures and targeted reinforcement learning against observed model weaknesses**."

### 3.3 但必须严格区分：这是**数据层**的自改进，不是 RSI 的完整形态

**【推断】** RSI（递归自我改进）严格的含义是"系统改进**自己**"。DeepSeek 这段描述的是：

| 维度 | DeepSeek V4.1 做的 | 完整 RSI 会要求 |
|---|---|---|
| 自生成**任务/环境** | ✅ 明确做了 | ✅ |
| 自评估**质量** | ✅ difficulty + correctness | ✅ |
| 自生成**训练算法** | ❌ **明确否认**（"refrain from introducing novel post-training algorithms"） | ✅ |
| 自改进**infra / 系统** | ❌ 未提及 | ✅ |
| 自改进**架构** | ❌ 未提及 | ✅ |

**所以准确表述是**：**论文在做"数据与环境的自改进（self-improving data engine）"，刻意不做"算法与架构的自改进"，并且全文回避 RSI 这个提法。**

### 3.4 一个我认为很关键的连接

**DeepSeek 在同一个报告里，同时给出了"agent 还差什么"的最强证据。**

竞赛方案 `report.pdf` p3 §6.3 原文：
> "roughly half of the contest traces have a valid token count that fits inside K = 2048... The **human noticed this pattern while reading the trace metadata**. We then re-ran the autonomous loop multiple times to see whether it would re-discover the optimization on its own. **It did not.** The pattern repeated across runs: **the agent reasoned over the kernel's code, not over the kernel's inputs.**"

**【推断】** 把两件事放在一起看：
1. **§5.1.1**：我们正在训练模型自己造任务（自改进数据引擎）
2. **竞赛报告 §6.3**：agent 反复重跑都**发现不了**人类一眼看出的 workload 级优化，因为它"读代码不读输入分布"

**第二条精确地指出了第一条的天花板在哪** —— 一个只能推理代码、不能推理输入分布的 agent，造出来的任务会系统性地偏。**这解释了报告为什么自己承认"this capability remains far from perfect"。**

👉 **这是我认为整个 V4.1 生态里最重要的"自我认知"**：他们知道自改进的瓶颈不在生成能力，而在**对真实分布的观察能力**。

---

## 四、四条线的层级关系（我的归纳）

论文实际上在四个层次上分别做"自改进"，且**只有第二层是明确声称的**：

```
① 数据/任务层   模型自造任务 + 自审计 + 失败回放        ← §5.1.1 明确声称
② 后训练层      RL + OPD（多 teacher 蒸馏）             ← §5.2.4 明确声称
③ 环境层        DSec：百万级并发沙箱                    ← §5.1.3 基础设施
④ Harness 层    DSH 的 Minimal/Standard/PTC + 跨 scaffold 联合训练  ← §5.3.4
```

**【推断，必须标注为推断】** 把 ① 和 ④ 合起来看，就是一个"模型参与塑造自己训练环境与交互协议"的方向。但论文**从未把它表述为 RSI 或自进化**，只说 model–harness **co-design**。

**为什么这个区分重要**：如果你要引用"DeepSeek 在做 RSI"，**论文里没有这个主张**。准确的说法是"**DeepSeek 在做自改进的数据引擎 + 模型-harness 协同设计，并明确否认算法创新**"。

---

## 六、无法确认 / 需注意

1. **下一代架构完全未知**。论文只说"a new starting point for our continued scaling efforts"，**没有透露任何架构方向**。
2. **§5.1.1 的"iteratively train the model to construct better tasks"具体怎么实现，报告没给细节**（没给数据量、没给迭代轮数、没给质量提升曲线）。
3. **"essentially all of the observed gains 来自数据"这个论断没有消融实验支撑**（至少报告里没给）。**【推断】** 这是一个强主张，但读者只能信，不能验。
4. **我不确定 DeepSeek 是否在内部讨论 RSI** —— 只能说**公开报告里零提及**。
5. **竞赛报告与 V4.1 报告是不同作者群**（竞赛方案是单作者 Doğaç Eldenk / Northwestern，"Human-AI collaboration"），我把它和 §5.1.1 并列是**我的推断**，不是论文的原话。


## 文献版本说明

本文所引 DeepSeek-V4.1-Flash 技术报告为原笔记记录的 2026-09-10、51 页版本（文件名 `DeepSeek_V41_Tech_Report.pdf`），页码对应这一版本。本次整理未获得可独立确认的公开下载链接，未将 PDF 打包进本站；涉及该报告的数值沿用原笔记，仍需对照原文复核。
