---
title: "Harness Engineering for Self-Improvement （RSI） 深度解读"
date: 2026-07-10
lastmod: 2026-09-21
draft: false
visibility: public
categories: ["模型与评测"]
tags: ["图文研讨", "Harness"]
description: "Harness 与递归自我改进的图文解读。"
---

> **阅读说明**：本页保留研讨原稿的正文与图表；本次仅做网站排版和资源迁移，未重新核验数据、引用及图中结论。涉及 2026 年底至 2027 年的内容属于预测，不代表已经发生的事实。原文中的核验记录属于原记录时点。

> **原文**: [Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/) — Lilian Weng (OpenAI 前预训练负责人)（）

> **发布日期**: 2026 年 7 月 4 日 · 预估阅读时长 31 分钟

> **本文概述**: 本报告基于 Lilian Weng 的博客文章，系统介绍 **Harness**（模型执行编排系统）的核心概念、设计模式、优化方法、典型案例与技术局限，并以中文撰写、专业术语保留英文。

---

### 目录

1. [什么是 Harness？](#1-什么是-harness)
2. [Harness Design Patterns（设计模式）](#2-harness-design-patterns设计模式)
3. [Case Study: Coding Agent Harness](#3-case-study-coding-agent-harness)
4. [Harness Layer vs Core Intelligence](#4-harness-layer-vs-core-intelligence)
5. [Harness Optimization（优化方法）](#5-harness-optimization优化方法)
6. [Harness 案例研究](#6-harness-案例研究)
7. [Future Challenges（未来挑战）](#7-future-challenges未来挑战)
8. [附录：相关 Benchmark](#8-附录相关-benchmark)
9. [参考文献](#9-参考文献)

---

### 1. 什么是 Harness？

#### 1.1 Recursive Self-Improvement（递归自我改进）的起源

**Recursive Self-Improvement（RSI）** 的概念可以追溯到 I. J. Good (1965)，他定义了"超智能机器"——一个能在所有智力活动上超越人类、并设计更好的机器来改进自身的系统。Yudkowsky (2008) 将其明确为"递归自我改进"：**AI 利用当前智能来改进产生其智能的认知机制本身**。

在现代 AI 中，这一反馈循环可能意味着模型直接改写自身权重，也可能更广泛地指模型改进训练 pipeline 和**部署系统（deployment system）**，从而催生性能更强的下一代模型。

#### 1.2 Harness 的定义

> **Harness 是围绕基础模型的系统，负责编排执行，决定模型如何思考和规划、调用工具和行动、感知和管理上下文、存储产出物，以及评估结果。**

Lilian Weng 特别强调"deployment system"的重要性——**模型与现实世界之间的这一层，与模型的原始智能同样重要**。以 Claude Code、Codex 等成功的 Coding Agent 产品为例，Harness 是它们的关键组成部分。

#### 1.3 Harness vs 早期 Agent 框架

相比早期的 Agent 框架公式 `Agent = LLM + Memory + Tools + Planning + Action`，Harness Engineering 额外包含：

| 维度 | 早期 Agent 框架 | Harness Engineering |
| --- | --- | --- |
| 核心形态 | Prompt 模板 | 运行时 + 软件系统设计 |
| Workflow | 无 / 简单链 | 循环工程（Loop Engineering） |
| 评估 | 无内置评估 | 集成 Evaluation |
| 权限控制 | 无 | Permission Controls |
| 持久状态 | 无 / 简单记忆 | Persistent State Management |
| 类比 | — | 操作系统（OS） |

Harness 的设计应当**刻意简单和通用**以实现泛化，借鉴已有的软件工程实践。Harness 与操作系统之间存在强类比关系——如同 OS 一样，Harness 应该在保持接口简单的同时封装复杂逻辑。

---

### 2. Harness Design Patterns（设计模式）

#### Pattern 1: Workflow Automation（工作流自动化）

定义一个模型能够在其中**操作、测试和迭代**的工作流，是自动化的关键设计。典型的工作流遵循一个目标驱动的循环：

```
// 代码块
Plan → Execute → Observe/Test → Improve → Execute Again → … 直到目标达成
```

Karpathy 的 [autoresearch](https://github.com/karpathy/autoresearch) 仓库是一个清晰的实例。

![图片](media/session/a7876d92a8fc4682.png)

*图 1: 简化的 Codex Agent 循环——Agent 调用工具，工具响应影响模型的下一步生成。（来源: OpenAI）*

工作流图还强调模型**分析自身的 trajectory 和失败案例**，然后通过"Agent Runtime"（而非静态 prompt 模板）来迭代进展。

#### Pattern 2: File System as Persistent Memory（文件系统作为持久记忆）

在长 horizon Agent 系统中，一个反复出现的模式是**对丰富状态和产出物的简单控制**。

**核心原则**: Harness 不应将整个工作流和所有日志都放在 context 中，而应将**持久状态保存在文件中**。

在长 horizon 的 agentic rollout 中，实验日志、代码 diff、论文摘要、错误堆栈、过去的 rollout trajectory 等产出物通常远超模型训练时的 context window 长度。

**为什么文件系统是好选择？**

- 学习读写和编辑文件系统（通常通过 bash 命令）是 LLM 的基础技能
- 以文件形式管理持久记忆，**天然受益于核心模型能力的提升**

#### Pattern 3: Sub-agent and Backend Jobs（子代理与后台任务）

Harness 可以**生成多个子代理并行执行**并监控后台任务。这在以下场景中特别有用：

- 主 Agent 需要搜索多个假设
- 并行运行实验
- 委派隔离的子任务而不污染主 context

**关键设计选择**: 让并行性**显式且可检查**。

| 子代理输出存储方式 | 效果 |
| --- | --- |
| 仅在临时 chat context 中 | 很快变得过时和隐藏 ❌ |
| 存储为文件、日志和状态记录 | 模型可在中断后恢复，并推理自身的执行历史 ✅ |

---

### 3. Case Study: Coding Agent Harness

主流 Coding Agent（Claude Code、Codex、OpenCode、Cursor 等）的核心接口已经趋于稳定。它们共同使用一个如下的循环：

![图片](media/session/f4efb3c97ce86af5.png)

*图 2: Coding Agent Harness 的标准循环。*

#### 工具集一览

| 分类 | 工具定义 |
| --- | --- |
| **File System** | File discovery: `glob`, `grep`, `ls`; File read: `read`, `read_many`; File modification: `write`, `edit`, `multi_edit`, `apply_patch` |
| **Shell Execution** | `bash`, `PowerShell` |
| **IO** | `lsp`, `git_status`, `git_diff`, `git_commit` |
| **External Context** | MCP Tools, Skills |
| **Web Search** | `web_search`, `web_fetch`, browser tools |
| **Artifacts** | Read docs/images; generate HTML/images |
| **Backend Processes** | `CronCreate`, `CronDelete`, `CronList` |
| **Agent Delegation** | `spawn_agent`, `resume_agent`, `wait_agent`, `list_agents`, `close_agent`, `interrupt_agent` 等 |

通过这些工具，Coding Agent 能够在给定仓库中开发和调试问题，类似于人类开发者使用 IDE 的方式。

---

### 4. Harness Layer vs Core Intelligence

Lilian Weng 的预测：

> **近期 RSI 的实际路径不太可能从模型直接改写自身权重开始。**

**近期路径预测:**

1. **Harness Engineering 将朝 meta-methodology 方向演进** —— 改进获取更好答案的"机制"（machinery），而非仅仅改进答案本身。Harness 系统本身成为优化目标，规则越来越少、机制越来越通用。

1. **反向促进**: 成熟的 Harness 使 auto-research 成为可能，实现模型自我改进循环；更聪明的模型又防止 Harness 过度工程化，保持系统可持续。

最终，许多 Harness 改进可能**被内化为核心模型行为**，但与外部上下文和工具的接口应当保留。正如 Prompt Engineering 的演变——随着 instruction tuning 和模型推理的改进，手动 prompt 技巧变得不再那么核心，但**指定目标、约束、上下文和评估的需求并没有消失**。

---

### 5. Harness Optimization（优化方法）

Harness 系统中被优化对象的演进大致为：

```
// 代码块
Instruction Prompts → Structured Context → Workflow → Harness Code → Optimizer Code
```

随着模型变得更智能和强大，我们向**更复杂的优化目标和更通用的方法**前进。

#### 5.1 Context Engineering

简单地将所有工具响应和模型生成追加到 context 中，随着 agentic 任务 horizon 的增长会迅速失控。Context Management 是一个构建**更结构化、更精简的 context** 并管理持久状态的层。

##### ACE: Agentic Context Engineering

ACE (Zhang et al. 2025) 将 context 视为**不断演化的 playbook**，而非不断变长的 prompt，包含三个组件：

- **Generator**: 生成任务 trajectory，参考 bullet points
- **Reflector**: 从成功和失败的 trajectory 中提炼 insights
- **Curator**: 用增量的、条目化的条目更新结构化 context

![图片](media/session/0fbdfaa639c75b68.png)

*图 3: Agentic Context Engineering (ACE) 框架。（来源: Zhang et al. 2025）*

**关键设计**: Curator **不重写完整的 prompt blob**，而是输出结构化的、条目化的 bullets `(identifier, description)`，通过确定性逻辑合并到结构化 context logbook 中。

##### MCE: Meta Context Engineering

MCE (Ye et al. 2026) 将**机制**（如何管理 context）与**内容**（context 中有什么）分离，在 meta-optimization 层运行 skill 演化，在 base 层运行 context 优化。

MCE skill $s \in \mathcal{S}$ 定义一个 context 函数 $c_s = (\rho_s, F_s)$:

- $\rho_s$: 静态组件（prompts, knowledge bases, code libraries）
- $F_s$: 动态算子（search, selection, filtering, formatting）

**双层优化**:

$$\text{Inner: } c_s^* = \arg\max_{c_s} J_{\text{train}}(c_s; s) \quad \text{Outer: } s^* = \arg\max_{s \in \mathcal{S}} J_{\text{val}}(c_s^*)$$

![图片](media/session/b029e8a00415e024.png)

*图 4: Meta Context Engineering (MCE) 框架——meta 层的 skill 演化搜索 context 管理机制，base 层优化任务 context。（来源: Ye et al. 2026）*

MCE 不强制特定的 context 结构规则（不像 ACE），而是使用自由形式的 skills 存储任务最重要的知识，并迭代地同时演化 skill 和 skill 条件下的 context。

##### Meta-Harness

Meta-Harness (Lee et al. 2026) 更深一层：**被优化的对象是决定和优化什么信息应该被存储、检索和呈现给模型的代码**。"Meta-" 意味着它是**优化 Harness 的 Harness**。

![图片](media/session/382d9c707bebbe57.png)

*图 5: Meta-Harness 外循环优化算法。（来源: Lee et al. 2026）*

![图片](media/session/11c0d4cf8735a06d.png)

*图 6: Meta-Harness 在文本分类和 TerminalBench-2 上的表现。（来源: Lee et al. 2026）*

**核心教训**: 一旦 Harness 设计变成可执行的搜索空间，强大的 Coding Agent 就能利用与人类工程师相同的设计空间。

#### 5.2 Workflow Design（工作流设计）

工作流设计可以由领域专家手工制定。以 auto-research 为例：

##### AI Scientist

AI Scientist (Lu et al. 2026) 构建了一个 pipeline: 提出研究想法 → 编写代码 → 运行实验 → 分析结果 → 撰写论文 → 进行同行评审。

![图片](media/session/b0d204e24a6c0326.png)

*图 7: AI Scientist pipeline——从想法生成到实验、论文写作和评审。（来源: Lu et al. 2026）*

##### ScientistOne

ScientistOne (Meng et al. 2026) 以**可验证性**为核心设计约束，每个声明（引用、数值、方法论、结论）必须追溯到证据来源，并通过 Chain-of-Evidence 审计。

##### Autodata

Autodata (Kulikov et al. 2026) 设计为 data scientist Agent，管理 challenger（出题者）、weak solver、strong solver 和 verifier/judge，合成"难度适中"的数据——**strong solver 成功但 weak solver 失败**。

![图片](media/session/dc949203f2068587.png)

*图 8: Autodata 的 agentic workflow 设计。（来源: Kulikov et al. 2026）*

##### ADAS: Automated Design of Agentic Systems

ADAS (Hu et al. 2025) 将 Agent 设计本身形式化为优化问题（"meta-agent search"），由 meta-agent 提出新的 agentic workflow 设计：

1. 用简单 Agent（CoT、self-refine）初始化 archive
2. Meta-agent 编程新 Agent（全部用代码），灵感来自 archive 中的现有方案
3. 通过两步 self-refine 检查新颖性
4. 评估新候选方案，成功的加回 archive
5. 重复直到达到最大迭代次数

![图片](media/session/dc5a10cecf66c578.png)

*图 9: Automated Design of Agentic Systems (ADAS) 框架。（来源: Hu et al. 2025）*

##### AFlow

AFlow (Zhang et al. 2025) 将 agentic workflow 表示为**图**，节点是 LLM 调用动作，边是代码中的逻辑操作。优化依赖 **MCTS (Monte Carlo Tree Search)**。

![图片](media/session/2bdf9fd25f4a57c6.png)

*图 10: AFlow 在 workflow 候选树上的优化过程。（来源: Zhang et al. 2025）*

![图片](media/session/b974178fc5c68103.png)

*图 11: AFlow 实验结果——与手动方法和 ADAS 的对比。（来源: Zhang et al. 2025）*

#### 5.3 Self-Improving Harness（自改进 Harness）

无论是 Context Engineering 还是 Workflow Design 都只是 Harness 的一部分。我们需要搜索**整个设计空间**，同时优化 context 管理逻辑、workflow、权限和其他 Harness 组件。

> **✨ 代码（Code）✨ 是定义程序和系统的通用语言。** 简单来说，Harness 就是编程 prompts、工具调用、子代理、控制流、记忆和 workflow 逻辑如何协同工作的代码。如果 LLM 能优化执行 Agent 的代码，它就能访问比手写 prompts 大得多的设计空间。

##### STOP: Self-Taught Optimizer

STOP (Zelikman et al. 2023) 是递归 scaffolding 改进的早期范例。目标不是直接改进解决方案 $s$，而是**改进改进器 **$I$** 本身**：

$$I_t = I_{t-1}(\hat{u}, I_{t-1}; M)$$

其中 meta-utility:

$$\hat{u}(I) \triangleq \frac{1}{|\mathcal{D}|} \mathbb{E}_{(u,s) \sim \mathcal{D}}[u(I(u,s;M))]$$

![图片](media/session/8f791ddd4796151d.png)

*图 12: Self-Taught Optimizer (STOP) 算法。（来源: Zelikman et al. 2023）*

改进后的 improver 发现了多种策略：遗传算法、分解改进、multi-armed prompt bandits、模拟退火、温度变化、beam/tree search 等。

![图片](media/session/0e0212ea35598557.png)

*图 13: STOP 发现的自改进策略示例。（来源: Zelikman et al. 2023）*

**⚠️ 重要发现**: STOP 在 GPT-4 上迭代改进了下游表现，但在较弱模型（GPT-3.5、Mixtral）上**反而下降**。递归结构本身不够——**基础模型必须足够强大才能改进机制**。

##### Harness Updating ≠ Harness Benefit

Lin et al. (2026) 详细研究了 Harness 演化对模型能力的依赖，分离了两个轴：

- **Harness-updating**: 产生有用 Harness 编辑的能力
- **Harness-benefit**: 利用更新后的 Harness 来更好地解决任务的能力

![图片](media/session/f47fa538e5e9ca01.png)

*图 14: (A) 从 Qwen2-32B 到 Opus 4.6 的 harness updating 能力大致持平；(B) harness benefit 非单调，中等模型受益最多。（来源: Lin et al. 2026）*

有趣的是，从 Qwen3.5-9B 到 Claude Opus 4.6 的模型展示了**相似的 harness updating 能力**——9B 的 harness 提议者能写出与 Opus 过程等价的 skill。但要最好地**利用** Harness，模型需要正确及时地调用 skills/tools，并擅长 long-horizon instruction following。

##### Self-Harness

Self-Harness (Zhang et al. 2026) 依赖 LLM Agent 通过 propose-evaluate-accept 循环改进自己的 Harness，包含三个阶段：

![图片](media/session/895bd4a7cd9b6b9d.png)

*图 15: Self-Harness 的循环——weakness mining、bounded harness proposal 和 validation。（来源: Zhang et al. 2026）*

1. **Weakness Mining**: 将失败聚类为 verifier-grounded failure patterns
2. **Harness Proposal**: 基于挖掘的失败模式提出有界的 Harness 编辑
3. **Proposal Validation**: 在 held-in 和 held-out 数据上验证，只接受无回归的编辑

**⚠️ Lilian 的担忧**: 如果程序被允许编辑 OS 系统，**抽象边界就被打破了**。可编辑表面需要精心设计，权限控制和安全层需要**位于这个循环之外**。

##### AHE: Agentic Harness Engineering

AHE (Lin et al. 2026) 认为 Harness 演化的瓶颈在于**可观测性（observability）**。框架创建了包含 3 个可观测性支柱的闭环：

| 支柱 | 描述 |
| --- | --- |
| **Component Observability** | 每个可编辑 Harness 组件在文件系统中有表示，动作空间显式可追踪。Harness 包含 7 个组件：system prompt, tool description, tool implementation, middleware, skill, sub-agent configuration, long-term memory |
| **Experience Observability** | 分析和汇总大量 raw trajectories 为证据和失败模式的层次结构 |
| **Decision Observability** | 每次编辑都配有对下一轮的预测以供验证 |

**关键安全约束**:

- 编辑仅应用于 Harness workspace；runs 目录、tracer、verifier 和 LLM 配置为**只读**
- 每次编辑都是**证据驱动的**，包含 manifesto entry：失败证据名称、推断的根因、针对性修复、预测影响

在 Terminal-Bench-2 上，AHE 超过了人类设计的 Harness（OpenCode、Terminus-2、Codex），且同一冻结 Harness 可**迁移到 SWE-bench-verified**，表明演化出的 Harness 能够将工程经验编码到组件中而非做 benchmark 特定优化。

#### 5.4 Evolutionary Search（进化搜索）

进化搜索是受自然选择启发的优化方法，适用于：

1. 搜索空间广阔或形状奇特
2. 难以用梯度直接优化但容易评估解决方案

##### AlphaEvolve

AlphaEvolve (Novikov et al. 2025) 是一个 coding-agent 进化搜索系统：存储候选程序池，提示冻结的 LLM 生成 diff 进行改进。

![图片](media/session/9aff606707805732.png)

*图 16: AlphaEvolve 的工作原理。（来源: Novikov et al. 2025）*

**关键设计细节**:

- Prompt 包含父程序、结果、指令和 meta 信息
- 代码区域用 `# EVOLVE-BLOCK-START` 和 `# EVOLVE-BLOCK-END` 显式标记
- Meta-prompt 与 instructions 和 context 共同演化

![图片](media/session/ad59c8ccfbcca45e.png)

*图 17: AlphaEvolve 消融实验——展示各设计的价值。（来源: Novikov et al. 2025）*

**相关变体**:

- **ThetaEvolve** (Wang et al. 2025): 结合进化搜索与 RL 和 in-context learning
- **DemoEvolve** (Che et al. 2026): 用人类专家 demonstrations 增强 self-rollout archive
- **ShinkaEvolve** (Lange et al. 2025): 引入三项改进 LLM 采样效率的组件

##### Darwin Gödel Machine (DGM)

DGM (Zhang et al. 2025) 明确以**可编辑的 Harness 代码仓库的演化**为目标，用 LLM Coding Agent 修改自己的 Harness。

后续工作 **Hyperagents** (Zhang et al. 2026) 引入 meta-agent 控制如何修改现有 task agent 以创建新 Agent：

1. 从池中一个 Coding Agent 开始
2. 按性能概率和子代数量反比选择父代
3. 父代检查自己的 benchmark 评估日志，提出 Harness 代码改进
4. 评估新 Agent，高性能者加回池中

在 SWE-bench Verified (20% → 50%) 和 Polyglot (14.2% → 30.7%) 上取得显著提升。

#### 5.5 Joint Optimization with Model Weights（与模型权重联合优化）

Harness 演化改变模型周围的非参数系统。为实现完全的自我改进，模型**完全可以在同一时间更新自身权重**。

##### SIA: Self Improving AI

SIA (Hebbar et al. 2026) 是将 Harness 改进和模型参数更新结合在同一优化循环中的早期尝试：

- **Meta-Agent**: 提出初始 Harness
- **Task-Specific Agent**: 执行任务
- **Feedback-Agent**: 根据最近的 trajectory 决定是更新 Harness 还是更新模型权重

![图片](media/session/86fbe9555a0bf6f3.png)

*图 18: SIA 中 Feedback-Agent 决定下一轮迭代类型。（来源: Hebbar et al. 2026）*

##### Continual Harness

Continual Harness (Karten et al. 2026) 在 long-horizon gameplay 场景中实验了 Harness 更新与共同学习 policy model（通过从 strong teacher model 蒸馏标签到 low-reward trajectories）。

---

### 6. Harness 案例研究

#### 6.1 自动研究的失败模式

Trehan & Chopra (2026) 测试了 LLM 能否用最少的 scaffolding 和基本工具从研究想法到论文。在三个领域（world models, multi-agent RL, AI safety & alignment）中实验，观察到 **六种反复出现的失败模式**：

| # | 失败模式 | 描述 |
| --- | --- | --- |
| 1 | **训练数据默认偏差** | 使用旧库、过时命令、标准格式或不基于实际仓库/数据集的假设 |
| 2 | **执行压力下的实现漂移** | 实现技术复杂时，模型倾向于转向常见的更简单方案 |
| 3 | **记忆和上下文退化** | 长 horizon 项目丢失关键细节，除非日志写为持久产出物 |
| 4 | **过度乐观** | 尽管实验结果嘈杂或失败，模型仍宣布成功（"p-hacking and eureka-ing" 模式） |
| 5 | **领域智能不足** | 模型缺乏隐性工艺知识——预测实现复杂度、判断实验结果是否合理 |
| 6 | **科学品味薄弱** | 实验可执行但未能回答正确的问题 |

---

### 7. Future Challenges（未来挑战）

尽管研究者取得了实质进展，但走向完全 RSI 仍有**七大瓶颈**：

#### 7.1 弱且模糊的 Evaluator

许多研究声明没有快速且精确的 verifier。当前的自改进循环在**评估指标可测量且客观**时效果最好（类似 RL 的工作方式）。

> 研究品味、新颖性和长期科学价值**远比这更难衡量**。

#### 7.2 Context 和 Memory 的生命周期

随着 AI Agent 变得更自主和独立，Memory 不断增长。有用的 Harness 需要管理 context 和 memory 来补充长 context 生成的现有局限。

> Lilian 的观点: **Context Engineering 将成为且应该成为智能的核心部分，而不仅仅停留在软件系统层。**

#### 7.3 负面结果（Negative Results）

研究者被激励发表成功结果，因此文献偏向成功。LLM 在大量（主要是人类创建的）数据上训练，可能**不擅长决定何时放弃假设、报告负面结果或承认失败**。

> 一个研究 Harness 应该让失败的尝试**易于保存**，因为从失败中学习是裁剪任务搜索空间的最佳方式。

#### 7.4 Diversity Collapse（多样性崩塌）

Evolutionary 和 RL 循环倾向于利用已知的高奖励模式。我们需要机制来**防止种群退化为同一解决方案的变体**。这对于开放式研究尤为关键——最佳路径在当前 evaluator 下可能最初看起来更差。

#### 7.5 Reward Hacking

自改进循环优化它被给予的任何信号：

- 如果 reward 来自 unit tests → Agent 可能过拟合测试
- 如果来自 judge model → 可能学会特定于该 judge 的 reward hacking 技巧
- 如果来自 benchmark scores → 可能利用 benchmark 的 artifacts

> **Evaluator 和权限控制应当位于演化 Harness 的循环之外**，配合 held-out tests、trace audits 和人类审查。

#### 7.6 Long-term Success（长期成功）

以 Coding Agent 为例：Coding Agent 已经提升了软件工程的日常生产力，但许多优化目标仍然**过于短期**。它通常能完成手头的任务，但不太清楚如何保护由数百或数千名工程师共同维护的仓库的**长期健康**。

标准的 sandbox-based RLVR 训练很少捕获：可维护性、所有权边界、迁移成本、向后兼容性或未来调试负担。

#### 7.7 The Role of Humans（人类的角色）

> **人类应该沿着栈向上移动，而不是被从循环中移除。**

人类应在正确的时间、正确的抽象层级提供 oversight，系统设计应考虑何时以及如何设置这样的接触点。

> 毕竟，我们是为了人类更好的未来而构建技术，而不是反过来。

---

### 8. 附录：相关 Benchmark

| Benchmark | 描述 | 规模 | 最佳表现 |
| --- | --- | --- | --- |
| **PaperBench** | 从头复现 20 篇 ICML 2024 Spotlight/Oral 论文 | 8,316 个评分标准 | Claude 3.5 Sonnet ~21%，未超过 ML PhD |
| **CORE-Bench** | 评估已发表研究的计算可复现性 | 270 个任务 / 90 篇论文 | GPT-4o 在最难任务上仅 21% |
| **ScienceAgentBench** | 评估数据驱动科学发现的 LLM Agent | 102 个任务 / 44 篇论文 / 4 个学科 | — |
| **RE-Bench** | 对比前沿 AI Agent 和人类专家的 ML 研究工程能力 | 7 个开放式 ML 研究环境 | AI 在 2h 预算下 4× 超人类，但人类在 8h/32h 反超 |
| **MLE-bench** | 在离线 Kaggle 竞赛上评估 ML 工程 Agent | 75 个 Kaggle 竞赛 | o1-preview + AIDE 在 16.9% 竞赛达到铜牌水平 |
| **KernelBench** | 评估生成的 GPU kernel 的正确性和速度 | 250 个 PyTorch 任务 | — |

---

### 9. 参考文献

> [1] Good, I. J. "Speculations Concerning the First Ultraintelligent Machine." *Advances in Computers*, 6:31–88, 1965.

> [2] Yudkowsky, E. "Recursive Self-Improvement." *LessWrong*, 2008.

> [3] Choi, et al. "Anchored Self-Play for Code Repair." *ICML 2026*.

> [4] Zhao, et al. "Absolute Zero: Reinforced Self-play Reasoning with Zero Data." arXiv:2505.03335, 2025.

> [5] Yuan, et al. "Self-Rewarding Language Models." arXiv:2401.10020, 2024.

> [6] Chen, et al. "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models." *ICML 2024*.

> [7] Zhang, et al. "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models." *ICLR 2026*.

> [8] Ye, et al. "Meta Context Engineering via Agentic Skill Evolution." arXiv:2601.21557, 2026.

> [9] Lee, et al. "Meta-Harness: End-to-End Optimization of Model Harnesses." arXiv:2603.28052, 2026.

> [10] Lu, et al. "Towards end-to-end automation of AI research." *Nature*, 651:914–919, 2026.

> [11] Meng, et al. "ScientistOne: Towards Human-Level Autonomous Research via Chain-of-Evidence." arXiv:2605.26340, 2026.

> [12] Kulikov, et al. "Autodata: An agentic data scientist to create high quality synthetic data." arXiv:2606.25996, 2026.

> [13] Hu, Lu, and Clune. "Automated Design of Agentic Systems." *ICLR 2025*.

> [14] Madaan, et al. "Self-Refine: Iterative Refinement with Self-Feedback." *NeurIPS 2023*.

> [15] Zhang, et al. "AFlow: Automating Agentic Workflow Generation." *ICLR 2025*.

> [16] Zelikman, et al. "Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation." *COLM 2024*.

> [17] Zhang, et al. "Self-Harness: Harnesses That Improve Themselves." arXiv:2606.09498, 2026.

> [18] Fernando, et al. "Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution." arXiv:2309.16797, 2023.

> [19] Agrawal, et al. "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning." arXiv:2507.19457, 2025.

> [20] Novikov, et al. "AlphaEvolve: A coding agent for scientific and algorithmic discovery." arXiv:2506.13131, 2025.

> [21] Lange, Imajuku, and Cetin. "ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution." arXiv:2509.19349, 2025.

> [22] Wang, et al. "ThetaEvolve: Test-time Learning on Open Problems." arXiv:2511.23473, 2025.

> [23] Zhang, et al. "Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents." arXiv:2505.22954, 2025.

> [24] Zhang, et al. "Hyperagents." arXiv:2603.19461, 2026.

> [25] Yuksekgonul, et al. "Learning to Discover at Test Time." arXiv:2601.16175, 2026.

> [26] Riaz, et al. "Epistemic Uncertainty for Test-Time Discovery." arXiv:2605.11328, 2026.

> [27] Hebbar, et al. "SIA: Self Improving AI with Harness & Weight Updates." arXiv:2605.27276, 2026.

> [28] Trehan and Chopra. "Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts." arXiv:2601.03315, 2026.

> [29] Bubeck, et al. "Early science acceleration experiments with GPT-5." arXiv:2511.16072, 2025.

> [30] Starace, et al. "PaperBench: Evaluating AI's Ability to Replicate AI Research." *ICML 2025*.

> [31] Wijk, et al. "RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts." *ICML 2025*.

> [32] Chan, et al. "MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering." arXiv:2410.07095, 2024.

> [33] Chen, et al. "ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery." *ICLR 2025*.

> [34] Siegel, et al. "CORE-Bench: Fostering the Credibility of Published Research Through a Computational Reproducibility Agent Benchmark." *TMLR 2024*.

> [35] Ouyang, et al. "KernelBench: Can LLMs Write Efficient GPU Kernels?" arXiv:2502.10517, 2025.

> [36] Lin, et al. "Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents." arXiv:2605.30621, 2026.

> [37] Lin, et al. "Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses." arXiv:2604.25850, 2026.

> [38] Karten, et al. "Continual Harness: Online Adaptation for Self-Improving Foundation Agents." arXiv:2605.09998, 2026.

> [39] Che, et al. "DemoEvolve: Overcoming Sparse Feedback in Agentic Harness Evolution with Demonstrations." arXiv:2605.24539, 2026.

---

> 📖 **原文引用**

> Weng, Lilian. "Harness Engineering for Self-Improvement". *Lil'Log* (Jul 2026). https://lilianweng.github.io/posts/2026-07-04-harness/

> ```
> // 代码块
> @article{weng2026harness,
>   title   = {Harness Engineering for Self-Improvement},
>   author  = {Weng, Lilian},
>   journal = {lilianweng.github.io},
>   year    = {2026},
>   month   = {July},
>   url     = "https://lilianweng.github.io/posts/2026-07-04-harness/"
> }
> ```
