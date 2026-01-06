---
title: Infra
description: 从硬件到系统，从理解原理到性能实践。
---

建议沿着 **系统全貌 → 核心组件 → 实验与复盘** 的顺序阅读。

- [系统架构]({{< article "infra/2026-04-15-deepseek-stack.md" >}})：先读四层地图，再看模型设计如何改变系统约束。
- [推理引擎]({{< article "infra/vllm-pr56214-analysis.md" >}})：沿模型接入和代码差异理解运行语义。
- [缓存与存储]({{< article "infra/reading-sglang.md" >}})：从前缀复用边界进入分层缓存（SGLang 阅读笔记汇总）。
- [算子与通信]({{< article "infra/flashmla-dsa-kernels.md" >}})：关注流水线、数据布局与正确性边界。
- [训练基础设施]({{< article "infra/2026-08-21-async-rl-infrastructure.md" >}})：理解异步 RL 的样本版本与状态管理。
- [模型与评测]({{< article "infra/model-evaluation-method.md" >}})：区分模型能力、运行环境与测量口径。
