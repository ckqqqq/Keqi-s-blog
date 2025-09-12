---
title: "SGLang 官方博客阅读笔记汇总"
date: 2025-09-10
lastmod: 2026-09-21
visibility: public
draft: false
categories: ["推理引擎"]
tags: ["SGLang","阅读笔记"]
description: "按时间顺序汇总 SGLang / LMSYS 官方博客的十篇中文阅读导引，覆盖 HiCache、稀疏 attention、EPD 分离、统一 Radix Cache、H20 优化与 Rollout 数据传输。"
---

> 本文为技术阅读导引汇总，不是原文全文转载。各节日期对应来源存档标注的原文日期（2025-09-10 至 2026-09-10）；中文导引整理于 2026-09-21。来源内容按【文档声明】理解，本次未复现性能，也未重新核验所有版本状态。

## 目录

1. [2025-09-10 · SGLang HiCache：多级 KV 缓存与存储后端](#2025-09-10--sglang-hicache多级-kv-缓存与存储后端)
2. [2025-09-29 · SGLang DSA：从候选选择到稀疏执行](#2025-09-29--sglang-dsa从候选选择到稀疏执行)
3. [2026-01-12 · SGLang EPD：编码器独立扩缩容](#2026-01-12--sglang-epd编码器独立扩缩容)
4. [2026-04-10 · HiSparse：稀疏计算之后，KV 容量怎么办](#2026-04-10--hisparse稀疏计算之后kv-容量怎么办)
5. [2026-04-25 · SGLang 与 Miles：模型接入中的推理和训练一致性](#2026-04-25--sglang-与-miles模型接入中的推理和训练一致性)
6. [2026-06-01 · 异构 EPD：用设备感知路由组合 CPU 与 GPU](#2026-06-01--异构-epd用设备感知路由组合-cpu-与-gpu)
7. [2026-08-11 · 统一 Radix Cache：共享前缀，不共享复用边界](#2026-08-11--统一-radix-cache共享前缀不共享复用边界)
8. [2026-08-19 · H20 推理优化：按负载选择服务配置](#2026-08-19--h20-推理优化按负载选择服务配置)
9. [2026-08-20 · Miles × Mooncake：把碎片化 Rollout 组织成批量 I/O](#2026-08-20--miles--mooncake把碎片化-rollout-组织成批量-io)
10. [2026-09-10 · V4.1 的引擎适配：跨层共享、Engram 与 Replay](#2026-09-10--v41-的引擎适配跨层共享engram-与-replay)

---

## 2025-09-10 · SGLang HiCache：多级 KV 缓存与存储后端

**要解决的问题**：【文档声明】HiCache 将 KV 从 GPU 延伸至主机和外部存储，使较长的历史前缀有机会被复用。

**阅读实现时抓住什么**：【推断：阅读方法】阅读缓存命中、预取、逐出及后端接口，区分减少重算与增加传输这两项成本。

**复现与适用边界**：原文含社区案例和作者测试，均按【文档声明】理解；不同模型、对话轮数和命中率的数据不可直接横比。

**来源与署名**：

- 原文标题：SGLang HiCache: Fast Hierarchical KV Caching with Your Favorite Storage Backends
- 原作者：SGLang / LMSYS 原文作者（署名见来源页）
- 资料日期：2025-09-10（沿用原始存档，未重新核验）
- 阅读入口：[原文](https://www.lmsys.org/blog/2025-09-10-sglang-hicache)

---

## 2025-09-29 · SGLang DSA：从候选选择到稀疏执行

**要解决的问题**：【文档声明】该资料介绍 DeepSeek-V3.2 稀疏 attention 的引擎适配，核心是将轻量索引器的候选结果交给 attention 计算。

**阅读实现时抓住什么**：【推断：阅读方法】把模型支持、kernel 依赖和调度接口分开看；部署成功只证明某个配置可运行，不说明所有负载都已优化。

**复现与适用边界**：保留原文的镜像及硬件约束；对新增路径先比较正确性，再测延迟与吞吐。

**来源与署名**：

- 原文标题：SGLang Day 0 Support for DeepSeek-V3.2 with Sparse Attention
- 原作者：SGLang / LMSYS 原文作者（署名见来源页）
- 资料日期：2025-09-29（沿用原始存档，未重新核验）
- 阅读入口：[原文](https://www.lmsys.org/blog/2025-09-29-deepseek-V32)

---

## 2026-01-12 · SGLang EPD：编码器独立扩缩容

**要解决的问题**：【文档声明】EPD 将视觉编码与语言模型的 prefill/decode 解耦，编码器可独立扩容，并通过不同后端传递视觉 embedding。

**阅读实现时抓住什么**：【推断：阅读方法】关注视觉结果缓存、传输后端和请求路由，画清完整数据流，而不是仅观察部署进程数量。

**复现与适用边界**：图片负载较小时，传输开销可能超过编码侧收益；应按图像数量和并发强度分别测试。

**来源与署名**：

- 原文标题：EPD Disaggregation: Elastic Encoder Scaling for Vision-Language Models in SGLang
- 原作者：SGLang / LMSYS 原文作者（署名见来源页）
- 资料日期：2026-01-12（沿用原始存档，未重新核验）
- 阅读入口：[原文](https://www.lmsys.org/blog/2026-01-12-epd)

---

## 2026-04-10 · HiSparse：稀疏计算之后，KV 容量怎么办

**要解决的问题**：【文档声明】每一步只访问部分 KV，不意味着全部上下文 KV 都能从容量账本中消失。HiSparse 将稀疏访问与分层存储结合。

**阅读实现时抓住什么**：【推断：阅读方法】分析被选 KV 如何加载，数据搬运与 attention 如何重叠，以及并发上升后容量和传输的瓶颈如何变化。

**复现与适用边界**：原文包含 H20 部署示例和 H200 性能条件，应分别对待。H100/H20 实验必须重新测量，不能引用 H200 数字作为本机结果。

**来源与署名**：

- 原文标题：HiSparse: Turbocharging Sparse Attention with Hierarchical Memory
- 原作者：SGLang / LMSYS 原文作者（署名见来源页）
- 资料日期：2026-04-10（沿用原始存档，未重新核验）
- 阅读入口：[原文](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse)

---

## 2026-04-25 · SGLang 与 Miles：模型接入中的推理和训练一致性

**要解决的问题**：【文档声明】这篇资料同时讨论推理与 RL 支持，覆盖前缀缓存、稀疏 attention、kernel 集成及训练侧的模型语义。

**阅读实现时抓住什么**：【推断：阅读方法】先区分前向算子的能力与引擎调度能力，再看训练如何重现 serving 的数值行为和路由信息。

**复现与适用边界**：Day-0 支持是特定时点的范围声明，路线图不等于已实现功能；部署前应核对版本与硬件。

**来源与署名**：

- 原文标题：DeepSeek-V4 on Day 0: From Fast Inference to Verified RL with SGLang and Miles
- 原作者：SGLang / LMSYS 原文作者（署名见来源页）
- 资料日期：2026-04-25（沿用原始存档，未重新核验）
- 阅读入口：[原文](https://www.lmsys.org/blog/2026-04-25-deepseek-v4)

---

## 2026-06-01 · 异构 EPD：用设备感知路由组合 CPU 与 GPU

**要解决的问题**：【文档声明】视觉编码主要发生在 prefill 之前，可把部分编码任务交给 CPU；不同设备的处理能力需要通过路由策略协调。

**阅读实现时抓住什么**：【推断：阅读方法】按设备权重、请求排队、视觉结果传输与下游消费的顺序阅读，理解异构资源如何共同参与服务。

**复现与适用边界**：CPU 的指令支持和模型大小会影响结论；应比较端到端 TTFT、TPOT 和负载下吞吐，而非只测编码时间。

**来源与署名**：

- 原文标题：Heterogeneous CPU + GPU EPD Disaggregation to Boost VLM Serving
- 原作者：SGLang / LMSYS 原文作者（署名见来源页）
- 资料日期：2026-06-01（沿用原始存档，未重新核验）
- 阅读入口：[原文](https://www.lmsys.org/blog/2026-06-01-hetero-epd)

---
