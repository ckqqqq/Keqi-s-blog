---
title: "MORI-IO：单机内的 Prefill/Decode 分离"
date: 2026-04-07
visibility: public
draft: false
categories: ["推理引擎"]
tags: ["vLLM","阅读笔记"]
description: "该资料研究单节点中把 prefill 与 decode 放到不同 GPU，通过 MORI-IO 传输 KV，以控制混合负载下的生成延迟。"
---

> 本文为技术阅读导引，不是原文全文转载。页面日期对应来源存档标注的原文日期 2026-04-07；中文导引整理于 2026-09-21。来源内容按【文档声明】理解，本次未复现性能，也未重新核验所有版本状态。

## 要解决的问题

【文档声明】该资料研究单节点中把 prefill 与 decode 放到不同 GPU，通过 MORI-IO 传输 KV，以控制混合负载下的生成延迟。

## 阅读实现时抓住什么

【推断：阅读方法】对照 read 与 write 两种模式，追踪请求何时发出、KV 何时就绪、decode 如何等待；再分析 TTFT 与 ITL 之间的取舍。

## 复现与适用边界

原文以 8 卡 MI300X 为硬件前提，复现需要单独申请该设备。比较 goodput 时必须固定 SLO，不能将它等同于裸吞吐。

## 来源与署名

- 原文标题：Next-Level Inference: Why Your Single-Node vLLM Setup Needs Prefill-Decode Disaggregation
- 原作者：AMD and Embedded LLM
- 资料日期：2026-04-07（沿用原始存档，未重新核验）
- 阅读入口：[vLLM 官方博客源仓目录（按原文标题检索）](https://github.com/vllm-project/vllm-project.github.io/tree/main/_posts)
- 原存档未保存准确的公开文章 URL，本次未补猜链接；可在源仓中按标题与日期定位。
