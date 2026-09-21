---
title: "vLLM DSA：索引器与稀疏注意力如何接入引擎"
date: 2025-09-29
visibility: public
draft: false
categories: ["注意力算法"]
tags: ["vLLM","阅读笔记"]
description: "稀疏注意力需要先生成候选位置，再读取相应 KV。引擎适配既要处理 attention kernel，也要维护索引器缓存。"
---

> 本文为技术阅读导引，不是原文全文转载。页面日期对应来源存档标注的原文日期 2025-09-29；中文导引整理于 2026-09-21。来源内容按【文档声明】理解，本次未复现性能，也未重新核验所有版本状态。

## 要解决的问题

【文档声明】稀疏注意力需要先生成候选位置，再读取相应 KV。引擎适配既要处理 attention kernel，也要维护索引器缓存。

## 阅读实现时抓住什么

【推断：阅读方法】沿 prefill 与 decode 两条路径阅读，重点检查候选索引如何与物理 KV 页对应，以及连续批处理如何表达不同请求的有效长度。

## 复现与适用边界

区分减少实际注意力计算和减少常驻 KV 容量；前者不会自动保证后者。

## 来源与署名

- 原文标题：DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in Action
- 原作者：vLLM Team
- 资料日期：2025-09-29（沿用原始存档，未重新核验）
- 阅读入口：[vLLM 官方博客源仓目录（按原文标题检索）](https://github.com/vllm-project/vllm-project.github.io/tree/main/_posts)
- 原存档未保存准确的公开文章 URL，本次未补猜链接；可在源仓中按标题与日期定位。
