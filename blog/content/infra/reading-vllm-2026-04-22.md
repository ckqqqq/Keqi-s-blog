---
title: "FP8 KV Cache：容量、精度与 attention 后端"
date: 2026-04-22
lastmod: 2026-09-21
visibility: public
draft: false
categories: ["缓存与存储"]
tags: ["vLLM","阅读笔记"]
description: "KV 量化改变存储量，也涉及 attention 计算路径、缩放与精度。混合 attention 中不同层未必适合统一量化。"
---

> 本文为技术阅读导引，不是原文全文转载。页面日期对应来源存档标注的原文日期 2026-04-22；中文导引整理于 2026-09-21。来源内容按【文档声明】理解，本次未复现性能，也未重新核验所有版本状态。

## 要解决的问题

【文档声明】KV 量化改变存储量，也涉及 attention 计算路径、缩放与精度。混合 attention 中不同层未必适合统一量化。

## 阅读实现时抓住什么

【推断：阅读方法】先读数值问题与 kernel 修复，再比较 prefill、decode、不同 head dimension 和跳过 SWA 层的情况。

## 复现与适用边界

按原文的模型、后端和硬件分组复核，不把 Hopper 与 Blackwell 数据合并成一个结论。B200 路径需单独申请设备。

## 来源与署名

- 原文标题：The State of FP8 KV-Cache and Attention Quantization in vLLM
- 原作者：Jonas Kübler* (AWS), Eldar Kurtić* (Red Hat AI), Lucas Wilkinson (Red Hat AI), Matthew Bonanni (Red Hat AI), Michael Goin (Red Hat AI), Alexandre Marques (Red Hat AI), Kailash Budhathoki (AWS) (* Equal Contribution)
- 资料日期：2026-04-22（沿用原始存档，未重新核验）
- 阅读入口：[vLLM 官方博客源仓目录（按原文标题检索）](https://github.com/vllm-project/vllm-project.github.io/tree/main/_posts)
- 原存档未保存准确的公开文章 URL，本次未补猜链接；可在源仓中按标题与日期定位。
