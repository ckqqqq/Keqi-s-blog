---
title: "GB300 上的 DeepSeek：量化与并行配置阅读笔记"
date: 2026-02-13
lastmod: 2026-09-21
visibility: public
draft: false
categories: ["算子与通信"]
tags: ["vLLM","阅读笔记"]
description: "该资料围绕 GB300 上的 DeepSeek 部署，讨论 NVFP4、TP/EP 配置、投机解码和 prefill/decode 分离。"
---

> 本文为技术阅读导引，不是原文全文转载。页面日期对应来源存档标注的原文日期 2026-02-13；中文导引整理于 2026-09-21。来源内容按【文档声明】理解，本次未复现性能，也未重新核验所有版本状态。

## 要解决的问题

【文档声明】该资料围绕 GB300 上的 DeepSeek 部署，讨论 NVFP4、TP/EP 配置、投机解码和 prefill/decode 分离。

## 阅读实现时抓住什么

【推断：阅读方法】先读硬件和模型配置，再比较不同并行方式，分清每卡吞吐、系统吞吐与单请求延迟。

## 复现与适用边界

原文依赖 GB300；复现实测需单独申请对应设备。H100/H20 上可分析配置与算法，但不能复现原硬件的原生指令或直接沿用跑分。

## 来源与署名

- 原文标题：DeepSeek-V3.2 on GB300: Performance Breakthrough
- 原作者：The DaoCloud and vLLM team
- 资料日期：2026-02-13（沿用原始存档，未重新核验）
- 阅读入口：[vLLM 官方博客源仓目录（按原文标题检索）](https://github.com/vllm-project/vllm-project.github.io/tree/main/_posts)
- 原存档未保存准确的公开文章 URL，本次未补猜链接；可在源仓中按标题与日期定位。
