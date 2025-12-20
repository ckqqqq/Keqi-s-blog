---
title: "vLLM EPD：将视觉编码从文本生成中拆开"
date: 2025-12-15
lastmod: 2026-09-21
visibility: public
draft: false
categories: ["推理引擎"]
tags: ["vLLM","阅读笔记"]
description: "视觉编码、文本 prefill 与 decode 的计算特征不同，将它们分开可以独立配置资源，但也引入 embedding 传输。"
---

> 本文为技术阅读导引，不是原文全文转载。页面日期对应来源存档标注的原文日期 2025-12-15；中文导引整理于 2026-09-21。来源内容按【文档声明】理解，本次未复现性能，也未重新核验所有版本状态。

## 要解决的问题

【文档声明】视觉编码、文本 prefill 与 decode 的计算特征不同，将它们分开可以独立配置资源，但也引入 embedding 传输。

## 阅读实现时抓住什么

【推断：阅读方法】依次跟踪 encoder 输出、远端缓存检查、scheduler 状态变化以及 worker 读取路径，理解请求何时真正具备执行条件。

## 复现与适用边界

评估收益需要覆盖图像多与图像少两类请求；新增网络等待可能抵消编码侧的节省。

## 来源与署名

- 原文标题：Encoder Disaggregation for Scalable Multimodal Model Serving
- 原作者：Multimodality Workstream @ vLLM
- 资料日期：2025-12-15（沿用原始存档，未重新核验）
- 阅读入口：[vLLM 官方博客源仓目录（按原文标题检索）](https://github.com/vllm-project/vllm-project.github.io/tree/main/_posts)
- 原存档未保存准确的公开文章 URL，本次未补猜链接；可在源仓中按标题与日期定位。
