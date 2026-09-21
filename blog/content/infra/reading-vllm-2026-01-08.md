---
title: "KV Offloading Connector：异步卸载与重算的取舍"
date: 2026-01-08
visibility: public
draft: false
categories: ["缓存与存储"]
tags: ["vLLM","阅读笔记"]
description: "将暂时不用的 KV 保存到 CPU 内存，在再次需要时取回，可在传输成本与重新 prefill 之间作权衡。"
---

> 本文为技术阅读导引，不是原文全文转载。页面日期对应来源存档标注的原文日期 2026-01-08；中文导引整理于 2026-09-21。来源内容按【文档声明】理解，本次未复现性能，也未重新核验所有版本状态。

## 要解决的问题

【文档声明】将暂时不用的 KV 保存到 CPU 内存，在再次需要时取回，可在传输成本与重新 prefill 之间作权衡。

## 阅读实现时抓住什么

【推断：阅读方法】按 connector 接口、调度侧决策、工作侧传输和完成通知的顺序阅读。重点是哪些步骤可与计算重叠，以及 KV 的布局如何影响复制效率。

## 复现与适用边界

同时记录命中率、主机容量、传输字节数与重算时间，不能只比较一次 H2D 复制带宽。

## 来源与署名

- 原文标题：Inside vLLM’s New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput
- 原作者：Or Ozeri, Danny Harnik (vLLM Team at IBM Research)
- 资料日期：2026-01-08（沿用原始存档，未重新核验）
- 阅读入口：[vLLM 官方博客源仓目录（按原文标题检索）](https://github.com/vllm-project/vllm-project.github.io/tree/main/_posts)
- 原存档未保存准确的公开文章 URL，本次未补猜链接；可在源仓中按标题与日期定位。
