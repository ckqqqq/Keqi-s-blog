---
title: "分布式逐层卸载：权重分片、预取与 AllGather"
date: 2026-08-17
visibility: public
draft: false
categories: ["缓存与存储"]
tags: ["vLLM","阅读笔记"]
description: "这一方案卸载的是 DiT 模型权重，不能与请求的 KV 卸载混为一谈。各设备保存权重分片，在层执行前重建所需权重。"
---

> 本文为技术阅读导引，不是原文全文转载。页面日期对应来源存档标注的原文日期 2026-08-17；中文导引整理于 2026-09-21。来源内容按【文档声明】理解，本次未复现性能，也未重新核验所有版本状态。

## 要解决的问题

【文档声明】这一方案卸载的是 DiT 模型权重，不能与请求的 KV 卸载混为一谈。各设备保存权重分片，在层执行前重建所需权重。

## 阅读实现时抓住什么

【推断：阅读方法】按 meta/mmap 加载、主机分片、双缓冲预取、AllGather 和并发调度顺序阅读，画清每层权重的驻留区间。

## 复现与适用边界

区分实际测量的模型与更大模型的容量估算；同时检查主机内存、锁页内存、通信和设备容量。

## 来源与署名

- 原文标题：Distributed Layerwise Offload: Scaling Toward 200B+ DiT Models Efficiently in vLLM-Omni
- 原作者：vLLM-Omni Diffusion Team
- 资料日期：2026-08-17（沿用原始存档，未重新核验）
- 阅读入口：[vLLM 官方博客源仓目录（按原文标题检索）](https://github.com/vllm-project/vllm-project.github.io/tree/main/_posts)
- 原存档未保存准确的公开文章 URL，本次未补猜链接；可在源仓中按标题与日期定位。
