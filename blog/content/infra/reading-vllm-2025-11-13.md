---
title: "共享内存 IPC：减少多进程推理的重复数据搬运"
date: 2025-11-13
visibility: public
draft: false
categories: ["推理引擎"]
tags: ["vLLM","阅读笔记"]
description: "多模态输入在协调进程与工作进程之间传递时，序列化与拷贝可能成为额外成本。共享内存缓存让多个进程引用同一份大对象。"
---

> 本文为技术阅读导引，不是原文全文转载。页面日期对应来源存档标注的原文日期 2025-11-13；中文导引整理于 2026-09-21。来源内容按【文档声明】理解，本次未复现性能，也未重新核验所有版本状态。

## 要解决的问题

【文档声明】多模态输入在协调进程与工作进程之间传递时，序列化与拷贝可能成为额外成本。共享内存缓存让多个进程引用同一份大对象。

## 阅读实现时抓住什么

【推断：阅读方法】阅读对象存储的写入、索引、读取和回收过程，明确数据所有权与进程退出后的清理责任。缓存命中也必须满足输入身份一致。

## 复现与适用边界

把数据准备、IPC 和模型执行分别计时；留意读写并发、对象大小分布及共享区容量。

## 来源与署名

- 原文标题：Shared Memory IPC Caching: Accelerating Data Transfer in LLM Inference Systems
- 原作者：Donglu Wang (Cohere)
- 资料日期：2025-11-13（沿用原始存档，未重新核验）
- 阅读入口：[原文](https://cohere.com/blog/making-data-transfer-in-llm-systems-faster-leaner-and-more-scalable)
