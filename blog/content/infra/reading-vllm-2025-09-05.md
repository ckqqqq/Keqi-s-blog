---
title: "vLLM 引擎全景：从请求到调度循环"
date: 2025-09-05
lastmod: 2026-09-21
visibility: public
draft: false
categories: ["推理引擎"]
tags: ["vLLM","阅读笔记"]
description: "以一次生成请求为主线，串起引擎初始化、调度、模型前向和结果返回；再观察连续批处理与分页 KV 如何改变运行方式。"
---

> 每日一读

## 要解决的问题

【文档声明】以一次生成请求为主线，串起引擎初始化、调度、模型前向和结果返回；再观察连续批处理与分页 KV 如何改变运行方式。

## 阅读实现时抓住什么

【推断：阅读方法】先画出 scheduler 与 model runner 的边界，再区分前缀缓存、分块预填充和投机解码各自复用了什么。最后阅读多进程执行器与多机服务，避免把单卡路径直接等同于分布式路径。

## 复现与适用边界

比较延迟与吞吐时固定输入输出长度、并发量和调度配置；先确认量度的是用户请求还是输出 token。

## 来源与署名

- 原文标题：Inside vLLM: Anatomy of a High-Throughput LLM Inference System
- 原作者：Aleksa Gordic
- 资料日期：2025-09-05（沿用原始存档，未重新核验）
- 阅读入口：[原文](https://www.aleksagordic.com/blog/vllm)
