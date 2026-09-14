---
title: "分层 KV Offloading：以主机内存为中转层"
date: 2026-09-10
lastmod: 2026-09-21
visibility: public
draft: false
categories: ["缓存与存储"]
tags: ["vLLM","阅读笔记"]
description: "资料采用 host-centric 设计，在加速器与文件系统、对象存储或远端节点之间以主机内存中转 KV。"
---

> 本文为技术阅读导引，不是原文全文转载。页面日期对应来源存档标注的原文日期 2026-09-10；中文导引整理于 2026-09-21。来源内容按【文档声明】理解，本次未复现性能，也未重新核验所有版本状态。

## 要解决的问题

【文档声明】资料采用 host-centric 设计，在加速器与文件系统、对象存储或远端节点之间以主机内存中转 KV。

## 阅读实现时抓住什么

【推断：阅读方法】重点看两条生命周期规则：D2H 完成后何时释放设备页，重载数据到主机就绪后何时分配设备页。再检查异步 I/O 与共享存储接口。

## 复现与适用边界

逐层测量传输和等待成本，并观察存储吞吐、主机缓存命中与请求端延迟；容量增加不等于每个请求都更快。

## 来源与署名

- 原文标题：Tiered KV Cache Offloading in vLLM
- 原作者：Or Ozeri, Danny Harnik, Ronen Schaffer, Itay Etelis, Varun Sundar Rabindranath
- 资料日期：2026-09-10（沿用原始存档，未重新核验）
- 阅读入口：[vLLM 官方博客源仓目录（按原文标题检索）](https://github.com/vllm-project/vllm-project.github.io/tree/main/_posts)
- 原存档未保存准确的公开文章 URL，本次未补猜链接；可在源仓中按标题与日期定位。
