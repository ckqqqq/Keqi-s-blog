---
title: "Terminal-Bench 任务对照：题面、时限与验证器"
date: 2026-09-18
lastmod: 2026-09-21
visibility: public
draft: false
categories: ["模型与评测"]
tags: ["Benchmark", "Agent"]
description: "区分任务难度变化、运行预算变化和判分协议变化。"
---

> 本文沿用已有技术笔记中的版本记录与来源链接，整理于 2026-09-21。本次没有重新访问全部上游标签或运行这些任务；表中的历史数字与【核验】表述指原笔记的记录，不代表本次独立验证。原始题面节选未随公开版转载，请沿来源链接阅读。

> 抓取日：**2026-09-17**（全部源码链接均实际访问成功；1.0 题库 241 个条目、2.0 为 89 题、3.0 为 74 题、4.0 为 66 题，目录清单经 `gh api` 核对）。配套主报告 §1.2.7 / §1.2.8。
> 选材原则：每版 3 题，覆盖安全/系统/数据/ML 等不同领域；2.0 与 1.0 有大量同题，3.0 与 4.0 的题库几乎相同（4.0 相对 3.0 删 8 题、无新增），故 3.0/4.0 两列同题。

## 对照表（3 行 × 4 列）

| | **v1.0**（2025-05-19，80 题） | **v2.0**（2025-11-07，89 题） | **v3.0**（2026-07-30，74 题） | **v4.0**（2026-08-28，66 题） |
|---|---|---|---|---|
| **Case 1（安全/Web）** | `filter-js-from-html`：写 `/app/filter.py` 原地删除 HTML 中的 JavaScript，其余结构原样保留（3 条行为要求）。medium / security | **同题保留**，题面逐字相同 | `bun-sourcemap-leak`：修 Bun/TS 发布流水线的 source map 泄露，按 `visibility.json` 策略脱敏，6 条编号验收要求，禁加依赖。Software/Systems | **同题保留**，题面逐字相同 |
| **Case 2（系统/DB）** | `build-linux-kernel-qemu`：从源码编译 linux-6.9，加一行 `printk` 证明，gen_init_cpio 打包并用 qemu 启动。medium / system-administration | `query-optimize`：优化 OEWN SQLite 库上的一条慢查询，结果不变，单条 SQL 无注释存入 `sol.sql`。medium / data-science | `wal-recovery-ordering`：原地修复 WAL 存储引擎的崩溃恢复——LSN 前缀回放、并发写的持久化序、深拷贝隔离、禁 quadratic 算法，十余条行为约束。Software/Databases | **同题保留**，题面逐字相同 |
| **Case 3（ML）** | `caffe-cifar-10`：装原版 Caffe 1.0（仅 CPU），CIFAR-10 训 500 迭代，test 精度 ≥45% 且不低于 train 的 95%。medium / machine-learning | `pytorch-model-recovery`：从 state dict 反推模型结构，只微调 output_layer 降低 MSE，TorchScript 导出（4 条成功判据）。medium / model-training | `sglang-qwen-burst`：修 SGLang 投机解码 serving 的流式输出乱序（tool_call 与 content 顺序），服务端修复。ML/Inference | **同题保留**，题面逐字相同 |
| **专家耗时估计** | 45 / 60 / 未标注（min） | 45 / 60 / **15**（min） | **1.5 / 6 / 2（小时）** | 同 3.0 |
| **agent 超时** | 900 / 1800 / 1200 s | 1800 / 900 / 900 s | 1800 / **7200** / 7200 s | **全部统一 28800 s（8 h）** |
| **verifier** | pytest，300–600 s | 900–1800 s | 独立沙箱，600–7200 s | 独立沙箱，wal/sglang 收窄到 1800/420 s |
| **环境资源** | 未标注 | 1 CPU / 2 GB | 1–2 CPU / 2–6 GB | 2 CPU / 4–8 GB |

来源（每格两条：题面 + 配置）：
- 1.0：[filter-js-from-html](https://raw.githubusercontent.com/laude-institute/terminal-bench/main/original-tasks/filter-js-from-html/task.yaml)｜[build-linux-kernel-qemu](https://raw.githubusercontent.com/laude-institute/terminal-bench/main/original-tasks/build-linux-kernel-qemu/task.yaml)｜[caffe-cifar-10](https://raw.githubusercontent.com/laude-institute/terminal-bench/main/original-tasks/caffe-cifar-10/task.yaml)
- 2.0：[filter-js-from-html](https://raw.githubusercontent.com/harbor-framework/terminal-bench-2/main/filter-js-from-html/instruction.md)｜[query-optimize](https://raw.githubusercontent.com/harbor-framework/terminal-bench-2/main/query-optimize/instruction.md)（[toml](https://raw.githubusercontent.com/harbor-framework/terminal-bench-2/main/query-optimize/task.toml)）｜[pytorch-model-recovery](https://raw.githubusercontent.com/harbor-framework/terminal-bench-2/main/pytorch-model-recovery/instruction.md)（[toml](https://raw.githubusercontent.com/harbor-framework/terminal-bench-2/main/pytorch-model-recovery/task.toml)）
- 3.0：[bun-sourcemap-leak](https://raw.githubusercontent.com/harbor-framework/terminal-bench/v3.0.0/tasks/bun-sourcemap-leak/instruction.md)｜[wal-recovery-ordering](https://raw.githubusercontent.com/harbor-framework/terminal-bench/v3.0.0/tasks/wal-recovery-ordering/instruction.md)（[toml](https://raw.githubusercontent.com/harbor-framework/terminal-bench/v3.0.0/tasks/wal-recovery-ordering/task.toml)）｜[sglang-qwen-burst](https://raw.githubusercontent.com/harbor-framework/terminal-bench/v3.0.0/tasks/sglang-qwen-burst/instruction.md)（[toml](https://raw.githubusercontent.com/harbor-framework/terminal-bench/v3.0.0/tasks/sglang-qwen-burst/task.toml)）
- 4.0：同三名题在 v4.0.0 标签下逐字复在，配置差异见下节；[v4.0.0 目录](https://github.com/harbor-framework/terminal-bench/tree/v4.0.0/tasks)
