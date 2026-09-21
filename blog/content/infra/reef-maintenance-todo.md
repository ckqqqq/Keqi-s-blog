---
title: "Reef 仓库维护 TODO"
date: 2026-09-21
lastmod: 2026-09-21
draft: false
visibility: public
categories: ["开源维护"]
tags: ["Reef", "Agent Infra", "持续学习"]
description: "fork 的 reef 仓库的维护与跟进来清单：上游同步、Issue 跟踪、重点模块精读与本地验证计划。"
---

> 这是个人维护清单，面向 [Human-Agent-Society/reef](https://github.com/Human-Agent-Society/reef)（本地 fork：`ckqqqq/reef`）。Reef 是 Agent 持续学习基础设施：推理 → 反馈 → 学习 → 版本化发布的闭环。本页持续更新，完成的条目就地打勾并注明日期。

## 上游同步

- [ ] 建立定期的 upstream 同步节奏（上游 9 天 200 个 commit，不跟就会迅速落后）
- [ ] 同步时重点 review `training_mode hybrid`（#320）、harness requests（#311/#312）两条主线的后续改动
- [ ] 跟踪上游 LoRA checkpoint 容量（#328）、Ray executor 清理（#326）等稳定性修复的合入情况

## 模块精读（按优先级）

- [ ] `reef/runtime/`：artifact 热更新机制——服务不中断换权重/harness 是怎么实现的
- [ ] `reef/service/`：`/reef/report` 反馈上报与 receipt 机制（`x-reef-agent-record-id`）的数据流
- [ ] `reef/train/cordis_backend/`：harness 进化后端（免 GPU 路线，优先于权重训练路线精读）
- [ ] `reef/artifact/`：基于 git-lfs 的版本化 artifact 仓库设计
- [ ] `reef/train/slime_backend/`：Slime + SGLang + Megatron 的胶水层（与 DeepSeek infra 主线呼应）

## 本地验证

- [ ] 跑通 `tutorials/evolve-your-harness/`（纯 API 路线，无 GPU 门槛）
- [ ] CPU 侧 pytest 套件本地跑通（覆盖率门槛 80%）
- [ ] recipes 里挑一个做端到端复现：`skillclaw`（免 GPU）优先；`tttd`（Qwen3-8B LoRA）需要 H100/H20，单独申请算力窗口
- [ ] `docker/Dockerfile.reef` 完整运行时构建验证

## 社区参与

- [ ] 通读 `CONTRIBUTING.md`、`.github/MAINTAINER.md` 与 RFC 模板，明确贡献流程
- [ ] 从 issue 里挑一个 good-first-issue 级别的修复练手（优先 harness adapter 或文档类）
- [ ] 按 `recipes/AGENTS.md` 的目录约定，尝试提交一个自包含的 example
