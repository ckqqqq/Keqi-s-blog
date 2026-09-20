---
title: "DeepSeek Infra Stack：Deepseek infra 开源仓库学习，持续更新中 (From V3 to V4.1)"
date: 2026-09-15
lastmod: 2026-09-21
draft: false
visibility: public
categories: ["系统架构"]
tags: ["DeepSeek","架构地图"]
description: "DeepSeek infra 开源仓库的分层整理：算子、通信、运行时、服务。附 V4.1 技术报告的系统设计笔记。"
---

> 硬件前提先放这：文里 Hopper 和 Blackwell 的东西都有。H100/H20 能跑 Hopper 路径；SM100/SM103 那些实验我没有卡，只读了代码，别当成我复现过。性能数字全是原文的，硬件前提以原文为准。

---

## 四层地图

这几个仓库一开始是一个个孤立读的，读到后面才拼出一张图：

```
┌─────────────────────────────────────────────────────────────┐
│ 第 4 层  服务与调度层（Serving / Scheduling）                  │
│   · nano-vllm        学习用的最小实现（不是生产目标）           │
│   · DSec             Agent 沙箱平台，百万级并发容器             │
│   核心问题：KV Cache 分层管理、PD/EPD 分离、调度、前缀缓存        │
├─────────────────────────────────────────────────────────────┤
│ 第 3 层  运行时与代码生成层（Runtime / Codegen）                │
│   · DeepJIT          CUDA + 昇腾 双后端 JIT 编译/缓存/加载       │
│   · TileLang/Triton  DSL 层（nano-vllm、TileKernels 都用）      │
│   核心问题：一次编写多硬件运行、编译缓存、分布式缓存共享           │
├─────────────────────────────────────────────────────────────┤
│ 第 2 层  通信层（Communication）                               │
│   · DeepEP           MoE dispatch/combine，EP/PP/CP 原语        │
│   · 关键演进：NVSHMEM → NCCL Gin，SM 占用大幅下降               │
│   核心问题：多机多卡 all-to-all、零 SM 占用、FP8 通信            │
├─────────────────────────────────────────────────────────────┤
│ 第 1 层  算子层（Kernels）                                     │
│   · FlashMLA                    MLA / DSA 注意力 kernel        │
│   · DeepSeek-Sparse-Attention-Kernels  竞赛方案（Triton）       │
│   · DeepSelect（新）             DSA TopK kernel               │
│   · DeepGEMM                     Mega-Gate/mHC/MoE 融合 kernel  │
│   · TileKernels（新）            TileLang 写的训练侧 kernel      │
│   核心问题：attention 融合、DSA 三段式、FP4/FP8、SM100           │
└─────────────────────────────────────────────────────────────┘
```

图本身不复杂，有意思的是它是被模型架构赶着走的：

- V3.2 加了 DSA，第 1 层就得新写 indexer、top-k、sparse attention 三类 kernel
- V4.1 上 FP4 KV Cache，第 1 层支持 FP4 还不够，第 2 层的通信也得跟着支持
- V4.1 搞 SWA Bounded Replay，第 4 层的 KV 分层和逐出策略要重做
- Engram 要条件记忆，第 2 层要有 RMA，第 1 层要有 gating kernel
- 昇腾这些国产卡，压力全在第 3 层的跨硬件 JIT 上

找方向的话，这张图基本就是答案：模型每改一次架构，infra 就有一层要重写。

---

## V4.1 报告的笔记

从阅读笔记里挑出来的系统设计部分，数字都是报告口径，以原文为准。

### 模型规模
- 552B backbone + 196B Engram。Engram 这个参数量，已经不能当外挂看了
- CED 架构，prefill 激活 8B、decode 激活 16B，两边不一样
- 40 层 = 20 encoder + 20 decoder，前两层 SWA，剩下 CSA2
- 上下文 100 万 token

### KV Cache 三级存储
| 层级 | 存什么 | 容量/寿命 |
|---|---|---|
| HBM | global KV | 890 bytes/token（V4-Flash 的 1/4，V1 的 1/437） |
| Host DRAM | encoder SWA KV | 每机 10% 内存做分布式池，TTL 几分钟 |
| SSD / Host | persistent KV | global KV 保 72 小时以上 |

KV cache 做到这份上，性质已经变了——不是显存里一块 buffer，是个跨 HBM/DRAM/SSD、带 TTL 和逐出策略的存储系统。开源引擎具体支持到哪步，得按版本核对。

### SWA Bounded Replay
V4 的时候说不缓存 SWA KV、miss 了重算（Zero SWA Caching），但精确恢复要跑 `L × n_win` 个 token 的前向，生产上跑不起。V4.1 退一步：只 replay 最近 `n_win` 个 token，承认是近似。

- query 在位置 `i`，只 attend `[max(s, i-W+1), i]`
- encoder 侧：replay 段只重生成 SWA KV，global KV 用缓存的，不重算不覆盖
- decoder 侧：前向压进 `n_win` 个 token，prefill 计算量差不多砍半

代价写得很坦白：replay 状态和完整前向数学上不等价，只是实测质量影响可以忽略。为了稳，post-training 阶段也模拟同样的 replay（train-aware adaptation）。

### Kernel 融合
CSA2 Reuse Mode 层，prefill 15 个 kernel、decode 11 个。涉及：

- FlashMLA：fused-RoPE-attention-RoPE-cast
- DeepGEMM：Mega-Gate、Mega-mHC、Mega-MoE
- TileKernels：Engram / mHC 相关
- DeepSelect：TopK

Single-Pass mHC 那个 Mega-mHC kernel，activation 内存流量比原来 4-kernel 的版本减半：理论下界 `(2n+2)d`，原实现 `(4n+4)d`，改完 `(3n+2)d`。

### 部署
EPD 分离：Encoder / Prefill / Decode 各自独立扩缩容、重叠执行。PD 分离之上又拆了一层。

### 训练侧
CSA2 训练要处理跨层共享 KV / indexer K / Top-K 索引，引出三个机制：

- Shadow indexers：共享参数只有一个逻辑 owner 管优化和 checkpoint，各 pipeline stage 放轻量副本跑
- Pipeline payload extensions：跨 stage 传中间表示和稀疏路由，按 CP 分区
- Micro-batch 级共享状态管理：跟踪并发 micro-batch 的共享状态生命周期，配激活重计算

Engram 训练：embedding 表按行切到独立进程组，优化器状态再分片；FP8 存和读，检索值和 scale 直接喂 GEMM；Sinkhorn 归一化避免整表重写；RL rollout 期间表常驻显存。

多模态长序列：图像按 CP rank 匀着切，每张图只读一次。判据 `ρ < (B_IO/B_GPU)·C`，跟序列长度无关。

### 后训练（异步 RL）
- rollout 和 training 同机 colocat，时间片共享
- sample-level dispatch。batch-level 和 prompt-level 都试过，不行
- concatenated routing-replay：跨 checkpoint 的样本，各段专家路由拼起来用，不丢不重算
- token 级中断，任意 token 边界能停
- rollout 状态按 token 粒度落盘（KV cache + 专家路由），换 checkpoint 直接用，省掉重新 prefill
- 集群抢占也用同一套快速中断 + 恢复
- OPD：40+ 异构 teacher，架构可以不一样，切换开销可忽略

### DSec（Agent 沙箱）
- 百万级并发容器，跨 harness / 平台 / 代码库 / 依赖
- 没用 K8s，自研 placement engine：不要强一致，多副本异步决策，节点侧硬准入兜底
- 单机并发从 ~1000 提到 2500+（sub-NUMA 分区 + 绑核 + 内存超分）
- latency-sensitive 的执行类用 `SCHED_IDLE` + core scheduling 挡超线程干扰
- 防 agent 作恶是认真做过的：per-sandbox AppArmor + 细粒度 eBPF 网络策略。报告里列了真出过的状况——agent 钻 XFS 权限、触发 AppArmor 非法内存访问、从镜像服务偷答案、删库

### 报告自己承认没解决的
> "the newly introduced architectural changes also create **robustness boundaries that have yet to be fully characterized**"
> "Potential **selection errors in CSA2** and **approximate state reconstruction in SWA Bounded Replay** may still cause capability degradation in untested boundary cases."
> "we will continue to expand our **stress-testing and evaluation stack**, with particular attention to **sparse retrieval over long contexts** and **SWA state reconstruction at cache-resumption boundaries**."

要找研究切入点的话，这几句基本是指路牌。

### 报告给的方向
- 数据、模型容量、RL 协同 scaling
- 大规模数据合成 + RL scaling
- model–harness co-design

---

## 版本说明

引用的 DeepSeek-V4.1-Flash 技术报告是 2026-09-10 的 51 页版本（`DeepSeek_V41_Tech_Report.pdf`），页码对应该版。没找到可确认的公开下载链接，PDF 没放进本站；数字沿用笔记，建议对原文复核。
