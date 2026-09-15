---
title: "六个 Infra 项目的技术职责与实现边界"
date: 2026-09-15
lastmod: 2026-09-21
draft: false
visibility: public
categories: ["系统架构"]
tags: ["开源项目","源码解读"]
description: "六个 Infra 项目的技术职责与实现边界；保留技术细节、出处与适用边界。"
---

> 这是技术笔记的公开整理版，原始记录日期为 2026-09-15，本次编辑于 2026-09-21。保留原笔记的公式、代码位置与证据分级；本次仅整理内容，未重新运行 GPU 实验或逐项复核上游。标为【码】【核验】的内容指原记录的核查结果，不代表当前版本仍然如此。

> **硬件边界**：本文涉及 Hopper 与 Blackwell 的不同实现。H100/H20 可用于适配的 Hopper 路径；SM100/SM103 专属实验需单独申请 B200/B300/GB300 等对应设备。没有对应设备时，只能阅读代码或进行不依赖设备的逻辑验证，不能声称复现文中性能。所有跑分沿用原文标注的硬件前提。


---

## 一、逐仓库要点

### 1. DeepEP —— MoE 专家并行通信库

- **定位**:MoE 训练/推理的 all-to-all(dispatch/combine)通信底座,被 SGLang/vLLM/DeepSeek 训练栈实际使用。设计目标是"0 或极少 SM 占用"(V1 用 24 SM → V2 降到 4–6)【文档声明】。
- **核心技术**【码】:已从 NVSHMEM 迁移到 **NCCL Gin**(NCCL 2.30+ 设备端 API + GDAKI/GPUDirect RDMA);`csrc/kernels/elastic/` 是 V2 核心;自研 JIT(`csrc/jit/`)做到安装零编译。支持 FP8 低精度传输。
- **新颖度证据**:V1→V2 是换后端级重构(2026-04);NVFP4、zero-copy、Compute Fabric Transport 仍在实验分支【文档声明】。Wide-EP(EP 规模上千)是当前 MoE infra 最热的方向。
- **门槛**:CUDA kernel + PyTorch distributed + RDMA verbs + Hopper/Blackwell 特性(TMA/mbarrier/cluster)。建议路径:README → `deep_ep/buffers/elastic.py` API 语义 → `csrc/kernels/elastic/dispatch.hpp` → legacy 与 JIT。

### 2. FlashMLA —— MLA/稀疏注意力 kernel 库

- **定位**:DeepSeek-V3 / V3.2 / V4.1 生产模型的注意力算子层,含 DSA 稀疏 prefill/decode、FP8/FP4 KV cache、V4.1 融合 kernel(norm+RoPE+attention+cast 一体化)。
- **核心技术**【码】:纯手写 CUDA(CUTLASS 辅助),按 SM90(H800)/SM100(B200)分目录;WGMMA/UMMA、TMA、paged KV cache + tile scheduler 负载均衡、在线 softmax split-combine。
- **成熟度**:被 SGLang/vLLM 集成;沐曦、摩尔线程、海光、天数、AMD 均做了移植(README 有 Community Support 章节)——生态影响力的直接证据【文档声明】。性能达硬件极限(660 TFLOPS@H800 / 1450 TFLOPS@B200;⚠️ 这两个数字分别是 H800/B200 上测的,不可搬到 H20)。
- **新颖度证据**【推断】:DSA、FP4 KV cache、kernel 融合都是近一年内出现的;SM100 仍在持续适配(最近一次提交就是今天)。

### 3. DeepJIT —— 跨后端 kernel JIT 基础设施

- **定位**:把 DeepEP `csrc/jit/` 抽成公共库,CUDA + **昇腾 Ascend** 双后端统一的运行时编译/缓存/launch;DeepGEMM 新版本已迁移使用。
- **核心技术**【码】:C++20 header-only ~9千行;include 递归追踪的确定性 cache key;原子 rename + `.committed` 标记的多进程安全发布;支持分布式文件系统共享缓存。
- **新颖度判断**【推断】:JIT 编译+缓存本身是成熟问题域(torch cpp_extension、jitify 早有),新颖点在**双后端统一(国产 NPU 场景稀缺)**与**多机共享缓存语义**。属"成熟问题上的新工程实现",不是研究前沿。
- **活跃度**:2026-09-08 才开源,3 提交、3 贡献者、⭐310(发布一周)【文档声明】。出身正统(LyricZhao 主导,FlashMLA/DeepGEMM 作者)。

### 4. DeepSeek-Sparse-Attention-Kernels —— DSA 竞赛冠军作品

- **定位**:MLSys 2026 FlashInfer Kernel 竞赛 DSA 赛道人机协作组第一名(平均 37.07x 加速)【文档声明】。实现了 DSA 解码的两个核心算子:Top-K Indexer(FP8 打分选 top-2048)与 Sparse Attention。
- **核心技术**【码】:Triton 为主(indexer_fused.py 309 行 / sparse_fused.py 378 行)+ CUDA radix top-K;针对 B200 优化(tcgen05 MMA tile 对齐、atomic counter 单 launch barrier)。
- **性质**:个人单作者作品、赛后一次性开源(仅 1 次提交),**不是可长期跟进的活跃项目**;但作为学习材料质量极高,
- **门槛**:MLA 结构、paged KV cache、FP8 per-token scale、Triton 进阶、split-combine 解码技巧。代码仅 1300 行,门槛在背景知识而非代码量。

### 5. Engram —— 条件记忆(第二稀疏轴)论文 demo

- **定位**:官方论文《Conditional Memory via Scalable Lookup》的配套演示。提出与 MoE 互补的**条件记忆**:N-gram 哈希 O(1) 查表替代部分神经计算;确定性寻址允许把巨大 embedding 表 **offload 到 host DRAM**,推理时按 hash 异步预取。
- **代码现状**【码】:仅 `engram_demo_v1.py` 420 行纯 PyTorch,Attention/MoE/mHC 全是 mock;**无 CUDA kernel、无预取流水线、无测试**,2026-01 发布即归档(4 次提交)。
- **门槛**:读懂 demo 只需 PyTorch 中级(1–2 天);做出 infra 贡献需补 embedding lookup kernel、pinned memory 异步预取、PCIe 带宽建模——**正是仓库缺失的部分,边补边学**。

### 6. nano-vllm —— vLLM 教学复刻

- **定位**:~1200 行 Python 复刻 vLLM 核心(PagedAttention、continuous batching、prefix caching、CUDA Graph、TP),教学用途。
- **技术栈**【码】:PyTorch + Triton(30 行 KV cache kernel)+ 现成 flash-attn + NCCL TP。重活全部外包,不手写 CUDA。
- **新颖度判断**【推断】:覆盖的是 2023–2024 已定型的范式(必修课),**不覆盖**投机解码、PD 分离、稀疏注意力等前沿。学完是地基,差异化要靠别的仓库。
- **价值**:读懂 vLLM/SGLang 源码前的最佳垫脚石;也是承接 Engram 空白(把查表模块接进推理引擎)的实验田。

---


## 文献版本说明

本文所引 DeepSeek-V4.1-Flash 技术报告为原笔记记录的 2026-09-10、51 页版本（文件名 `DeepSeek_V41_Tech_Report.pdf`），页码对应这一版本。本次整理未获得可独立确认的公开下载链接，未将 PDF 打包进本站；涉及该报告的数值沿用原笔记，仍需对照原文复核。
