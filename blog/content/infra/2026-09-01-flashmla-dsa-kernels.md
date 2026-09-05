---
title: "FlashMLA 与 DSA：内核、流水线和正确性边界"
date: 2026-09-01
lastmod: 2026-09-21
draft: false
visibility: public
categories: ["算子与通信"]
tags: ["FlashMLA","DSA","GPU"]
description: "FlashMLA 与 DSA：内核、流水线和正确性边界；保留技术细节、出处与适用边界。"
---

> **硬件边界**：本文涉及 Hopper 与 Blackwell 的不同实现。H100/H20 可用于适配的 Hopper 路径；SM100/SM103 专属实验需单独申请 B200/B300/GB300 等对应设备。没有对应设备时，只能阅读代码或进行不依赖设备的逻辑验证，不能声称复现文中性能。所有跑分沿用原文标注的硬件前提。

> 采集基线：FlashMLA `ba89a34`（2026-09-15）、DeepSeek-Sparse-Attention-Kernels `9734cf8`（2026-05-13，单次提交）。
> 证据分级：**【代码】**= 直接读源码，含 `file:line` / **【文档声明】**= README·blog·report.pdf 所述 / **【推断】**= 子代理推理 / **无法确认**= 证据不足。
> **全程只读**：分析过程未修改任何仓库文件。

---

## 0. 阅读指引

本文覆盖两个仓库，它们是**同一技术的两端**：

| | FlashMLA | DSA 竞赛方案 |
|---|---|---|
| 源码行数 | **31,997** | **2,113** |
| kernel 数 | **13 个 `__global__`** / 58 个编译单元 | **6 个**（5 Triton + 1 CUDA 模板） |
| 贡献者 / 提交 | **22 人 / 64 次** | **1 人 / 1 次** |
| Tag / CI / 结果产物 | 0 / 无 / — | 0 / 无 / **无** |
| 定位 | 生产级 kernel 库 | 研究原型 |

**一句话**：FlashMLA 是"**给定 indices 之后怎么做注意力**"的工业级答案；竞赛方案是"**怎么把 indices 算出来**"的研究级答案。**中间那段"生产级 indexer"，两个仓库都没做。**

---

## 1. 仓库总览与规模

### 1.1 FlashMLA

| 指标 | 数值 | 证据 |
|---|---|---|
| 源码总行数 | 31,997 | `find` + `xargs cat` + `wc -l`（不含 .git 与未检出的 cutlass） |
| `csrc/kernels` / `csrc/api` / `csrc/kerutils` | 22,944 / 1,853 / 2,534 | 分目录统计 |
| `flash_mla`（Python 胶水层）/ `tests` / `benchmark` | 669 / 2,771 / 547 | 同上 |
| `.cu` / `.cuh` / `.h+.hpp` | 51 / 22 / 59 | `find` 计数 |
| CUDA kernel 入口（`__global__`） | 13 | `grep -rn "__global__" csrc` |
| 被编译的 `.cu` 翻译单元 | 58 | `setup.py:67-143` 逐条计数 |
| 提交总数 / 贡献者 | 64 / 22 | `git rev-list --count`；`git shortlog -sne HEAD` |
| Tag / Release / CI | **0 / 0 / 无** | `git tag -l` 为空；无 `.github/` |

**目录职责**：`flash_mla/` 是 Python 门面；`csrc/api/` 是 7 个 C++ 胶水文件（校验、架构/格式分发、workspace、tile-scheduler metadata）；`csrc/kernels/sm90/` 与 `sm100/` 是两代架构内核；`csrc/kernels/smxx/` 是与架构无关的 `get_decoding_sched_meta` 与 `combine`；`csrc/kerutils/` 是自研设备端工具库（`ku::` 命名空间）；`csrc/cutlass/` 是 submodule，**本机未检出**，pin 在 `147f5673d0c1c3dcf66f78d677fd647e4a020219`。

**贡献者前 8**（`git shortlog -sne HEAD`）：
```
14 Shengyu Liu <shengyuliu@deepseek.com>
11+6 Jiashi Li
6  ljss
4  hpp
3  Sijia Chen <sijiac@meta.com>
3  Zeyu WANG <zeyuw@nvidia.com>
2  zhang <dianzhangc@nvidia.com>
2  zhengsize <zheng.size@bytedance.com>
```
外部贡献者还有 ByteDance、UC Berkeley。**【推断】** 形成"DeepSeek 内部 2 人主导 + 大厂外援"的官方内核库形态。

**提交节奏（按月）**：2025-02:**26**（开源首月爆发）/ 03:2 / 04:7（新 dense decode + 博客）/ 08:6（NVIDIA SM100 MHA PR）/ 09:**12**（DSA 稀疏内核）/ 10:1 / 2026-01:3 / 02:1 / 03:1 / 04:1 / 07:1 / 09:3（V4.1 内核）。
**【推断】** 2026 年的提交以补丁式修复为主（MSVC 兼容、CUDA 13 编译、grid 维度越界），**说明主架构已稳定**。

**构建**：`setup.py` + `torch.utils.cpp_extension.CUDAExtension`，扩展名 `flash_mla.cuda`。**需要 NVCC**（58 个 `.cu` 即时编译）。架构 flag 为 `compute_100a/sm_100a`、`compute_103a/sm_103a`、`compute_90a/sm_90a`（`setup.py:42-48`），注释说明选 arch-specific 而非 `sm_100f` 以求更好 SASS。CUDA ≥12.8，SM100 需 ≥12.9；PyTorch ≥2.0；C++20。

**两个"内行信号"**：`--use_fast_math`（`setup.py:158`）与 `--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage,--warn-on-double-precision-use`（`setup.py:159`）——**把寄存器分配策略当一级调优旋钮**。环境开关 `FLASH_MLA_DISABLE_FP16/SM90/SM100`（`setup.py:21-23,36-37`）。

**fused 内核测试另需** TileLang / Tile-Kernels / DeepGEMM（`README.md:60`）。

### 1.2 DeepSeek-Sparse-Attention-Kernels

| 文件 | 行数 |
|---|---|
| `solution/triton/topk.cuh` | 463 |
| `solution/triton/sparse_fused.py` | 378 |
| `scripts/run_eval.py` | 315 |
| `solution/triton/indexer_fused.py` | 309 |
| `annotated/indexer_fused_annotated.py` | 304 |
| `scripts/pack_solution.py` | 111 |
| `solution/triton/topk_ext.py` | 93 |
| `solution/triton/topk_binding.cu` | 90 |
| `README.md` / `config.toml` | 29 / 21 |
| **合计** | **2,113** |

去掉 304 行的本地 `annotated` 文件恰为 **1,809 行** = 唯一提交 `9734cf8` "initial commit" 的 insertion 数。`report.pdf` 5 页 / 163,536 B。**1 位作者（Doğaç Eldenk，Northwestern University），1 次提交，无 tag。**

**★B1 仓库卫生**（已亲自 `git ls-files` / `git status` 核实）：**`solution/.DS_Store` 被 git 跟踪**（在 initial commit 里），而 **`annotated/` 未跟踪**（`git status --short` 输出 `?? annotated/`）。即中文注释版是本地学习产物，**不属于提交内容**。`solution/` 下只有 `triton/`，没有 `cuda/`、没有 `cutedsl/`、没有 `solution.json`、**没有任何 benchmark 结果或日志** —— **37.07x 在仓库内不可独立验证**。

**★B2 `annotated` 文件不是忠实副本（教学陷阱）**：`annotated/indexer_fused_annotated.py:125-130` 把 `_score_kernel` 的 **12 个 stride 参数声明为普通运行时标量**，而正式版 `solution/triton/indexer_fused.py:134-147` **全部声明为 `tl.constexpr`**。其余仅为注释差异。后果：annotated 版会以运行时 stride 编译（**失去常量折叠/向量化特化**），codegen 不同。
→ **建议：数值/性能相关一律读 `solution/triton/indexer_fused.py`；`annotated/` 只当中文阅读辅助。**

---

## 2. 内核清单（含文件路径与目标架构）

### 2.1 FlashMLA 全部 13 个 `__global__` 内核

| # | 内核符号 | 文件:行 | 阶段 | 架构 | dtype |
|---|---|---|---|---|---|
| 1 | `get_decoding_sched_meta` | `csrc/kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu:11` | 调度元数据 | smxx | int32 |
| 2 | `flash_fwd_mla_combine_kernel` | `csrc/kernels/smxx/decode/combine/combine.cu:19` | decode split-KV 归约 | smxx | 模板 ElementT + fp32 LSE |
| 3 | `sparse_attn_fwd_kernel` | `csrc/kernels/sm90/prefill/sparse/phase1.cuh:569` | sparse prefill | SM90a | bf16 |
| 4 | `flash_fwd_splitkv_mla_kernel` | `csrc/kernels/sm90/decode/dense/splitkv_mla.cuh:960` | dense decode | SM90a | bf16/fp16 |
| 5 | `flash_fwd_splitkv_mla_fp8_sparse_kernel` | `csrc/kernels/sm90/decode/sparse/splitkv_mla.cuh:681` | sparse decode | SM90a（2-CTA cluster） | FP8 e4m3 KV → bf16 |
| 6 | `sparse_attn_fwd_kernel` | `csrc/kernels/sm100/prefill/sparse/fwd/head64/phase1.cuh:697` | sparse prefill | SM100f | bf16 |
| 7 | `sparse_attn_fwd_kernel` | `csrc/kernels/sm100/prefill/sparse/fwd/head128/phase1.cuh:627` | sparse prefill | SM100f | bf16 |
| 8 | `..._for_small_topk_kernel`（prefill） | `.../fwd_for_small_topk/head128/phase1.cuh:966` | sparse prefill（小 topk 特化） | SM100f | bf16 + FP8/FP4 KV |
| 9 | 同上（decode） | `.../fwd_for_small_topk/head128/phase1.cuh:972` | sparse decode（h=128,d_qk=512） | SM100f | FP8 e4m3 / e8m0，可选 FP4 extra |
| 10 | `permute_wv_proj_kernel` | `.../fused_norm_rope_attn_rope_cast_fwd/permute_wv_proj/kernel.cu:33` | 权重预处理 | SM100 目标 | bf16 + FP8 |
| 11 | `run_fused_norm_rope_attn_rope_cast_fwd_kernel` | `.../core_attn/kernel.cuh:1450` | 融合 prefill/decode | SM100f（2-CTA cluster） | bf16 Q/KV → FP8 e4m3 出 + ue8m0 scale |
| 12 | `permute_q_b_proj_kernel` | `.../permute_q_b_proj/kernel.cu:23` | 权重预处理 | 同上 | 同上 |
| 13 | `flash_fwd_splitkv_mla_fp8_sparse_kernel` | `csrc/kernels/sm100/decode/sparse/head64/kernel.cuh:806` | sparse decode（h=64） | SM100f | FP8 e4m3 / ue8m0 / FP4 e2m1+e4m3 |

**注**：SM100 的 **dense prefill/backward 不在上表** —— 它们不是手写 kernel，而是从 NVIDIA CUTLASS `cutlass::fmha` collective 派生的模板实例，入口是 `fmha_cutlass_fwd_sm100.cu` 的 `FMHACutlassSM100FwdRun` / `run_fmha_fwd<...>` 与 `fmha_cutlass_bwd_sm100.cu` 的 `FMHACutlassSM100BwdRun`；dtype 固定为 `cutlass::bfloat16_t`。

**编译期实例化矩阵**（`setup.py:82-142`）：SM90 dense decode 2（fp16/bf16）；SM90 sparse decode 4（v4/v32 × h64/h128 persistent）；SM90 sparse prefill 4（k512/k512_topklen/k576/k576_topklen）；SM100 sparse prefill 5；SM100 fused norm-RoPE 16（`{V4,V41,V41_FP4}`×`{h64,h128}`×`{prefill,decode}`×`{norm,nonorm}`）；SM100 sparse decode 14（head64 的 4 种 KV 格式 × split/no-split = 8；small_topk head128 decode 6）。**总计 58 个 `.cu`。**
**【推断】** 编译时间是最现实的工程痛点：58 个 `.cu` × 3 个 arch target，`NVCC_THREADS` 默认 32（`setup.py:51`）正是为此。

### 2.2 竞赛方案的 6 个内核

5 个 `@triton.jit`（`indexer_fused.py:26, :50, :125`；`sparse_fused.py:19, :177`）+ 1 个 CUDA `__global__` 模板（`topk.cuh:181`，实例化 8 次：BT∈{512,1024} × VEC_SIZE∈{1,2,4,8}）。

| 内核 | 文件:行 | 阶段 | dtype |
|---|---|---|---|
| `_passthrough_single_page_kernel` | `indexer_fused.py:26-47` | indexer 快路径（单页） | int32 |
| `_passthrough_multi_page_kernel` | `indexer_fused.py:50-81` | indexer 快路径（多页） | int32 |
| `_score_kernel` | `indexer_fused.py:125-208` | indexer 打分 | FP8 e4m3 入 / FP32 累加 / BF16 存 |
| `FilteredTopKUnifiedKernel` | `topk.cuh:181-403` | top-k + page-table 融合 | bf16 分数 / int32 索引 |
| `_small_t_kernel` | `sparse_fused.py:177-274` | sparse attention（T<=2 特化） | bf16 |
| `_fused_split_combine_kernel` | `sparse_fused.py:19-174` | sparse attention（split-K + combine 单次发射） | bf16 / fp32 累加 / bf16 partial |

关键常量：`PAGE_SIZE=64, HEAD_COUNT=64, HEAD_DIM=128, TOPK=2048`（`indexer_fused.py:17-20`）；sparse 侧 `H=16, D_ckv=512, D_kpe=64`（`sparse_fused.py:8`）。

> **★【推断，极重要】** `config.toml` 定义的 workload 是 `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`（**active**）与 `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`（**被注释掉**）。sparse attention 只有 **16 个 query head**，而 FlashMLA 生产内核最小配置是 `h_q=64`（`sparse_decode.cpp:118` 的 `TORCH_CHECK(params.h_q == 64, ...)`）。**这个竞赛方案无法直接替换 FlashMLA，它是被缩小过的代理 workload。** 看到"37x 击败 FlashInfer"必须带上这个上下文。

---

## 3. MLA 解码内核技术剖析

### 3.1 数据布局：为什么是 656 字节

MLA 解码等价于 MQA：`h_q=128, h_kv=1, head_dim_k=576, head_dim_v=512`。
FP8 KV cache 每 token **656 字节**三段拼接：前 **512 B** = 512 × `float8_e4m3`（量化 NoPE）；中 **16 B** = 4 × `float32` scale（**每 128 个 e4m3 共享一个 scale**）；后 **128 B** = 64 × `bfloat16`（RoPE，**故意不量化**）。

**【代码】** 硬编码在 `csrc/kernels/sm90/decode/sparse/components/config.h:16-19`：`QUANT_TILE_SIZE=128`；`NUM_SCALES=HEAD_DIM_NOPE/QUANT_TILE_SIZE=4`；`NUM_BYTES_PER_TOKEN = 512 + 4*4 + 64*2 = 656`。

V4/V4.1 改为 **SOA**（一个 page block 先存 data 行再存 scale 行），scale 类型从 fp32 换成 `float8_e8m0`，粒度细化到 **64/32/16**。**【代码】** 由 `csrc/kernels/kv_cache_format.h:8-28` 的 `KVCacheFormat<ModelType>` 统一描述，`BYTES_PER_TOKEN` 依次 **656 / 584 / 528 / 288**。

### 3.2 split_kv 为什么存在

解码时 `s_k=128K` 但 `s_q=1`，若一个 CTA 吃整条序列，grid 只有 `batch_size` 个 block，小 batch 大 seqlen 下 SM 严重闲置。解法是 Flash-Decoding 式 split-KV：切 `num_splits` 段，各自在线 softmax 得局部 `(m_j,l_j,acc_j)`，再由 combine kernel 按 `out = Σ acc_j·e^{m_j-m} / Σ l_j·e^{m_j-m}` 合并（`combine.cu:36-38`）。

**关键设计：`num_splits` 不是静态的。** `get_decoding_sched_meta` 内核（`get_decoding_sched_meta.cu:64-105`）在 GPU 上做负载均衡，把 `(请求, block, split)` 三元组贪心分给 `num_sm_parts` 个 SM-part，结果写入 `DecodingSchedMeta{begin_req_idx, end_req_idx, begin_block_idx, end_block_idx, begin_split_idx, is_first_req_splitted, is_last_req_splitted}`（`params.h:10-17`）。

**【代码】** `num_sm_parts` 随架构/head 数变化：SM90 = `num_sms/s_q/(h_q/64)`（`sparse_decode.cpp:71`）；SM100 head64 = `num_sms/s_q`（:108）；SM100 head64x2 = `num_sms/s_q`（:147）；SM100 真 head128 = `num_sms/s_q/2`（:198）。

**【代码】** 何时开 split-KV 是启发式：`bool enable_split_kv = arch.is_sm90a() || !(topk + extra_topk <= 640);`（`sparse_decode.cpp:268`）—— **SM90 永远开，SM100 只在 topk>640 时开**。

**★【代码 vs 文档不一致】** `docs/20250422-new-kernel-deep-dive.md:60-61` 声称用 Programmatic Dependent Launch 重叠 `splitkv_mla` 与 `combine`，但 SM90 稀疏解码里 PDL **被注释掉了**：`splitkv_mla.cuh:757-758` "`// NOTE Don't use PDL because of potential compiler bugs!`"；而 `combine.cu:57-59` 仍保留 `cudaGridDependencySynchronize()`，说明 dense 路径仍在用。

### 3.3 密集解码：seesaw 调度（FlashMLA 最核心的技术资产）

**【文档声明】** 瓶颈推导（`docs/20250422:11-15`）：`FLOPs ≈ 2·h_q·s_q·s_k·(d_k+d_v)`；访存 `≈ 2·s_k·d_k` 字节；算术强度 `≈ 2·h_q·s_q`。H800 实际（降频后）≈ 865 TFlops / 3.35 TB/s ⇒ **分水岭在 `h_q·s_q ≈ 128`**。DeepSeek 线上推理**不用 TP**，`h_q=128` ⇒ **compute-bound**。

**【文档声明】** 寄存器墙（:21）：`64×512` 的 fp32 输出矩阵占 `64×512 = 32,768` 个 32-bit 寄存器，而每 SM 只有 **65,536** 个 ⇒ **直接封死了 FlashAttention-3 的 ping-pong 双缓冲方案**（需两份输出矩阵）。

**【文档声明】** 解法是 seesaw 调度（:25-42）：每轮取两个 KV block（`K0,K1,V0,V1`），输出纵切为 `O_L`（warpgroup 0，64×256）与 `O_R`（warpgroup 1）。11 步序列的关键在于 `scale_0` 与 `scale_1` 分属两个 warpgroup，通过 `p0 ← p0·scale_1`（第 9 步）与 `O_R ← O_R + p0·V0R`（第 10 步）补回交叉项 —— **数学上等价于标准 online softmax，但让两个 warpgroup 交替占用 Tensor Core 与 CUDA Core**。

**【代码】** 佐证：输出被 `local_tile(sOutputBuf, Shape<_64,_256>{}, make_coord(_0{}, warpgroup_idx))` 切成两半（`splitkv_mla.cuh:635`）；两个 warpgroup 用 `NamedBarrier` 做 `sScale0Ready/sScale1Ready/sP0Ready/rO1sP0sV0RIssued` 四阶段同步（:779,:792,:800,:808,:890,:892,:913,:920）；exp 强制走 `exp2f`（:383-384,:456-457）配合 `scale_softmax_log2`。

**【文档声明】** 其他细节（:52-61）：**细粒度 TMA-GEMM 流水** —— 一个 64×576 K block 拆成 **9 次 64×64 TMA copy**，每完成一片即启动对应 GEMM；**Cache hint** `cute::TMA::CacheHintSm90::EVICT_FIRST` 提升 L2 命中；结果 **80% Tensor Core 利用率 + 3 TB/s**。

### 3.4 稀疏解码：crossover + DSM（第二篇 deep-dive 核心）

**【文档声明】** 问题定义（`docs/20250929:17-25`）：H800 每 SM 每 cycle 4096 MMA Flops（989 TFlops/1830 MHz/132 SMs）；64 个 query head 处理 1 个 KV token 的 MMA 成本 `64×(576+512)×2/4096 ≈ 34 cycles`；但 H800 **不支持 fp8→bf16 直接转换**，须 `e4m3→half→fp32→bf16→乘 scale`，每 token 约 `(1/64+1/64+1/16+1/256)×512 ≈ 50 cycles` ⇒ **内核是 dequantization-bound**。

**【文档声明】** crossover 方案（:31-45）：(1) 两个 CTA 组成 cluster of 2；(2) 各只加载**一半**量化 KV；(3) 各自在 CUDA Core 反量化；(4) 写进自己的 shared memory；(5) **同时用 `st.async` 写进 peer CTA 的 shared memory**（Distributed Shared Memory）；(6) 用 cluster transaction barrier 同步。

**【代码】** 逐条对应（`sm90/decode/sparse/splitkv_mla.cuh`）：
- `:93` `const int idx_in_cluster = CLUSTER_SIZE == 1 ? 0 : head_block_idx % 2;`
- `:456` `cutlass::arch::warpgroup_reg_dealloc<152>();` // 生产者 warpgroup 让出寄存器
- `:460` `static constexpr int NUM_TOKENS_PER_ROUND = 32;`
- `:584` `fp8x16 cur_fp8x16 = load_128b_from_gmem<fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(...);`
- `:591` / `:621` `st_async_128b(...)` 写 peer CTA
- `:669` `sync_all_threads_in_cluster();`
- `:681` `__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE)`
- 注释说明 `EVICT_LAST` 是因为 V3.2 的 `gK_base` 可能不是 32B 对齐（:584）

**【文档声明】** 收益：**410 TFLOPS vs 优化前 250 TFLOPS**（:48，配置 `batch=128, heads=128, s_q=2, topk=2048`）；`topk=32768` 时可达 460 TFLOPS（:50）。

### 3.5 SM100 侧解码：路线完全不同

**【代码】** `head64/kernel.cuh:1-14` 头注释写得很清楚："Uses FP8 KV cache with dequantization, **UTCMMA/TMEM** architecture, and TMA for data transfers. Supports split-KV partitioning for large topk values. ... `NUM_THREADS=384` (3 warpgroups)"。

要点：
- **UTCMMA(tcgen05) + TMEM** —— `cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data())`（`kernel.cuh:72`），TMEM 列分配 `O=0..256 / Q=256.. / P=400`（`config.h:64-72`）
- **TMA gather4** 按 index 直接 gather 稀疏 KV（`config.h:56-57` 讨论 `RAW_TOKEN_SMEM_STRIDE` 与 `4*stride % 128 == 0`）
- **反量化器 `KVBlockDequantizer`**（`dequant_utils.cuh:74+`）用 1 个 warpgroup 把 `B_TOPK` 个 token 的 fp8/fp4 行搬进 smem 并转成 bf16 写进 canonical SW128 K-major 布局；注释透露 bank-conflict 级细节 —— "8 threads per token cover one swizzle-atom column ... so the STS.128 of a wavefront (8 lanes) is conflict-free"，fp4 行额外 pad 32 B 让 LDS.32 无冲突（`config.h:60-62`）
- **KV 格式对作为模板参数**：`for_each_kv_block<Orig,Extra>`（`dequant_utils.cuh:20-35`），注释解释为何拆成两循环 —— "so that the code of the two formats is never interleaved (**ptxas would otherwise merge the live ranges of both**)" **【推断】** 这是被寄存器压力逼出来的写法

---

## 4. DSA 三段式流水线剖析

### 4.0 ★ 首要发现：FlashMLA **没有** indexer，也**没有** top-k

对整个 FlashMLA 仓库 `grep -rni "indexer|lightning|topk_select|top_k"` —— **零命中**。

FlashMLA 的稀疏内核只接受**已算好的 indices**（`flash_mla_interface.py:53-70`）。indices 语义被精确定义为**物理 token id**：`indices_in_kvcache[i][j][k] = (page block index)*page_block_size + (offset in page)`，因此**不需要 block_table**（`README.md:147`；`flash_mla_interface.py:86-87`）；无效项写 `-1`（`README.md:148`）。稀疏 **prefill** 契约更极端：`flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512, ...)` **没有 batch 维**（`README.md:165`），indices 是**逻辑索引**（直接 index 进 `kv[s_kv, d_qk]`），完全不走 page table。

> **【推断】这条分工是理解整个技术栈的钥匙**：**上游（模型侧）** lightning indexer 打分 + top-k 在 DeepSeek-V3.2-Exp 里用 TileLang + `torch.topk` 实现，FlashMLA 里根本没有；**下游（内核侧）** 给定 indices 的稀疏注意力才是 FlashMLA 的职责。**这正好解释了竞赛为什么存在** —— 上游两步在生产里是性能灾难。

### 4.1 Stage 1 — Lightning Indexer

**【文档声明】** 数学定义（`DeepSeek_V3_2.pdf` p.0, Eq.1）：`I_{t,s} = Σ_{j=1..H_I} w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)`。论文明确说 "We choose ReLU as the activation function for throughput consideration"，且 "the lightning indexer has a small number of heads and can be implemented in FP8"。**关键结构：`k^I_s` 没有 head 维**（所有 indexer head 共享同一 key 投影），整个打分退化成 **1 次 GEMV 而非 GEMM**。

**【代码】** 参考实现（`DeepSeek-V3.2-Exp/inference/kernel.py:530-636`，TileLang）：
- `:568-572` 输入 `q:(b,m,h,d)` FP8 / `q_s:(b,m,h)` FP32 / `k:(b,n,d)` FP8 / `k_s:(b,n)` FP32 → `o:(b,m,n)` FP32
- `:576` `grid=(batch, query, ceil(n/512))`
- `:579-580` q 钉进 shared memory（q 小且复用，k 大且流式，与 FlashAttention 同思想）
- `:586` `T.Pipelined(4, num_stages=2)`
- `:599-606` `T.gemm(k_smem, q_smem, logits, transpose_B=True, clear_accum=True)`
- `:610-611` `logits = max(logits,0) * q_s_frag[i_h]`
- `:616` `T.reduce_sum(logits, logits_sum, dim=1)`
- `:619-620` `logits_sum *= k_s_frag[i_n]`

精度【文档声明】：`q,k ∈ F_{8,e4m3}^128`；`w ∈ R float32`；scale ∈ float32；**score 输出在 kernel 内 fp32，存储 bf16**。

**【代码】** Host 侧参考（`model.py:435-487`）：
```python
:478  weights = weights_proj(x.float()) * n_heads ** -0.5
:479  weights = weights.unsqueeze(-1) * q_scale * softmax_scale
:480  index_score = fp8_index(q_fp8, weights, k_cache[:bsz,:end_pos], k_scale_cache[...])
:483  topk_indices = index_score.topk(min(index_topk, end_pos), dim=-1)[1]
```

> **【推断】`model.py:479` 是很漂亮的代数变形**：把 query 侧 fp8 反量化 scale 与 `1/sqrt(d)` 全折进 per-head 权重 `w`，于是打分 kernel 里**不需要对 q 做任何反量化** —— "把标量尽量往便宜的地方推"的经典手法。

### 4.2 Stage 2 — Top-K Selection

参考实现是 `torch.topk`（`model.py:483`）—— 正是性能灾难所在。

**【推断】** 生产中真实难度：`index_score` 形状 `[batch, seq_len]`，128K 下每行 128K 个元素；top-k 逐行，天然需 radix select 或 bitonic sort；传统 radix top-k 对 32-bit float 需 **3–4 趟 GMEM 往返**（`report.pdf` p1 §3.2 原话）；输出还要再做一次 page-table 展开（第二次全量访存）。

> **【推断】FP8 的隐性红利**：分数存 bf16（16 bit）而非 fp32，radix select 从 4 趟（32bit / 8bit per pass）降到 **2 趟** —— 竞赛方案正是吃了这个红利。

### 4.3 Stage 3 — Sparse Attention

FlashMLA 契约：`indices (batch, seq_len_q, topk)`，要求 `is_fp8_kvcache=True` 且 `causal=False`（`flash_mla_interface.py:160-163`），返回 `(out, softmax_lse)`。

**【代码】** 权威语义定义在 `tests/ref.py:70-99`：
```python
gathered_kv = blocked_k.view(-1,d_qk).index_select(0, indices.view(-1))
attn_weight = q @ gathered_kv.transpose(-1,-2) * sm_scale
# invalid_mask 处置 -inf
lse = logsumexp
output = exp(attn_weight - lse) @ gathered_kv[...,:d_v]
```
`invalid_mask = (indices == -1)` 或位置超过 `topk_length`（`ref.py:75-77`）；`attn_sink` 通过 `output *= 1/(1+exp(attn_sink - lse))` 生效（`ref.py:102-103`）。

MLA 的 MQA 形态让算术强度极低：`h_kv=1, h_q=128`；V 就是 CKV latent 本身（`d_v=512`），K 是 `[CKV; K_RoPE]` 拼接（`d_qk=576`）。**抢的是 KV cache 带宽，不是 FLOPs。**

**谁主导成本**：**稀疏 prefill** `FLOPs = 2·s_q·h_q·topk·(d_qk+d_v)`，topk 个 token 被所有 `s_q` 个 query 共享 —— DeepSeek 报 **640 TFlops @H800、1450 TFlops @B200**（`README.md:52`）。**稀疏 decode** topk 个 token 每个只有一个 query token 消费，算术强度翻不上去 → 接近 memory-bound；docs 里 410 TFLOPS 是 `batch=128,s_q=2,topk=2048` 下的"compute-bound 配置"。
**【文档声明】** `docs/20250929:50`："With a smaller topk, the relative overhead of the kernel's prologue and epilogue becomes larger compared with dense decoding" —— **topk 越小，prologue/epilogue 摊销越差**。

---

## 5. 竞赛方案 37x 加速的具体手段

### 5.1 先把"37x"的口径说清楚

**【文档声明】** 三处一致：`README.md:5` "an average speedup of 37.07x"；`report.pdf` p0 abstract "37.07x mean speedup over FlashInfer benchmarks"；`report.pdf` p3 §6.4。

**口径澄清**：

1. 37.07x 是 **flashinfer-bench 的 per-workload `speedup_factor` 均值**，**不是端到端推理加速**（"end-to-end" 一词在 README 与 report 中**从未出现**）。
2. 两个 kernel 是**两个独立 definition、两套不相交 trace**（indexer 128 条、sparse 23 条，`report.pdf` p0 §2），分两次提交。**无法确认**均值是对全部 151 条 trace 求的还是两个 track 均值的平均。
3. **★A2 分母的描述自相矛盾**：`report.pdf` p0 §1 说是 "a reference **FlashInfer** implementation"，但 `scripts/run_eval.py:87` 称之为 "the **100-iter PyTorch baseline** measurement per workload"。两处未在任何地方调和。准确说法应是"竞赛的 reference baseline（report 称 FlashInfer，`run_eval.py:87` 称 PyTorch）"。**per-workload 基线延迟不在仓库里。**
4. **★A1 "0.009 ms" 是宽松取整**：`README.md:7` 与 `report.pdf` p3 §6.4 都说两个 kernel 都是 0.009 ms，但 `report.pdf` p2 Table 2 给出 indexer "Overall 128 workloads **~7.2 us**" = "Passthrough 69 ~2.0 us" + "Score + Radix 59 ~13.2 us"。三者精确自洽：`(69×2.0 + 59×13.2)/128 = 7.1625 us`（已复算）。所以 sparse 的 0.009 ms 正确（与 Table 6 `NUM_SPLITS=16 → 9.4 us` 相符），但 **indexer 是 ~0.0072 ms**。**引用时应写 7.2 us。**
5. **【代码】测量口径细节**：`scripts/run_eval.py:44-51` 注释说官方镜像里的 flashinfer-bench 0.1.2 计时会 fallback 到 CUDA events，**包含每次调用 2–4 us 的 kernel launch 开销**，故强制 `pip install cupti-python` 以获 CUPTI 精确计时。**【推断】** 在 9 us 量级 kernel 上，2–4 us 是 **20–40% 系统误差** —— **这场比赛在"测量"上与"优化"同等重要**。
6. **★B4 top-K 输出顺序不属于评分契约**：`run_eval.py:29-34` 原文说 0.1.2 的评测器 "compares indexer output tensors element-by-element with `required_matched_ratio=1.0`, **which rejects our valid-but-reordered top-K output**"，因此他们强制从 git main 重装 flashinfer-bench。即**评测只查集合成员，不查顺序**。

### 5.2 手段一：Workload-aware 快路径（**贡献最大的单项**）

**【文档声明】** `report.pdf` p3 §6.3，标题即 "Workload-aware reasoning"：
> "The largest single improvement on the indexer came from an observation about **the workload, not the kernel**: roughly half of the contest traces have a valid token count that fits inside `K = 2048`, collapsing top-k selection to a structural mapping. The human noticed this pattern while reading the trace metadata. We then re-ran the autonomous loop multiple times to see whether it would re-discover the optimization on its own. **It did not.** The pattern repeated across runs: **the agent reasoned over the kernel's code, not over the kernel's inputs.**"

**【代码】** 落地为 `kernel()` 里的一个分支（`indexer_fused.py:305-307`）：`if max_seq_padded <= TOPK: _run_passthrough(...); return`。`_passthrough_multi_page_kernel` 只把逻辑 token 号经 page table 换成物理号，**跳过全部打分和排序**（:53-77）。效果：**69/128 个 workload 从 ~13.2 us 掉到 ~2.0 us**。

**★B6【代码】CUDA kernel 内部还有第二道 per-row passthrough**：`topk.cuh:205-214`，`if (length <= top_k)` 直接 emit `[0,length)` 加 `-1` 填充 —— 与 Triton 层那道独立，只要**真实**长度够短就触发，即使 padding 页数不够。

> **【推断】这条对性能分析的启示**：冠军方案里最大的单项收益**不是 kernel 技巧，而是对输入分布的观察**。**纯 kernel 优化能力在这里是必要不充分条件。**

### 5.3 手段二：FP8 打分 kernel 的融合链

**【代码】** `_score_kernel`（`indexer_fused.py:125-208`）在一个 kernel 里融合 5 步：(1) 页表查表 + 物理页定位（:153）；(2) `tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)`（:198）；(3) ReLU（:199）；(4) 逐头加权 `scores_tile * weight[:, None]`（:200）；(5) 跨头求和 `tl.sum(scores_tile, axis=0) * scale`（:201）。

**【文档声明】** `report.pdf` p1 §3.1 论证顺序：
> "The kernel applies the per-token FP8 scale `scale_{p,r}` **after** the cross-head sum, which is mathematically equivalent to dequantising first because `scale_{p,r}` is per-`(p,r)`, not per-element, and lets the cross-head reduction read straight from the FP8 `tl.dot` accumulator."

**【推断】** 干净的代数优化：scale 是 per-(page,row) 标量而非 per-element 向量，可提出求和号外，**省掉 64×128 次乘法**。

**padding 处理【代码】**：整页越界直接写 `-1e30` 后 return（:146-150）；页内部分越界写 `-1e30`（:198）。**★注意 `report.pdf` p0 说写 "-inf"，代码写的是 `-1e30`**（:165, :205）—— 轻微文档/代码不一致（bf16 与 fp32 同指数范围，`-1e30` 可表示，故无实际影响）。

**空间布局零拷贝 trick【代码】**（:209-233）：deep_gemm 的 KV cache 声明形状 `[P,64,1,132] int8`，真实内存是 `[每页 8192 B fp8][每页 256 B scale]`。用 `as_strided` 造两个视图：`stride_page_i8 = 64*132 = 8448 B/page`；`stride_page_f32 = 2112 f32/page`；`k_fp8` 形状 `(P,64,128)` 步长 `(8448,128,1)`；`k_scale = cache_f32_flat[:, 2048:]`。**【推断】** "不搬数据，只换视角"，避免一次完整 KV cache 重排。

### 5.4 手段三：`num_warps=2` 与 B200 的 occupancy 数学

**【代码】** `indexer_fused.py:290-291`：`num_warps=2, num_stages=2`（源内注释 :120-122 复述理由）。

**【文档声明】** `report.pdf` p2：B200 有 **148 SM**、**每 SM block 上限 32** ⇒ 最大并发 block = `32×148 = 4736`；他们最大 grid = `30×91 = 2730` ⇒ **128/128 workload 全部单波完成**，最大 shape 利用率 `2730/4736 ≈ 57.6%`。

Table 4（B200，128 workload 平均）：`num_warps=2` → 64 threads / 32 blocks/SM / No-Skip 13.2 us / **All 7.2 us**；`num_warps=4` → 128/16/18.0/9.4；`num_warps=8` → 256/8/17.9/9.3。

**★B9 作者自己的诚实限定**（`report.pdf` p2 §5）："...but **the dominant effect is occupancy, not tile shape**."

> **【推断】非常"系统级"的论证**：`num_warps=2` 之所以赢不是单 block 更快，而是让**整个 grid 塞进一波** —— 典型的 occupancy × wave quantization 权衡，不是 SIMD 效率权衡。

### 5.5 手段四：CUDA 侧两阶段 Filtered Radix Top-K（融合 page-table）

**【代码】** `topk.cuh`。算法（:123-127 注释即权威描述）："Two-stage filtered radix top-k (bf16-specialized): 1. 8-bit coarse histogram over the full row -> find threshold bin. 2. For the threshold bin only, one 8-bit refinement pass using the remaining 8 bits..."

**不是 bitonic sort，也不是 CUB** —— 全仓库零 CUB 引用，是**手写 filtered radix select**，用 `atomicAdd` on shared counters + `__syncthreads()` + `__launch_bounds__(BT)`。

**Stage 1**（:235-303）：(a) 清零 `RADIX+1 = 257` 个 slot（:236）—— **257 而非 256 很重要**，因为阈值搜索读 `s_histogram[tx+1]`（:284,:295）；(b) 用可复用 lambda `for_each_score_full` 全行遍历，向量化主体 + 标量尾巴（:243-257）；(c) `atomicAdd(&s_histogram[ToCoarseKey(raw_input)], 1)`（:258-261）；(d) `run_cumsum` 做 **8 轮 Hillis-Steele 后缀和**（非前缀和），在 `s_histogram_buf[2]` ping-pong（:266-280，+128 padding 界定 `tx+j` 的读范围）；(e) 阈值 bin 判定 `s_histogram[tx] > topk && s_histogram[tx+1] <= topk`（:284-289,:295-299）；(f) `topk -= s_histogram[threshold_bin+1]`（:303）。

**Stage 2**（:305-393）：预算为 0 时只收集严格大于的 bin（:309-319）；否则重零直方图（:322-323），`filter_and_add_to_histogram` 一趟做三件事 —— 大于阈值 bin 的直接进 `s_indices`（:331-332），等于的缓存进 `s_input_idx[0]` **并同时**用低 8 位建第二层直方图（:334-338）；`run_refine_round_last(0, FIRST_SHIFT)` 重算后缀和找子 bin 阈值，再用 `collect_with_threshold_last_round` 走 SMEM 候选、从 global 重读分数、用 `s_last_remain` 原子倒计数把精确并列项放到输出**尾部**：`const auto pos = atomicAdd(&s_last_remain, -1); if (pos > 0) { s_indices[top_k - pos] = idx; }`（:357-360）。

**Emit 阶段**（:395-402）：`dst[base] = src_page_entry[u/PAGE_SIZE]*PAGE_SIZE + (u%PAGE_SIZE)`。**【文档声明】** `report.pdf` p1 §3.2："The remap is fused inside the radix CTA's emit phase; the kernel writes `token_id` directly to the destination tensor in a single store, **with no intermediate buffer of logical indices**."

**两个 DSA 特化**（把通用代码剪成专用代码）：

**(a) SMEM 缓冲尺寸的"安全性论证"**（:226-229）："DSA invariant: `threshold_bin` size is bounded by `max_len`, and DSA's `max_len ≤ 5824 < 8192`, so the buffer never overflows — **no overflow-fallback code path here**." **【代码】** `topk_binding.cu:57-60` 进一步压到 6K：`kSmisAll = 6*1024`，注释 "DSA invariant `max_len ≤ 5824 < 6144` ... Reduces dynamic SMEM per block by 16 KB → potentially higher occupancy."
**【推断】** 用已知输入上界换代码简化 + occupancy，**代价是通用性：换 workload 就静默出错（连 overflow 检查都没有）**。

**(b) 按 `max_len` 切换 threads-per-block**（`topk_binding.cu:52-80`）：`kSmallBT=512`（`max_len<=4096`）、`kBigBT=1024`（否则）。**【文档声明】** `report.pdf` p1 §3.2："We also dispatch threads-per-block at runtime: BT=512 for `max_len ≤ 4096` (more CTAs per SM), BT=1024 above (more compute per CTA)."

**【代码】来源**（`topk.cuh:17-28`）：明确标注**改编自 FlashInfer 的 FilteredTopK**，三处裁剪（DType=nv_bfloat16；output mode=PageTable；index type=int32_t）+ 一处扩展（templated PAGE_SIZE 融合 page-table lookup）。
**【推断】重要的"诚实度"信号**：**冠军方案的 top-k 不是原创**，是把 FlashInfer 的 kernel 剪到刚好够用。`report.pdf` p3 §6.4 也承认："We kept the autonomous loop's Triton sparse attention and FP8 scoring, while replacing Top-K with the **collaboration phase's CUDA FilteredTopK**."

**【代码】VEC_SIZE 选择**（:406-421）：`MAX_VEC = 16/sizeof(DType) = 8`（bf16）；`ComputeFilteredTopKVecSize` 返回 `gcd(max_len, MAX_VEC)`。因为 scratch 行数是 `next_pow2`（`indexer_fused.py:211-212`），gcd 恒为 8，**实际走的是 8-wide 16 字节路径**。

**【代码】CUDA↔Triton 边界**：`topk_ext.py:16-52` 在 import 时用 `torch.utils.cpp_extension.load(...)` JIT 编译，构建目录 `$DSA_TOPK_BUILD_DIR` 默认 `/tmp/topk_ext_build`（:26），模块缓存在 `_MODULE`（:12,:16-19）。pybind 名 `top_k_page_table_transform`（`topk_binding.cu:87-90`），DPS/out-param 调用。host launcher 对每个实例 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)`（`topk.cuh:447-448`），因 48 KB 超过默认上限。

### 5.6 手段五：Sparse Attention 的 split+combine 单次发射 + atomic barrier

**【代码】** `sparse_fused.py`。**注意：竞赛的 sparse attention 也是 MLA 形态**（`D_ckv=512, D_kpe=64`，:8），但 `H=16`。

**核心创新：把 split 与 combine 塞进同一个 grid launch，用 atomic counter 当 barrier**（:121-127）：
```python
tl.atomic_add(Counter_ptr + t, 1, sem="release")
count = tl.load(Counter_ptr + t, volatile=True)
while count < target_count:
    count = tl.load(Counter_ptr + t, volatile=True)
tl.debug_barrier()
```
`target_count` 由 host 侧**单调递增 generation** 维护：`target_count = gen * NUM_SPLITS`（:277-290），counter buffer 按 `(device, num_tokens)` 缓存复用（:282-290）。
**【推断】** 为避免每次调用 `cudaMemset` —— 很实际的 launch-overhead 优化。
**★注意**：该 int32 计数器每次调用 `+16`，累计约 `2^31` 次 split 后溢出；且自旋用 volatile load 但**无 acquire 语义、无 backoff**（形式上的缺失 acquire fence，实践中可用）。

**split 的 stride 切分（不是连续切分）**（:78, :84）：`offs_split = s + tl.arange(0, SPLIT_SIZE) * NUM_SPLITS`。
**【文档声明】** `report.pdf` p2 §4.1.2："The stride pattern (instead of a contiguous `[jK/J, (j+1)K/J)` slice) spreads the prefix-heavy valid positions evenly across split. **Contiguous splits would dump all valid keys onto split 0** on workloads where the index list is short."
**【推断】** 因为 indices 用 `-1` 填充到固定 2048，短序列有效项**全部集中在前面**，连续切分会让 split 0 干完所有活、其余空转。

**提前退出【代码】**（:80-81）：`num_valid = tl.sum((idx_scan >= 0).to(tl.int32), axis=0)`；`max_bn = ((num_valid + BLOCK_N - 1)//BLOCK_N)*BLOCK_N`；`for bn in range(0, max_bn, BLOCK_N)`。

**★A3 split 循环在出厂配置下恰好只跑一次**（比"提前退出"更强）：`NUM_SPLITS=16`（:314）、`SPLIT_SIZE = 2048/16 = 128`（:50）、`BLOCK_N=128`（:376）。因 `num_valid <= SPLIT_SIZE = 128`，`max_bn` 只能是 0 或 128 ⇒ 循环**至多执行一次**。**2048 个 key 是靠 split 并行度覆盖的，不是靠循环深度。**
这也精确解释了 `NUM_SPLITS=32` 的崩溃：`SPLIT_SIZE=64 < BLOCK_N=128`，`offs_n` 可达 127，gather 索引 `s + 127*32` 达到 **4095 > 2047**（已独立复算该算术）→ 读到下一个 token 的行 / 最后一个 token 越界。这就是 `report.pdf` p3 Table 6 脚注 "† `BLOCK_N=128 > SPLIT_SIZE=2048/32=64`; out-of-range gather."

**在线 softmax 与 log2 域【代码】**（:96-106）：
```python
logits = logits * sm_scale_log2e
m_new = tl.maximum(m_i, tl.max(logits, axis=1))
m_new_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
alpha = tl.where(m_i == NEG_INF, 0.0, tl.exp2(m_i - m_new_safe))
p = tl.exp2(logits - m_new_safe[:, None])
acc = tl.dot(p.to(tl.bfloat16), kc, acc=acc)
```
**【文档声明】** `report.pdf` p2："The kernel actually runs this in log2 space (`e^{m_j-m}` implemented as `exp2(m_j - m)`, with logits pre-scaled by `√d^{-1} · log2 e` so the softmax base change cancels): mathematically identical to the equation above, faster on hardware."
`m_new_safe` 的三元表达式是容易被 agent 漏掉的数值细节（`-inf - (-inf) = NaN`）。

**两个 `tl.dot` 拆分 q_nope / q_pe**（:94-95）。**【文档声明】**："the inner product is split as `<q^nope, ckv> + <q^pe, k^pe>` and accumulated by two back-to-back `tl.dot` calls **so the CKV and PE blocks are paged into registers separately**."

**Combine 阶段复用 `program_id(1)` 当 D 切片号**（:130-131, :312-314）：`d = s`；`NUM_SPLITS=16`；`BLOCK_D = D_ckv // NUM_SPLITS = 512/16 = 32`。覆盖校验 `16×32 = 512 = D_ckv` ✓。
**★【推断】** 16 个 program 各自**冗余重算**完整的 16 路 m/l 合并，只写自己那 32 列 —— 用冗余换掉第二次 kernel launch。combine 用 `tl.static_range(NUM_SPLITS)`（:137）完全展开。

**partial accumulator 用 bf16 而非 fp32**（:301-306）：源内注释 "bf16 `partial_acc` halves GMEM/L2 roundtrip for the partial→combine handoff. Accuracy impact is ~0 on our 23 workloads (partials are pre-divided by the running max, so values are bounded in `[0, exp2(m_split - m_global)]`)." **【文档声明】** `report.pdf` p2 §4.2："Using bfloat16 for intermediate weighted sums reduces register pressure and L2 bandwidth vs. float32."
**★注意**：**寄存器内**的 `acc`/`acc_comb` 仍是 fp32（:76,:135,:225），只有 GMEM 交接是 bf16。

**T<=2 的 D-parallel 特化 kernel**（:177-274, :332-350）：完全不走 split。**【代码】** 它只为 P·V dot 载入**自己那一段 `kc_slice` 列**（:255-257），而 split kernel 必须 gather 全 512 列（:89）—— 两条路径的真实带宽差异。**★另注意** `_small_t_kernel` 的 Q 载入**没有** eviction hints（:219-220），而 split kernel 有 `evict_last`（:71-72）—— 两条路径的不一致。其 `BLOCK_N=64`（:347）与 split kernel 的 128 不同，故最多 32 次迭代。

**实测配置【文档声明】** `report.pdf` p3 Table 6：`num_warps ∈ {2:15.0, **4:9.4**, 8:10.4, 16:13.1}` us overall；`NUM_SPLITS ∈ {4:14.7, 8:10.9, **16:9.4**, 32: runtime error}`。
**【推断】** `NUM_SPLITS=32` 的失败暴露脆弱性：`BLOCK_N=128` 硬编码（:376），与 `NUM_SPLITS` 强耦合，**没有任何 `static_assert` 保护**。**修法就是一行 `tl.static_assert(BLOCK_N <= SPLIT_SIZE)` 或 `BLOCK_N := min(BLOCK_N, SPLIT_SIZE)`。**

### 5.7 手段六：Roofline 视角的自我诊断

**【文档声明】** `report.pdf` p3 Figure 1 是作者用 Nsight Compute 计数器画的 B200 roofline（HBM peak 8 TB/s, BF16 TC peak 2250 TFLOP/s, FP8 TC peak 4500 TFLOP/s）。Table 5 分类占用率：

| 类别 | Active SM | L1 | L2 | DRAM | Compute |
|---|---|---|---|---|---|
| Sparse easy (T<3) | **1–2/148** | 42% | 0.9% | 0.3% | 2.2% |
| Sparse hard (T>=3) | **6–8/148** | 38% | 3.8% | 3.0% | 9.2% |
| Indexer easy (passthrough) | 1–30/148 | 4.4% | 0.5% | 0.0% | 0.3% |
| Indexer hard (score+radix) | 5–86/148 | 9.9% | 0.8% | 0.9% | 1.4% |

**★B9 `report.pdf` p2 §5 作者自述（verbatim）**：
> "Three patterns drive the under-utilisation. **Sparse attention is L1-bound. Indexer hard is dominated by kernel launch and the radix top-K kernel that runs after the score phase: one CTA per batch row, serial within each CTA. Indexer easy is trivially small, the gap is from the kernel launch.**"

另有 "**Our choices for those parameters are mostly empirical.**"

> **【推断】这是整份报告信息量最大的一页**：所有 kernel 的 compute 占用率都低于 10%，DRAM 低于 4%，**Active SMs 只有 1–86/148**。说明当前所有"优化"都还在 **latency-bound / launch-overhead-bound / 并行度不足** 区域，**离 roofline 边界极远**。37x 主要来自"去掉 reference 实现里的愚蠢开销"，而不是"把 Tensor Core 榨干"。**真正的技术前沿不是"再快 37x"，而是把 active SM 从 8 提到 148。**

---

## 6. 技术前沿与缺口

> **诚实声明**：**两个仓库都没有 "Future Work" 章节**。FlashMLA `README.md` 只有 "News"；`report.pdf` 第 4 页从 "Acknowledgments" 直接跳到 "References"，中间无任何未来工作段落。故本节主要是【代码】层可验证缺口 +【推断】。

### 6.1 FlashMLA 的硬性功能缺口（均有 `TORCH_CHECK` 或 `static_assert` 佐证）

| 缺口 | 证据 | 影响 |
|---|---|---|
| **SM100 没有 dense decode** | `csrc/api/dense_decode.cpp:25-26`：`if (!arch.is_sm90a()) TORCH_CHECK(false, "Dense decode MLA is only supported on SM90a architecture")`；`README.md:75` 支持矩阵 "Dense Decoding = SM90" | B200 上跑 V3/V3.1 dense MLA 解码只能退回其他实现 |
| **SM90 没有 dense prefill** | `README.md:77` 支持矩阵 "Dense Prefill = SM100" | Hopper 训推一体无法用同一套 kernel |
| **稀疏解码只支持 MQA** | `sparse_decode.cpp:274`：`TORCH_CHECK(h_kv == 1, ...)` | 真 MHA/GQA 稀疏注意力无实现 |
| **FP4 只能当 `extra_kv`** | `sparse_decode.cpp:338`：`TORCH_CHECK(model_type != ModelType::V41_FP4, "The fp4 KV cache is only supported as extra_kv")`；`kv_cache_format.h:45-48` 白名单只有 `(V41, V41_FP4)` | **全 FP4 KV cache 未实现** |
| **稀疏解码 `h_q` 只有 64/128** | `sparse_decode.cpp:359-365` | 其他 head 数需重新实例化 |
| **★V3.2 的 h=128 sparse decode 在 SM100 上靠"两次 h=64"** | `sparse_decode.cpp:126-168`，类名 `Decode_Sm100_Head64x2_Impl`，注释 "An implementation that calls the head64 kernel twice to process head128 / Necessary for running V3.2 shape (i.e. h = 128, d_qk = 576) on SM100f"；`csrc/kernels/sm100/decode/sparse/head128/README.md` 全文仅一句："Head128 decoding kernels are located at `.../phase1_decode_k512.cu` (for k_dim = 512) or **simulated using 2x head64 kernel** (for k_dim = 576)" | **一个真正的 h=128/`d_qk=576` SM100 kernel 不存在** —— 最有价值的"待补"缺口之一 |
| **稀疏 decode 只支持 `d_qk ∈ {512,576}`、`d_v = 512`** | `sparse_decode.cpp:275-276` | 维度泛化未做 |
| **fused norm-RoPE 不支持 split-KV** | `core_attn/config.h:66-67` 头注释："**Split-KV is not supported (the fused RoPE + FP8-quant epilogue cannot be combined across splits)**" | h=128 长 topk 下无法用 split-KV 提并行度 |
| **fused kernel decode 模式 batch 必须为 1** | `core_attn/config.h:63`："Batch size must be 1; the grid covers the s_q query tokens" | 只能 batch=1 推理 |
| **SM100 dense prefill 只支持 bf16** | `fmha_cutlass_fwd_sm100.cu:46-47` | 无 FP16/FP8 dense prefill |
| **SM100 backward 不支持 GQA** | `flash_mla_interface.py:291-293`：`# TODO: fix bwd GQA` + `raise ValueError("SM100 bwd doesn't support GQA now...")` | 训练场景受限 |
| **架构只覆盖 `sm90a/sm100a/sm103a`** | `setup.py:42-48` | **没有 `sm_120`（消费级 Blackwell / RTX 50）、`sm_110`、`sm_121`** |
| **稀疏 prefill 没有 batch 维** | `README.md:165` | 多 batch 需手动 reshape + 改 indices |
| **dense decode 的 `page_block_size` 硬编码 64** | `dense_decode.cpp:65` | 与 vLLM/SGLang 可配 block size 冲突 |
| **`num_splits` bucket 上限 256** | `combine.cu:165-190` 的 `MLA_NUM_SPLITS_SWITCH` 展开到 `<=256` | 百万级序列长度下 split 数可能不够 |
| **`fixed_overhead_num_blocks = 5`、`block_size_topk = 64` 硬编码** | `sparse_decode.cpp:106-110, :145-149, :196-200` | 调度启发式写死 |

### 6.2 FlashMLA 的工程/流程缺口

- **无 CI**（无 `.github/`）
- **无 release/tag**（版本号靠 `git rev-parse --short HEAD` 拼，`setup.py:175-186`）
- **benchmark 无回归基线**（`bench_flash_mla.py:507` 写 CSV 但无自动比对）
- **22 个 TODO/FIXME/XXX**，其中多处是**真实正确性约束**（如 `core_attn/kernel.cuh:345` 的 `static_assert` 在 `FOLD_FACTOR=4 && MODEL_TYPE==V4` 时不成立）
- **cutlass submodule 未初始化**
- **编译时间是隐性成本**（58 `.cu` × 3 arch）

**★"被遗弃的泛化"层**（多轮尽调累积）：

| 项 | 证据 |
|---|---|
| **dense API 静默给出错答** | `fmha_cutlass_fwd_sm100.cu:74-77` 与 `bwd .cu:75-77`：不支持的 head-dim 只 `std::cout` 一行然后**正常返回、输出张量未被触碰**；只有 `(192,128)` 和 `(128,128)` 被实例化 |
| **`MaskMode::kNone/kCustom` 无法选到** | `NoMask` 类**已在** `collective/fmha_fusion.hpp:41` 实现且有 `NeedMask` 编译期分支（`sm100_fmha_mla_fwd_mainloop...:759`），但 API 只传 `CausalMask<false>` 或 `ResidualMask`（`fmha_cutlass_fwd_sm100.cu:50-61`）→ **`kNone` 静默退化成 `ResidualMask`** |
| **`FLASH_MLA_ASSERT` 杀进程** | `common/helper.h:64-70` 用 `std::abort()` 而非异常 → 参数传错会**杀掉 Python 解释器** |
| **两个 MMA atom 是死代码** | `SM100_MMA_F16BF16_{TS,SS}_NOELECT`（`gemm.cuh:493, :596`）在 `gemm.cuh` 之外**零引用** |
| **一个 layout factory 是死代码** | `make_umma_canonical_layout`（`helpers.cuh:129`）只有定义无调用者 |
| **11 个 option tag 只有 1 个被消费** | 只有 `Tag::kIsPersistent` 被使用（`fmha_cutlass_fwd_sm100.cuh:80`、`.cu:23-24`） |
| **fp8 trait 别名零使用** | `common/utils.hpp:33` 的 `cutlass_dtype_t` |
| **`Pod2` 无用户** | `common/pow_2.hpp`（92 行）在 dense tree 无调用者 |
| **`gather_tensor.hpp` 是 demo** | 215 行，在 `namespace example`，未接入任何 kernel |
| **四个 `cta_group::1` atom 无 arch guard** | `gemm.cuh:80-111, :182-212, :493-527, :596-628` 的 fma body 无 `CUTE_ARCH_TCGEN05` 守卫（只有两个 `cta_group::2` 有，`:301/:409`）→ 非 SM100 编译时会在 **PTX 汇编期**失败而非干净 guard |

**【推断】** 这是快速迭代的 kernel fork 的正常现象，但**恰好是新贡献者可以高置信度、低风险清理的表面**。

### 6.3 竞赛方案的缺口（逐条可证）

| 缺口 | 证据 |
|---|---|
| **`NUM_SPLITS=32` 直接崩** | `report.pdf` p3 Table 6 脚注 †；根因 `sparse_fused.py:376` 硬编码 `BLOCK_N=128`，与 `NUM_SPLITS` 无 `static_assert` |
| **top-k 完全没有 overflow 兜底** | `topk.cuh:226-229 / :325-327 / :390-391` 三处注释均写 "no overflow-fallback code path here" / "no overflow check here"。**一旦 `max_len > 6144` 就静默越界** |
| **★注释谎报安全网（强证据：仓库欠维护）** | `topk.cuh:126-127` 仍声称 "(threshold-bin candidates buffered in SMEM, **with a correctness-first rescan fallback on SMEM-buffer overflow**)"，但 **fallback 已被删除**：`:325-327` 与 `:390-391` 均说无检查，且 `:367` 是**静默 clamp**（`num_input = min(raw_num_input, SMEM_INPUT_SIZE)`）—— 溢出时会**静默丢弃候选**而非 rescan |
| **★其他过期注释** | `topk.cuh:153-155` 说 "Per-buffer SMEM set to **8K**" 但 launcher 实际传 **6K**（`topk_binding.cu:60`）；`topk_binding.cu:52-54` 也仍写 "512/8K ... 1024/8K"；`FILTERED_TOPK_SMEM_DYNAMIC`（`topk.cuh:159`）由默认 8192 算得 65536 B，而 launcher 独立重算 49152 B（:433）—— 未被使用故无害，但是给后来者的陷阱 |
| **`FILTERED_TOPK_MAX_K = 2048` 无运行时保护** | `topk.cuh:156` 用它定 `s_indices` 大小与 emit 上界（:397），**没有 `top_k <= 2048` 的检查** |
| **子 bin 阈值搜索缺"无满足 bin"保护** | `topk.cuh:284` 的谓词不满足时会沿用**陈旧的 `s_threshold_bin_id`**（仅并列边缘情形）。继承自上游 FlashInfer |
| **sparse attention 只支持 H=16** | `sparse_fused.py:8`。**与生产 V3.2 的 `h_q=128` 差 8 倍** |
| **top-k 输出不保序** | `run_eval.py:29-34` |
| **完全不使用 TMA / tensor descriptor** | 两个 Triton kernel 只用 `tl.make_block_ptr` + `tl.load`（`indexer_fused.py:173-191`），无 `tl.make_tensor_descriptor`。是否降级为 `cp.async` 属编译器决定，**无法确认** |
| **没有 warp specialization** | 两个 Triton kernel 都是"所有 warp 做同样的事"。`report.pdf` p3 §6.1 承认 warp specialization 是**人类提出但未进入最终 kernel** 的方向 |
| **没有跨层 index 复用 / 层次化 indexer** | 全仓库 grep 无痕迹；`DeepSeek_V3_2.pdf` 也只描述单层 indexer |
| **没有低精度 KV cache 的反量化融合** | sparse attention 直接读 fp8 视图并用 `tl.dot` 消费（`sparse_fused.py:327-328`），**没有 kernel 内反量化到 bf16** —— 对比 FlashMLA 的 `KVBlockDequantizer` 是一整套精心机制 |
| **假设 KV cache 页连续** | `sparse_fused.py:327-328`：`ckv_flat = ckv_cache.view(num_pages * page_size, D_ckv)` —— **要求所有物理页显存连续**，因此 attention kernel 内**不做 page-table 查表**（查表被推到 top-k 的 emit 阶段）。**【推断】** 为省 gather 而假设的强条件 |
| **没有 FP4 支持** | 全无 |
| **没有数值容差文档** | `report.pdf` 未报任何精度指标 |
| **★第二块 24 KB 动态 SMEM 分配但未使用** | 48 KB 是双缓冲（`topk.cuh:230, :433`），但 bf16 路径只写 index 0（:335），且 `run_refine_round_last` 以 `r_idx=0` 调用（:392），`NUM_REFINE_ROUNDS==1` |
| **★stride 当 size 用的潜在 bug** | `indexer_fused.py:81` 的 store mask 写成 `mask=offsets < stride_out` —— **把 stride 当 size**；仅因输出连续（`stride(0)==2048`）才无害 |
| **★LSE 单位是 log2** | `sparse_fused.py:172, :272`：`lse = m + tl.log2(l)`，即 `以 2 为底的 log-sum-exp` 而非 `ln(LSE)`。评测器是否要求自然对数**无法确认** |
| **★harness 缺陷** | `run_eval.py:265-266` `if mode == "probe": probe.remote()` —— **全文未定义 `probe` 函数**，`mode="probe"` 直接 `NameError`；`run_eval.py:263` 文档化了 `DSA_SPARSE_K_SPLITS_CAP` / `DSA_SPARSE_BLOCK_K` 两个环境变量但**方案里无人读取**；`README.md:21-22` 让用户创建 Modal volume `flashinfer-trace`，而 `run_eval.py:19` 实际用的是 `mlsys26-contest` |
| **★打包会带上全部文件** | `pack_solution.py:46-74` 打包**整个 `source_dir`**，故 indexer 提交必然携带 `sparse_fused.py` / `topk_ext.py` / `topk.cuh` / `topk_binding.cu` |

**★竞赛容差（已确定）**：`scripts/run_eval.py:129-134` —— `warmup_runs=10, iterations=50, num_trials=3, rtol/atol=0.01, timeout_seconds=300`，"matches official eval"。只有权威的 `EVALUATION.md` 文本缺失。

### 6.4 我认为真正值得投入的四个方向（全为【推断】，但每条锚定上述可验证缺口）

1. **SM100 上真 `h=128 / d_qk=576` 稀疏解码 kernel**。目前靠 `Decode_Sm100_Head64x2_Impl` 打两次，意味着 Q 被读两次、调度元数据被消费两次、L2 复用被切断。这是 FlashMLA 里**唯一一处"用工程 workaround 代替内核实现"**的地方，也是最容易做出可量化收益的入口。

2. **把 indexer + top-k + sparse attention 合成真正的流水线**。FlashMLA 完全不碰 indexer，竞赛方案碰了但只有 H=16。**中间地带的"生产级 indexer"是当前无人认领的生态位**：需要 FP8 + paged cache + 128 head + var-len 的真实形态，且必须与 FlashMLA 的 indices 契约（物理 token id、`-1` 填充）严格对齐。

3. **并行度而不是微内核**。`report.pdf` p3 Table 5 显示 sparse-hard 只有 6–8/148 SM active、compute 9.2%。**【推断】** 方向是 split-KV 更激进 + combine 融合，或把 `NUM_SPLITS` 与 `BLOCK_N` 解耦后扫到 64/128。当前 `NUM_SPLITS=32` 就崩，说明这条路根本没走完。

4. **SM120（消费级 Blackwell）与 FP4 主 cache**。`setup.py:42-48` 只编 `sm_90a/sm_100a/sm_103a`；FP4 被限制在 `extra_kv`（`sparse_decode.cpp:338`）。
   **★重要限定（四重验证）**：**全仓库只有 bf16 MMA** —— `grep -rn "kind::" csrc/` 恰好 6 处命中**全是 `kind::f16`**；`kerutils/device/sm100/` 里 fp8/fp4 dtype token **零命中**；无 `kind::f8f6f4`；fp8 trait 别名零使用。**FP8/FP4 只是存储+反量化格式，从来不是 MMA 操作数类型。**
   → **加 FP4 存储格式是反量化 kernel 问题（中等门槛）；加 FP4 MMA 需要新的 `kind::f8f6f4` atom，完全是另一个量级。**

---
