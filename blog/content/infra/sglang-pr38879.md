---
title: "SGLang PR #38879：性能优化与工程边界"
date: 2026-09-15
lastmod: 2026-09-21
draft: false
visibility: public
categories: ["推理引擎"]
tags: ["SGLang","源码解读"]
description: "SGLang PR #38879：性能优化与工程边界；保留技术细节、出处与适用边界。"
---

> **硬件边界**：本文涉及 Hopper 与 Blackwell 的不同实现。H100/H20 可用于适配的 Hopper 路径；SM100/SM103 专属实验需单独申请 B200/B300/GB300 等对应设备。没有对应设备时，只能阅读代码或进行不依赖设备的逻辑验证，不能声称复现文中性能。所有跑分沿用原文标注的硬件前提。

> **标题**：`[DeepSeek-V4.1] Optimize DSpark verify and MoE kernels on Blackwell`
> **作者**：BBuf（Xiaoyu Zhang）｜**Co-author**：@DarkSharpness（Ziyi Xu）
> **目标分支**：`dsv4.1` ← `BBuf:bbuf/dsv41-dspark-kernel-stack`
> **规模**：16 文件，**+1524 / −136**，3 commits
> **时间**：2026-09-10 12:19 创建 → **13:58 合并**（约 100 分钟）｜merge commit `c36636b7da60`

---

## 0. 为什么这个 PR 值得逐行读

它是**一份真实的生产级 V4.1 部署优化**，而且**几乎命中我们讨论过的每一条主线**：

| 我们的主线 | 本 PR 的对应 |
|---|---|
| DSpark 投机解码 | **target verify + 三个 draft stage** 全部优化 |
| MoE | router 打包 / 反量化 / finalize 融合 |
| 通信（EP / all-reduce） | 自研 **push all-reduce** 替换 FlashInfer oneshot |
| mHC（Single-Pass mHC） | **mHC 统计重叠 0.18% → 93.50%** |
| CSA2 分层索引器 | `candidate_blocks.py` = **block 级 max + 可见性掩码** |
| kernel launch 开销 | **target graph kernel 数 2209 → 1965** |
| KV cache | `--max-total-tokens 33554432`（32M token 池） |

**最值得学的不是任何一个优化，而是它的工程规范**（见 §5）。

---

## 1. 基本信息与硬件前提

### 1.1 测试环境【原文】

| 项 | 值 |
|---|---|
| GPU | **4× B300 SXM6 AC**（Blackwell Ultra） |
| 并行 | **TP4 / EP4**（单节点） |
| PyTorch | 2.13.0+cu130 |
| FlashInfer | 0.6.18 |
| Triton | 3.7.1 |
| sglang-kernel | 0.4.6.post1 |
| CUTLASS DSL | 4.6.2 |
| nvcc / driver | 13.0.88 / 580.126.20 |
| 模型 | DeepSeek-V4.1，**DSpark block size 5，真实接受率** |

**⚠️ 重要**：这是 **B300（SM100/Blackwell）**，**不是你能直接租的 H100/H20**（按 `AGENTS.md` §5.1，其他型号需单独申请）。所以：

> **这个 PR 的优化大概率不能直接搬到 H100/H20 上**，原因见 §4.3。**但它的方法论完全可迁移。**

### 1.2 基线选择【原文】

- Base：公开 `dsv4.1` @ `7bdebdab7db4befb71c64ae0d6f0eb37fe7d8402`
- Candidate：`1b742acd2a49ebd7acd017032875552099d12391`（= 本 PR 第 3 个 commit）

---

## 2. 性能结果

### 2.1 吞吐【原文】

| Workload | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| BS1，输入 4096 / 输出 1024，post-first-event tok/s | 569.88 | **764.29** | **+34.1%** |
| **BS64，输入 4096 / 输出 2048，stable decode tok/s** | 6,555.95 | **13,472.51** | **+105.5%** |
| BS64，同 workload，full batch tok/s（含 prefill） | 4,114.78 | 6,434.55 | +56.4% |

**接受长度几乎没变**（BS1: 5.658→5.818；BS64: 5.407→5.479）—— **说明加速不是靠"猜得更准"，是纯粹的执行效率**。

### 2.2 GPU trace（BS1，4096 输入，各 20 次 graph replay）【原文】

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| **Target verify graph 中位数** | 9.049 ms | **6.927 ms** | **−23.4%** |
| Draft graph 中位数 | 0.833 ms | 0.719 ms | −13.7% |
| Target-start → next target-start | 10.050 ms | 7.810 ms | −22.3% |
| **Target kernels per graph** | **2209** | **1965** | **−244** |
| Draft kernels per graph | 196 | 181 | −15 |

### 2.3 ⭐ Kernel launch 数变化（这张表是全文最有解释力的）

完整 20 周期 trace 的 launch 计数：

| Kernel 路径 | Baseline | Candidate |
|---|---:|---:|
| 单独的 router padding mask | 860 | **0** |
| 单独的 router ID packing | 860 | **0** |
| 独立的 MoE finalize | 860 | **0** |
| **融合的 MoE finalize + shared add + all-reduce** | 0 | **860** |
| FlashInfer oneshot all-reduce | 1760 | **0** |
| **自研 push all-reduce** | 40 | **940** |

**读法**：`860` 次/20 周期 = **43 次/周期**。也就是**每个 decode step 少掉约 86 次 launch**（padding mask + ID packing + 独立 finalize 各 43），换成 43 次融合 launch。

**原笔记将其与竞赛测量作如下比较**：竞赛报告说 "Indexer hard is dominated by kernel launch"、所有 kernel 的 compute 占用率 < 10% —— **在 V4.1 的生产服务里，同样的规律成立**。

### 2.4 ⭐ mHC 统计的重叠率

> **mHC statistics overlap ... increases from 0.18% to 93.50% of its kernel duration.**

**作者自己加了免责声明**（这点很值得学）：
> "This measures time overlap in the trace; it is **not an occupancy metric** or a guarantee that the overlapped work has zero cost."

**【推断】** 从 0.18% 到 93.5% 意味着**原来 mHC 统计几乎完全串行在关键路径上**。这是一个"把已有工作挪到别的 CUDA stream 上"的优化 —— 不算新算法，纯调度。

---

## 3. 八处改动逐条拆解

### 3.1 ⭐ 自研 push all-reduce（最大的单项，+465 行 CUDA）

**文件**：`kernels/jit/csrc/distributed/all_reduce_fusion.cuh`（465 行，全新）+ `kernels/ops/communication/all_reduce_fusion.py`（243 行）

**头部注释直接说明了它的血统**【码】：

> "A **generalisation of the K3 `finalize_push_norm` kernel** (`csrc/kimi_k3/comm/ar_fusion.cuh`)"

**即：这个 kernel 最初是为 Kimi K3 写的**，本 PR 把它泛化成模板化的、可复用的实现。这是一条**跨模型的 infra 复用**证据。

**融合了三件事**【码】：

```
stage 1（只用寄存器）：
  local[t] = Σ_k expert_weights[t,k] · gemm2_out[idx[t*top_k + k]]
             (+ shared_output[t])
stage 2（跨 rank）：
  out[t]   = Σ_ranks local[t]
stage 3（可选 kNorm）：
  out[t]   = out[t] · rsqrt(mean(out[t]²) + eps) · w
```

**三个我认为值得学的设计点**：

**① 关键中间结果从不落到 global memory**
> "The **rank-local finalize never materializes in global memory**: each thread computes one 16B vector of it and pushes it straight into every peer's push slot with unicast `st.relaxed.sys` stores"

② **push-plane 协议的巧思**（我认为最漂亮）：
> "**`+0.0` payload words are remapped to `-0.0`** (numerically identical) so **a written word is never 0** and `word == 0` means **'not arrived yet'**"

**用一个数值上等价的变换（+0.0 → −0.0）换来一个免标志位的就绪协议。** 消费者轮询自己的 slot，"还剩 +0.0 标记"就意味着没到齐。

③ **phase counter 的兼容性设计**：
> "the phase counters are per block of the GENERIC push kernel, which launches `num_blocks` blocks and flips one counter each. This kernel uses **one counter per row cluster** (flipped by the cluster's leader block after a cluster barrier... ) and **a trailing 'bumper' cluster flips every remaining one**, so the whole array keeps one parity and the **two kernel families can share the plane freely**"

**【推断】** 这是为了"新老 kernel 能共存于同一个 push plane"而做的兼容层 —— 一个很实际的工程约束（你不可能一次替换掉所有调用点）。

**④ 数值等价性被显式承诺**【码】：
> "Accumulation is fp32; **the bf16 rounding points are exactly the unfused path's**: the routed combine (what TRT-LLM's finalize returns), the `+ shared` (torch's bf16 add) and the all-reduce output ... and **the staged vector is bit-identical to what the unfused path reduces**"

**并且给出了等价的目标链**：`TRT-LLM finalize` → `shared.add_(routed)` → `fp32 累加的 bf16 all-reduce（按 rank 顺序）`。

**约束**【码】：
- 每个 rank 必须用相同的 `num_tokens / hidden / top_k / epilogue` 调用
- 单流序列化（`single-stream calls are serialized`）

### 3.2 ⭐ 分层索引器的 block 级打分（`candidate_blocks.py`，+115 行）

**新文件**，两个 Triton kernel。这**正是 V4.1 §2.3.2 Hierarchical Sparse Indexer 的 block 级选择**：

```python
scores = tl.reduce(values, axis=1, combine_fn=_maximum_with_nan)     # block 内取 max
scores = tl.where((length > 0) & (blocks == (length - 1) // GROUP),
                  float("inf"), scores)                              # 最后一块强制可见
```

**三个细节值得注意**：

**① 显式的 NaN 语义**：`tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)` —— 不是默认行为，是刻意选的。

**② 最后一块给 `+inf`**：保证"当前正在写的那个不完整块"永远可见。**【推断】** 这是正确性要求，不是优化 —— 否则一个序列的最后几个 token 可能看不到自己。

**③ 融合了"可见性掩码"**：原来是独立的 mask kernel（对应 §2.3 表里"fuse index candidate masking"），现在融进同一个 kernel。

**调用点**（`deepseek_v4_backend.py`，+60 行）：加了**一长串守卫条件**才走新路径 —— `is_cuda` / `ndim==2` / `stride(1)==1` / `seq_lens` 设备与 dtype 与连续性 / `shape` / `numel>0` / `0 < block_size <= 1024`，**不满足就回退老路径**。

**【推断】** 这种"宽守卫 + 保守回退"是生产 PR 的典型写法，值得学：**新 kernel 只在完全确认的形态下启用**。

### 3.3 MoE router 的打包与反量化（`mxfp4_flashinfer_trtllm_moe.py`，+168/−38）

**关键是引入了 `Mxfp8RoutedInputPreQuant` 这个 `NamedTuple`**【码】：

> "MXFP8 linear-layout quant of the routed MoE input, **produced ahead of `apply()`** ... **e.g. on a side stream while the gate GEMM and the router run on the main stream**. `ready` is the event recorded on the producing stream after the quant; `apply` makes its stream wait on it right before the routed MoE op"

**即：把输入量化提前到旁路 stream 上，与 gate GEMM 和 router 重叠。** 用 CUDA event 做 stream 同步。

**配套的 guard**（也是"延迟到最后一刻才同步"的写法）：
> "apply makes its stream wait on it **right before the routed MoE op**, whose first kernel (routing) precedes the GEMM that reads `x_q`/`x_sf`"

**【推断】** 注意这里有个微妙之处：**等 event 的位置被刻意往后推**（推到真正要读 `x_q` 之前的那个 kernel），这样中间的 kernel 还能先跑。这是**把同步点尽量后移**的经典手法。

### 3.4 mHC 的 `num_stages` 平台特化（`mhc.py`，+11/−2）

```python
def _num_stages_for(m, k):
    # GB300 verify batches benefit from a smaller shared-memory footprint.
    # This changes memory scheduling only; K tiles and reduction order stay fixed.
    if get_platform().is_blackwell and k == 20480 and 64 <= m <= 384:
        return 1
    return _HC_MIX_NUM_STAGES
```

**⭐ 这条注释是我在这个 PR 里最欣赏的一处**：

> "This changes **memory scheduling only**; **K tiles and reduction order stay fixed**."

**它明确声明了"这个改动不影响数值结果"**，理由是 **K 方向的 tile 数和归约顺序都没变**。

**【推断】** 这是一个**正确性论证**，不是性能说明。改 `num_stages` 会改 shared memory 的流水线深度，但**不会改浮点累加顺序** —— 所以结果逐位一致。**能说清这一点，说明作者真的懂数值语义。**

**条件极其狭窄**：`is_blackwell` + `k == 20480` + `64 <= m <= 384`。**只对 GB300 的 verify batch 生效。**

### 3.5 WO-A 输出直接写成 token-major（`deepseek_v4.py`，+147/−13）

新增 `_apply_wo_a_bf16_matmul`，注释说明：

> "**Single-token decode uses a GEMV** for the validated TP4 shape. **Blackwell verify batches up to 384 rows write token-major output directly to avoid the layout copy before wo_b.** ROCm decode can use aiter batched GEMM; **other cases use torch.einsum**."

**替换了什么**：原来是 `torch.einsum("bgd,grd->bgr", o, wo_a)`。

**为什么能省**：einsum 出来是 `[T, G, R]`，而 `wo_b` 要的是 token-major —— 所以中间要一次 **layout copy**。新 kernel **直接按目标布局写**，省掉这次拷贝。

**同样有一长串 shape/stride 守卫**：`o.shape[1:] == (2, 4096)` / `wo_a.shape == (2, 1024, 4096)` / dtype 都是 bf16 / `o.stride(2)==1` / `o.stride(1)==4096` / `o.stride(0)>=8192` / `wo_a.is_contiguous()`。

**【推断】** `[T, G, R]` → token-major 的布局转换在 decode 的临界路径上是纯浪费。**"改输出布局以省掉一次转置"** 是一类常见且低风险的优化。

### 3.6 MoE top-k 输出格式的解耦（`topk.py`，+112/−31）

**删掉了"实验开关"下的分支**【码】：

```python
# 删除：
# ===== TO BE REFACTORED ====
# The experimental fused topk+pack carrier only exists under the master switch.
if _SGLANG_EXPERIMENTAL_LORA_OPTI:
    return isinstance(topk_output, (StandardTopKOutput, StandardTopKOutputPacked))
# ===== END TO BE REFACTORED ====
return isinstance(topk_output, StandardTopKOutput)

# 改为：
return isinstance(topk_output, (StandardTopKOutput, StandardTopKOutputPacked))
```

**注释解释了为什么 `Packed` 也算 standard**：
> "`StandardTopKOutputPacked` is the standard `(weights, ids, logits)` triple **plus the FlashInfer routed-MoE packed ids** the router emitted alongside them; **every standard-format consumer reads it by field name**"

**【推断】** 这是"**把一个实验性载体提升为标准**"的典型动作：因为下游都按字段名读，多带一个字段是向后兼容的。删掉 `TO BE REFACTORED` 标记和实验开关，是**降低技术债**。

### 3.7 自定义 all-reduce 的默认值收敛（`overrides.py`，+13/−1）

```python
prefer_custom_dsv41 = (
    getattr(hf_config, "model_type", None) == "deepseek_v41"
    and getattr(hf_config, "hidden_size", None) == 5120
    and get_platform().is_blackwell
    and view.tp_size == 4
    and view.nnodes == 1
    and not view.disable_custom_all_reduce
)
```

**注释**：
> "V4.1 TP4 uses the custom push plane for decode and fused MoE finalize. **Keep the existing default for other models and parallel configurations.**"

**⭐ 这对应第 2 个 commit**（`Limit custom all-reduce default to the validated V4.1 configuration`）—— 见 §5.2。

**六个条件全部满足才启用**：模型类型 + hidden_size==5120 + Blackwell + TP4 + 单节点 + 用户没禁用。

### 3.8 其它

| 文件 | 变化 | 说明 |
|---|---|---|
| `dsv41_sparse.py` | **−1 行** | **纯删除一行** —— 最小改动 |
| `flashinfer_trtllm.py` | +5 | MoE runner 适配 |
| `standard.py`（token dispatcher） | +8/−5 | 适配新载体 |
| `deepseek_v4_dspark.py` | +33/−2 | 接上新的 WO-A matmul |
| `deepseek_v2.py` | +147/−13 | 公共 decoder layer |

---

## 4. 我认为最重要的四个观察

### 4.1 ⭐ 加速的主要来源是"减少 launch 和消除布局转换"，不是新算法

看 §2.3 那张 launch 表：**每个 decode step 少了约 86 次 kernel launch**（860/20=43，三类各 43）。

**这与我在 FlashMLA/DSA 尽调里的发现完全一致**：
- 竞赛报告：*"Indexer hard is **dominated by kernel launch**"*
- 竞赛所有 kernel 的 **compute 占用率 < 10%**、active SM 只有 1–86/148

**【推断】** **"kernel launch 开销 + 布局转换 + stream 同步"是现代推理引擎的主要战场**，而不是"更快的 GEMM"。这个 PR 的 +105.5% 是这一判断的又一个强证据 —— 它**一行新的矩阵乘算法都没写**。

### 4.2 数值等价性被反复显式承诺（这是可学的核心工艺）

PR 里**至少四处**主动声明"这个改动不改数值"：

| 位置 | 承诺 |
|---|---|
| all-reduce kernel 注释 | "bf16 rounding points are **exactly the unfused path's**"、"**bit-identical**" |
| mHC `num_stages` | "This changes **memory scheduling only**; K tiles and **reduction order stay fixed**" |
| commit 3 标题 | "**Clarify fusion rounding guarantees**" |
| PR body | "**Preserve the BF16 rounding points** and synchronize phase-counter reuse" |

**【推断】** 这是**内核工程师和"能跑就行"的人之间最本质的区别**。融合 kernel 最容易出的错就是**悄悄改了浮点累加顺序**，在大 batch 上表现为精度退化、在小 batch 上测不出来。**作者每处融合都交代了舍入点在哪。**

### 4.3 ⚠️ 大量 Blackwell / shape 专属特化 → 可移植性差

**几乎每个优化都带了平台或形状守卫**：

```
mhc:               is_blackwell and k==20480 and 64<=m<=384
overrides:         is_blackwell and tp_size==4 and nnodes==1 and hidden==5120
WO-A:              is_blackwell and o.shape[1:]==(2,4096) and stride 精确匹配
all-reduce default: model_type=="deepseek_v41" and hidden==5120 and ...
candidate_blocks:  9 个前置条件
```

**【推断】** 这正是 `AGENTS.md` §5.1 提到的约束的体现：**这个 PR 的收益基本无法搬到 H100/H20**，因为：
- `is_blackwell` 直接排除 Hopper
- `k == 20480` 是 V4.1 的特定维度
- TP4 单节点的 push plane 依赖特定互联拓扑

**这反而是一个机会**：**H20 上的同类优化几乎没人做**（因为大家的重心都在 Blackwell）。

### 4.4 跨模型复用：kernel 从 Kimi K3 泛化而来

all-reduce kernel 的注释明确说是 **K3 `finalize_push_norm` 的泛化**。

**【推断】** 这说明 SGLang 内部已经积累了**跨模型的 kernel 复用路径**。`csrc/kimi_k3/` 这个目录的存在本身就是一个信号：**推理框架正在变成"多模型 kernel 库 + 调度层"**。

---

## 5. ⭐ 工程规范：这才是最值得学的部分

### 5.1 测试覆盖（粒度极细）

> "**60 kernel/integration tests and 224 subtests passed**"
> "Four-GPU all-reduce/finalize suites: **99 + 61 cases passed**, including **graph replay, mixed token counts, independent unfused/FP32 references, and delayed-reader phase-counter probes**"

**⭐ `independent unfused/FP32 references`** —— 不只是"新老对比"，还有**独立的 FP32 参考实现**。这是**正确性论证**而不是回归测试。

**⭐ `delayed-reader phase-counter probes`** —— 专门测"消费者延迟读到 phase counter"的竞态。**这是并发正确性测试，不是功能测试。**

覆盖清单还点名了：`router packing/padding`、`routed-input quantization events`、`WO-A layout`、`mHC target/draft overlap`、**`source-stream dependency guards`**（跨 stream 依赖）、`candidate masking`、`verify compressor state/cache writes`。

### 5.2 三个 commit 的演进（**最值得学的工程习惯**）

| # | 时间 | 内容 |
|---|---|---|
| 1 | 12:05 | 主体优化（激进） |
| 2 | 12:10 | **`Limit custom all-reduce default to the validated V4.1 configuration`** |
| 3 | 12:41 | `Clarify fusion rounding guarantees and remove duplicate import` |

**⭐ commit 2 是核心工艺**：**先做激进的优化，然后把默认启用范围收窄到"已验证的配置"**。

**【推断】** 这是把"性能"和"风险"分开处理的正确顺序：
1. 先证明优化有效（激进实现）
2. 再决定**在哪里默认开启**（保守范围）

而且 commit 3 里同时做了"**澄清舍入保证**"和"**删除重复 import**" —— 说明作者在自查。

### 5.3 精度披露得极其诚实

| 测试 | Baseline | Candidate | 预算截断 |
|---|---:|---:|---:|
| GSM8K serial 100 | 97/100 | 98/100 | 0/0 |
| GSM8K 5-shot 1314，并发 64 | 1268/1314 | **1266/1314** | 0/0 |
| AIME 2026 serial 30，temp 0 | 25/30 | 25/30 | 4/5 |
| AIME 2026 30×16，并发 64，temp 1 | 446/480 | **451/480** | 13/18 |

**⭐ 但作者主动补充了这条**（我认为是整个 PR 最体现诚实的部分）：

> "GSM8K has **14 correct-to-incorrect and 12 incorrect-to-correct changes** under the **unchanged prompt and scorer**. **Generated text is not bitwise identical.** These results and the kernel checks **do not establish exact model equivalence for every input**."

**【推断】** 聚合分数几乎持平（1268 vs 1266），但**逐题有 14 个从对变错**。作者**主动说了**，而不是只报聚合分。

**为什么这很重要**：融合 kernel 改了浮点顺序后，**逐样本翻转是预期的**（数值不同 → 采样路径不同）。但如果只报聚合分，读者会以为"完全等价"。**主动披露逐样本翻转，才让人能判断风险。**

对照（这也是常见反面写法）：
- ❌ "精度无退化，GSM8K 96.50% → 96.35%"（隐藏了 14 个翻转）
- ✅ 本 PR 的做法

### 5.4 可复现性做到了极致

PR body 里包含：
- **完整的 server 启动命令**（含所有参数）
- **完整的 benchmark Python 脚本**（`bench_bs1.py` / `bench_bs64.py` / `gsm_eval.py` 全文）
- **数据集 SHA256**（`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`）
- **精确的 worktree 复现命令**
- 评测器版本与 prompt 修订号（`sgl-eval==0.1.0`、NeMo-Skills revision `645cf567...`）

**⭐ 甚至交代了测量口径的陷阱**：
> "**These metrics should not be compared interchangeably.**"
> "BS64 decode samples require **consecutive log intervals with exactly 64 running requests and CUDA graphs active**, excluding intervals across prefill."

**【推断】** 这条直击我们之前讨论过的**测量陷阱**（竞赛里 CUDA events vs CUPTI 差 2–4 μs、在 9 μs kernel 上是 20–40% 误差）。**明确区分口径、拒绝混用，是可信评测的前提。**

### 5.5 影响范围隔离

> "This PR contains **production changes only**; **standalone validation harnesses are kept outside the source diff**."

**【推断】** validation harness 不进 diff，意味着**评审面更干净、review 更快**（100 分钟合并）。这是"怎么让自己的 PR 被快速接受"的实操技巧。

---

## 6. ⚠️ 值得注意的问题与风险

### 6.1 CI 实际上是失败的

> "GPU CI is **blocked** before tests by the global requirement to include `main` commit `3700c4ee26a1`, which also **rejects this `dsv4.1`-based PR**"

三条 CI 状态全是 ❌：
- PR Test (Base)：❌
- PR Test (Extra)：❌
- AMD ROCm 10：❌

**【推断】** 这是**分支策略冲突**：CI 强制要求 PR 包含 main 的某个 commit，而本 PR 基于 `dsv4.1` 分支，必然不满足。作者用"**直接在指定 base/candidate 上采集 B300 结果**"绕过了 CI。

**风险**：**这个 PR 合并时没有跑通 CI**。虽然人工验证很详尽，但这是一个流程上的红灯。

### 6.2 "+105.5%" 的口径需要小心

BS64 decode 的 13,472 tok/s 是在**特定筛选条件**下测的：
> "consecutive log intervals with exactly 64 running requests and CUDA graphs active, excluding intervals across prefill"

而同一个 workload 的 full-batch 指标只涨了 **+56.4%**。

**【推断】** **稳态 decode 的收益（+105%）明显大于含 prefill 的整体收益（+56%）**。引用时**必须带口径**，否则会误导。这与竞赛那个"37× 不是端到端"是同一种陷阱。

### 6.3 多次测量样本量偏小

每个 arm 两次 server 启动 × 3 次测量 = 6 个样本，取中位数。**个体差异不小**：
```
base-a:    6491.16, 6577.62, 6534.28
candidate-a: 13771.38, 13120.99, 13846.63
base-b:    6501.85, 6849.70, 6691.80   ← base-b 最高 6849
```
**【推断】** candidate 的取值范围（12,872~13,846）与 baseline（6,491~6,849）**完全不重叠**，所以 +105% 这个结论应该是稳的。但**没有给出方差或置信区间**。

### 6.4 我无法验证的部分

- **PR 已合并，但我没有 B300 环境**，无法独立复现任何数字
- **`csrc/kimi_k3/comm/ar_fusion.cuh`**（被泛化的那个原版 kernel）我**没看到**，所以无法对比泛化前后的差异
- **`sgl_kernel` 的 push-plane 协议实现**（`communicator.cuh`）不在本 PR 内，无法核对

---

## 7. 可复现的验证问题

**【推断】按门槛排序**：

1. **读 `all_reduce_fusion.cuh` 的头部注释**（约 40 行）—— 它是**一份完整的融合 kernel 设计文档**：数据流、数值等价性、push-plane 协议、phase counter 兼容性。**这是最好的学习材料，比任何教程都值。**

2. **把 `candidate_blocks.py`（115 行 Triton）跑通并读懂** —— 它直接对应 V4.1 §2.3.2 的 Hierarchical Sparse Indexer 的 block 级选择。**这是我们之前讨论过的方向，而且有真实生产实现可对照。**

3. **`mhc.py` 的 `_num_stages_for`（11 行）** —— 最小改动、最清晰的价值示范：**只改内存调度、不改归约顺序**，就能拿到收益。

4. **⭐ 最有价值的方向：把这些优化搬到 H20**。理由：
   - 本 PR 的守卫里 `is_blackwell` 直接排除了 Hopper → **H100/H20 上这些优化都不生效**
   - 按 `AGENTS.md` §5.2，**H20 是"算力稀缺、带宽宽裕"的相反画像**
   - 而 §4.1 的结论（瓶颈在 launch 和布局，不在算力）**在 H20 上只会更成立**（算力更少 → 调度开销占比更高）


---

## 8. 证据分级与来源

| 结论 | 来源 | 等级 |
|---|---|---|
| PR 元数据、性能表、launch 表、精度表 | GitHub API 的 PR body | **【原文】** |
| kernel 设计细节（push plane、+0.0 处理、舍入点） | 分支源码 `all_reduce_fusion.cuh` 注释 | **【码】** |
| `_num_stages_for` / `_apply_wo_a_bf16_matmul` / `prefer_custom_dsv41` 条件 | 分支源码 | **【码】** |
| 三个 commit 的演进 | GitHub API commits | **【原文】** |
| "加速主要来自 launch 减少"、"H20 是空白"等 | 我的推理 | **【推断】** |
| 原版 K3 kernel、push-plane 协议实现 | **未看到** | **无法确认** |
| PR 数字本身的正确性 | **无 B300 环境** | **无法验证** |

**未确认项**：
1. `csrc/kimi_k3/comm/ar_fusion.cuh` 原版内容（不在本 PR）
2. `sgl_kernel/distributed/communicator.cuh` 的 push-plane 实现
3. 作者身份的具体背景（BBuf 是 SGLang 常见贡献者，但我未核实其所属）
4. 该 PR 是否随后被 cherry-pick 到 main


## 文献版本说明

本文所引 DeepSeek-V4.1-Flash 技术报告为原笔记记录的 2026-09-10、51 页版本（文件名 `DeepSeek_V41_Tech_Report.pdf`），页码对应这一版本。本次整理未获得可独立确认的公开下载链接，未将 PDF 打包进本站；涉及该报告的数值沿用原笔记，仍需对照原文复核。

上游入口：[SGLang PR #38879](https://github.com/sgl-project/sglang/pull/38879)。
