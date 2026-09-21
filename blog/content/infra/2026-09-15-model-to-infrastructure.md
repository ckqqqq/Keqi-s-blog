---
title: "从模型架构到系统约束：V4.1 工程问题映射"
date: 2026-09-15
lastmod: 2026-09-21
draft: false
visibility: public
categories: ["系统架构"]
tags: ["DeepSeek","系统设计"]
description: "从模型架构到系统约束：V4.1 工程问题映射；保留技术细节、出处与适用边界。"
---

> 这是技术笔记的公开整理版，原始记录日期为 2026-09-15，本次编辑于 2026-09-21。保留原笔记的公式、代码位置与证据分级；本次仅整理内容，未重新运行 GPU 实验或逐项复核上游。标为【码】【核验】的内容指原记录的核查结果，不代表当前版本仍然如此。

> **硬件边界**：本文涉及 Hopper 与 Blackwell 的不同实现。H100/H20 可用于适配的 Hopper 路径；SM100/SM103 专属实验需单独申请 B200/B300/GB300 等对应设备。没有对应设备时，只能阅读代码或进行不依赖设备的逻辑验证，不能声称复现文中性能。所有跑分沿用原文标注的硬件前提。

> 本地代码核对：FlashMLA、DeepEP（含 `engram_fetch` 实现）

---

## 0. 核心方法论

读架构报告不能只读"模型怎么做"，要读**"这个设计给系统出了什么新题目"**。本文按这个思路逐组件拆解。

先建立坐标系——V4.1 报告的原文把 KV Cache 成本拆成**三个可乘的维度**：

> "These costs can be reduced along three multiplicative dimensions: the **entry size**, where GQA reduces the number of KV heads and MLA shares a small latent across heads; the **sequence dimension**, where every m tokens are compressed into one entry, like CSA and HCA in DeepSeek-V4; and the **layer dimension**, where some layers reuse the caches and selections of other layers instead of keeping their own..."

这句话是理解 V4.1 的钥匙：

| 维度 | 手段 | 成熟度 |
|---|---|---|
| entry size（每条的字节数） | GQA（2023）→ MLA（2024）→ **FP4 KV（2026）** | FP4 是新的 |
| sequence（多少 token 一条） | **CSA/CSA2 压缩**（m 个 token 压成 1 条） | 新的 |
| layer（几层共用一条） | **跨层 KV / indexer K / Top-K 索引复用** | **最新，且最缺系统实现** |

报告还说，前人的工作（IndexCache、YOIO、HySparse）**只覆盖了其中一个维度**，"more importantly, **none of these methods covers all three multiplicative dimensions**"。CSA2 是第一个三个维度一起做的——**所以它的系统复杂度也是三者叠加的**。

**这就是"可以冲"的根源：架构维度叠加 = infra 复杂度相乘 = 空白相乘。**

---

## 1. 逐个新组件 → infra 题目映射

### 1.1 CED（Causal Encoder-Decoder）—— 新

**机制**（报告 2.2 节）：40 层劈成两半，decoder 层（l > L/2）的 global KV **不从自己的隐状态算**，而是直接从第 L/2 层的隐状态投影出来：

```
C_l = H_{L/2} · W^KV_l ,   Z_l = H_{L/2} · W^Z_l ,   l > L/2
```

其中 C 是 KV entry，Z 是对应的压缩权重。SWA 仍然逐层算（保持局部 KV 生成的"计算深度"）。

**收益**：prefill 复杂度从 `O(N·L)` 降到 `O(N·L/2 + n_win·L/2) ≈ O(N·L/2)`，**prefill 计算量几乎减半**。prefill 只激活 8B 参数，decode 激活 16B。这对 agentic 场景（频繁工具调用产生大量 prefill）是关键优化。

**给 infra 出的题目**：

| 题目 | 现状 | 缺口 |
|---|---|---|
| encoder 末尾的 KV 投影 kernel | FlashMLA 已有 | 基本饱和 |
| Decoder SWA Bounded Replay | 报告 3.2.2 描述了 | 需集成进服务框架 |
| **EPD 分离（Encoder/Prefill/Decode 三段独立扩缩容 + 重叠执行）** | 报告说是当前部署形态；vLLM/SGLang 生态只做到 **PD 两段分离** | **三段分离的调度与资源编排是空白** |

**注意**：报告提了一句很重要的话——"prior work (Chen et al., 2025) has shown that the **actual effective receptive field of SWA is much smaller than the theoretical `n_win × L/2`**"。这解释了他们为什么敢做 bounded replay：**经验上有效感受野远小于理论上界**。这也是一个可验证、可攻击的研究点。

---

### 1.2 CSA2 的三种模式 + 跨层复用 —— 最新，系统复杂度最高

**机制**（报告 2.3.1）：每个 CSA2 层被**静态指定**三种模式之一。三种模式都自己算 query 和 SWA KV，区别在于 main KV / indexer K / Top-K 索引从哪来：

| 模式 | main KV | indexer K | Top-K 索引 | 代价 |
|---|---|---|---|---|
| **Full** | 本层算 | 从 main KV 投影 | 本层算（全量扫描） | 完整路径 |
| **Reindex** | 复用最近 Full 层 | 复用 | **本层重新算**（在候选池内） | 中间 |
| **Reuse** | 复用 | 复用 | **复用** | 最省，indexer Q 都不算 |

**三重复用带来的"系统噩梦"**（报告 3.1.2 亲口承认）：

> "layers that share attention components **may be placed on different pipeline stages**, making direct module reuse incompatible with conventional stage-local execution."

也就是说：**要复用的 KV 在一个 pipeline stage，要消费它的层在另一个 stage**。这在传统训练框架里根本不是问题（因为没人做过跨层共享带梯度的状态）。DeepSeek 为此加了三个机制：

1. **Shadow indexers（影子 indexer）**：共享参数只有**一个逻辑 owner** 负责优化和 checkpoint；每个参与的 pipeline stage 放一个**轻量可执行副本**；通过参数同步和梯度聚合让副本保持一致。好处是 pipeline scheduler 不需要把共享层当特殊执行单元处理。
2. **Pipeline payload extensions**：跨 pipeline 边界传中间表示和稀疏路由信息；这些状态被塞进**已有的 point-to-point 通信路径**，并按 context parallelism 一致地分区，避免多余复制、维持梯度流。
3. **Micro-batch 级共享状态管理**：跟踪并发活跃 micro-batch 的共享状态，协调它们在 forward / 激活重计算 / backward 之间的生命周期；状态保留到**最后一个消费者**完成才释放；同一套运行时抽象还负责 stage 放置和"生产者-消费者"关系解析，让 attention 实现**不依赖物理 pipeline 布局**。

**给 infra 出的题目**：

| 题目 | 现状 | 缺口 |
|---|---|---|
| CSA2 三模式的训练支持（shadow indexer + payload + 生命周期） | **只有报告描述，无开源实现** | 🔴 **最大的空白**：这是"读论文 → 自己实现 → 能对比验证"的完整闭环 |
| 跨层共享状态的显存管理 | 报告只给了机制描述 | 没有代码、没有数字 |
| Reuse 模式的缓存一致性 / 竞态 | 未提及 | 并发正确性问题 |
| CSA2 选错 Top-K 的鲁棒性 | 报告承认"未完全刻画" | 见 1.6 |

**为什么说这块最值得冲**：它是**算法+系统交叉**，不是纯 kernel。门槛不在"会写 CUDA"，而在"能读懂 attention 语义 + 能改造分布式训练框架"。这类人极少。

---

### 1.3 Hierarchical Sparse Indexer（分层稀疏索引器）—— 新，且 kernel 实现不完整

**机制**（报告 2.3.2）：只在 CED 的 decoder 用。decoder 里**第一个 Full Mode 层**建一个共享候选池：

1. 该层先对所有因果可见的 main KV 位置打分（全量扫描），产出自己的 Top-K（报告例子是 Top-512）
2. 同时做**块级别的候选选择**：每个 block 取其内部位置 indexer score 的**最大值**作为该 block 的分数，选出分数最高的 block
3. 把选中 block 覆盖的所有位置收集成**候选池**（比最终 Top-K 集合大），例如选 **2048 个 block × 每块 8 个位置 = 16384 个候选位置**
4. 后续 **Reindex Mode 层只在这个池子里打分**，各自选自己的 Top-K

**关键收益**：候选池大小固定时，后续 indexer 的 per-query 成本**从 context 长度的线性变成常数**。但**第一个 Full 层仍然要全量扫描**——报告明确说 "Hierarchical indexing therefore reduces the cost of later indexer evaluations **while retaining the initial full-range pass**"。

**给 infra 出的题目**：

| 题目 | 现状 | 缺口 |
|---|---|---|
| Top-K 选择 kernel | **DeepSelect 已开源（2026-09-09）** | 已覆盖 |
| **块级 max 归约 + 块选择 + 候选池构建的 fused kernel** | 报告只画了 Figure 5，**未开源** | 🟡 明确的 kernel 空白 |
| **在受限候选域（16384 而非全量 N）上做 Top-K** | **DeepSelect 的算法假设是全量 O(N) 扫描** | 🟡 见下方"深挖点" |
| 固定 topk=512 / 候选池大小是编译期还是运行期 | 未说明 | 待确认 |

**深挖点（我认为这是最漂亮的一个切入点）**：

我读了 DeepSeek 官方发布的 DeepSelect 算法解析（`docs/DeepSelect-deep-dive.zh.md`），它的核心是**阈值过滤 + 随机块序**：

```
DeepSelectTopk(x[0:N), k, B, B2):
    topk_candidate = []            # 放在 shared memory
    topk_threshold = -inf          # 当前第 k 大元素的值
    p = random_permutation(ceil_div(N, B))   # 与输入无关的随机块顺序
    for block in p:
        for j in block:
            if x[j] > topk_threshold:          # 阈值过滤
                topk_candidate.append((x[j], j))
        if len(topk_candidate) >= k + B2:
            topk_candidate = RadixSelectTopK(topk_candidate, k)   # 压缩
            topk_threshold = min(v for v, _ in topk_candidate)     # 阈值单调上升
    return RadixSelectTopK(topk_candidate, k)
```

它的性能保证依赖"**每个元素恰好从全局内存读一次，按连续块访问**"，期望处理元素总量为 `O((1 + k/B₂)(k + B + B₂)·log(N/B))`。

**注意**：这个算法是为**全量扫描**设计的。而 Hierarchical Sparse Indexer 的 Reindex 层面对的是**受限候选域**（16384 个候选位置，而非全部 N 个）。这两个场景的访存模式、最优块大小、是否需要随机块序，**都不一样**——官方文档明确划了支持范围："`topk`: small (must be ≤ 4096; larger values are not supported)"，且只覆盖 Lightning Indexer 和 Sampling 两个场景，**没有覆盖"候选池内的 Top-K"**。


---

### 1.4 Single-Pass mHC + Mega-mHC —— 已实现，空间在收窄

**机制**（报告 2.4.1）：mHC 在相邻 Transformer block 之间维护 `n` 条残差流 `X_l ∈ R^{n×d}`：

```
X_{l+1} = B_l · X_l + C_l · F_l(A_l · X_l),    (A_l, B_l, C_l) = H(X_l)
```

**报告的定量分析（很有价值）**：

- 单个理想融合 map 需要读 `(n+1)d`、写 `(n+1)d`，activation 内存流量下界是 `(2n+2)d`
- V4 的**多趟实现**是 3 个串行 kernel：残差更新、系数计算 H、输入混合。共读 `(n+1)d + nd + nd`，写 `(n+1)d`，带 pre-norm 后**总流量 `(4n+4)d` = 下界的 2 倍**
- **两趟实现**：把 normalization 权重离线折进投影权重，RMS 除法放到投影之后；于是「残差更新（不需要跨 hidden 维归约）」可以和「H 的投影累积 + 平方和累积」**共用一次 residual 遍历**。但输入混合**不能**融进这一趟，因为 `A_l` 要等所有 hidden tile 的归约完成才可用，所以需要第二次读 `X_l`（这趟顺便做 input pre-norm）。总流量 `(3n+2)d`
- **结果**：Mega-mHC kernel 把 activation 流量**减半**（相比原来的 4-kernel 实现）

**现状**：DeepGEMM 有 Mega-mHC；TileKernels 有完整的 mHC kernel 套件（`expand`、`head_compute_mix`、`multilayer_recompute`、`norm_fn`、`post`、`pre_apply_mix`、`pre_big_fuse`、`pre_split_mixes`、`sinkhorn`）。


---

### 1.5 Engram 条件记忆 —— 全新范式，缺口最大

**机制定位**（报告 2.1）：Engram 是 **"sparsely accessed conditional memory"**，与 DeepSeekMoE 的共享专家/细粒度路由专家**并列**。V4.1 有 **552B backbone 参数 + 196B Engram 参数**。

**推理侧**：报告 3.1.3 说得很清楚——

- Engram embedding 表**按行切分**到独立的进程组（`engram parallel size`），组大小控制"单设备内存占用 vs embedding lookup 通信范围"的权衡
- 优化器状态再在各个表分区的副本间**进一步分片**
- Engram 的 lookup 索引**只依赖输入 token 序列**，所以可以在每个 pipeline stage 处理 micro-batch **之前**就为整个 local batch 发起 embedding **预取**，把干扰降到最低
- Embedding 梯度在 backward 期间**缓冲**，backbone backward 完成后**归还原属 rank**
- Embedding 以 **FP8 存储和读取**，取出的值和 scale factor **直接喂给后续 GEMM**
- 表更新用 **Sinkhorn 归一化**，跨迭代维护行/列缩放向量，**避免重复写整个归一化矩阵**；行归一化 + 列统计的偏和累积**融进单个 kernel**
- RL rollout 期间 Engram 表**常驻显存**，减轻主机内存压力、避免主机内存碎片导致的 OOM

**训练侧/优化器**：报告 2.5 节说，对 Engram 参数用 Adam 会让优化器状态内存暴涨，所以改用**动量 + Sinkhorn 平衡**（Algorithm 1），只保留一个动量缓冲，经验上还优于 Adam。

**分布式内存访问（最关键的发现）**：

我在 **DeepEP 源码里找到了 Engram 的 RDMA 实现**——`deep_ep/buffers/elastic.py`：

- `get_engram_storage_size_hint(num_entries, hidden, ...)` —— 标注 **(Experimental)**，用于估算 buffer 尺寸，**CPU buffer 必须是 2 MB 对齐**（"e.g. for Engram storage"）
- `engram_write(storage, sf)` —— 把 Engram 存储写进 buffer，标注 **(Experimental)**，storage 形状 `[num_entries, hidden]`，支持 bfloat16 或 FP8
- `engram_fetch(indices, num_qps, use_tma_aligned_col_major_sf)` —— **通过 RDMA 从远端 rank 拉取 Engram entry**，标注 **(Experimental)**；FP8 模式下 scale factor 来自 `engram_write` 时提供的 `sf` 张量
- 底层 kernel：`csrc/kernels/elastic/engram.hpp`（`EngramFetchRuntime`，JIT 编译 `engram_fetch_impl<...>` 10 个模板参数）、`include/deep_ep/impls/engram_fetch.cuh`、`engram_fetch_wait.cuh`
- **代码里明确的 TODO**：`# TODO(tianr22): revise the QP count in consideration of Engram`（QP = Queue Pair，RDMA 的核心资源）
- `elastic.py:199` 的注释直接把用途写成 "**Engram (remote KV cache fetch, using RDMA)**"

**结论**：Engram 的**分布式内存访问内核已经有了雏形**（在 DeepEP 的 elastic buffer 里），但是：

1. **全部标注 Experimental**
2. **有未完成的 TODO（QP 数量规划——这是 RDMA 性能的关键调参点）**
3. **只在 DeepEP 这一个仓库里有，没有任何服务端/存储引擎/训练框架的集成代码**
4. 本地 `Engram` 仓库**只有 4 个文件**（README + demo + 论文 + 图），是纯粹的算法演示，**没有一行生产代码**

| 题目 | 现状 | 缺口 |
|---|---|---|
| Engram gating / hash / 梯度归约 kernel | **TileKernels 已开源**（训练侧） | 已覆盖 |
| Engram RDMA 预取 | DeepEP `elastic.py` + `engram_fetch.cuh` | 🔴 Experimental，QP 调优 TODO 未做 |
| **Engram 的服务端存储引擎**（百万级表、FP8、跨机调度、热度感知放置） | **不存在** | 🔴 完全空白 |
| **Engram 训练并行框架**（engram parallel size、优化器状态分片、预取调度） | **只有报告描述** | 🔴 完全空白 |
| Sinkhorn 更新 kernel | TileKernels 有 `sinkhorn` | 基本覆盖 |
| Engram 在 vLLM/SGLang 里的接入 | **不存在** | 🔴 空白 |


---

### 1.6 FP4 KV Cache —— 新，且有明确的硬件不对称（我认为这是最"实"的冲点）

> ⚠️ **本节已修正**（2026-09-15）：初版把结论写成"SM90 完全没有 V4.1 kernel"。逐文件核实后，**实际缺口比那个说法更窄、但更精确**。以修正版为准。

**机制**：报告 2.1 说 "we compress the main KV cache to **FP4**"；结论说 cross-layer KV reuse + FP4 KV caching 把 global KV 压到 **890 bytes/token**（注意：890 B/token 是**全局聚合值**，不是单个 kernel 的 KV 布局宽度）。

**我在 FlashMLA 源码里核对到的事实**（本文最硬的一段证据）：

`csrc/kernels/params.h` 定义四种模型类型：

```cpp
enum class ModelType {
    V32,        // DeepSeek V3.2 (d_qk=576)
    V4,         // DeepSeek V4 (d_qk=512)
    V41,        // DeepSeek V4.1 (d_qk=512, RoPE fp8, quant tile size 32)
    V41_FP4     // DeepSeek V4.1 (d_qk=512, fp4 e2m1, quant tile size 16, e4m3 scales)
};
```

**KV 布局的精确参数**（`tests/quant.py`，返回 `(d, d_nope, d_rope, tile_size, num_tiles)`）：

| 布局 | 参数 | 每 token 字节数公式 | 数值 |
|---|---|---|---|
| `V32_FP8Sparse` | `(576, 512, 64, 128, 4)` | `d_nope + num_tiles*4 + 2*d_rope` | 512+16+128 = **656** |
| `V4_FP8Sparse` | `(512, 448, 64, 64, 7)` | `d_nope + 2*d_rope + num_tiles + 1` | 448+128+7+1 = **584** |
| `V41_FP8Sparse` | `(512, 448, 64, 32, 16)` | `d_nope + d_rope + num_tiles` | 448+64+16 = **528** |
| `V41_FP4` | `(512, 448, 64, 16, 32)` | `d // 2 + num_tiles` | 256+32 = **288** |

**关键点**：`quant tile size` 从 V4 的 **64** 变成 V41 的 **32**（FP8）和 **16**（FP4）；scale 数量从 V4 的 **8** 变成 **16**。这**不是一个新 dtype 的问题，而是 KV 布局整体换了**。

**实例化清点结果**（逐文件读过 `cat`）：

| 架构 | 稀疏解码实例化 | V4.1（V41 / V41_FP4）支持 |
|---|---|---|
| **SM90（Hopper，H800/H100）** | 只有 `v32_persistent_h64/h128.cu` 和 `v4_persistent_h64/h128.cu` **共 4 个** | ❌ **无任何 V41 实例化**；`grep -i fp4 csrc/kernels/sm90/` **返回空** |
| **SM100（Blackwell，B200）** | 含 `v41_h64{,_no_split}.cu` 与 `v41fp4_h64{,_no_split}.cu` | ✅ FP8 + FP4 |

**修正后的结论（更精确）**：

1. **SM90 完全不支持 V4.1 的稀疏 KV 格式**（V41 FP8 布局 / V41_FP4），这是硬缺口。
2. **SM90 上连"用 V4 path 假装跑 V4.1"都会算错**，而且可能是**静默错误**：
   - `sm90/decode/sparse/config.h` 里 `NUM_SCALES = MODEL_TYPE == V32 ? 4 : 8`，注释写 "For DeepSeek-V4: **7 fp8_e4m3 + 1 padding**"
   - V4.1 的 `num_tiles = 16`（即 16 个 scale），**远超 SM90 的 8**
   - `splitkv_mla.cuh:585` 的 scale 索引是 `scales[MODEL_TYPE == V32 ? dim_idx/2 : dim_idx]`——**没有 V41 分支**，只有 `if constexpr (MODEL_TYPE == V32)` / `== V4`
3. **所以 SM90 现在只有 V32 和 V4 是安全的**，V4.1 必须上 SM100。
4. 测试文件自己写着：`tests/test_flash_mla_sparse_decoding.py:25`
   > `# The DeepSeek-V4.1 KV cache formats (V41 / V41_FP4) are only supported on SM100f`
   > `supports_v41 = torch.cuda.get_device_capability()[0] >= 10`

**这个缺口为什么重要**：

- DeepSeek 自己的 benchmark 数字来自 **H800**（README 里 660 TFLOPS / 410 TFLOPS / 3000 GB/s 全是 H800 SXM5 上测的），说明**生产集群里 Hopper 仍是主力**
- 但 Hopper **没有原生 FP4 TensorCore**。所以 FP4 在 Hopper 上的合理路径是：**FP4 做存储/带宽压缩，读出后反量化到 bf16 再算**——而这**恰好就是现有 FP8 稀疏解码 kernel 已经在做的事**（`splitkv_mla.cuh:586` 的 `dequant_and_save_bf16x8` lambda 把 FP8 反量化成 bf16 再算）
- 也就是说：**把 FP4 的 KV cache 布局 + 反量化路径 + 新的量化粒度（tile 16 / 16 scales）加到 SM90 上，是"有现成骨架、目标明确、可验证"的工程任务**——不需要发明新算法

| 题目 | 现状 | 缺口 |
|---|---|---|
| SM100 的 V4.1 FP8/FP4 稀疏解码 | ✅ 已开源 | 饱和 |
| **SM90 的 V4.1 FP8 稀疏解码**（tile 32 / 16 scales） | ❌ 无 | 🟢 **明确空白，有骨架可循** |
| **SM90 的 V4.1 FP4 稀疏解码**（e2m1 / tile 16 / e4m3 scales） | ❌ 无 | 🟢 **明确空白，价值最高**（Hopper 是生产主力） |
| SM90 上误用 V4 path 跑 V4.1 → 静默错误 | ⚠️ 可能发生 | 🟡 **可以加一道断言/防御** |


---

### 1.7 KV Cache 分层存储与 SWA Bounded Replay —— 新，且是系统层空白

这是**第 4 层（服务层）最实的变化**，前面在总览里列过，这里补充它的"题目属性"：

| 题目 | 现状 | 缺口 |
|---|---|---|
| HBM / DRAM / SSD 三级 KV 存储 + 不同 TTL | 报告描述了生产形态 | vLLM/SGLang **原生不支持三级层级 + 差异化 TTL** |
| SWA KV 放主机内存池（10% DRAM）、TTL 几分钟 | 生产已用 | 无开源实现 |
| global KV 保证 72 小时寿命 | 生产已用 | 无开源实现 |
| **Bounded Replay 的近似性刻画** | 报告说 "barely compromises response quality"、有 train-aware 模拟 | 🔴 **报告自己列为未解决问题** |
| 缓存恢复边界的正确性 | 🔴 报告明确说 "SWA state reconstruction at cache-resumption boundaries" 需要继续做 | **研究机会** |

**注意这里有个重要的正确性陷阱**（报告原文承认）：

> "By design, the replayed prefix state is **approximate**, so the global KV and SWA KV computed for the uncached suffix **depend on the cache-hit position** and are **not mathematically identical across positions**."

意思是：**同一个 suffix，因为 cache 命中位置不同，算出来的 KV 会不一样。** 这直接破坏"prefix caching 是确定性函数"这个隐含假设。对服务系统来说，这意味着：
- 不能假设 prefix cache 命中后结果与全量重算一致
- 需要新的**一致性测试/校验机制**
- 影响**可复现性**（同一个请求在不同 cache 状态下可能得到不同输出）——这会波及评测、A/B、回归检测


---

## 3. 报告自己列的"未解决问题"（最值得抄的作业）

报告结论章（Section 6）原文列的 robustness boundaries：

1. **CSA2 的 selection errors**（选错 Top-K 对下游能力的影响未被刻画）
2. **SWA Bounded Replay 的近似状态重建**（近似性边界未刻画）
3. 重点关注方向：
   - **sparse retrieval over long contexts**（长上下文稀疏检索）
   - **SWA state reconstruction at cache-resumption boundaries**（缓存恢复边界的 SWA 状态重建）
4. 方法论承诺：扩大 **stress-testing and evaluation stack**，系统性刻画 failure modes 和 robustness boundaries



---


## 文献版本说明

本文所引 DeepSeek-V4.1-Flash 技术报告为原笔记记录的 2026-09-10、51 页版本（文件名 `DeepSeek_V41_Tech_Report.pdf`），页码对应这一版本。本次整理未获得可独立确认的公开下载链接，未将 PDF 打包进本站；涉及该报告的数值沿用原笔记，仍需对照原文复核。
