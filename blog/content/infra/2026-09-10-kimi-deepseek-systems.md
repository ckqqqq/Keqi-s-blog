---
title: "Kimi vs DeepSeek Infra"
date: 2026-09-10
lastmod: 2026-09-21
draft: false
visibility: public
categories: ["系统架构"]
tags: ["KV Cache","MoE","沙箱"]
description: "Kimi 与 DeepSeek：缓存、通信与沙箱的系统对照；保留技术细节、出处与适用边界。"
---


> 证据基础：
> - **DeepSeek**：V4.1-Flash 技术报告（51 页）+ 本地源码级尽调（FlashMLA / DeepEP / DeepJIT / Engram / DeepSelect）
> - **Kimi**：K3 技术报告（arXiv:2607.24653v2，47 页，2026-08-07 修订）+ K2.5（arXiv:2602.02276）+ 已核实开源的 MoonEP / AgentENV
> kimi和deepseek infra stack解构～技术栈比较

---

## 0. 一句话结论

**两家在"模型架构"上分道扬镳（DeepSeek 走稀疏，Kimi 走线性），但在"infra 该做什么"上高度收敛——收敛点就是机遇所在。**

两家独立撞上了同一批问题，且都公开承认没解决完：

| 收敛点 | DeepSeek | Kimi |
|---|---|---|
| KV cache 不再是单一结构 | global KV + SWA KV（**两种 TTL**） | MLA KV + KDA recurrent state（**两种生命周期**） |
| 前缀缓存粒度被"最慢的状态"绑架 | bounded replay 破坏确定性 | block size 被迫 1024–6144 |
| 内存分层到 DRAM/SSD | HBM 890B/token + DRAM TTL 分钟级 + SSD 72h | GPU→CPU DRAM write-back pool + NVMe 卸载训练状态 |
| 负载均衡 | 报告承认未做 | MoonEP：动态冗余专家 + ILP 规划 |
| Agent 沙箱 | DSec：百万级容器 | AgentENV：microVM，**5121 万**沙箱 |

**【推断】速度（延迟/吞吐）的机会已经不在 kernel 里了**，理由见第 3 节。

---

## 1. 架构分道扬镳：这决定了 infra 的题目不同

### 1.1 DeepSeek V4.1 的路线【原文】

- **CED（Causal Encoder-Decoder）**：40 层劈成 20 encoder + 20 decoder，decoder 的 global KV **直接从第 20 层隐状态投影**得到 → prefill 计算量减半
- **CSA2**：沿**三个可乘维度**同时压缩 KV —— entry size（MLA）、sequence（压缩）、**layer（跨层复用 KV / indexer K / Top-K 索引）**
- **SWA Bounded Replay**：只 replay 最近 `n_win` 个 token，**主动接受近似状态**
- **FP4 KV cache**：global KV → 890 B/token
- 报告自评："none of these methods covers all three multiplicative dimensions"（指前人工作）

### 1.2 Kimi K3 的路线【原文】

- **Kimi Delta Attention（KDA）**：**线性注意力**，用**固定大小的 recurrent state** `S ∈ R^{d_k × d_v}` 取代增长的 KV cache
- **Attention Residuals**：改善跨模型深度的信息流
- **Stable LatentMoE**：每 token 激活 16 / 896 个 routed experts
- 规模：**2.8T 总参数 / 104B 激活 / 1M 上下文**
- 每 block = **3 层 KDA + 1 层 Gated MLA**

### 1.3 关键对比

| 维度 | DeepSeek V4.1 | Kimi K3 |
|---|---|---|
| 长上下文手段 | **稀疏 + 压缩**（保留 softmax 注意力） | **线性注意力**（recurrent state） |
| KV cache 性质 | 大、只读、可压缩、可跨层复用 | **可变、原地更新、固定大小** |
| 前缀缓存对象 | 只有 KV | **KV + recurrent state（两种）** |
| 并行策略难点 | 跨层复用的 pipeline 放置 | **recurrent 的串行依赖** |

**【推断，重要】** 这个分叉意味着：**谁做出"通用"的 infra，谁的方案就能被两边复用**。而"通用"恰恰是当前开源生态最缺的——见第 3 节。

**顺带一个有意思的信号**：DeepSeek V4.1 引用 PowerAttention（arXiv:2503.03588）来支撑 SWA 有效感受野的论断。PowerAttention 是**稀疏模式设计**的工作，但它和线性注意力是一条思路上的东西（都是在攻"感受野如何随深度扩张"）。**【推断】** 这暗示 DeepSeek 内部**也在看线性注意力这条线**，尽管 V4.1 选了稀疏。**无法确认**他们是否会转向。

---

## 2. 两家的 Infra 章节逐条对照（这是最有价值的部分）

### 2.1 KDA / KDA Context Parallelism（Kimi）

**【原文，§5.1.2】** KDA 的状态更新是 `S_t = M_t S_{t-1} + β_t k_t v_t^T`，其中 `M_t := (I − β_t k_t k_t^T) Diag(α_t)`。

问题：**vanilla 线性注意力的直接求和对 KDA 不成立**——因为 delta rule 先把 token 相关的矩阵 `M_t` 作用在"进入的状态"上，所以**一个局部段的效应依赖于进入该段的状态**，无法只从 `S=0` 推出的状态得到。

解法（KCP）：把每个段的效应**分解成两个可局部计算的量**：
- `M_{T_{i+1}←1}`：累积转移矩阵（cumulative transition）
- `ẽS_{T_{i+1}}`：从零开始本地生成的状态

于是 `S_t^{[i+1]} = ẽS_t^{[i+1]} + M_{t←1}^{[i+1]} S_{T_i}^{[i]}`。这些 rank 级更新**可结合**，所以用 **prefix scan** 恢复每个 rank 的进入状态：
> "Each rank first computes `M_{T_i←1}^{[i]}` and `ẽS_{T_i}^{[i]}` locally, then exchanges both tensors with **one all-gather**. ... KCP requires only a **fixed-size all-gather** for recurrent-state synchronization and achieves linear compute scaling."

**【核验】** 论文说实现已进 FLA PR #691。

**⭐ 这是"线性注意力的 CP 比 softmax 的 CP 通信量小"的严格论证**：
- softmax CP：交换的 KV block **随序列长度增长**
- 线性 CP：交换**固定大小**的 recurrent state
- 代价：KDA 需要把 delta rule 的矩阵因子也传过去（不是直接求和）

### 2.2 MoonEP（Kimi）—— 直接对标 DeepEP

**【原文，§5.2.1】** 开篇就点名："MoonEP preserves the overall computation flow of conventional schemes such as **DeepEP** and additionally introduces **online planning and migration of redundant experts**."

| 项 | 内容 |
|---|---|
| **目标** | 每个 rank 恰好收到 `S × K` 个 token（`S`=序列长，`K`=每 token 选几个专家）→ 所有 rank 计算量完全相同 |
| **理论结果** | **证明了平衡方案总是存在，且每 rank 最多 `E/R` 个冗余专家（`E`=专家数，`R`=EP size），且该界本质上是紧的（§E）** |
| **对比 ECHO / UltraEP** | 它们预设冗余专家数或设 per-rank token 上限 → **无可行方案时训练被迫停止**，且上限要手工调 |
| **在线规划** | 离线用 **ILP** 求精确最优作为参考；线上设计 **GPU 规划 kernel**，近最优、开销可忽略、始终满足 `E/R` 上界 |
| **零拷贝通信** | 融合 permute/unpermute，规划 kernel 预计算每个 token 的目的地，**消除中间拷贝** |
| **sync-free** | 完美平衡 ⇒ 所有层的形状**静态已知** ⇒ **消除每层 MoE 的 host 同步**，缓解 host 侧 kernel launch 开销 |
| **Expert-GEMM 调度** | 即使总量平衡，rank 内 per-expert token 数仍偏斜 → 用**工作负载感知的调度器**，启动前按当前 token 分布调参，执行中固定；用硬件指标的**解析代价模型** + 离线 autotune 标定系数 |

**⭐ 缓冲区的定量对比【原文】**：
> "Under worst-case imbalance, supporting the same copy-free data path in DeepEP requires a communication buffer of size **`S × K × R`**, whereas MoonEP requires only a fixed **`S × K`** buffer owing to the perfect balance."

**【核验】** MoonEP 已开源：`github.com/MoonshotAI/MoonEP`，1131 stars，MIT，创建 2026-07-24。
仓库描述："MoonEP: A Perfectly Balanced Expert Parallelism Library via Dynamic Redundant Experts"

**这与 DeepSeek 的对照非常有信息量**：DeepSeek V4.1 报告说他们有一个 TODO —— `# TODO: support do_expand and allow_multiple_reduction`，且 `get_theoretical_num_sms` 的 docstring 自认 "**assumes a balanced gate distribution**"、**"For V3.0's group-limited gate, please do not use this function"**。

**【推断】** 也就是说：**DeepSeek 公开承认自己的 EP 解析模型不支持非均衡 gate，而 Kimi 把"完美均衡"做成了一个有理论保证的开源库。** 这是竞品之间罕见的"一方补上另一方缺口"的例子。

### 2.3 内存效率（Kimi）—— 抽象层次的提升

**【原文，§5.2.2】** 几个设计我觉得层次很高：

**① Unified activation manager（统一激活管理器）**
> "every tensor saved for the backward pass is associated with a **pluggable storage backend**. **Recomputation, quantization, and offload/remote-offload are merely storage policies** under this abstraction and can be freely composed at tensor granularity; policies are declared via lightweight annotations on tensors, **fully decoupled from the model code**."

**【推断】** 这是把"重算 / 量化 / 卸载"统一成**一种策略**而不是三套机制——很好的抽象。可以类比成"激活的存储虚拟化"。

**② MoE 梯度依赖的数学改写**
> "the gradient computation of permuted probs depends on the forward output. Inspired by SonicMoE, we **rewrite this gradient through a mathematical transformation** into a form that depends only on the intermediate activation and the upstream gradient, **eliminating the backward dependency on output**"

**③ P2P-based Muon orthogonalization**
> "the Newton–Schulz orthogonalization in Muon **requires the full parameter matrix**，necessitating a communication step to gather complete parameters before each update. The naive approach performs an **all-gather over the entire parameter buffer on every rank** [74], which incurs a substantial memory footprint... Instead, each rank retrieves **only the shards of its locally owned parameters via P2P** communication with the corresponding owner ranks"

**【推断】** Muon 的分布式实现是当前热点（DeepSeek 也在用 head-wise Muon，且报告说 "head-wise Muon is also validated in GLM 5 and **Kimi-K3**"）。这里的 P2P 优化是一个真实且通用的技术点。

**④ 跨 PP rank 的激活均衡**
> "activations are unevenly distributed across PP ranks due to pipeline warmup... To avoid OOM, we **remotely offload activations to the memory of other PP ranks** using the **Mooncake Transfer Engine**"

### 2.4 百万 token Agentic RL（Kimi）

**【原文，§5.3.1】**

**① External KV cache pool（外部 KV cache 池）**
> "We decouple prefix retention from GPU residency with a **write-back** design. Active decoding blocks remain in GPU KV cache, while reusable idle prefixes are **written back to an external KV cache pool in CPU DRAM only when it is evicted from GPU**, and is prefetched back before the next reuse. **KDA states are offloaded and prefetched together with the corresponding MLA KV cache blocks, keeping their lifecycles aligned.**"

关键对比：
> "Compared with a **write-through** strategy, this policy incurs CPU DRAM usage and transfer bandwidth **only for prefixes that leave the active decode path**, avoiding redundant CPU copies of blocks that are still resident and active on GPU."

**【推断】** "write-back vs write-through" 这个选择很关键：write-through 会把每个块都拷到 CPU，包括还在活跃使用的——浪费带宽。**这是一个纯粹的缓存策略问题，Mac 上就能建模分析。**

**② Rollout auto-throttling scheduler**
用运行时信号（活跃请求数、排队请求数、KV cache 利用率）**动态控制发给推理引擎的请求数**，避免早段欠载与后段过载。

**③ 用梯度缓冲复用参考模型权重**
> "We keep these weights in CPU memory and materialize them only when needed, **backing their parameter tensors with the policy model's FP32 gradient-buffer storage**... one slot is used for the current forward computation while the other prefetches the next chunk"

**【推断】** 很妙的显存复用：参考模型权重"寄居"在策略模型的梯度缓冲里，因为梯度缓冲在真实梯度计算时会被覆盖，所以安全。

### 2.5 AgentENV（Kimi）vs DSec（DeepSeek）

**【原文，§5.3.2 + 脚注 4】**

| 项 | Kimi AgentENV | DeepSeek DSec |
|---|---|---|
| 隔离机制 | **Firecracker microVM** | 容器 + **per-sandbox AppArmor + eBPF 网络策略** |
| 为什么 | "container-based sandbox runtimes, we observed several **kernel panics and deadlocks** caused by unintended agent operations" | "agents exploited recently disclosed vulnerabilities, including permission issues from the XFS driver, illegal memory access in AppArmor, leaking answers from package mirror services" |
| 规模 | **51,219,741 个沙箱 / 1,505,678 个镜像** | **百万级并发容器** |
| 密度 | 内存超分比 **6.5×**；OverlayBD 镜像 + 自研 **ublk** 驱动 + 存储层共享 + P2P 传输 | 单物理节点 **1000 → 2500+** 容器（sub-NUMA 分区 + 绑核） |
| checkpoint/resume | **133 ms / 49 ms**（增量：只存脏页） | 抢占安全恢复 |
| 高级操作 | **Pause/Resume**（暂停不占内存 CPU）、**Fork**（无副作用做 reward judging）、**Snapshot** | — |
| 关键洞察 | "a sandbox can be paused while the agent is waiting for the model's inference result, which can account for **as much as 98% of the sandbox lifetime**" | — |

**【核验】** AgentENV 已开源：`github.com/kvcache-ai/AgentENV`，**3465 stars**，**Rust**，创建 2026-07-23，**最近推送 2026-09-15（今天）**。

**【推断】** **"98% 的沙箱生命期在等模型推理"** 这个数字是整个 agentic RL infra 里信息量最大的一个——它说明沙箱的主要成本不是计算，而是**等待**。所以 pause/resume + fork + snapshot 不是锦上添花，而是把 CPU/内存占用降到接近零的核心手段。

**【推断】两家都撞上了"agent 作恶"**：DeepSeek 遇到 XFS/AppArmor 漏洞利用、删库跑路；Kimi 遇到 kernel panic 和死锁。**这是 agentic training 独有的新 infra 问题**，不是传统分布式训练的问题。

### 2.6 部署侧（Kimi）—— 最有借鉴价值的一节

**【原文，§5.4.1】**

**① Unified cache layout for hybrid KDA–MLA**
> "The MLA KV cache grows with sequence length and is paged per token, whereas the KDA recurrent state is **fixed in size with a single copy per request**. Maintaining a separate manager for each would duplicate the allocation, eviction, and transfer logic. We therefore **pack KDA states into the same paged block pool as MLA KV**, unifying pages to the same byte size so that both page types **share one implementation of allocation, reference counting, and eviction**."

细节：页内所有 head 的状态**按 head 连续存放**，使每个 head 的字节流自包含，成为**跨节点传输的最小单位**。PD 分离时若 prefill/decode 节点 TP 度不同，**在传输路径上重布局，GPU 侧零 reshuffle**。

**【原文，一句很有意思的话】**：
> "This asymmetry proved useful during development: **any type-confused access yields garbage rather than plausible data — a zero-overhead sanity check** on the pooled layout."

**② 前缀缓存粒度的耦合问题（我认为这是全文最漂亮的分析）**

> "Block-hash-based prefix caching reuses the KV cache at the granularity of one physical block: only complete blocks are hashed, so only block-aligned prefixes are reusable. **This coupling breaks down in Kimi K3.** Block-hash matching requires **one block size shared by all layers**, and a prefix hit is reusable only if **the KDA state at the hit boundary has been persisted**. A KDA layer maintains a **single large recurrent state per sequence rather than per-token entries**, so state snapshots are **affordable only at sparse boundaries**; the shared block size is therefore **forced to 1024–6144 tokens**—and, since hashing is tied to the storage block, the hash granularity as well, **although MLA's per-token entries alone would tolerate much finer blocks**. At such a coarse granularity **caching is nearly useless**: requests shorter than one block can never be reused, and chunked prefill exports no cacheable prefix until it crosses a full block boundary."

解法（**解耦两种粒度**）：
- 前缀哈希跑在 MLA 页内的**细 hash block（512 token）**
- 物理块仍是粗分配单位
- KDA checkpoint **只在 MLA 的 hash 端点（的稀疏子集）保存**——因为那是 lookup 唯一会引用的位置
- **Lookup 两阶段**：先按 chained hash 匹配整个物理块；在第一个缺失块处**回退到块内的 hash 端点**，于是部分填充的页仍可命中；KDA 阶段要求候选边界在每个 KDA cache group 都有 checkpoint
- **命中边界永远是 hash block 的倍数，从不要求是物理块的倍数**——论文例子里请求前 2800 token 匹配，**命中在 `B = 2560 = 5 × 512`，深入 6144 的物理块内部**

**③ 并发调度下的一致性（三个具体失败模式）**
> "First, all cache groups draw blocks from **one shared free list**, so allocating a private copy for one group could evict a block that another group has just hit; every hit block is therefore **pinned across all groups before anything is allocated**. Second, the copy into the private block executes on the GPU **immediately before the forward pass**, so a block allocated or registered within the current scheduling step would still hand the previous owner's bytes to a reader; such blocks are **excluded from matching until their copies land**. Third, a checkpoint can restore a request only if it exists in **every** KDA group, so **evicting one group's checkpoint atomically invalidates its siblings**."

### 2.7 KDA 解码 kernel 与投机解码的状态回滚（Kimi）

**【原文，§5.4.2】** 这是我认为**最有"新工程问题"味道**的一段：

> "the evolving recurrent state... is **updated in place at every decoding step**. **This in-place update becomes problematic in MTP-based speculative decoding: if verification rejects a subset of the drafted tokens, the state has already advanced beyond the last accepted token and cannot be trivially rolled back.** Maintaining a state snapshot for each draft position would enable rollback, but would also **multiply state traffic — a cost that dominates at the large batch sizes typical of online serving**."

解法：
> "The state after any accepted draft prefix, however, is **fully determined by the projected inputs of the draft tokens, which are far smaller than the state itself**. We therefore **cache only these projected inputs, rebuild the states of accepted tokens on-chip, and write back the states of the verified and bonus tokens**... The replayed tokens, the bonus token, and the next draft window share **one recurrent loop inside a single fused kernel** covering short convolution, input normalization, gating, the KDA recurrence, and output normalization."

**【推断】** 这是一个**二阶**问题：投机解码本来是为了加速，结果引入了"状态回滚"这个新问题。**这类问题在大规模部署前不会被发现** —— 正是"新"的标志。（论文提到 concurrent work ReplaySSM [25] 独立提出同样设计。）

---

## 3. 速度（延迟/吞吐）的机会在哪 —— 我的判断

### 3.1 先说一个反直觉的结论

**两家都已经把 kernel 做得很好，"再快 37x"式的机会基本没了。** 证据：

- DeepSeek 的 SM90 dense decode seesaw 调度：**80% Tensor Core 利用率 + 3 TB/s**，两篇官方 deep-dive，2025-04 后无实质改动【码】
- 所审查的 FlashMLA 快照只有 6 个 MMA atom，**全是 bf16**；FP8/FP4 只是存储+反量化格式【码】
- 竞赛方案 37x 的构成：**最大单项来自"读 workload 而不是读 kernel"**（69/128 trace 走 passthrough，13.2→2.0 μs），而不是 kernel 技巧【原文】
- 竞赛所有 kernel 的 compute 占用率 **< 10%**、DRAM **< 4%**、active SM 只有 **1–86 / 148**【原文】

**所以"速度"的机会不在算术强度，而在别处。**

### 3.2 机会在四个地方（按我认为的确定性排序）

#### 机会 A：KV cache 的"分层 + 差异化生命周期"（确定性最高）

**为什么**：两家**独立**都做了这件事，且都还没做完。

- DeepSeek：HBM 890 B/token / DRAM（10% 内存，**TTL 分钟级**）/ SSD（global KV **≥72 小时**）
- Kimi：GPU KV cache → **CPU DRAM 外部池**（write-back）+ **NVMe** 卸载训练状态

**【推断】具体的未解问题**：
1. **"辅助状态与主 KV 的生命周期对齐"是新的通用需求**。Kimi 的解法是硬编码的："KDA states are offloaded and prefetched **together with** the corresponding MLA KV cache blocks"。**一个通用的"衍生状态跟随主状态"抽象还不存在。**
2. **write-back vs write-through 的策略选择缺乏理论指导**。Kimi 只说了 write-back 更好（避免冗余拷贝），但没给驱逐策略、预取时机、容量规划的模型。
3. **DeepSeek 的 bounded replay 破坏确定性**（同一 suffix 因 cache 命中位置不同，KV 不同），**而 Kimi 的细粒度前缀缓存明确保证了确定性**（"every registered state always corresponds to exactly its declared token prefix"）。**这两种设计的取舍没有比较研究。**


#### 机会 B：前缀缓存的粒度解耦（工程性最强）

Kimi 把 block size 从"被迫 1024–6144"压回到"hash 512 / 物理块粗分配"的两级结构。

**【推断】但问题没完**：
- Kimi 的解法**深度绑定"KDA 状态只在稀疏边界可截"** 这个具体性质。**DeepSeek 的 CSA2 跨层 KV 复用 + SWA bound replay 是完全不同的约束**，能不能用同一套两级结构？**未知。**
- **层级更多时怎么办**？如果未来一个模型同时有：per-token KV（MLA）、per-request 状态（KDA）、per-layer 共享 KV（CSA2 Reuse 模式），**前缀缓存的"一致性边界"要怎么定义？**
- Kimi 的三个并发一致性失败模式（共享 free list / 本步注册的块 / checkpoint 跨 group 失效）**在更多 cache group 时会组合爆炸**。

**【推断】** 这是一个**可以纯软件、纯逻辑做**的方向 —— Mac 上就能写模拟器验证。

#### 机会 C：MoE 负载均衡的"规划"范式（Kimi 已给出参考答案）

**【核验】MoonEP 的核心贡献**：
- **理论**：证明平衡方案总存在，每 rank ≤ `E/R` 冗余专家，**且界是紧的**
- **方法**：离线 ILP 做参考 + 线上 GPU 规划 kernel（近最优、开销可忽略）
- **收益**：消除 host 同步（形状静态已知）+ 零拷贝 permute/unpermute + **缓冲从 `S×K×R` 降到 `S×K`**

**【推断】DeepSeek 的缺口已被我自己在源码里证实**：`get_theoretical_num_sms` 自认 "assumes a balanced gate distribution"，且明确写 **"For V3.0's group-limited gate, please do not use this function"** + `assert num_scaleout_topk == 0`。

**这意味着什么**：**Kimi 用一个理论结果 + 一个开源库，把 DeepSeek 公开承认没做的部分给做了。** 这种"补竞品缺口"的位置，是最清晰的机遇形态。

#### 机会 D：agentic RL 的沙箱与长上下文 rollout（最被低估）

**【推断】我认为这是被严重低估的方向**，理由是 Kimi 那个数字：

> "a sandbox can be paused while the agent is waiting for the model's inference result, which can account for **as much as 98% of the sandbox lifetime**"

**98% 的时间在等推理。** 这意味着：
- 沙箱 infra 的优化目标**不是算得快，而是"等的时候不占资源"**
- 所以 pause/resume（**133 ms checkpoint / 49 ms resume**）、fork（无副作用 reward judging）、snapshot 是核心
- 内存超分 **6.5×**、镜像 sub-second 启动、**5121 万个沙箱 / 150 万个镜像**

**两家的具体缺口**：
- DeepSeek DSec 用容器 + AppArmor/eBPF；Kimi 转向 **Firecracker microVM**，明确理由是容器"**kernel panics and deadlocks**"
- **【推断】"容器 vs microVM"这个问题没有被解决，只是被选择了**。而且 agent 能力越强，隔离要求越高——这是一个**会持续恶化**的问题。
- 长上下文 rollout：Kimi 的 auto-throttling 用运行时信号调并发；DeepSeek 的 sample-level dispatch。**两套都是启发式，没有理论。**

### 3.3 一句话总结"速度的机会"

> **延迟的大头已经不是 compute，而是"状态在内存层级之间搬动"以及"等待"。**
>
> - 算得慢 → 已经很好了（80% TC、<10% compute 占用率说明根本不是瓶颈）
> - **搬得慢 → 新战场**（KV/状态的分层、生命周期对齐、前缀缓存粒度）
> - **等得久 → 新战场**（沙箱 98% 在等、rollout 长尾、auto-throttling）

---

## 5. 必须说清楚的证据缺口

1. **我不了解两家的内部路线图**。本文全部基于**公开论文 + 源码**。他们"未来会押注什么"**本质上是推断**，不是事实。
2. **Kimi K3 的 infra 数字未经复现验证**（51M 沙箱、6.5× 超分、133/49 ms、2.5× scaling efficiency 全部引自论文）。
3. **DeepSeek 的 890 B/token 口径无法从报告确认**——`tests/quant.py` 里 `V41_FP8Sparse` 主 KV 是 528 B/token，加 FP4 `extra_kv` 288 B/token，对不上 890。**【核验】我算不出来，报告也没给推导。**
4. **两家是否真的会继续这两条路线**：可能的变数——
   - DeepSeek 是否会转向线性注意力？**无法确认**（只看到他们引用了 PowerAttention）
   - Kimi 的 KDA 是否会在更大规模暴露问题？**无法确认**
5. **我没有 Kimi 的源码级尽调**（不像 DeepSeek 有本地 6 个仓库）。**MoonEP 和 AgentENV 可以作为补做的起点。**
6. **"哪家 infra 更强"我无法判断** —— 两家的报告都是自述，没有第三方基准。

---

## 附：文献与核验

| 内容 | 来源 |
|---|---|
| Kimi K3 技术报告 | arXiv:2607.24653v2（2026-08-07 修订，47 页） |
| Kimi K2.5 | arXiv:2602.02276v2 |
| MoonEP | `github.com/MoonshotAI/MoonEP`｜1131★｜MIT｜创建 2026-07-24 |
| AgentENV | `github.com/kvcache-ai/AgentENV`｜3465★｜Rust｜创建 2026-07-23｜推送 2026-09-15 |
| KDA CP 实现 | FLA PR #691（论文脚注 2） |
| DeepSeek V4.1 | `DeepSeek_V41_Tech_Report.pdf`（51 页） |
| DeepEP 缺口 | `deep_ep/buffers/elastic.py:750, :755-757`（源码核验） |


## 文献版本说明

本文所引 DeepSeek-V4.1-Flash 技术报告为原笔记记录的 2026-09-10、51 页版本（文件名 `DeepSeek_V41_Tech_Report.pdf`），页码对应这一版本。本次整理未获得可独立确认的公开下载链接，未将 PDF 打包进本站；涉及该报告的数值沿用原笔记，仍需对照原文复核。
