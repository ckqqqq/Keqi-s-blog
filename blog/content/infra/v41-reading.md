---
title: "DeepSeek V4.1：模型架构与系统设计精读"
date: 2026-09-16
lastmod: 2026-09-21
draft: false
visibility: public
categories: ["模型架构"]
tags: ["DeepSeek","KV Cache"]
description: "DeepSeek V4.1：模型架构与系统设计精读；保留技术细节、出处与适用边界。"
---

> 这是技术笔记的公开整理版，原始记录日期为 2026-09-16，本次编辑于 2026-09-21。保留原笔记的公式、代码位置与证据分级；本次仅整理内容，未重新运行 GPU 实验或逐项复核上游。标为【码】【核验】的内容指原记录的核查结果，不代表当前版本仍然如此。


---

## 零、一句话主线

报告标题即主线:**"Pushing the Limits of KV Cache Compression"**。V4.1 的一切架构选择都服务于一件事——把长上下文 agentic 负载(input-heavy)的部署成本打下来:prefill 只激活 8B(decode 16B)【p1】,全局 KV cache(常驻 HBM)压到 **890 bytes/token ≈ V4-Flash 的 1/4**【p1】,持久 KV cache(SSD/host 内存)压到 **≈ 1/8**【p1】。相对 V1 降 437 倍【p1 Fig.1 注】。

---

## 一、模型配置(全部数字原文照录,§4.2.1)

- 40 层 Transformer,hidden dim 5120;**CED = 20 层 causal encoder + 20 层 causal decoder**【p21-22】
- 前 2 层纯 SWA;encoder 其余 18 层 CSA2,压缩率 m=2,3 组 × 6 层(首层 Full,余 5 层 Reuse)【p22】
- decoder 20 层 CSA2,m=1,5 组 × 4 层:第一组首层 Full,其余四组首层 **Reindex**,余皆 Reuse【p22】
- indexer:32 heads × dim 128,**top-k=512**;query 64 heads × dim 512,compression dim 1280【p22】
- 分层索引器:2048 blocks × 8 positions = 16384 候选位置【p22】;SWA 窗口 n_win=128【p22】
- MoE:每层 1 shared + **384 routed experts**,中间维 2304,每 token 激活 6 个【p22】
- mHC expansion factor=4;**Engram 196B 参数**,两个模块放在第 1、14 层【p13】
- 总量:552B backbone + 196B Engram;prefill 激活 8B / decode 16B【p22】
- 训练:45T tokens,64K 起步无 dense warmup,34T 处扩到 1M;batch 100.6M tokens;无 instability【p22】

## 二、四个核心架构机制

### 2.1 CED(Causal Encoder-Decoder)——prefill 减半【§2.2, p9】

decoder 的全局 KV 不由本层产生,而从 encoder 末层 hidden states 投影:C_l = H_{L/2}·W_l^{KV}。于是 prefill 只过 encoder:
> "CED reduces the prefill complexity from O(NL) to O(NL/2 + n_win × L/2) ≈ O(NL/2)"【原文 p9】

代价:decoder SWA KV 仍需逐层生成——用 **Decoder SWA Bounded Replay** 兜底(见 2.4)。动机是 agent 工作流频繁 tool call 导致 prefill 请求爆炸。

### 2.2 CSA2——三维压缩 + 三种模式【§2.3, p9-12】

压缩可在三个相乘的维度上做:entry size、序列维(每 m token 压一条)、层维(跨层复用)。批评前人:"none of these methods covers all three multiplicative dimensions"【p10】。

每层属于三种模式之一:
- **Full**:自己算 main KV + 跑 indexer 出新 Top-K
- **Reindex**:复用前层 KV,用自己的 query 重打分出新 Top-K
- **Reuse**:KV 和 Top-K 全复用,最省——"Reuse Mode layers execute with only 15 kernels during prefill and 11 during decode"【p6/p18-19】

相对 V4 的 CSA:去掉相邻压缩条重叠与绝对位置嵌入,indexer K 改从 main KV 投影【p10】。

**Hierarchical Sparse Indexer**(仅 decoder)【§2.3.2, p11-12】:首个 Full 层全范围打分选 top-512,并按 block(8 位置/块)选 2048 块构成 16384 候选池;后续 Reindex 层只在池内打分——"changes the per-query cost of deeper indexers from linear in context length to constant"【p11-12】。训练感知:候选限制在训练与推理中一致施加【p12】。

### 2.3 FP4 主 KV Cache【§2.4.4, p14】

- 格式:OCP MXFP4(E2M1 + 每 16 通道一个 E4M3 scale),**省掉 NVFP4 的第二级全局 scale**——明确承认是为硬件兼容性牺牲精度:"to support as many hardware platforms as possible, despite the higher accuracy of alternative formats"【p14】
- 数值论证:RMSNorm 后 512 通道 latent 的 L2 范数 ≤ √512,单通道上界 ≈22.6,实测最大 ≈10,格式上限 448×6=2688,故无需全局 scale【p14】
- RoPE 之后量化;SWA KV 保留 FP8(量化敏感)【p14】;效果:相对 V4 的 FP8 主 KV 近乎减半【p14】

### 2.4 SWA Bounded Replay——持久化压到 1/8 的关键【§3.2, p19-20】

问题:SWA KV"访问模式与长期保留策略不匹配"——只在会话内分钟级窗口复用,存 SSD 又贵又无效【p19】。V4 的精确重建要重放 L×n_win token,"proved prohibitive in production"【p19】。

V4.1 方案:
1. SWA KV 不进持久 cache,改存**每台机器 10% host DRAM 的分布式内存池**,TTL 几分钟;global KV 仍留持久 cache ≥72 小时【p19】
2. miss 时只重放最近 **n_win 个 token** 近似重建(而非 L×n_win):"turns a catastrophic miss into a graceful, inexpensive degradation"【p19】
3. **Decoder Bounded Replay**:每次 prefill 重放 prompt 最后 n_win token 过 decoder 层,所得 SWA KV 只用于 decode 不进 cache——"nearly halving total prefill computation"【p20】

承认的近似性:"not mathematically identical across positions",但实验显示质量影响可忽略;且 post-training 中模拟同样 replay 做 train-aware 适配【p20】。

## 三、架构扩展三件套【§2.4, p12-14】

- **Single-Pass mHC**:输入混合系数移位一个 block(用 A_{l-1} 而非 A_l),消除依赖,单次遍历。部署融合 kernel **Mega-mHC** 把残差更新+输入混合+系数预测+pre-norm+FP8 cast 合一,达到理想下界 (n+1)d 读 + (n+1)d 写,激活显存流量减半【p13】。trade-off:"this shift incurs negligible performance degradation"【p13】
- **Engram**:196B 参数分两模块,N-gram orders {2,3,4}、8 哈希头、每头 ~16M 条目(素数表)、FP8 存储【p13】。推理时"deterministic addressing enables embeddings to be prefetched from host memory via background RDMA transfers"【p13】。去掉了短因果卷积:"performance gains do not justify the added complexity"【p13】
- **DSpark**(投机解码):3 个 Transformer block(滑窗 128),单次前向并行算 5 个草稿位置 + Markov head 建模依赖;confidence head 预测接受概率,调度器按实测吞吐曲线动态选验证长度【p14】。预训练后专设阶段冻结主干训练;RL/OPD rollout 也用它加速【p14】

## 四、训练与推理基建【§3, p16-20】

**训练侧**:
- 对比学习阶段两次 all-gather 全部藏进计算:视觉特征在文本前向时 gather,文本特征在文本反向时 gather【p16-17】
- vision encoder 解耦执行:每步分三相(vision forward → LLM fwd/bwd → vision backward),LLM 阶段零视觉计算【p17】
- 超长序列:图片按负载均衡分片到各 CP rank、每张只加载一次;瓶颈判据 ρ < (B_IO/B_GPU)·C 与序列长度无关——"production-scale models remain compute-bound"【p17】
- CSA2 跨 pipeline stage 共享:shadow indexers(单一逻辑 owner)+ pipeline payload 扩展 + micro-batch 级共享状态生命周期管理【p17-18】
- Engram 训练:表按行切分到专属进程组;每个 micro-batch 处理前对整个本地 batch 预取 embedding;FP8 存取;Sinkhorn 归一化融合单 kernel【p18】

**推理侧**(明确点名开源仓库!):
> "the fused-RoPE-attention-RoPE-cast kernel in **FlashMLA**, the Mega-Gate, Mega-mHC, and Mega-MoE kernels in **DeepGEMM**, the kernels in **TileKernels**, and the TopK kernel in **DeepSelect**"【原文 p18】

- 部署用 **EPD(Encoder-Prefill-Decode)解耦**,三者独立扩缩容、重叠执行【p19】
- 通信-计算重叠、分片 Engram 表、kernel 融合【p6】

## 五、优化器【§2.5, p14-15】

三路并行:**Muon**(线性层)+ **AdamW**(归一化/bias/scale)+ **Sinkhorn 平衡更新**(所有 embedding 与 prediction head,含 196B Engram——省掉 Adam 的庞大 optimizer state,且"empirically outperforming Adam"【p15】)。Query/Key 用 **head-wise Muon**(每 head 独立预条件器),GLM-5 与 Kimi-K3 亦验证此优势【p15】。Engram 学习率 ×5【p22】。

## 六、后训练:无算法创新,全是数据与基建【§5, p25-36】

作者明确:"our post-training introduces no algorithmic innovation"【p6】;"systematic improvements in the scale, diversity, and verifiability of synthesized data and environments account for essentially all of the observed gains"【p25】。

- **任务合成**:任务 = (problem, environment, verification) 三元组,按 difficulty + correctness 迭代训练模型自己造任务;环境构建由多个专职 agent 流水线完成(判定→搭建→解题→质检→修复),含"清除解答痕迹""hackability 风险审查"环节【p25-26】
- **DSec 沙箱平台**:数百万并发沙箱实例【p27-28】;**不用 K8s**,自研 placement engine(最终一致性即可,节点本地硬拒绝超额放置);sub-NUMA 绑定使单机密度 1000→2500+ 容器【p28】;per-sandbox AppArmor + eBPF 网络策略治理 reward hacking(观察到 agent 利用 XFS 权限问题、从包镜像泄露答案、删文件系统)【p28-29】
- **可控推理 effort**:b∈{1..100} 作为 RL 条件信号,token 惩罚系数随 effort 指数衰减 k(b)=k₀·exp(−(b−b_min)/τ);附录推导证明该形式使"effort 与偏好长度呈仿射关系"(局部近似)【p29-30, p49-51】。生产 API 三档:max/high/low = 100/75/50【p30】。effort 25→100:推理均值 67.1%→76.3%,代价 ~2.5× token【p34】
- **异步 RL 基建**:rollout 与训练共置分时;三种派发粒度试错(batch→prompt→**sample 级**最终胜出);token 级中断 + KV cache/routing token 粒度持久化 + sample 级 GC;跨 checkpoint 样本用 concatenated routing-replay【p30-31】
- **长度偏置**:异步下短样本先完成主导早期 batch → dispatcher 限流 + 丢弃过早短样本【p31】;off-policy → 上限 + loss masking 剔除高 staleness token【p31】
- **OPD**:**超过 40 个架构异构的 teacher** 全词表蒸馏,teacher 切换近乎零成本【p31-32】
- **多智能体**:DSH Agent Team(spawn_teammate/共享任务板/derived-latency 惩罚——按关键路径计费,鼓励并行);每个 wall-clock deadline 上多智能体均优于单智能体(ProgramBench 8h:30.04% vs 20.39%)【p35-36,标注为初步结果】

## 七、作者承认的局限(原文照录)

1. 【p37】"no finite test suite can cover every extreme input and deployment condition. Potential selection errors in CSA2 and approximate state reconstruction in SWA Bounded Replay may still cause capability degradation in untested boundary cases."
2. 【p37】"this parity does not imply that the model matches the frontier capabilities of leading closed-source systems on complex, high-difficulty reasoning and edge cases."
3. 【p6】科学向 agentic 任务(Terminal-Bench 4.0)与巨型模型仍有差距;多模态整体落后于闭源巨型系统
4. 【p32-33】评测基建易被 gaming(观察到反编译 Ubuntu 核心包找漏洞),呼吁社区设计下一代 benchmark 时优先缓解

## 附:精读中发现的细节存疑点

- 摘要与 §1 正文对 8B/16B 的表述顺序相反(数值一致)【p1 vs p4】
- "V4-Pro ≈ 1.6T 总参数"可由 p24 Table 1 直接证实(Backbone Params 1.6T),非推断
- effort 训练的 k₀、τ、C_max、L_norm 具体数值全文未给出【p30】
- Figure 6(BPB)与 Figure 7-12 的精确曲线值需查原图,文本提取不含


## 文献版本说明

本文所引 DeepSeek-V4.1-Flash 技术报告为原笔记记录的 2026-09-10、51 页版本（文件名 `DeepSeek_V41_Tech_Report.pdf`），页码对应这一版本。本次整理未获得可独立确认的公开下载链接，未将 PDF 打包进本站；涉及该报告的数值沿用原笔记，仍需对照原文复核。
