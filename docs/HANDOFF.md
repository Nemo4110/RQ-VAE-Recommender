---
updated_at: 2026-09-01
branch: repro/tiger-paper-audit
base_branch: fork-main
scope: TIGER / RQ-VAE 复现审计
status: RQ-14 3k→200k 已完成；出现晚期恢复但未超过 2k 峰值，训练长度方向冻结，当前无获准 GPU 实验
---

# TIGER 复现审计交接文档

> 本文是当前会话在上下文压缩后的单一方向锚点。后续 Agent 必须先读“实验结果总表”“当前结论”和“停止线”，再决定是否运行新实验。
>
> 本文只覆盖 TIGER、RQ-VAE、Semantic ID、decoder 和对应评估协议；不要扩展到无关模型或数据集。

## 0. 文档维护规则

1. 每次新增实验，必须同时更新开头的“实验结果总表”和对应实验方向章节。
2. 每个实验方向必须明确记录五项内容：**动机、依据、改动、结果、分析**。
3. 推荐指标只比较 `paper_candidate_unconstrained` 结果；不得与 `native_constrained` 或 Temporal-v1 teacher-forced full-catalog 混比。
4. “论文达成率”只用于描述数值差距，不表示协议已经等价：
   - RQ-VAE 论文基准：Recall@10=`0.0648`，NDCG@10=`0.0384`；
   - LSH 论文基准：Recall@10=`0.0533`，NDCG@10=`0.0309`；
   - Random-ID 论文基准：Recall@10=`0.0434`，NDCG@10=`0.0250`。
5. 使用占位内容特征的 `dataset/amazon-p5` RQ-VAE/LSH 结果一律标记为“语义结论无效”；Random-ID 不依赖内容特征，仍可作为 decoder 控制实验。
6. 不删除旧 checkpoint、日志或实验目录；新实验必须使用新的 `out/...` 路径。
7. 每次 decoder 训练结束后，必须立即运行 beam-100 `paper_candidate_unconstrained` evaluator。
8. 不安全加载历史 pickle-rich artifact；历史机制应从源码重建，并明确标注与旧 artifact 是否字节等价。

## 1. 实验结果总表

> 达成率计算公式：本地指标 ÷ 对应论文同类方法指标。标记为“不适用”的行，不得用于论文复现结论。

| 编号 | 实验/方向 | 数据与语义源 | 关键改动 | 训练预算 | H/Recall@10 | NDCG@10 | Recall 达成率 | NDCG 达成率 | raw top-10 invalid | expanded beam invalid | 有效性/结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| P-RQ | 论文 Beauty RQ-VAE | 论文 | 论文报告值 | 200k decoder | 0.06480000 | 0.03840000 | 100.00% | 100.00% | 约 0.1%–1.6% | 未公开精确 beam | 目标基准 |
| P-LSH | 论文 Beauty LSH | 论文 | 论文报告值 | 论文预算 | 0.05330000 | 0.03090000 | 100.00% | 100.00% | 未公开 | 未公开 | LSH 基准 |
| P-RND | 论文 Beauty Random-ID | 论文 | 论文报告值 | 论文预算 | 0.04340000 | 0.02500000 | 100.00% | 100.00% | 未公开 | 未公开 | Random-ID 基准 |
| RQ-01 | corrected E1 1k | `amazon-p5-st5` / RQ-VAE | Kaiming、reset1、no-separator | 1k | 0.02660645 | 0.01443039 | 41.06% | 37.58% | 0.047847% | 0.081161% | 第一个有效内容语义点，但低于论文 |
| RQ-02 | historical-E1 1k | `amazon-p5-st5` / RQ-VAE | 历史 epoch 顺序、reset50、标准初始化重建 | 1k | 0.02852927 | 0.01497015 | 44.03% | 38.98% | 0.040692% | 0.096856% | 明显优于 corrected E1 1k |
| RQ-03 | historical-E1 2k | 同上 | 从 1k 精确恢复 optimizer/scheduler | 2k | **0.02884228** | **0.01507586** | **44.51%** | **39.26%** | 0.017887% | 0.075884% | 当前最佳有效 RQ-VAE 点 |
| RQ-04 | historical-E1 3k | 同上 | 再续训 1k | 3k | 0.02750078 | 0.01436449 | 42.44% | 37.41% | 0.027724% | 0.118186% | 相对 2k 回落，但仍处于 10k 恒定 LR 阶段，不足以排除长程恢复 |
| RQ-14a | historical-E1 长程 total-5k | 同上 | 单进程从 total-3k 继续；只改训练长度 | 5k | 0.02884228 | 0.01434907 | 44.51% | 37.37% | 0.036221% | 0.252023% | Recall 与 2k 持平，但 NDCG 低 4.82% |
| RQ-14b | historical-E1 长程 total-10k | 同上 | 到达恒定 LR 区间末端 | 10k | 0.02530966 | 0.01308892 | 39.06% | 34.09% | 0.432411% | 2.619818% | 相对 2k 下降 12.25%/13.18% |
| RQ-14c | historical-E1 长程 total-20k | 同上 | 进入 inverse-square-root decay | 20k | 0.02499665 | 0.01276273 | 38.58% | 33.24% | 0.666726% | 3.631758% | 长程轨迹最低点 |
| RQ-14d | historical-E1 长程 total-50k | 同上 | 保持同一单进程/seed/optimizer | 50k | 0.02709833 | 0.01327331 | 41.82% | 34.57% | 0.648392% | 3.279479% | 相对 20k 出现恢复，但仍低于 2k |
| RQ-14e | historical-E1 长程 total-100k | 同上 | 同上 | 100k | 0.02736663 | 0.01327803 | 42.23% | 34.58% | 0.594732% | 2.683003% | 恢复趋缓，仍低于 2k |
| RQ-14f | historical-E1 长程 total-200k | 同上 | 与论文训练步数对齐；其余合同不变 | 200k | 0.02745607 | 0.01343658 | 42.37% | 34.99% | 0.639449% | 2.589545% | 相对 2k 仍低 4.81%/10.87%；长度方向冻结 |
| RQ-05 | corrected E4 1k | `amazon-p5-st5` / RQ-VAE | normalized codebook + reset1 | 1k | 0.02736663 | 0.01460464 | 42.23% | 38.03% | 0.059026% | 0.085856% | 健康但下游低于 historical-E1 1k，冻结 |
| RQ-06 | 论文切分：不暴露 validation | `amazon-p5-st5` / historical-E1 | `train_subsample_include_validation_target=False` | 1k | 0.02110629 | 0.01118454 | 32.57% | 29.13% | 0.028172% | 0.084067% | 更符合论文切分，但本地性能显著下降 |
| RQ-07 | T5X-inspired paper stack | `amazon-p5-st5` / historical-E1 | batch256、LR .01、10k 恒定后逆平方根、Adafactor 近似、不暴露 validation | 1k | 0.01184993 | 0.00558019 | 18.29% | 14.53% | 0.000000% | 18.185127% | 本地近似失败，禁止扩展 |
| OPT-01 | T5X Adafactor CPU 数值兼容门禁 | 固定小 tensor + 13M HF T5 参数结构 | T5X 默认衰减/因子化/参数缩放/update clipping/state 语义 | CPU 2 step + preflight | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 4/4 专项测试通过；对应 RQ-09/RQ-10 已完成并冻结 |
| RQ-09 | T5X-compatible 1k | `amazon-p5-st5` / historical-E1 | 官方 T5X 默认 Adafactor 数学、paper split、batch256、LR .01 | 1k | 0.01846801 | 0.00996719 | 28.50% | 25.96% | 0.419890% | 2.555561% | 显著优于旧 HF 近似，但低于 paper-split AdamW 1k；允许精确续训到 2k |
| RQ-10 | T5X-compatible total 2k | `amazon-p5-st5` / historical-E1 | 从 1k 精确恢复 optimizer/scheduler，再加 1k update | 2k | 0.01502482 | 0.00757808 | 23.19% | 19.73% | 4.680946% | 11.630282% | loss 继续降但检索与 invalid 显著恶化；冻结，不扩 3k/10k |
| RQ-08 | literal HF `d_model=128` | `amazon-p5-st5` / historical-E1 | 仅将 `d_model` 384→128 | 1k | 0.02647230 | 0.01398943 | 40.85% | 36.43% | 0.041139% | 0.118186% | 低于 384 维且仅 5.11M 参数，冻结 |
| ARCH-01 | T5X T5-1.0 tied shared-vocab CPU gate | 13M HF T5 结构 | ReLU 对应官方 T5-1.0；共享 1024 item vocab、logits via embedding | CPU forward/preflight | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | CPU gate 通过、14.93M 参数；对应 RQ-11 已完成并冻结 |
| RQ-11 | T5X tied shared item-vocab 1k | `amazon-p5-st5` / historical-E1 | shared 1024 item vocab、logits via embedding、T5X-compatible optimizer | 1k | 0.01591915 | 0.00839680 | 24.57% | 21.87% | 0.458346% | 3.127264% | 低于 per-position T5X 1k，冻结 head 方向 |
| LOSS-01 | T5X z-loss CPU gate | 固定 logits + 现有模型 | 官方 T5-1.0 `z_loss=1e-4`，只改 loss | CPU formula/preflight | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | CPU gate 通过；对应 RQ-12/RQ-13 已完成并冻结 |
| RQ-12 | T5X z-loss 1k | `amazon-p5-st5` / historical-E1 | per-position T5X-compatible + `z_loss=1e-4` | 1k | 0.01855744 | 0.01001847 | 28.64% | 26.09% | 0.394849% | 2.757725% | 相对 no-z 仅微升约 0.5%；允许 total-2k 稳定性判定 |
| RQ-13 | T5X z-loss total 2k | `amazon-p5-st5` / historical-E1 | 从 z-loss 1k 精确恢复，再加 1k update | 2k | 0.01475652 | 0.00770087 | 22.77% | 20.05% | 4.896928% | 12.311228% | 未阻止回落/invalid 激增；T5X framework-default 近似全部冻结 |
| RQ-H1 | corrected E1 RQ-VAE health | `amazon-p5-st5` | 100 epoch、reset1 | 1200 step | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | health 通过；best epoch55，loss `0.1419834` |
| RQ-H2 | historical-E1 RQ-VAE recovery | `amazon-p5-st5` | reset50、500 epoch | 6000 step | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | health 通过；best epoch299，loss `0.1216636` |
| RQ-H3 | corrected E4 health/export | `amazon-p5-st5` | normalized codebook + reset1 | 120/1200 step | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 10/100 epoch 均通过，full ID 唯一 |
| RQ-H4 | historical-E2 zero-mass EMA | `amazon-p5-st5` | zero-mass EMA、reset50、历史顺序 | 1 step | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 第一步坍塌：usage `[2,9,6]`，最大 bucket 5042 |
| RND-01 | Random-ID seed29 1k | 交互数据 / Random-ID | 4×255/256 空间、no-separator | 1k | 0.02915530 | 0.01537338 | 67.18% | 61.49% | 0.054107% | 0.132988% | 有效 decoder 控制实验 |
| RND-02 | Random-ID seed29 2k | 同上 | 精确续训到 2k | 2k | **0.03197245** | **0.01717991** | **73.67%** | **68.72%** | 0.069311% | 0.145463% | Random-ID 当前最佳，2k 局部峰值 |
| RND-03 | Random-ID seed29 3k | 同上 | 再续训到 3k | 3k | 0.03072039 | 0.01652435 | 70.78% | 66.10% | 0.065286% | 0.359657% | 回落，冻结长度方向 |
| RND-04 | Random-ID seed43 2k | 同上 | 第二 decoder seed 复核 | 2k | 0.03130170 | 0.01649155 | 72.12% | 65.97% | 0.076913% | 0.215535% | 复现 2k 优势 |
| RND-05 | Random-ID train-only | 同上 | 不将 validation 放回训练 | 1k | 0.02450476 | 0.01211243 | 56.46% | 48.45% | 0.055002% | 0.132764% | 明显低于 validation-inclusive 控制 |
| LSH-01 | 本地 LSH-8 控制 | `amazon-p5` 占位内容 / LSH | 8-bit SimHash、4 token | 1k | 0.02794795 | 0.01493994 | 52.44%* | 48.35%* | 0.069758% | 0.114743% | `*` 内容语义无效，不得作论文结论 |
| LSH-02 | real-content LSH-8 preflight | `amazon-p5-st5` / LSH | 4 token、每位 8 hyperplanes | 未训练 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 仅 9478/12101 唯一，论文 collision 合同不明，禁止 GPU decoder |
| SYN-01 | 早期 `amazon-p5` RQ-VAE 系列 | 占位文本/随机 smoke feature | LR、optimizer、batch、hash、head、separator、checkpoint 等多组诊断 | 多组 | 不适用 | 不适用 | 不适用 | 不适用 | 多组 | 多组 | 只能保留机制诊断，不是 Beauty 内容语义复现 |

## 2. 当前结论与选择

### 2.1 当前最佳有效点

```text
语义源: historical-E1 real-content RQ-VAE
decoder: nominal total 2k, seed 29, no separator
H@10:    0.02884228
NDCG@10: 0.01507586
论文达成率: Recall@10 44.51%，NDCG@10 39.26%
```

该结果仍包含以下协议差异，因此只能称为“本地有效内容语义最佳点”，不能称为严格论文复现：

- 训练 sampler 会高概率重新暴露 validation target；
- 当前使用 HF T5 近似，而非作者 T5X 配置；
- user hashing 只做到 2000 桶容量对齐，未证明 hash 函数完全一致；
- Sentence-T5 revision、pooling、normalization 和文本序列化未完全确认；
- beam width 与 tie-break 未公开。

### 2.2 已冻结方向

- historical-E1 decoder 长度：`RQ-14` 已完成；10k/20k 继续下降，50k～200k 虽晚期恢复，但 200k 仍比 2k 低 `4.81%/10.87%`，因此训练长度方向现已充分冻结。
- E4 normalized-codebook：health 通过但下游回落；禁止继续 decoder。
- historical-E2 zero-mass EMA：第一步坍塌；禁止扩大。
- validation-exclusion：论文切分更准确但 1k 明显下降；禁止本地续训。
- T5X-inspired batch/LR/Adafactor 近似：性能和 expanded invalid 均失败；禁止扩大。
- literal HF `d_model=128`：低于 384 维；禁止组合或续训。
- Random-ID：2k 为局部峰值；禁止 3k 以上、mapping sweep 或更多 seed。
- real-content LSH：collision-resolution 合同未公开；禁止自行加第五 token、丢弃碰撞 item 或挑 seed。

### 2.3 当前停止线

T5X z-loss total-2k 未能阻止回落：相对 z-loss 1k，H@10/NDCG@10 下降 `20.48%/23.13%`，raw/expanded invalid 升至 `4.90%/12.31%`；相对 no-z 2k 也没有一致优势。至此官方 T5X framework-default 的 optimizer、tied/shared item head 与 z-loss 三条 bounded gate 均已完成并冻结。T5X 系列禁止继续 3k/10k/50k、组合 sweep、更多 seed 或 3024-way user+item 输出。

`RQ-14` 已把 historical-E1 从 total-3k 单进程连续训练到 total-200k。轨迹在 10k/20k 继续恶化，随后 50k～200k 出现真实但有限的晚期恢复；最终 200k 为 `H@10=0.02745607`、`NDCG@10=0.01343658`，相对 2k 峰值仍低 `4.81%/10.87%`，论文达成率仅 `42.37%/34.99%`。因此弱命题“长程训练可能从中期低谷恢复”成立，但强命题“200k 会进入显著更好、足以追平论文的局部最优”被否定。训练长度方向现已充分冻结。

截至 2026-09-01，公开官方来源仍未提供原 TIGER 作者代码、T5X/SeqIO gin、官方 Semantic ID/checkpoint、exact user hash、beam 或 LSH collision 合同。当前没有获准的 GPU 实验；下一次启动必须满足第 11 节的新证据条件。

## 3. 不可偏离的论文与语义事实

### 3.1 RQ-VAE 与第四 token

- 论文 RQ-VAE 只有 **3 层** residual quantization，每层 codebook `K=256`，latent dimension `32`。
- 第四 token 是训练完成后的 deterministic collision-resolution suffix，不是第四个可训练 codebook。
- 论文固定完整 Semantic ID 为 4 个位置，每个位置 cardinality 256。
- 不能把 collision suffix 描述成“第四层 RQ-VAE”。

### 3.2 论文 Beauty 关键指标

```text
RQ-VAE: Recall@10=0.0648, NDCG@10=0.0384
LSH:    Recall@10=0.0533, NDCG@10=0.0309
Random: Recall@10=0.0434, NDCG@10=0.0250
```

### 3.3 论文 decoder 公开配置

- 4 层 encoder + 4 层 decoder；
- 6 个 attention heads，每个 head dimension 64；
- MLP 1024；
- dropout 0.1；
- batch size 256；
- 2000 user tokens + Hashing Trick；
- Beauty 训练 200k steps；
- 前 10k steps learning rate 0.01，之后 inverse-square-root decay；
- TeX 同时写了“input dimension 128”、`6×64=384` 和约 13M 参数，三者存在未解决歧义。

### 3.4 数据切分

论文明确：

- 最后一个 interaction 用于 test；
- 倒数第二个 interaction 用于 validation；
- 其余 interaction 用于 training；
- 训练 history 最长 20。

本地默认 sampler 会将 validation target 追加回可采样训练序列；真实数据审计测得 validation target 被作为训练目标的平均概率为 `0.88174275`。这解释了本地高分的一部分，但关闭该暴露后性能明显下降。

## 4. 数据与内容特征事实

### 4.1 `amazon-p5` 与 `amazon-p5-st5`

| 属性 | `dataset/amazon-p5` | `dataset/amazon-p5-st5` |
|---|---:|---:|
| item 数 | 12,101 | 12,101 |
| user/interaction 序列 | 与 st5 完全一致 | 与 p5 完全一致 |
| item text | `p5 beauty smoke item {i}` | 真实 Title/Brand/Categories/Price |
| feature norm mean | 27.70944 | 1.00000 |
| 内容语义有效性 | 无效 | 有效 |

结论：

- `amazon-p5` 是占位 smoke artifact，不是 Beauty 内容语义数据。
- `amazon-p5-st5` 才是当前有效的真实内容特征路径。
- 旧 `amazon-p5` RQ-VAE/LSH 结果不能用于论文内容语义复现。
- Random-ID 与 feature 无关，因此旧 Random-ID 结果仍可作 decoder/交互协议诊断。

### 4.2 real-content protocol audit

审计文件：

```text
out/audits/tiger_beauty_st5_protocol_audit_20260831.json
```

关键结果：

```text
items/features:                12,101 / [12,101,768]
feature SHA-256:               8c9631129784004195effa417548ba424735c682fb066d40a896996d43aa2fdf
text SHA-256:                  a0aa232f2a35ac83f77d65701ee770ab79fe4fb16ddfbe615e67d3740056b0ef
train/test users:              22,363 / 22,363
same ordered user IDs:         true
test history contract:         100% = last20(train + validation)
train/validation catalog:      12,092 items
unique test-target catalog:    7,978 items
test targets outside train/val catalog: 9 items
```

当前可将未知边界进一步收窄为：

- 本轮官方模型仓库核对确认：当前 `sentence-transformers/sentence-t5-xxl` 由 Transformer、mean pooling、无 bias 的 Dense projection、Normalize 四个模块组成，输出 768 维；这与本地 `p5-st5` 的 768 维、单位范数特征一致。
- 但 processed artifact 没有记录下载时的精确 model revision 或本机 cache 来源，因此仍不能证明它使用了当前官方 revision。
- 作者用于 TIGER 的标题/品牌/类别/价格序列化是否与本地字符串逐字符一致，仍未公开。

## 5. 评估协议隔离

### 5.1 `native_constrained`

- fork 原生生成/评估路径；
- 可能使用 prefix 约束；
- 不能与论文候选 evaluator 直接混比。

### 5.2 `paper_candidate_unconstrained`

当前 canonical 规则：

- 4 token autoregressive deterministic beam；
- 每个位置使用完整 head vocabulary；
- 不使用 valid-prefix mask；
- 完整 ID 映射后统计 invalid；
- 过滤 invalid；
- 对 valid item 去重；
- 再计算 H/Recall@K 与 NDCG@K；
- canonical `beam_size=100`；
- 同时记录 raw top-10 invalid 与 expanded beam invalid。

### 5.3 Temporal-v1 teacher-forced full-catalog

- teacher-forced 全 catalog scorer；
- 与生成式 beam evaluator 不是同一任务；
- 不得放在同一表中声称“复现率”或架构优劣。

## 6. 实验方向记录

### 6.1 evaluator 与 beam/K 审计

- **动机**：确认低指标是否由 invalid ID、beam 太小或 prefix 过滤策略造成。
- **依据**：论文只说明 beam search 和过滤 invalid ID，没有给出精确 beam width 与 tie-break。
- **改动**：实现 `evaluate_tiger_paper_candidate.py`；支持 beam 10/20/50/100/200、多个 K、raw/expanded invalid、valid-item 去重和 underfilled 统计。
- **结果**：代表性旧 checkpoint 在 beam 100 后已基本饱和；beam 100 与 200 的 H@10 相同，top-10 raw invalid 极低。
- **分析**：beam width 不是当前 2 倍以上论文差距的主要原因；继续增大 beam 只增加成本，不能改善模型质量。

### 6.2 paper-strict RQ-VAE 与坍塌诊断

- **动机**：按论文 Adagrad、LR 0.4、batch 1024、3×256、latent32 重建 Semantic ID。
- **依据**：论文明确给出 RQ-VAE 架构与优化器超参数，并要求 codebook usage ≥80%。
- **改动**：构建 paper-strict trainer、fixed-256 collision gate、usage/entropy/unique triple/max bucket 诊断。
- **结果**：最字面 PyTorch 实现立即坍塌：三个层级 usage 均接近 `1/256`，12,101 items 聚到单一/极少 ID，suffix capacity 失败。
- **分析**：有限 loss 不等于健康；作者实现显然包含未公开或框架相关的初始化/优化细节。禁止盲目重跑同一 strict 配方。

### 6.3 历史 collapse-fix 分支事实核查

- **动机**：确认历史上“成功修复 RQ-VAE 坍塌”的机制是否已经合入 `fork-main`。
- **依据**：历史分支 `exp/rqvae-collapse-fix-matrix-20260728` 存在 E1/E2/E3/E4/E5 诊断。
- **改动**：核查 commits `2f259d7`、`4b8a8a8`、`373b000` 与 `fork-main` ancestry，并审计当前默认 trainer。
- **结果**：这些 commits 均不是 `fork-main` ancestors；默认 `train_rqvae.py` 没有 EMA、dead-code reset、Kaiming repair、fixed-256 gate 等关键机制。
- **分析**：历史成功经验不能被视为已合并能力；当前只允许在隔离的 healthy trainer 中移植最小、可测试机制。

### 6.4 healthy-RQ-VAE 基础设施

- **动机**：在不修改默认 trainer 的前提下，建立可阻止长跑浪费的健康门。
- **依据**：历史 strict run 表明只看 loss 会错过严重 codebook 坍塌。
- **改动**：新增 `train_rqvae_healthy.py`、`run_rqvae_healthy.py`、`modules/rqvae_healthy.py`，记录 first-step 梯度、参数变化、usage、unique triples、collision bucket、full-ID uniqueness、tensor-only checkpoint。
- **结果**：支持 Kaiming、EMA、dead-code reset、normalized codebook、历史 epoch permutation、donor sampling 和 RNG policy；全量回归通过。
- **分析**：该 trainer 是“healthy-local-improvement”基础设施，不是论文 exact trainer；默认 `train_rqvae.py` 保持不变。

### 6.5 EMA 系列：E5、标准初始化、corpus bootstrap

- **动机**：验证 EMA 是否能替代梯度更新并稳定 codebook。
- **依据**：历史 E2 使用 EMA，但当前最初实现的 unit-pseudocount 与历史 zero-mass 不同。
- **改动**：依次测试 Kaiming+EMA、标准初始化+EMA、corpus-assignment mass bootstrap、禁用 EMA 的隔离对照。
- **结果**：多条 EMA 路径在完整 epoch 的后期 usage gate 失败；corpus bootstrap 也未解决 step12 失效。
- **分析**：EMA 成败强依赖初始化和内部状态，不能仅 sweep decay/pseudocount；这促成后续 zero-mass 历史源码重建。

### 6.6 corrected E1：Kaiming + per-step reset

- **动机**：寻找当前代码中最稳定、最小的 anti-collapse 基线。
- **依据**：Kaiming 修复 encoder/decoder 几何，dead-code reset 能恢复未使用 code。
- **改动**：Kaiming-ReLU 初始化、AdamW `.001`、warmup50、reset every step、真实 `p5-st5` 内容。
- **结果**：100 epoch/1200 step 全部健康；best epoch55，loss `0.1419834`；1k decoder 得 `0.02660645/0.01443039`。
- **分析**：这是第一个有效内容语义结果，但只有论文 RQ-VAE Recall@10 的 41.06%，说明健康 codebook 不是充分条件。

### 6.7 reset cadence 与 checkpoint selection

- **动机**：判断 reset 过于频繁是否破坏语义结构，以及 train loss 最优 checkpoint 是否更适合下游。
- **依据**：历史 E1 使用 reset50；当前 reset1 的 proxy 与下游未必一致。
- **改动**：测试 reset2/reset4/reset12、epoch36/best-loss/final state、dense decoder checkpoints、early-decay resume。
- **结果**：小幅 proxy 改善没有稳定转化为 candidate 指标；长 decoder 经常在 1k 后回落。
- **分析**：训练 loss、usage 和 collision 只能作为健康门，不能作为最终 checkpoint selector；下游 candidate evaluator 才是选择依据。

### 6.8 decoder learning rate、schedule 与训练长度

- **动机**：解释为什么本地 1k–10k 与论文 200k 差距巨大。
- **依据**：论文公开 LR `.01`/10k 恒定/逆平方根，但作者 optimizer 未公开。
- **改动**：测试 AdamW `.01/.001/.00075/.0005`、early decay、exact resume、1k/2k/3k/10k、多 seed。
- **结果**：真实 historical-E1 在 2k 达峰，3k 回落；`RQ-14` 继续到 5k/10k/20k/50k/100k/200k，对应 H@10 为 `.02884/.02531/.02500/.02710/.02737/.02746`。20k 后存在晚期恢复，但 200k 仍未回到 2k 峰值，且 invalid 明显高于早期点。
- **分析**：用户对“3k 不足以排除长程恢复”的质疑成立，原冻结依据确实过弱；完整 200k 轨迹确认了先降后升，但恢复幅度不足。现在可以基于完整预算而不是相邻两个早期点冻结训练长度方向。

### 6.9 decoder optimizer、batch 与输出结构

- **动机**：隔离 T5X/Adafactor、论文 batch256、shared head、tied output 等实现差异。
- **依据**：论文 T5X 配置未公开，当前 HF T5 参数约 15.33M。
- **改动**：测试 fixed-LR Adafactor、batch256、shared vocabulary head、tied output、scaled tied output、mean loss、separator/no-separator。
- **结果**：旧占位语义诊断中 batch256 和 LR `.01` 组合明显失败；no-separator 相对更好；shared/tied/mean-loss 未形成稳定提升。
- **分析**：这些旧结果只能作 decoder 机制诊断。真实内容 T5X-inspired 组合进一步失败，expanded beam invalid 达 18.19%。

### 6.10 user-token 容量与 hashing

- **动机**：核对用户 embedding 数量应由数据集用户数决定，还是按论文固定 2000。
- **依据**：论文明确使用 2000 user tokens 和 Hashing Trick；Beauty 有 22,363 users。
- **改动**：实现 `paper_fixed`、`dataset_capped`、显式 user bin policy；测试 modulo hash 与 stable non-periodic hash。
- **结果**：论文比较基线固定 2000 bins；stable hash 没有改善本地 Recall@10。
- **分析**：2000 是论文容量设计，不是每个用户独立 embedding；可做 per-user 诊断，但不能替代论文基线。当前 modulo mapping 只能称为容量对齐，尚未证明 hash 函数 exact。

### 6.11 Random-ID 控制

- **动机**：隔离 decoder/交互协议问题与内容语义质量问题。
- **依据**：Random-ID 不依赖 item feature，因此不受 `amazon-p5` 占位内容影响。
- **改动**：实现 deterministic collision-free Random-ID，测试 decoder seed、mapping seed、1k/2k/3k、train-only sampler。
- **结果**：seed29 2k 得 `0.03197245/0.01717991`，达到论文 Random-ID 的 73.67%/68.72%；3k 回落；seed43 复现 2k 优势；train-only 明显下降。
- **分析**：decoder 本身能学习随机 ID，但仍远低于论文；2k 局部峰值和 validation exposure 效应均可复现。停止 Random-ID GPU 工作。

### 6.12 LSH 控制

- **动机**：复现论文 semantic-ID 替代方法，判断 RQ-VAE 优势是否来自内容语义结构。
- **依据**：论文报告 LSH 指标，但没有公开 seed、centering/whitening、碰撞 suffix 合同。
- **改动**：实现 deterministic LSH SimHash、记录 seed/hyperplane、4 token preflight。
- **结果**：占位内容 LSH 指标语义无效；真实 `p5-st5` LSH 只有 9478/12101 unique full IDs，每位观测值 `[83,101,114,75]/256`。
- **分析**：不能自行追加第五 token、丢弃 collision items 或挑 seed。该方向是论文协议歧义，不是简单性能失败。

### 6.13 真实内容 artifact 校正

- **动机**：解释旧 RQ-VAE/LSH 结果为何与论文语义表现不一致。
- **依据**：对 `amazon-p5` 和 `amazon-p5-st5` 的文本、feature hash、norm、同 item cosine 做差分审计。
- **改动**：新增 `validate_non_placeholder_tiger_content`；RQ-VAE/LSH 拒绝 placeholder 内容，Random-ID 允许继续。
- **结果**：确认 `amazon-p5` 为 smoke artifact；所有后续内容语义实验切换到 `amazon-p5-st5`。
- **分析**：这是整个实验计划的关键纠偏。旧 synthetic RQ-VAE 指标不能继续用于论文达成率或 source selection。

### 6.14 historical-E1 真实内容重建

- **动机**：验证历史 reset50/common-init 机制能否在当前分支和真实内容上恢复健康，并提升下游。
- **依据**：历史 E1 初期会严重坍塌，但在 reset50 后长期恢复；不能把前 50 epoch 的 collapse 当成最终失败。
- **改动**：从历史源码重建 epoch permutation、reset50、without-replacement donors、global RNG；不加载不安全 legacy artifact。
- **结果**：500 epoch/6000 step 后 usage `[256,256,256]`，unique triples 11891，max bucket4，full IDs 全唯一；best epoch299。下游 1k/2k/3k 如总表。
- **分析**：这是唯一同时通过真实内容 health 和下游改善的历史机制；2k 是局部峰值，仍只有论文 Recall@10 的 44.51%。

### 6.15 E4 normalized-codebook

- **动机**：检验单位球 codebook 几何能否改善 proxy 和下游语义树。
- **依据**：历史 E4 能保护后两层，但必须与 reset recovery 结合。
- **改动**：真实 `p5-st5`、Kaiming、reset1、`codebook_normalize=True`，先 10 epoch，再 100 epoch export，再单次 decoder。
- **结果**：health 全通过，best state full-ID audit 12101/12101 唯一；1k decoder `0.02736663/0.01460464`，低于 historical-E1 1k。
- **分析**：proxy 更健康不代表推荐更好；E4 下游方向冻结。

### 6.16 historical-E2 zero-mass EMA

- **动机**：重建历史 E2 与当前 unit-pseudocount EMA 的关键差异。
- **依据**：历史源码将 `cluster_size=zeros(K)`，但 `embed_avg` 初始化为 codebook weight。
- **改动**：新增显式 `ema_initialization="zero_mass"`；checkpoint 记录并校验模式；真实内容、reset50、无 warmup、历史顺序。
- **结果**：post-kmeans 健康，但第一步后 usage `[2,9,6]`、unique triples15、max bucket5042、full ID 不唯一。
- **分析**：zero-mass EMA 强依赖不可安全加载的历史 common-init 状态；当前可移植重建失败，禁止继续。

### 6.17 validation 切分兼容性

- **动机**：修正论文“validation 不属于 training”的明确差异。
- **依据**：TeX 数据章节和本地 protocol audit 均给出直接证据。
- **改动**：仅设置 `train_subsample_include_validation_target=False`，test context 仍包含 validation item。
- **结果**：H@10 从 historical-E1 1k 的 `0.02852927` 降至 `0.02110629`，NDCG@10 降至 `0.01118454`。
- **分析**：本地较高结果部分依赖 validation exposure；更忠实的切分在当前 1k recipe 下更差。不能一边保留 exposure，一边声称 exact paper split。

### 6.18 T5X-inspired paper stack

- **动机**：在真实内容和论文切分下，测试公开 batch/LR/schedule 的组合。
- **依据**：TeX 明确 batch256、LR .01、前10k恒定、之后逆平方根；optimizer 未公开。
- **改动**：使用本地 fixed-LR Adafactor 近似、batch256、LR .01、no-validation、`d_model=384`。
- **结果**：H@10 `0.01184993`、NDCG@10 `0.00558019`，expanded invalid `18.1851%`。
- **分析**：该本地近似明显失败，但不能据此否定作者未公开 optimizer/initializer。禁止扩到 10k/200k。

### 6.19 literal HF `d_model=128`

- **动机**：关闭 TeX “input dimension 128” 与 `6×64`/13M 参数的文本冲突。
- **依据**：旧 128 维实验使用了无效 synthetic 内容，需要在真实内容上重做。
- **改动**：historical-E1 1k baseline 仅将 `t5_d_model=384` 改为 `128`。
- **结果**：参数量 5,108,992；H@10 `0.02647230`、NDCG@10 `0.01398943`，均低于 384 维。
- **分析**：literal HF 128 不是本地优选，且参数量远低于论文约 13M；保留架构歧义，不再扩展。

### 6.20 T5X Adafactor CPU 数值兼容门禁

- **动机**：旧 `RQ-07` 使用 HF Transformers fixed-LR Adafactor，不能代表官方 T5X framework-default optimizer；需要先排除优化器数学和 state 语义差异，再决定是否消耗 GPU。
- **依据**：Google Research 官方 T5X commit `2045b332cf19887885a74ef1bd6b2adb2a7ca634` 的 Apache-2.0 `adafactor.py`；默认关键语义为 factored second moment、`1-(step+1)^-0.8` 衰减、参数 RMS 缩放、RMS update clipping、`epsilon1=1e-30`、`epsilon2=1e-3`、最小因子化维度 128、默认无 momentum/weight decay。
- **改动**：新增 `modules/t5x_adafactor.py` 和 `decoder_optimizer='t5x_adafactor_compatible'`；默认 AdamW 与旧 HF Adafactor 路径不变；checkpoint 新增 optimizer 名称并在恢复时校验；新增固定向量/矩阵两步参考、factored/unfactored state shape、state_dict 恢复及 rank 边界测试；新增唯一 GPU 配置 `configs/decoder_tiger_beauty_st5_historical_e1_t5x_adafactor_compatible_seed29_smoke_20260831.gin`。
- **结果**：4/4 optimizer 专项测试通过；全量 `53 passed`；13M 级实际 decoder 有 `15,326,208` 参数，参数秩分布为 74 个矩阵、22 个向量，没有 rank>2 参数；gin 解析确认 batch256、LR `.01`、warmup/恒定段 10k、paper split、weight decay 0、historical-E1 与独立输出目录。
- **分析**：CPU 门禁已经关闭当前 PyTorch HF T5 参数范围内的主要数学/state 风险，满足恢复单一 GPU gate 的条件。但实现有意拒绝 rank>2 scanned/fused 参数，未复刻 JAX logical-axis partitioning，也没有作者 TIGER gin，因此实验标签必须是“**T5X framework-default compatibility approximation**”，不能写成 exact reproduction。

### 6.21 T5X-compatible 1k GPU gate 与 total-2k 判定门禁

- **动机**：检验 CPU 对齐后的 T5X optimizer 是否只是比旧 HF 近似收敛更慢，还是在当前 paper split/historical-E1 上已经触顶。
- **依据**：1k 终态 loss `8.2760`，明显低于旧 HF fixed-LR 近似的约 `13.93`；H@10/NDCG@10 相对旧近似提升 `55.85%/78.62%`，expanded invalid 从 `18.1851%` 降至 `2.5556%`，但相对同切分 AdamW 1k 仍低 `12.50%/10.88%`。该组合支持一次短续训判定，不支持直接跳到论文 200k。
- **改动**：完成 1k real-content gate 和 beam-100 evaluator；checkpoint 确认 `decoder_optimizer=t5x_adafactor_compatible`、optimizer global step `1000`、scheduler `last_epoch=1000`、LR `.01`、validation 不暴露。新增精确恢复配置 `configs/decoder_tiger_beauty_st5_historical_e1_t5x_adafactor_compatible_seed29_resume_to2k_20260831.gin`，只增加 1,000 update。
- **结果**：1k H@10 `0.01846801`、NDCG@10 `0.00996719`，论文达成率 `28.50%/25.96%`；total-2k 降到 H@10 `0.01502482`、NDCG@10 `0.00757808`，达成率 `23.19%/19.73%`。同时 raw top-10 invalid 从 `0.419890%` 升至 `4.680946%`，expanded invalid 从 `2.555561%` 升至 `11.630282%`。
- **分析**：2k 明确否定“只需更多 optimizer update 即可追上”的解释；训练 loss 与候选检索质量发生反向变化，且 invalid 激增，说明 decoder 正在更强地拟合 token 目标却偏离有效完整 ID 流形。optimizer-only 方向已冻结；下一嫌疑应转向 T5X/HF initializer 与模型默认值，而不是继续训练长度或 seed sweep。

### 6.22 官方 T5X initializer 与 tied shared-vocabulary 审计

- **动机**：在 optimizer-only 2k 回落后，判断剩余差距更可能来自初始化分布，还是来自标准 T5X 的共享词表/输出投影结构。
- **依据**：官方 T5X T5-1.0 使用 ReLU、共享 vocabulary、`logits_via_embedding=True`；论文同样明确 ReLU，并称 item semantic codeword 构成 1024-token vocabulary。官方 T5X attention/embedding 初始化尺度与当前 HF T5 基本一致：embedding std 1、K/V/O 与 MLP fan-in 尺度一致、Q 再除 `sqrt(head_dim)`；主要 initializer 差异仅是部分 MLP 的 truncated-normal 尾部，目标方差相同。
- **改动**：不启动低依据 initializer-only GPU 实验；扩展现有 `output_embedding_mode='tied'` 使其支持 `decoder_head_mode='shared_vocab'`，训练和 paper-candidate beam 均可使用完整 1024 item-token logits，native 路径仍可按位置切片。新增 CPU forward、完整/局部 logits 等价、global target offset 与 embedding-tied 数值测试；新增 `configs/decoder_tiger_beauty_st5_historical_e1_t5x_tied_shared_vocab_seed29_smoke_20260831.gin`。
- **结果**：全量 `54 passed`；实际模型参数量从 per-position untied 的 `15,326,208` 降为 `14,932,992`。GPU 1k 得 H@10 `0.01591915`、NDCG@10 `0.00839680`，论文达成率 `24.57%/21.87%`；相对 per-position T5X 1k 下降 `13.80%/15.76%`，raw/expanded invalid 为 `0.458346%/3.127264%`。
- **分析**：直接对齐 shared/tied item vocabulary 没有提升，说明四个独立 head 并非当前主要性能瓶颈；继续加入 2000 个 user output token 只会同时改变更大词表与输入 embedding 语义，当前依据不足。该方向冻结，不续训、不扩 3024-way 输出。

### 6.23 官方 T5X z-loss 门禁

- **动机**：T5X-compatible 2k 出现“cross-entropy loss 继续下降、完整 ID invalid 激增”的反向变化，需要测试官方 T5-1.0 的 logits 正则是否能抑制过度增大的 partition function。
- **依据**：官方 T5X T5-1.0 base gin 明确设置 `Z_LOSS=0.0001`；公式为每个 target token 增加 `z_loss * logsumexp(logits)^2`。当前本地 loss 没有该项。
- **改动**：新增默认关闭的 `decoder_z_loss`；模型按官方公式计算 batch mean，checkpoint/evaluator 记录并校验该值；新增固定 logits 数值测试；配置 `configs/decoder_tiger_beauty_st5_historical_e1_t5x_zloss_seed29_smoke_20260831.gin` 仅在 per-position T5X-compatible 1k baseline 上设置 `0.0001`。
- **结果**：全量 `55 passed`。1k H@10 `0.01855744`、NDCG@10 `0.01001847`；total-2k 降到 H@10 `0.01475652`、NDCG@10 `0.00770087`，论文达成率 `22.77%/20.05%`。相对 1k 下降 `20.48%/23.13%`；raw top-10 invalid 从 `0.394849%` 升到 `4.896928%`，expanded invalid 从 `2.757725%` 升到 `12.311228%`。
- **分析**：z-loss 没有修复 loss 与检索质量背离，也没有抑制 invalid 激增；1k 的约 0.5% 微升属于局部噪声量级，不能作为扩展依据。optimizer/loss/head 的 T5X framework-default 近似全部冻结，后续应回到 RQ-VAE semantic hierarchy、作者未公开配置或安全历史 artifact，而不是继续 decoder 组合调参。

### 6.24 historical-E1 3k→200k 长程局部最优判定（RQ-14）

- **动机**：验证 2k 局部峰值和 3k 回落是否只是早期训练动力学，论文 200k 预算是否可能在更晚阶段进入泛化更好的参数盆地。
- **依据**：`RQ-03/RQ-04` 只覆盖 total-2k/3k；当前 inverse-square-root scheduler 在前 10k steps 保持 LR `.00075`，因此 3k 尚未进入衰减阶段，不能外推 200k。
- **改动**：新增显式 checkpoint milestone 支持；从 `RQ-04` checkpoint 在一个训练进程中连续增加 197k updates，不重启 RNG/数据流，不改变 RQ-VAE、数据暴露、AdamW、LR/scheduler、架构、seed 或 evaluator。只保存 total-5k/10k/20k/50k/100k/200k 六个 checkpoint，约新增 1 GB 制品；训练结束后逐个运行 beam-100 `paper_candidate_unconstrained`。
- **结果**：单进程 197k updates 与六个 beam-100 评估均成功完成，无错误。轨迹为：5k `.028842/.014349`、10k `.025310/.013089`、20k `.024997/.012763`、50k `.027098/.013273`、100k `.027367/.013278`、200k `.027456/.013437`。最终 checkpoint 记录 cumulative optimizer step `200000`、scheduler `last_epoch=200000`、LR `.0001677047`、AdamW、seed `29`、validation exposure 开启。
- **分析**：轨迹明确存在 20k 后的晚期恢复，但没有任何里程碑在 H@10 与 NDCG@10 上同时超过 `RQ-03`；200k 相对 2k 仍低 `4.81%/10.87%`。因此不启动 paper-split 200k：当前结果没有显示长度本身可追平论文，而且本实验的 validation exposure 只会使该结论更偏乐观。训练长度方向冻结。

## 7. 关键代码与测试状态

### 7.1 主要代码

```text
modules/tiger_policy.py
modules/checkpointing.py
modules/tokenizer/semids.py
modules/rqvae_healthy.py
modules/rqvae.py
modules/model.py
modules/t5x_adafactor.py
evaluate/tiger_native.py
evaluate_tiger_paper_candidate.py
train_rqvae_healthy.py
run_rqvae_healthy.py
train_decoder.py
audit_tiger_beauty_protocol.py
run_tiger_decoder_long_horizon_20260831.sh
```

### 7.2 主要测试

```text
tests/test_tiger_policy.py
tests/test_hf_rqvae_loading.py
tests/test_rqvae_healthy.py
tests/test_random_semantic_ids.py
tests/test_lsh_semantic_ids.py
tests/test_train_subsample_policy.py
tests/test_tiger_beauty_protocol_audit.py
tests/test_t5x_adafactor.py
tests/test_decoder_checkpoint_schedule.py
```

最新验证：

```text
57 passed, 16 existing TorchScript deprecation warnings
git diff --check passed
```

### 7.3 新增安全与 provenance 修复

- healthy-RQ-VAE metadata 记录实际 dataset root/name/split，不再硬编码 `amazon-p5`。
- RQ-VAE tensor-only state 使用 `weights_only=True` 加载。
- 不对历史 `common_initialization.pt` 使用 `weights_only=False` 或 allowlist 绕过。
- EMA checkpoint 记录 `unit_pseudocount`/`zero_mass` 初始化模式并拒绝错配。
- decoder 支持显式 `save_model_at` milestone，并在 checkpoint 记录 `cumulative_optimizer_steps`；`RQ-14` 因而能在单进程中只保存六个长程里程碑，避免 RNG/数据流分段重启和约 34 GB 的逐 1k checkpoint。

## 8. 分支、环境与 Git 边界

```text
当前分支: repro/tiger-paper-audit
相对远端: ahead 1
正确 Python: /root/autodl-tmp/recsys-roi-study/external/venvs/rqvae-recommender/bin/python
```

未经用户再次明确授权，不执行：

```text
git commit
git push
git merge
git rebase
git reset
git tag
remote 修改
```

GPU 实验应使用持久 PTY；不要使用会被执行环境回收的普通 `nohup` 后台进程。

## 9. 重要 artifact 索引

### 9.1 论文源码

```text
/root/autodl-fs/arXiv-2305.05065v3.tar.gz
```

### 9.2 当前选择的真实内容 RQ-VAE

```text
out/rqvae/tiger_beauty_st5_historical_E1_common_reset50_epoch500_recovery_20260831/
  model_best_total_loss_state_dict.pt
```

### 9.3 当前选择的 decoder 局部峰值

```text
out/decoder/tiger_beauty_st5_historical_e1_no_sep_seed29_resume_to2k_20260831/
  checkpoint_999.pt
  paper_candidate_beam100_total2k.json
```

### 9.4 corrected E4

```text
out/rqvae/tiger_beauty_st5_E4_normalized_reset_epoch100_export_20260831/
out/decoder/tiger_beauty_st5_e4normalized_no_sep_seed29_smoke_20260831/
```

### 9.5 protocol audit

```text
out/audits/tiger_beauty_st5_protocol_audit_20260831.json
```

### 9.6 RQ-14 长程轨迹

```text
out/decoder/tiger_beauty_st5_historical_e1_no_sep_seed29_longrun_to200k_20260831/
  checkpoint_1999.pt      # total-5k
  checkpoint_6999.pt      # total-10k
  checkpoint_16999.pt     # total-20k
  checkpoint_46999.pt     # total-50k
  checkpoint_96999.pt     # total-100k
  checkpoint_196999.pt    # total-200k
  trajectory.tsv
  paper_candidate_beam100_total*.json
```

## 10. 公开官方来源核对

用户已授权：完成本文档中文化、五字段结构化和宽表后，可以只访问公开的官方作者代码/配置来源进行检索与核对。

### 10.1 本轮核对范围与版本

| 来源 | URL | 本轮 revision / 发布日期 | 许可证 | TIGER 直接配置 |
|---|---|---|---|---|
| `google-research/t5x_retrieval` | `https://github.com/google-research/t5x_retrieval` | 远端 HEAD `e2fbef301f9275a87356220079a7af974a1d9efd`，commit date `2022-12-16`；2026-08-31 复核 | Apache-2.0 | 未发现 |
| `google-research/t5x` | `https://github.com/google-research/t5x` | 远端 HEAD `2045b332cf19887885a74ef1bd6b2adb2a7ca634`，commit date `2026-08-03`；2026-08-31 复核 | Apache-2.0 | 未发现 |
| `sentence-transformers/sentence-t5-xxl` | `https://huggingface.co/sentence-transformers/sentence-t5-xxl` | main revision `97f38f2f69860f732f602829da6df76ee5c4c0f9`，last modified `2025-03-06`；2026-08-31 复核 | Apache-2.0 | 只提供 Sentence-T5 模型 |
| `shashankrajput/P5` | `https://github.com/shashankrajput/P5` | 远端 HEAD `3eb464fa479d1e2e72467ddca3e26fffbcf9422f`，commit date `2023-02-02`；2026-08-31 复核 | MIT | 未发现 TIGER/RQ-VAE/T5X 配置 |
| TIGER 作者公开仓库列表 | `https://github.com/nikhil-dce?tab=repositories`、`https://github.com/shashankrajput?tab=repositories` | 2026-08-31 查询 | 各仓库单独声明 | 未发现名为 TIGER 或论文标题的作者仓库 |
| Nikhil Mehta 官方主页 | `https://nikhil-dce.github.io/` | 2026-08-31 查询 | 页面未声明软件许可 | TIGER 条目只链接论文，没有代码链接 |
| TIGER arXiv v3 | `https://arxiv.org/abs/2305.05065` | v3，`2023-11-03` | arXiv non-exclusive distribution license；非软件许可 | 论文/TeX，不含代码或制品 |
| NeurIPS 2023 官方 Paper 与 Supplemental | `https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html` | NeurIPS 2023 proceedings；2026-08-31 复核 | 论文发布许可；非软件许可 | 有方法/消融细节，无训练代码或 gin |
| 第一作者托管的 TIGER 预印本 | `https://shashankrajput.github.io/Generative.pdf` | 作者静态 PDF；2026-08-31 复核 | 未声明软件许可 | 写明“接收后发布代码与数据集”，但页面没有实际下载地址 |
| Google DeepMind `action_piece` | `https://github.com/google-deepmind/action_piece` | 远端 HEAD `ae8e61a89ade8d545a16119bcb5b3a43d9da852f`，commit date `2025-06-08`；2026-08-31 复核 | 软件 Apache-2.0；其他材料 CC-BY 4.0 | 后续工作代码；仓库中无 TIGER model/config/artifact |
| ActionPiece arXiv v3 | `https://arxiv.org/abs/2502.13581` | v3，`2025-08-15`；ICML 2025 Spotlight | CC-BY 4.0 | 提供后续工作对 TIGER 的二级配置证据，不是原 TIGER 作者 recipe |
| GenRetrieval Research 演讲稿 | `https://nikhil-dce.github.io/data/GenRetrieval%20Research.pdf` | `2026-01-23` | 页面未声明软件许可 | 有 Semantic ID v2 高层方向，无可执行配置或制品 |

核对方法只包括公开页面、公开仓库列表、远端 HEAD 查询、浅克隆后的静态文本搜索；没有执行外部脚本、没有加载外部模型权重，也没有上传任何本地内容。Google DeepMind `action_piece` 的静态审计覆盖 README、三份 YAML、dataset/split、tokenizer、model 与 generation 代码；全仓库没有命中独立的 `TIGER` 文件、模型注册、配置或 Semantic ID 下载制品。

### 10.2 已确认的新证据

#### A. Sentence-T5 preprocessing

- **动机**：缩小本地 `p5-st5` embedding provenance 的未知范围。
- **依据**：官方 T5X Retrieval 与官方 Sentence-Transformers 模型仓库。
- **改动**：只读检查模型 module/config，不重新生成 embedding。
- **结果**：官方 PyTorch 模型依次使用 Transformer、mean-token pooling、`1024→768` 无 bias Dense、Normalize；官方 T5X Retrieval 架构也显式使用 MeanPooling、L2Norm 和 768 维 projection。
- **分析**：本地 768 维、均值范数 1.0 的特征与官方 pipeline 强一致，因此“pooling/normalization 完全未知”可降级为“具体 revision 未记录”。但不能据此证明作者 TIGER 文本序列化逐字符一致。

#### B. T5X optimizer 与 schedule

- **动机**：判断旧 `T5X-inspired` 实验使用的 HF Adafactor 是否足以代表 T5X。
- **依据**：官方 T5X/T5X Retrieval base gin 与 scheduler 源码。
- **改动**：只读核对 optimizer/scheduler，不修改本地训练器。
- **结果**：官方 base 配置使用 T5X `Adafactor(decay_rate=0.8, step_offset=0)` 和 logical factor rules；T5X scheduler 通过 factor string 组合 `constant`、`linear_warmup`、`rsqrt_decay`、`rsqrt_normalized_decay`。本地 Transformers Adafactor 的 `scale_parameter=False/relative_step=False/warmup_init=False` 并不等价于 T5X Adafactor。
- **分析**：这一差异已经形成并完成 `RQ-09`～`RQ-13`；optimizer、tied/shared item head 与 z-loss 均未追平论文，不能继续把 T5X framework-default 当作尚未测试的解释。

#### C. 原 TIGER 官方代码/制品可得性

- **动机**：寻找可直接消除 optimizer、hash、beam、RQ-VAE 与 LSH 歧义的作者配置或制品。
- **依据**：第一作者托管预印本、Google Research 论文页、NeurIPS/OpenReview 官方页面、作者主页与作者公开 GitHub 仓库。
- **改动**：静态搜索 `TIGER`、完整论文标题、`Semantic ID`、`RQ-VAE`、`t5x`、论文 arXiv ID 等关键词；核对公开附件和代码链接。
- **结果**：第一作者托管预印本确实写明“接收后发布代码与数据集”；论文后来被 NeurIPS 2023 接收，但截至 `2026-08-31`，NeurIPS、OpenReview、Google Research 论文页和两位作者公开页面仍没有 TIGER 训练代码、T5X/SeqIO gin、官方 Semantic ID、RQ-VAE checkpoint 或 dataset artifact 下载地址。
- **分析**：“计划发布”不能替代可审计 artifact。第三方复现仓库即使实现完整，也不能用于声称 exact author recipe；当前原作者证据边界仍未突破。

#### D. NeurIPS Supplemental 对停止原因的排除力

- **动机**：判断切分、history、user token 或 invalid-ID 是否足以解释当前约 55% 的 Recall 缺口。
- **依据**：NeurIPS 2023 官方 Supplemental。
- **改动**：只读核对 preprocessing、user-ID 消融、invalid-ID 分析与多次运行方差；不修改代码。
- **结果**：官方补充材料明确按时间排序，最后一项 test、倒数第二项 validation、其余 train，最大 history 为 20；本地 `data/amazon.py` 已使用同一 leave-two-out 与 `max_seq_len=20`。Beauty 的 user-ID 消融从 `H@10=0.06479/NDCG@10=0.0367` 变为 `0.0648/0.0384`，说明 user token 几乎不改变 Recall，主要影响排序。论文对 top-20 beams 报告的 invalid 比例跨数据集/模型约 `0.3%–6%`；该口径与本地 raw top-10/expanded beam 不完全相同。
- **分析**：history 长度不是未测试差异；user-token 实现细节不可能单独解释 Recall 从 `0.02884` 到 `0.0648` 的缺口；invalid-ID 也不是当前最佳点的主瓶颈，因为 `RQ-03` raw/expanded invalid 仅 `0.017887%/0.075884%`。这批证据收窄了原因，但没有产生新的 GPU gate。

#### E. ActionPiece 提供的“后续官方二级证据”

- **动机**：检查 Google DeepMind 后续生成式推荐工作是否公开了 TIGER baseline recipe、Semantic ID 或 checkpoint。
- **依据**：ActionPiece arXiv v3 与 `google-deepmind/action_piece` 官方仓库；该论文包含 Google DeepMind 作者和 TIGER 原论文共同作者 Ed H. Chi。
- **改动**：只读审计论文 Appendix G/H、仓库 README、YAML、dataset/split 和 generation 实现；远端 HEAD 与本地浅克隆 commit 一致。
- **结果**：ActionPiece 论文将 TIGER 描述为 RQ-VAE residual-quantization baseline，并明确其后续实验中的 TIGER beam size 为 `50`；Appendix H 说 Sports/Beauty 的主表 TIGER 数字直接取自原论文，并在后续实现中按 Rajput et al. 使用 `d_model=128`、4+4 层、6 heads、`d_kv=64`。但 ActionPiece 仓库本身只有 ActionPiece 实现，未包含 TIGER model/config、RQ-VAE 训练器、Semantic ID、checkpoint 或数据制品；YAML 中的 PQ、AdamW/cosine、beam-50 是 ActionPiece recipe，不能整体冒充原 TIGER recipe。
- **分析**：这是比第三方复现更强、但仍低于原作者配置的二级证据。`d_model=128` 已由 `RQ-08` 隔离测试且低于 384 维；当前 beam-100 evaluator 对有效候选回填比 beam-50 更宽松，而最佳点 invalid 极低，因此改成 beam-50 不可能补回约 55% Recall 缺口。该来源解决了“后续 Google 实现倾向什么”的问题，却没有提供值得启动新 GPU 的单变量机制。

#### F. 2026 Semantic ID v2 高层方向

- **动机**：判断作者最新公开材料是否给出能改善当前 RQ-VAE semantic hierarchy 的具体机制。
- **依据**：Nikhil Mehta 官方主页链接的 `GenRetrieval Research` 演讲稿，日期 `2026-01-23`。
- **改动**：只读检查 Semantic ID v1/v2 页面，不执行任何实现。
- **结果**：演讲稿提出 multimodal、engagement-aware、multi-resolution codebook、progressive masking 等方向，但没有公开 TIGER Beauty recipe、损失权重、codebook schedule、训练代码、checkpoint 或对照 artifact。
- **分析**：这些方向说明“更好的语义 ID”确实可能是性能突破口，但当前是多因素研究议程，不是可审计的单一 patch；直接自行实现会重新进入无依据 sweep，不能触发 GPU。

### 10.3 已完成并冻结的 T5X bounded gates

官方 T5X Adafactor 与 HF Transformers Adafactor 的差异已经完整闭环，不再是“当前唯一获准 GPU 工作”：

1. **已完成 CPU 实现/门禁**：隔离实现 `t5x_adafactor_compatible`，覆盖全局 step、factored second moment、decay、参数 RMS 缩放、update clipping、epsilon、state 与 checkpoint round-trip；
2. **已完成 1k GPU gate**：`RQ-09` 为 `H@10=0.01846801`、`NDCG@10=0.00996719`；
3. **已完成 total-2k 精确续训**：`RQ-10` 相对 1k 回落 `18.64%/23.97%`，并出现 invalid 激增；
4. **已完成 tied/shared item-vocab**：`RQ-11` 低于 per-position T5X 1k；
5. **已完成 z-loss 1k/2k**：`RQ-12` 仅约 `+0.5%`，`RQ-13` 再次显著回落；
6. **结论**：optimizer、loss、head 三条 T5X framework-default 近似全部冻结；禁止继续 3k/10k、组合 sweep、更多 seed 或扩大输出词表。

### 10.4 仍未解决的检索目标

检索目标按“能否形成单变量 bounded gate”的优先级排序：

1. 原 TIGER 作者官方仓库、T5X/SeqIO gin、训练日志或发布配置；
2. 原作者官方 Beauty Semantic ID、RQ-VAE checkpoint 或可安全读取的 tensor/JSON artifact；
3. exact RQ-VAE 训练机制：初始化、loss 权重、codebook update/reset、checkpoint selection 与 collision suffix；
4. exact Sentence-T5 revision、item 文本序列化与 embedding artifact provenance；
5. exact optimizer factor string、initializer、loss reduction 与有效模型维度；
6. user Hashing Trick 的 hash 函数、bucket 规则和 OOV 语义；
7. 原 TIGER beam width、length normalization、tie-break、invalid filtering；ActionPiece 的 beam-50 只能作为后续二级证据；
8. LSH seed、预处理和 collision-resolution 方法。

外部来源使用规则：

- 只读取公开官方来源；
- 不上传本地代码、数据、日志、模型或内部文档；
- 不执行外部脚本，不加载外部模型权重；
- 记录 URL、commit/tag、发布日期、许可证和关键配置证据；
- 若没有找到官方实现，明确写“未公开/未找到”，不使用第三方猜测替代；
- 只有新证据能形成单一、可审计的 bounded gate 时，才恢复 GPU 实验。

### 10.5 当前停止原因：检索后的最终校准

- **动机**：避免把“GPU 空闲”误判成“应该继续扫参”，也避免后续 Agent 重复已经失败的 T5X/架构实验。
- **依据**：`RQ-01`～`RQ-14`、官方 TIGER Paper/Supplemental、作者公开页面、T5X/T5X Retrieval、ActionPiece 和 2026 作者演讲稿。
- **改动**：完成官方来源静态审计，并把已解决、已排除、仍未知三类问题拆开记录；随后用 `RQ-14` 将训练长度从早期 3k 一次性验证到论文 200k。
- **结果**：已经排除或显著降权的原因包括 history=20、leave-two-out 基本切分、user token 对 Recall 的主导作用、invalid-ID 作为最佳点主瓶颈、T5X optimizer/head/z-loss、literal `d_model=128` 和“仅延长 decoder 到 200k 即可追平”。仍有高解释力但缺证据的核心是：原作者 RQ-VAE/Semantic ID artifact 与训练机制、item embedding 精确 provenance、collision suffix、以及原作者未公开的完整训练 recipe。
- **分析**：`RQ-14` 证明长程训练确有晚期恢复，但不足以超过早期峰值，更不足以追平论文。当前重新回到证据门禁；没有新的官方/安全 artifact 前不继续占用 GPU。

## 11. 恢复实验的判定条件

T5X optimizer 与 `RQ-14` 长程训练两条历史恢复条件都已经消费完毕并冻结，不能再次作为启动理由。当前没有获准 GPU 实验；只有满足以下任一条件，才可再次启动 GPU：

1. 找到原 TIGER 作者官方代码/gin/log，能够提出只改变一个合同的实验；
2. 找到官方 Beauty Semantic ID、RQ-VAE checkpoint、embedding 或其他可校验 artifact，且能在不执行不可信反序列化的前提下读取；
3. 从安全历史 E2 tensor artifact 或官方材料中确认一个尚未测试、可单独实现的 RQ-VAE 语义质量机制，并先通过 CPU health/provenance gate；
4. 找到原作者 collision/hash/beam/LSH 合同，并证明该差异可能改变有效 top-10，而不是只改变已经极低的 invalid 统计。

以下材料**不满足**后续恢复条件：ActionPiece 自身 YAML/optimizer、beam-50 二级证据、Semantic ID v2 高层演讲、第三方 TIGER 复现、更多 seed、继续延长 historical-E1、paper-split 200k 或组合 sweep。否则保持当前停止线，不占用 GPU。
