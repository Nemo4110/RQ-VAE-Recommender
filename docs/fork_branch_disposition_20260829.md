---
document_status: active
document_type: fork-branch-disposition
date: 2026-08-29
base_ref: origin/main
action_branch: fork-main
---

# RQ-VAE-Recommender fork 分支取舍与合并说明

## 1. 目的与当前操作

本文说明 `Nemo4110/RQ-VAE-Recommender` fork 中各实验分支的用途、证据等级、是否进入 `fork-main` 以及未合并的原因。

本次操作基于本地可见的 `origin/main`:

```text
origin/main: a6732a55338c55e2d31689740ecf2ee917004258
```

从该基线创建了:

```text
fork-main
```

并只合入了低风险、通用的 local Hub RQ-VAE checkpoint loading 变更:

```text
824c3b0 feat: load local Hub RQ-VAE checkpoints
```

本次没有将四 token candidate、paper-strict trainer、collapse diagnostic harness、E1-E5 fix-matrix trainer 或 Adafactor experiment branch 直接合入 `fork-main`。

## 2. 为什么不能把所有实验分支合并

现有分支混合了三类不同内容:

1. **可复用的实现能力**，例如可选的离线 Hugging Face checkpoint 加载和架构校验。
2. **论文复现协议变体**，例如三层 RQ-VAE、训练后 collision-resolution suffix、decoder target policy、user-token policy 和 evaluator policy。
3. **一次性诊断与实验基础设施**，例如 collapse snapshot、common initialization、state hash、checkpoint transaction、resume 验证和 E1-E5 treatment harness。

如果把三类内容全部放进 `fork-main`,会产生以下风险:

- 默认行为被一次性实验处理改变;
- RQ-VAE 三层量化与训练后第四 suffix 被混淆;
- native fork evaluator 与研究仓库的 teacher-forced scorer 被混合;
- 旧 checkpoint 无法判断依赖哪一个 token/optimizer policy;
- 失败实验和诊断代码被误读为稳定实现;
- 后续无法区分 upstream-compatible baseline 与 modified third-party reproduction。

因此 `fork-main` 应保持为稳定、可解释的实现基线,实验分支继续保存可重建的研究谱系。

## 3. 分支处置矩阵

| 分支 | tip | 建议 | 是否进入 `fork-main` | 原因 |
|---|---|---|---|---|
| `origin/main` | `a6732a5` | 作为基线 | 是 | 当前 fork 的主线基准,只有 README DOI 更新相对 upstream pin |
| `origin/repro/main-hf-beauty-20260714` | `e5a03e5` | 合入通用 loader | **是** | loading-only 适配、架构 mismatch 检查和离线测试,不改变 RQ-VAE 层数和默认 optimizer |
| `origin/fix/4token-20260711` | `4a6772d` | 暂缓,拆分后重审 | 否 | 混合 token cardinality、decoder target、beam、item evaluator 等多项行为变化,不能再笼统称为“RQ-VAE 第四层修复” |
| `origin/exp/decoder-adafactor-20260729` | `f14d748` | 保留实验分支,之后抽取 | 否 | 依赖 `4a6772d`; optimizer 改动本身有价值,但当前提交边界和 decoder candidate 绑定 |
| `origin/repro/tiger-paper-strict-beauty-20260714` | `2e968ca` | 保留 paper-strict 实验分支 | 否 | Beauty-specific、包含失败的 paper-protocol RQ-VAE 尝试,不是稳定默认实现 |
| `origin/diag/rqvae-collapse-instrumentation-20260714` | `2f259d7` | 保留诊断分支 | 否 | 大量 collapse metrics、state hash、snapshot 和 diagnostic trainer,不应进入默认训练路径 |
| `origin/diag/rqvae-collapse-checkpoint-20260715` | `4b8a8a8` | 保留 checkpoint safety 分支 | 否 | 面向诊断实验的 checkpoint transaction/resume 加固,不是当前公共 API |
| `origin/exp/rqvae-collapse-fix-matrix-20260728` | `373b000` | 保留 E1-E5 实验分支 | 否 | 包含完整 diagnostic ancestry 和 treatment trainer; E2 结果作为 artifact/实验依据保留,但不改变 fork-main 默认 RQ-VAE |
| `origin/exp/rqvae-collapse-adagrad04-e500-20260714` | `2f259d7` | 保留实验标签 | 否 | 与 AdamW 标签指向相同 instrumentation tip,是运行谱系而非独立公共实现 |
| `origin/exp/rqvae-collapse-adamw1e3-e500-20260714` | `2f259d7` | 保留实验标签 | 否 | 同上 |

## 4. 已合入 `fork-main` 的内容

### 4.1 Local Hub RQ-VAE checkpoint loading

已合入 `e5a03e5`:

- `SemanticIdTokenizer` 支持 legacy checkpoint 或 local Hub checkpoint 二选一;
- local Hub checkpoint 使用 `local_files_only=True`;
- 加载时校验 input dimension、embedding dimension、hidden dimensions、codebook size、layer count 和 categorical feature count;
- 同时提供 legacy loading regression test 和 Hub loading regression test;
- 不改变默认 `num_user_bins=None`;
- 不改变 RQ-VAE 默认 `n_layers=3`;
- 不改变原始 decoder evaluator 或 optimizer 默认值。

这项变更的定位是:

> 可复用的离线 checkpoint loading capability。

它不能被解释为完整 TIGER reproduction fix。

## 5. 三层 RQ-VAE 与第四 suffix 的边界

论文的 RQ-VAE 是三层 residual quantization。前三个 codeword 由三个可训练 RQ-VAE codebook 产生。

多个 item 共享前三个 codeword 时,在 RQ-VAE 训练结束后,通过 collision lookup 为同一 bucket 内的 item 确定性追加 suffix:

```text
(c1, c2, c3) + collision_index
```

该 suffix:

- 不是第四层 RQ-VAE;
- 不是第四个可训练 codebook;
- 不参与 RQ-VAE loss;
- 不应被送入 RQ-VAE quantizer 训练;
- 应记录 collision bucket 的稳定排序、最大 bucket 和 suffix cardinality。

因此后续 fork 修改必须保留:

```text
RQ-VAE: n_layers = 3
```

如果 decoder 需要预测完整四 token Semantic ID,应把它作为独立的 decoder token policy,而不是将 RQ-VAE 改成四层。

## 6. 为什么暂不合并 `4a6772d`

`4a6772d` 实际同时修改了:

- valid-prefix index;
- token cardinalities;
- vocabulary offsets;
- decoder hierarchy count;
- decoder target;
- deterministic beam expansion;
- item-level evaluation;
- invalid-ID 处理;
- full-catalog evaluation。

它的实验价值不应被否定,但它不能作为一个未经拆分的“correctness fix”直接进入 `fork-main`。

后续应拆成至少两个独立变更:

1. `feat: clarify three-layer rqvae suffix semantics`
2. `feat: add explicit decoder token policy`

并显式支持:

```text
historical/native token policy
paper full-id token policy
```

每种 policy 必须统一决定:

- tokenizer output;
- encoder input;
- decoder target;
- generation length;
- valid-prefix lookup;
- invalid-ID accounting;
- evaluator target。

禁止通过 `[:, :vae_n_layers]` 之类的隐式切片静默丢弃 suffix,也禁止把 suffix 当作 RQ-VAE codebook 输出。

## 7. user-token policy

论文要求加入 2,000 个 user-specific tokens,并使用 Hashing Trick 映射 raw user ID。

当前 `fork-main` 合入的 `e5a03e5` 没有修复这一项,默认参数仍可能是:

```python
num_user_bins=None
```

因此 `fork-main` 目前不应宣称已经完成论文 user-token 对齐。

后续建议加入显式 paper mode:

```text
paper_aligned = True
num_user_bins = 2000
```

当 `paper_aligned=True` 时:

- `num_user_bins` 必须等于 2,000;
- 未设置时 preflight 失败;
- summary 记录 `user_token_policy`、`num_user_bins` 和 user ID 映射方式;
- 明确当前实现是 exact hash 还是 modulo bucket;
- 不把早期未启用 user bins 的结果与后期结果合并。

建议默认模式仍保留 upstream-compatible 行为,而不是静默将所有历史运行改成 2,000 user bins。

## 8. Adafactor 的处置

`f14d748` 的 Adafactor 结果有明确实验价值,但不应直接 cherry-pick 到当前 `fork-main`,因为它建立在 `4a6772d` 的 decoder 重构之上。

后续应在完成 token policy 后重新提取 Adafactor:

- AdamW 仍为默认 optimizer;
- Adafactor 必须显式选择;
- checkpoint resume 要区分 AdamW scheduler 与 Adafactor relative-step 状态;
- 增加 optimizer-specific checkpoint 测试;
- 将 HuggingFace Adafactor 标记为 T5X-style approximation,而不是 official T5X reproduction。

## 9. E1-E5 的处置

E1-E5 是 RQ-VAE 训练实验 treatment,不是 fork-main 默认实现。

应保留:

- `origin/exp/rqvae-collapse-fix-matrix-20260728`;
- E1-E5 的配置、summary、snapshot 和失败记录;
- E2 20k epoch checkpoint 的来源、commit 和 hash;
- 主研究仓库中的 evidence 文档和 artifact manifest。

不应直接合入:

- diagnostic trainer;
- collapse state machine;
- checkpoint transaction harness;
- 500 epoch bounded matrix runner;
- E2 训练处理作为无配置保护的默认 codebook update path。

如果未来决定将 E2 作为可复用能力,应重新提取最小实现并配置化:

```text
codebook_update_mode = gradient
codebook_update_mode = ema_reset
```

同时增加训练前后行为、checkpoint compatibility 和恢复轨迹测试。

## 10. 推荐的后续工作流

### 10.1 保持 `fork-main` 稳定

当前 `fork-main` 只包含:

- `origin/main` 基线;
- optional local Hub checkpoint loading;
- 后续经过单独审计的公共能力。

它不包含研究仓库的 Temporal-v1 adapter,因为该 adapter 使用 teacher-forced full-catalog scorer,属于主研究仓库的实验协议。

### 10.2 创建 TIGER candidate branch

建议从最新 `fork-main` 创建:

```bash
git switch -c repro/tiger-paper-audit fork-main
```

在 candidate branch 上完成:

1. 核实三层 RQ-VAE 输出和训练后 suffix;
2. 对 decoder token policy 做显式配置;
3. 核实 fork 原生 evaluator 的实际 token target;
4. 强制 paper mode 使用 2,000 user bins;
5. 对照论文源码的 collision handling、user token、beam 和 invalid-ID 描述;
6. 增加最小 regression tests;
7. 先完成 Beauty bounded/native evaluator 复核;
8. 通过后再决定是否把整理后的公共能力合入 `fork-main`。

### 10.3 分支合并门禁

任何候选功能进入 `fork-main` 前,至少满足:

- 不改变 RQ-VAE 三层定义;
- 没有静默改变历史默认行为;
- token policy、user-token policy 和 evaluator policy 有明确配置;
- 有 focused tests;
- checkpoint 能加载并在兼容模式下安全失败;
- summary 记录 upstream commit、fork commit、配置和 evaluator;
- 不把 diagnostic 结果写成 official reproduction;
- 主研究仓库的 manifest 和 patch digest 同步更新。

## 11. 当前未合并分支的总体结论

未合并不表示这些分支无价值,而是表示它们的价值属于不同层次:

- `e5a03e5`: 通用能力,已进入 `fork-main`;
- `4a6772d`: decoder full-ID candidate,等待 token policy 审计;
- `f14d748`: optimizer experiment,等待依赖整理;
- `2e968ca`: paper-strict 失败/诊断复现,保留为历史证据;
- `2f259d7` / `4b8a8a8`: collapse diagnostics and checkpoint safety,不进入默认路径;
- `373b000`: E1-E5 fix matrix,保留为实验分支和 checkpoint 来源。

这种取舍可以同时保持 fork-main 的可复用性、实验分支的可重建性和论文结论的证据边界。
