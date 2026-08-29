---
document_status: active
document_type: agent-handoff
date: 2026-08-29
scope: tiger-only
base_branch: fork-main
---

# TIGER audit handoff

## 1. Non-negotiable direction

本会话只推进与 TIGER 相关的工作。不要把主仓库中 ML-20M SASRec、HSTU、统计加固或服务成本实验带入当前 fork 的行动主线；它们属于主研究仓库的其他路线。

当前目标不是把所有实验分支合并进 `fork-main`，也不是立即启动新的全量 TIGER campaign。当前目标是关闭 TIGER 的实现协议缺口：

```text
三层 RQ-VAE + 训练后 collision-resolution suffix
→ 显式 decoder token policy
→ paper-fixed 或 dataset-capped user bins（协议必须显式）
→ fork 原生 evaluator compatibility audit
→ Beauty native evaluator 复核
→ 再决定是否扩展 Sports/Toys、更新 PNG 和合并公共能力
```

## 2. 已确认的证据边界

主研究仓库 `docs/experiment_gap_analysis.md`（2026-08-29）给出的 TIGER 结论：

- Beauty v4/v5、Sports v4、Toys v4 的 TIGER Temporal-v1 运行已有 `45/45` registry completion 和 `summary.json`/`summary.md`。
- 这些运行使用 `teacher-forced full-catalog scorer`，不是 fork/论文原生的 autoregressive semantic-ID generation + constrained beam evaluator。
- 现有结果可以保留为本地 teacher-forced scorer 下的条件化曲线和 diagnostic/modified third-party reproduction。
- 现有结果不能直接称为 fork 原生 TIGER Recall/NDCG，不能和 README/论文表格直接横比，也不能在 evaluator 等价性审计前进入跨模型质量/成本排名图。
- 现有 PNG 不包含 TIGER 是已知绘图集成缺口，不代表 TIGER campaign 没有运行。
- 早期 author-main HF 10k、p5/st5 未确认显式启用 `num_user_bins=2000`，只能作为 diagnostic。
- 四 token、D1/D3/D4 Adafactor 以及 Temporal-v1 v4/v5 显式设置了 `num_user_bins=2000`，但当前实现是对 user ID 取模的 modulo-hashed bucket，不能无条件声称 exact paper Hashing Trick reproduction。
- `num_user_bins=2000` 只属于 `paper_aligned + paper_fixed` 的固定论文协议；泛化/审计变体必须允许 `dataset_capped`，按 `min(user_bin_cap, dataset_user_count)` 解析有效上限，并在 metadata 中记录实际值。

参考：

- `~/recsys-roi-study/docs/experiment_gap_analysis.md:121-232`
- `docs/fork_branch_disposition_20260829.md:89-113,131-182,224-266`

## 3. RQ-VAE / Semantic ID 语义边界

论文对齐的准确解释必须保持：

```text
RQ-VAE: 3 trainable residual-quantization layers
full Semantic ID: 3 RQ-VAE tokens + post-training deterministic collision suffix
```

第四个位置：

- 不是第四层 RQ-VAE；
- 不是第四个可训练 codebook；
- 不参与 RQ-VAE loss；
- 在 RQ-VAE 训练结束后，根据 collision bucket 的稳定顺序确定性追加；
- 需要记录 collision bucket 稳定排序、最大 bucket、suffix cardinality 和完整 ID 唯一性。

当前 fork 的 tokenizer 已有此行为的雏形：`modules/tokenizer/semids.py` 中 `n_layers=3`，`precompute_corpus_ids()` 计算重复计数并追加一列。但接口和命名仍需显式化。

禁止把“四 token”描述成“第四层 RQ-VAE 修复”。E1-E5 只属于三层 RQ-VAE codebook collapse/usage/长期漂移实验，保留为证据，不合入默认路径。

## 4. 当前代码审计重点

当前基线 `fork-main` tip 在本会话开始时为 `70be388`，工作区干净，已包含 `824c3b0` local Hub checkpoint loader。

重点审计位置：

- `train_decoder.py:53`：默认 `vae_n_layers=3`；
- `train_decoder.py:65`：默认 `num_user_bins=None`；
- `train_decoder.py:133`：当前以 `tokenizer.cached_ids[:, :vae_n_layers]` 构造 decoder codebooks；
- `train_decoder.py:239`：当前 evaluation target 以 `sem_ids_fut[:, :vae_n_layers]` 截断；
- `modules/tokenizer/semids.py:103`：`sem_ids_dim = n_layers + 1`；
- `modules/tokenizer/semids.py:122-140`：训练后 collision/dedup 列追加；
- `modules/model.py:26-45`：当前 `_strip_dedup_col()` 是隐式丢弃 suffix 的旧路径；
- `modules/model.py:201-204`：user ID 通过 remainder 映射到 user embedding bucket；
- `modules/model.py:394-410`：当前生成入口默认只生成 `num_hierarchies` 个 token。

这些位置不能直接机械改成四层。必须先引入显式 token policy，并保持 historical/native 默认行为兼容。

## 5. 代码修改拆分

优先拆成三个可审查的独立逻辑变更（不直接 cherry-pick `4a6772d`）：

### A. `fix: clarify three-layer rqvae suffix semantics`

目标：为三层 RQ-VAE 和训练后 suffix 建立显式、可测试的接口和命名。

要求：

- 保持 RQ-VAE `n_layers=3`；
- 明确前三层 codebook cardinality 与 suffix cardinality 的不同来源；
- 提供稳定的 collision suffix/完整 Semantic ID 结构化处理；
- 校验 collision bucket 排序和完整 ID 唯一性；
- 不把 suffix 送入 RQ-VAE quantizer 训练。

### B. `feat: make tiger token policy explicit`

至少支持：

```text
historical/native  -> 三 token decoder
paper/full-id      -> 三 token + collision suffix decoder
```

一个 policy 必须同时决定：

- tokenizer output；
- encoder input；
- decoder target；
- generation length；
- valid-prefix lookup；
- invalid-ID accounting；
- evaluator target。

禁止通过 `[:, :vae_n_layers]` 静默决定协议。若 historical/native 需要三 token，必须通过有名称的 policy helper 显式选择；若 paper/full-id 需要完整 ID，model 的 token cardinalities、embedding offsets、output heads 和 generation 都必须使用完整 token policy。

### C. `fix: require explicit user-bin policy`

user bins 必须区分两种协议：

```text
paper_fixed:
    paper_aligned = True
    num_user_bins = 2000

dataset_capped:
    num_user_bins = min(user_bin_cap, dataset_user_count)
```

要求：

- `paper_aligned=True` 时必须使用 `paper_fixed` 和 `num_user_bins=2000`；
- 泛化/审计变体可以使用 `dataset_capped`，默认 cap 为 2,000，但有效值按数据集用户数解析；
- dataset-capped 结果不能写成 exact paper fixed-bin reproduction；
- summary/config metadata 记录 `user_token_policy`、`user_bin_mode`、`user_bin_cap`、`dataset_user_count`、有效 bin 数和 raw/indexed user ID 映射；
- 当前 modulo 实现只能标为 modulo-hashed bucket，不得宣称 exact hash reproduction。

## 6. 评估协议隔离

必须保留两条独立 evaluator/scorer 线：

1. **Temporal-v1 adapter**：teacher-forced full-catalog scorer，继续作为现有条件化曲线证据；
2. **fork/paper native line**：autoregressive semantic-ID generation、constrained beam、valid-prefix、item-level mapping、invalid-ID/tie-break 和原生 Recall/NDCG。

两条线永不混比。summary、artifact、图和结论必须标出 scorer/evaluator policy。

第一组新结果只做 Beauty。Beauty native evaluator 通过前，不启动 Sports/Toys native 全量复核，不把 TIGER 加入跨模型排名 PNG，不重新解释旧 summary。

## 7. 最小测试门禁

CPU 可执行的 focused tests 至少覆盖：

- 三层 RQ-VAE 约束；
- suffix 是训练后追加列而非第四 codebook；
- collision bucket 稳定排序；
- 完整四 token 唯一性；
- historical/native 与 paper/full-id target 选择不同且显式；
- generation length 与 token policy 一致；
- `paper_aligned` 固定协议必须是 `user_bin_mode=paper_fixed` + `num_user_bins=2000`；若采用 dataset-capped，则必须明确标记为非 exact paper fixed-bin 变体；
- modulo user bucket 的索引稳定；
- native evaluator 与 teacher-forced adapter policy 元数据隔离；
- 旧 checkpoint/config 在兼容模式安全加载或明确失败。

不得为了测试启动训练、下载模型、访问 W&B/Hugging Face 外部服务或占用 GPU。

## 8. 何时可以扩展/合并

Beauty native evaluator 通过以下门禁后，才考虑 Sports/Toys、PNG 和公共能力合并：

- 三层 RQ-VAE 与训练后 suffix 证据完整；
- 3-token/4-token decoder policy 显式且端到端一致；
- 2,000 user bins 状态和 modulo/hash 限制已记录；
- native evaluator 的 target、beam、valid-prefix、invalid-ID、item-level mapping 和 tie-break 有 focused tests；
- native 与 teacher-forced 两条线分开产出；
- summary 记录 upstream commit、fork commit、patch digest、config 和 evaluator；
- 旧 diagnostic artifacts 未被覆盖；
- 不能把结果写成跨架构普遍优越或 official reproduction，除非证据另行解锁。

Adafactor 必须等 token policy 完成后重新抽取，保持 AdamW 默认；E1-E5/diagnostic trainer 不进入 fork-main 默认路径。

## 9. GPU 停止点

在当前 VM GPU 正在跑实验期间：

- 只进行文档、代码、CPU unit tests、静态检查、导入检查和不加载真实数据的模型构造测试；
- 不启动 `train_rqvae.py` 或 `train_decoder.py`；
- 不启动 Beauty native evaluator 的真实数据/模型运行；
- 不访问外部服务；
- 当下一步需要真实 RQ-VAE checkpoint、真实 Beauty 数据或 GPU decoder/native evaluator 运行时，暂停并汇报，而不是影响现有 GPU 任务。

## 10. Handoff 完成定义

下一个 agent 继续时，先读取本文件，再检查：

```bash
git status --short --branch
sed -n '1,260p' docs/HANDOFF.md
```

然后只沿 TIGER audit 路线推进，不把未合并实验分支误当作稳定实现，不把现有 teacher-forced TIGER 结果误当作 native evaluator 结果。

## 11. 本轮已完成的 CPU-only 修改

截至 2026-08-29，本轮已完成并提交到 `repro/tiger-paper-audit` 的修改：

- 新增 `modules/tiger_policy.py`：token policy、user-bin preflight、token cardinality、suffix 唯一性和 checkpoint policy 校验；
- 新增 `evaluate/tiger_native.py`：完整 Semantic ID 到 frozen catalog item 的 native 映射、invalid-ID 统计和 item-level accumulator；明确不实现 Temporal-v1 teacher-forced scorer；
- `modules/model.py` 支持显式 historical/native 与 paper/full-id token width，以及按位置 token cardinality/embedding offset；
- `modules/tokenizer/semids.py` 明确 RQ-VAE 层数与 collision suffix，并提供 collision summary；
- `train_decoder.py` 接入 TIGER policy、paper-fixed/dataset-capped user bins、native evaluator 限定、目标 token 选择、checkpoint policy 校验和元数据；
- 新增 `tests/test_tiger_policy.py`，覆盖纯 policy 逻辑和条件式 CPU model forward；model 测试在缺少 Transformers 时按测试框架约定跳过；
- 新增项目级 `AGENTS.md` 及 `CLAUDE.md -> AGENTS.md`，用于保持后续 agent 方向一致。

提交序列：

- `f7dd2d7 docs: add TIGER audit handoff`
- `6e27291 feat: add TIGER policy contracts`
- `f657fe3 feat: wire TIGER native evaluation`
- `bccba13 test: add TIGER audit regressions`

正确环境是 `/root/autodl-tmp/recsys-roi-study/external/venvs/rqvae-recommender/bin/python`；该环境包含 pytest、transformers、einops、accelerate、gin 和 huggingface_hub。完整 pytest 和 compileall 已通过。
