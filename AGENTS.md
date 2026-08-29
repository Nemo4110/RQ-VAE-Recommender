# RQ-VAE-Recommender Project Rules

## Project Overview

- Goal: provide a PyTorch semantic-ID generative retrieval implementation based on a three-layer RQ-VAE.
- Domain: sequential recommendation and TIGER/RQ-VAE reproduction auditing.
- Primary framework: PyTorch with gin-config training entry points and Hugging Face T5 components.
- Supported runtime: Python 3.12.3 has been verified in the current VM; GPU training is supported by the existing scripts but is not started by CPU-only checks.
- Deployment target: local training and evaluation; no service deployment is defined in this repository.

## Instruction Precedence

- This project inherits `~/.agents/AGENTS.md` and the applicable files under `~/.agents/rules/`.
- Project-specific changes must preserve the distinction between upstream-compatible behavior, TIGER paper-aligned behavior, and diagnostic experiment infrastructure.

## Project-Specific Rule Selection

- Always applicable user rules: `~/.agents/rules/code-mode.md`, `~/.agents/rules/python.md`, `~/.agents/rules/pytorch.md`, `~/.agents/rules/data-pipeline.md`, `~/.agents/rules/model-persistence.md`, `~/.agents/rules/experiment-tracking.md`, `~/.agents/rules/hardware.md`.
- Additional project or subdirectory rules: none verified.

## Environment

- Package manager: `requirements.txt` is the repository dependency manifest; Python 3.12.3 is verified.
- Environment: use `/root/autodl-tmp/recsys-roi-study/external/venvs/rqvae-recommender/bin/python` and its matching `pip`/`pytest`; this environment was verified to contain the project test/runtime dependencies.
- Environment creation: not defined in repository files.
- Dependency synchronization: README documents `pip install -r requirements.txt`; do not switch to the base Miniconda interpreter for project validation.
- Hardware requirements: CPU-only static/unit checks must not start training; GPU experiments require an explicitly available and idle GPU runtime.
- Required configuration names: gin configuration files under `configs/`.

## Project Structure

- `modules/`: RQ-VAE, tokenizer, retrieval model, and supporting model code.
- `data/`: datasets, preprocessing, processed batches, and schemas.
- `tests/`: pytest regression tests.
- `configs/`: gin training configurations.
- `train_rqvae.py`: RQ-VAE training entry point.
- `train_decoder.py`: decoder training and generation/evaluation entry point.
- `evaluate/`: metric accumulators.
- `docs/`: branch disposition and handoff/audit documentation.

## Commands

- Focused test command: `/root/autodl-tmp/recsys-roi-study/external/venvs/rqvae-recommender/bin/pytest -q tests/test_tiger_policy.py tests/test_hf_rqvae_loading.py`.
- CPU syntax check: `/root/autodl-tmp/recsys-roi-study/external/venvs/rqvae-recommender/bin/python -m compileall -q modules evaluate train_decoder.py tests`.
- Full test suite: `/root/autodl-tmp/recsys-roi-study/external/venvs/rqvae-recommender/bin/pytest -q`.
- Lint, type check, smoke test, and build commands are not defined or verified in repository files.

## Training and Evaluation

- Training entry points: `python train_rqvae.py <gin-config>` and `python train_decoder.py <gin-config>`.
- Configuration source: gin files under `configs/`.
- Primary metrics: the decoder uses Recall/NDCG-style top-k evaluation through `evaluate/metrics.py`.
- Smoke-test limits: CPU-only tests and import/signature/config preflight checks; do not launch training as a smoke test.
- Distributed launch: existing training scripts use `accelerate`; no project-specific launch command is verified here.

## Experiment Tracking

- Tool: optional local/W&B logging through the existing training script.
- Required artifacts for audited TIGER runs: resolved configuration, upstream/fork commit identifiers, token policy, user-token policy, evaluator policy, and checkpoint/artifact identifiers.
- Offline behavior: do not initiate external W&B or model-hub access during CPU-only validation.

## Data and Checkpoints

- Data source/version: configured by existing dataset paths and gin files; no single repository-wide version identifier is defined.
- Split policy: use the existing dataset classes and configuration; do not silently change splits in an audit patch.
- Schema reference: `data/schemas.py`.
- Checkpoint location: configured by training arguments; do not add credentials or machine-specific absolute paths.
- Loading compatibility: preserve explicit architecture checks and safe `map_location` behavior for existing checkpoint loading.

## Project-Specific Permission Deltas

- Allowed without prompting: reversible documentation/code/test changes in this repository and CPU-only focused validation.
- Ask first: dependency changes, deletion or replacement of experiment/checkpoint artifacts, external services, GPU experiment launches, and Git history/remote operations.
- Never: claim native TIGER and teacher-forced scorer results are comparable without an audit; describe the collision suffix as a fourth trainable RQ-VAE layer; upload secrets, PII, or internal artifacts.
