#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/recsys-roi-study/external/RQ-VAE-Recommender"
PYTHON="/root/autodl-tmp/recsys-roi-study/external/venvs/rqvae-recommender/bin/python"
CONFIG="configs/decoder_tiger_beauty_st5_historical_e1_no_sep_seed29_longrun_to200k_20260831.gin"
OUT="out/decoder/tiger_beauty_st5_historical_e1_no_sep_seed29_longrun_to200k_20260831"
FINAL_CHECKPOINT="${OUT}/checkpoint_196999.pt"

cd "${ROOT}"
mkdir -p "${OUT}"
printf '%s\n' "${PYTHON} train_decoder.py ${CONFIG}" > "${OUT}/command.txt"
printf '%s\n' \
  "out/decoder/tiger_beauty_st5_historical_e1_no_sep_seed29_resume_to3k_20260831/checkpoint_999.pt" \
  > "${OUT}/source_checkpoint.txt"
git rev-parse HEAD > "${OUT}/git_head.txt"
git status --short > "${OUT}/git_status_at_launch.txt"

if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" train_decoder.py "${CONFIG}" \
    2>&1 | tee -a "${OUT}/train.log"
fi

milestones=(5000 10000 20000 50000 100000 200000)
local_steps=(1999 6999 16999 46999 96999 196999)

printf 'total_steps\th_at_10\tndcg_at_10\traw_top10_invalid\texpanded_beam_invalid\n' \
  > "${OUT}/trajectory.tsv"

for index in "${!milestones[@]}"; do
  total="${milestones[$index]}"
  local_step="${local_steps[$index]}"
  checkpoint="${OUT}/checkpoint_${local_step}.pt"
  summary="${OUT}/paper_candidate_beam100_total${total}.json"
  log="${OUT}/paper_candidate_beam100_total${total}.log"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing expected milestone checkpoint: ${checkpoint}" >&2
    exit 1
  fi

  if [[ ! -f "${summary}" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" evaluate_tiger_paper_candidate.py "${CONFIG}" \
      --decoder-checkpoint "${checkpoint}" \
      --output-json "${summary}" \
      --beam-size 100 \
      --batch-size 128 \
      --device cuda \
      2>&1 | tee "${log}"
  fi

  "${PYTHON}" -c '
import json
import sys
summary = json.load(open(sys.argv[2], encoding="utf-8"))
metrics = summary["metrics"]
print(
    sys.argv[1],
    metrics["h@10"],
    metrics["ndcg@10"],
    metrics["invalid_id_rate"],
    metrics["expanded_beam_invalid_id_rate"],
    sep="\t",
)
' "${total}" "${summary}" >> "${OUT}/trajectory.tsv"
done
