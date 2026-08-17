#!/usr/bin/bash
# step-reduction 对照 sweep のドライバ (断点续跑対応)。
#
#   bash launch/run_steps_bf16_sweep.sh [vit_size ...]     # 既定: large
#
# step ∈ {32,24,20,16} × config ∈ {baseline, N2} を回す。
#   baseline 列 = 「cache を使わず step を減らすだけ」の FID-延迟曲线
#   N2 列      = 各 step の上に layer cache を重ねた場合 (直交性の検証)
# step=32 の 2 行は既存の layercache_bf16_sweep と同一設定なのでそこから転記され、
# 再計算はしない (grep で既存行を判定するため、手で書いておけばスキップされる)。
cd /work/q-li/Halton-MaskGIT || exit 1

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(large)
STEPS=(32 24 20 16)
CONFIGS=(baseline N2)

for VIT in "${MODELS[@]}"; do
  SWEEP="results/halton_${VIT}384_steps_bf16_sweep.txt"
  if [ ! -f "$SWEEP" ]; then
    bash launch/_write_steps_header.sh "$VIT" > "$SWEEP"
  fi
  for STEP in "${STEPS[@]}"; do
    for CFG in "${CONFIGS[@]}"; do
      if grep -qE "^step${STEP}[[:space:]]+${CFG}[[:space:]]" "$SWEEP"; then
        echo "[skip] ${VIT}/step${STEP}/${CFG} — already in $SWEEP"
        continue
      fi
      echo "===== [$(date '+%F %T')] ${VIT}/step${STEP}/${CFG} ====="
      if ! bash launch/eval_fid_steps_bf16.sh "$VIT" "$STEP" "$CFG"; then
        echo "[FAIL] ${VIT}/step${STEP}/${CFG} — 続行 (再実行で再開可能)"
      fi
    done
  done
done

echo "[$(date '+%F %T')] STEPS-SWEEP-ALLDONE"
