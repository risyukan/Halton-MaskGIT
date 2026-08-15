#!/usr/bin/bash
# layer cache の bf16 FID sweep 全体 (3 モデル × 8 構成) を順に回すドライバ。
#
#   bash launch/run_layercache_bf16_sweep.sh            # large -> base -> small
#   bash launch/run_layercache_bf16_sweep.sh small base # 指定モデルのみ
#
# 各構成が終わるたびに results/halton_<vit>384_layercache_bf16_sweep.txt へ 1 行
# 追記する。既に同じ構成の行があればスキップするので、途中で落ちても同じコマンドで
# 再開できる (全体で 20 時間規模になるため必須)。
cd /work/q-li/Halton-MaskGIT || exit 1

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(large base small)
CONFIGS=(baseline N1 N2 N3 N4 N5 N6 norefresh)

for VIT in "${MODELS[@]}"; do
  SWEEP="results/halton_${VIT}384_layercache_bf16_sweep.txt"
  if [ ! -f "$SWEEP" ]; then
    bash launch/_write_sweep_header.sh "$VIT" > "$SWEEP"
  fi
  for CFG in "${CONFIGS[@]}"; do
    # 先頭カラム完全一致で既存行を判定 (N1 が N1x に誤マッチしないよう ^CFG[[:space:]])
    if grep -qE "^${CFG}[[:space:]]" "$SWEEP"; then
      echo "[skip] ${VIT}/${CFG} — already in $SWEEP"
      continue
    fi
    echo "===== [$(date '+%F %T')] ${VIT}/${CFG} ====="
    if ! bash launch/eval_fid_layercache_bf16.sh "$VIT" "$CFG"; then
      echo "[FAIL] ${VIT}/${CFG} — 続行します (再実行で再開可能)"
    fi
  done
done

echo "[$(date '+%F %T')] SWEEP-ALLDONE"
