#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 4 ]; then
  echo "usage: $0 BASE OCR_DIR LAYOUT_DIR STAGE [POLL]" >&2; exit 2
fi
BASE="$1"; OCR="$2"; OUT="$3"; STAGE="$4"; POLL="${5:-10}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
CLAIM="$BASE/00_MANIFEST/colab_layout_active_claim.tsv"
python3 "$ROOT/build_colab_layout_claim.py" --base "$BASE" --ocr-dir "$OCR" --layout-dir "$OUT" --stage "$STAGE" --claim "$CLAIM"
OCR_ABS="$OCR"; OUT_ABS="$OUT"
[[ "$OCR_ABS" = /* ]] || OCR_ABS="$BASE/$OCR_ABS"
[[ "$OUT_ABS" = /* ]] || OUT_ABS="$BASE/$OUT_ABS"
python3 -u "$ROOT/wait_colab_layout.py" --ocr-dir "$OCR_ABS" --layout-dir "$OUT_ABS" --stage "$STAGE" --poll "$POLL"
