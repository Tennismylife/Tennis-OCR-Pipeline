# TML 1904 — Colab-first / VPS-controlled architecture

The VPS is always the source of truth and controller. Colab is disposable compute only.

## Fixed profiles

- RapidOCR A100: 12 CUDA workers, 8 SFTP downloaders. Measured 1903 reference: 72.948 pages/min.
- Layout A100: 4 CUDA workers, 8 SFTP downloaders, PP-DocLayout_plus-L copied from the working VPS model cache. Measured 1903 reference: about 14.55 pages/min.
- ALTO/Gallica: one network stream, at least 16 seconds between requests. ALTO is independent and never gates RapidOCR or Layout.

## Hard rules

1. RapidOCR and Layout use separate Python environments. Do not mix Torch/ONNX and Paddle/NCCL stacks.
2. RapidOCR source images come from VPS by SFTP; Colab RapidOCR makes zero Gallica requests.
3. Layout model files come from `/home/andre/.paddlex/official_models/PP-DocLayout_plus-L` on the VPS, not Hugging Face or ModelScope.
4. GPU workers are preflighted before a large claim is downloaded.
5. Completed JSON/TXT pairs are skipped and never recomputed unless explicitly requested.
6. Every completed result is uploaded to VPS immediately.
7. VPS gates on output completeness, not on whether a Colab process is alive.
8. Claim files are written atomically by VPS and watched by Colab. Colab should stay alive and pick up the next claim instead of idling after one batch.
9. VPS local compute remains a compatible fallback; Colab and VPS outputs share the same output paths and skip completed work.

## Standard year paths

`BASE=/home/andre/GallicaJobs/gallica-${YEAR}-all-tennis/GALlica_${YEAR}_ALL_TENNIS`

- Layout claim: `$BASE/00_MANIFEST/colab_layout_active_claim.tsv`
- Layout completion flag: `$BASE/00_MANIFEST/colab_layout_quality_complete_${YEAR}.flag`
- Rapid claim: `$BASE/00_MANIFEST/colab_active_claims.tsv`

## Reusable components

- `colab/rapid_pool.py`: tuned RapidOCR A100 pool.
- `colab/rapid_watch.py`: watches VPS Rapid claim and automatically runs/retries batches.
- `colab/layout_pool_v3.py`: year-agnostic A100 layout worker with VPS model cache, preflight and live progress.
- `vps/build_colab_layout_claim.py`: generic atomic layout claim builder.
- `vps/wait_colab_layout.py`: generic VPS completeness gate.

For a new year, only the year-specific pipeline decides which OCR/layout stage becomes the active claim. The compute workers remain unchanged.
