# Colab ALTO + RapidOCR worker

This folder contains a manual Google Colab batch worker for Gallica pages.

## What it does

- Reads a claimed TSV manifest from the VPS over SFTP.
- Uses `mode=ALTO` rows for Gallica ALTO download.
- Uses `mode=RAPID` rows for RapidOCR.
- Falls back from an ALTO row to RapidOCR if usable ALTO is not returned.
- Uploads `.json`, `.txt`, and `.hits.txt` directly to the selected VPS OCR cache after each completed page.
- Optionally uploads native ALTO XML to the shared VPS ALTO cache.
- Re-running the same claim is resumable because remote JSON files are checked before work begins.

## Colab Secrets

Create these secrets in Colab. Never commit them to GitHub:

- `VPS_HOST`
- `VPS_USER`
- `VPS_SSH_KEY_B64` — base64-encoded private SSH key
- `VPS_PORT` — normally `22`
- `VPS_MANIFEST` — absolute path to the claimed TSV on the VPS
- `VPS_REMOTE_CACHE` — absolute final OCR output directory for the current stage
- `VPS_ALTO_CACHE` — optional shared ALTO cache root

## VPS claim coordination

The VPS-side pipeline supports `00_MANIFEST/colab_active_claims.tsv`. When a split is regenerated, pages in that claim file are assigned to the external `COLAB` branch and are excluded from new local ALTO/RapidOCR queues.

Create claims with `SCRIPTS/prepare_colab_claim_1903.py` before regenerating/restarting a local split. This avoids duplicate work between Colab and VPS.

## Notebook

Open `TML_Colab_ALTO_RapidOCR.ipynb` in Google Colab, add the required Secrets, then run the cells in order.
