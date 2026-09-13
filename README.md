# Tennis OCR Pipeline

Distributed OCR experiments and benchmarks for historical tennis newspaper research.

## Current benchmark

The current workflow reproduces the RapidOCR/ONNX newspaper OCR profile used in the Gallica historical tennis pipeline and runs it on three parallel GitHub-hosted Linux runners.

- 3 shards
- 4 workers per shard
- 12 OCR workers total
- Python 3.12
- RapidOCR 3.9.2
- ONNX Runtime 1.29.0
- HQ newspaper profile

The benchmark downloads a 60-page public sample directly from Gallica, processes the pages independently, records throughput metrics, and uploads short-lived OCR artifacts.

## Scope

This repository contains software, workflows, and small public benchmark manifests only. Large newspaper image collections and private project data are not committed here.

## Run

Open **Actions → OCR Benchmark → Run workflow**, or push changes affecting `ocr_benchmark/` or the benchmark workflow on `main`.
