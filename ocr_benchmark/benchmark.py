#!/usr/bin/env python3
import csv, json, os, time, urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
from rapidocr_newspaper import build_engine

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'sample_60.csv'
IMAGES = ROOT / 'images'
OUT = ROOT / 'out'
ENGINE = None


def engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = build_engine('HQ')
    return ENGINE


def download(ark, page):
    IMAGES.mkdir(exist_ok=True)
    p = IMAGES / f'{ark}_f{page}.jpg'
    if p.exists() and p.stat().st_size > 10000:
        return p
    url = f'https://gallica.bnf.fr/ark:/12148/{ark}/f{page}.highres'
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0','Accept':'image/*,*/*;q=0.8'})
    data = urllib.request.urlopen(req, timeout=90).read()
    if len(data) < 10000 or data[:2] != b'\xff\xd8':
        raise RuntimeError(f'invalid image payload for {ark} f{page}: {len(data)} bytes')
    p.write_bytes(data)
    return p


def process(path):
    t0 = time.time()
    result = engine()(str(path))
    boxes = result.boxes if result.boxes is not None else []
    txts = result.txts if result.txts is not None else []
    scores = result.scores if result.scores is not None else []
    rows = []
    for b, t, s in zip(boxes, txts, scores):
        xs = [float(q[0]) for q in b]
        ys = [float(q[1]) for q in b]
        rows.append({'text':str(t),'score':float(s),'x1':min(xs),'x2':max(xs),'y1':min(ys),'y2':max(ys)})
    OUT.mkdir(exist_ok=True)
    stem = path.stem
    (OUT / f'{stem}.txt').write_text('\n'.join(r['text'] for r in rows) + '\n', encoding='utf-8')
    (OUT / f'{stem}.json').write_text(json.dumps({'source':path.name,'ocr_profile':'HQ','rows':rows}, ensure_ascii=False), encoding='utf-8')
    return stem, time.time() - t0, len(rows)


def main():
    shard = int(os.environ.get('SHARD','0'))
    shards = int(os.environ.get('SHARDS','3'))
    workers = int(os.environ.get('WORKERS','4'))
    with MANIFEST.open(newline='', encoding='utf-8') as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for i, r in enumerate(all_rows) if i % shards == shard]
    print(f'CONFIG shard={shard}/{shards} pages={len(rows)} workers={workers}', flush=True)

    t_download = time.time()
    paths = []
    for i, r in enumerate(rows, 1):
        p = download(r['ark'], r['page'])
        paths.append(p)
        print(f'DOWNLOAD {i}/{len(rows)} {p.name} bytes={p.stat().st_size}', flush=True)
    download_sec = time.time() - t_download

    t_ocr = time.time()
    done, errors, page_secs, row_counts = 0, 0, [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process, p): p for p in paths}
        for fut in as_completed(futures):
            try:
                stem, sec, nrows = fut.result()
                done += 1
                page_secs.append(sec)
                row_counts.append(nrows)
                print(f'OCR {done}/{len(paths)} {stem} sec={sec:.2f} rows={nrows}', flush=True)
            except Exception as e:
                errors += 1
                print(f'ERROR {futures[fut].name}: {type(e).__name__}: {e}', flush=True)
    ocr_sec = time.time() - t_ocr
    metrics = {
        'shard': shard, 'shards': shards, 'workers': workers,
        'pages': len(paths), 'done': done, 'errors': errors,
        'download_sec': download_sec, 'ocr_wall_sec': ocr_sec,
        'pages_per_min': (done / ocr_sec * 60) if ocr_sec else 0,
        'mean_page_worker_sec': sum(page_secs)/len(page_secs) if page_secs else None,
        'mean_rows': sum(row_counts)/len(row_counts) if row_counts else None,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f'metrics_shard_{shard}.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('METRICS ' + json.dumps(metrics), flush=True)
    if errors:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
