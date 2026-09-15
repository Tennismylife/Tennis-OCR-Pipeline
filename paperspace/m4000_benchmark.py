#!/usr/bin/env python3
import argparse, csv, json, os, subprocess, sys, time, urllib.request
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'colab'))
from worker import build_engine, ocr_image  # noqa: E402

ENGINE = None


def init_engine(profile, ready_dir):
    global ENGINE
    ENGINE = build_engine(profile, 'cuda')
    Path(ready_dir, f'ready_{os.getpid()}').write_text('1', encoding='utf-8')


def bench_one(path):
    t0 = time.time()
    rows = ocr_image(path, ENGINE)
    return time.time() - t0, len(rows)


def gpu_snapshot():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'],
            text=True, timeout=5
        ).strip().split(',')
        return {
            'name': out[0].strip(), 'util_pct': float(out[1]), 'mem_used_mib': float(out[2]),
            'mem_total_mib': float(out[3]), 'power_w': float(out[4])
        }
    except Exception as e:
        return {'name': 'unknown', 'error': f'{type(e).__name__}: {e}'}


def read_sample():
    p = ROOT / 'ocr_benchmark' / 'sample_60.csv'
    with p.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def download(ark, page, image_dir):
    image_dir.mkdir(parents=True, exist_ok=True)
    dst = image_dir / f'{ark}_f{page}.jpg'
    if dst.exists() and dst.stat().st_size > 10000:
        return dst
    url = f'https://gallica.bnf.fr/ark:/12148/{ark}/f{page}.highres'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'image/*,*/*;q=0.8'})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read()
    if len(data) < 10000 or data[:2] != b'\xff\xd8':
        raise RuntimeError(f'invalid image payload {ark} f{page} bytes={len(data)}')
    dst.write_bytes(data)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-pages', type=int, default=18)
    ap.add_argument('--candidates', default='1,2,3,4')
    ap.add_argument('--profile', default='HQ', choices=['HQ','STANDARD'])
    ap.add_argument('--workdir', default='/notebooks/tml_m4000_benchmark')
    ap.add_argument('--out', default='/notebooks/tml_m4000_benchmark/result.json')
    a = ap.parse_args()

    snap = gpu_snapshot()
    print('GPU ' + json.dumps(snap), flush=True)
    if 'M4000' not in str(snap.get('name','')).upper():
        print('NOTE expected free M4000, continuing on detected GPU', flush=True)

    rows = read_sample()
    n = min(max(1, a.sample_pages), len(rows))
    idx = [round(i * (len(rows)-1) / max(1, n-1)) for i in range(n)]
    sample = [rows[i] for i in idx]
    wd = Path(a.workdir)
    imgdir = wd / 'images'
    wd.mkdir(parents=True, exist_ok=True)

    print(f'DOWNLOAD_START pages={len(sample)}', flush=True)
    paths = []
    t0 = time.time()
    for i, r in enumerate(sample, 1):
        p = download(r['ark'], r['page'], imgdir)
        paths.append(str(p))
        print(f'DOWNLOAD {i}/{len(sample)} {p.name} bytes={p.stat().st_size}', flush=True)
    print(f'DOWNLOAD_COMPLETE sec={time.time()-t0:.1f}', flush=True)

    candidates = []
    for x in a.candidates.split(','):
        x = x.strip()
        if x:
            candidates.append(int(x))
    candidates = sorted(set(x for x in candidates if 1 <= x <= 8))
    ctx = mp.get_context('spawn')
    results = []

    for workers in candidates:
        ready = wd / f'ready_{workers}'
        ready.mkdir(exist_ok=True)
        for p in ready.glob('*'):
            p.unlink()
        print(f'CASE_START workers={workers}', flush=True)
        case_t0 = time.time()
        page_times = []
        rows_total = 0
        errors = []
        try:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=ctx,
                initializer=init_engine, initargs=(a.profile, str(ready))
            ) as pool:
                warm = [pool.submit(time.sleep, 0.5) for _ in range(workers * 2)]
                deadline = time.time() + 180
                while time.time() < deadline and len(list(ready.glob('ready_*'))) < workers:
                    time.sleep(0.25)
                for f in warm:
                    f.result()
                t_ocr = time.time()
                futs = {pool.submit(bench_one, p): p for p in paths}
                done = 0
                for fut in as_completed(futs):
                    try:
                        sec, nrows = fut.result()
                        done += 1
                        page_times.append(sec)
                        rows_total += nrows
                        if done == 1 or done % 4 == 0 or done == len(paths):
                            print(f'CASE_PROGRESS workers={workers} done={done}/{len(paths)} sec={time.time()-t_ocr:.1f} gpu={json.dumps(gpu_snapshot())}', flush=True)
                    except Exception as e:
                        errors.append(f'{type(e).__name__}: {e}')
                elapsed = time.time() - t_ocr
        except Exception as e:
            elapsed = time.time() - case_t0
            errors.append(f'POOL_{type(e).__name__}: {e}')

        done = len(page_times)
        ppm = done * 60 / elapsed if elapsed and done else 0.0
        rec = {
            'workers': workers, 'pages_requested': len(paths), 'pages_done': done,
            'elapsed_sec': round(elapsed, 2), 'pages_per_min': round(ppm, 3),
            'avg_page_worker_sec': round(sum(page_times)/done, 2) if done else None,
            'rows_total': rows_total, 'errors': errors[:10], 'gpu_end': gpu_snapshot()
        }
        results.append(rec)
        print('CASE_RESULT ' + json.dumps(rec), flush=True)

    valid = [r for r in results if r['pages_done'] == len(paths) and r['pages_per_min'] > 0]
    best = max(valid, key=lambda r: r['pages_per_min']) if valid else None
    payload = {
        'platform': 'Paperspace Gradient Free GPU', 'expected_gpu': 'NVIDIA M4000',
        'profile': a.profile, 'sample_pages': len(paths), 'results': results,
        'best': best, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('BENCHMARK_COMPLETE ' + json.dumps(payload), flush=True)
    if best is None:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
