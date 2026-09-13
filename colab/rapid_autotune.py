#!/usr/bin/env python3
import argparse, json, os, subprocess, threading, time, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from worker import read_manifest, pg, connect_sftp, build_engine, ocr_image

ENGINE=None

def init_engine(profile, ready_dir):
    global ENGINE
    ENGINE=build_engine(profile,'cuda')
    Path(ready_dir, f'ready_{os.getpid()}').write_text('1')

def bench_one(path):
    t0=time.time(); rows=ocr_image(path,ENGINE); return time.time()-t0, len(rows)

def sample_gpu(stop, bucket):
    while not stop.is_set():
        try:
            out=subprocess.check_output([
                'nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw',
                '--format=csv,noheader,nounits'
            ], text=True, timeout=3).strip().split(',')
            bucket.append((float(out[0]),float(out[1]),float(out[2])))
        except Exception:
            pass
        stop.wait(0.5)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True); ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True)
    ap.add_argument('--vps-key-b64',required=True); ap.add_argument('--vps-port',type=int,default=2222)
    ap.add_argument('--profile',default='HQ'); ap.add_argument('--sample-pages',type=int,default=32)
    ap.add_argument('--candidates',default='4,8,12,16'); ap.add_argument('--downloaders',type=int,default=8)
    ap.add_argument('--workdir',default='/content/tml_autotune'); ap.add_argument('--out',default='/content/tml_autotune_result.json')
    a=ap.parse_args(); rows=read_manifest(a.manifest)
    rapid=[r for r in rows if (r.get('mode') or '').upper()=='RAPID' and (r.get('source_image') or '').strip()]
    if not rapid: raise SystemExit('No RAPID rows to benchmark')
    n=min(a.sample_pages,len(rapid)); idx=[round(i*(len(rapid)-1)/max(1,n-1)) for i in range(n)]
    sample=[rapid[i] for i in idx]
    wd=Path(a.workdir); imgdir=wd/'images'; imgdir.mkdir(parents=True,exist_ok=True)
    print(f'AUTOTUNE_START rapid_rows={len(rapid)} sample={len(sample)} cpu_count={os.cpu_count()} candidates={a.candidates}',flush=True)

    tls=threading.local()
    def get_sftp():
        if not hasattr(tls,'sftp'):
            tls.tr,tls.sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
        return tls.sftp
    def fetch(r):
        ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); dst=imgdir/f'{ark}_f{page}.jpg'
        if not dst.exists() or dst.stat().st_size<10000:
            get_sftp().get((r.get('source_image') or '').strip(),str(dst))
        return str(dst)
    with ThreadPoolExecutor(max_workers=max(1,a.downloaders)) as ex:
        paths=list(ex.map(fetch,sample))
    print(f'AUTOTUNE_IMAGES_READY n={len(paths)}',flush=True)

    raw=[int(x) for x in a.candidates.split(',') if x.strip()]
    cpu=os.cpu_count() or 8
    raw += [max(2,cpu//2), cpu]
    candidates=sorted({x for x in raw if 2<=x<=16})
    results=[]; ctx=mp.get_context('spawn')
    for workers in candidates:
        ready=wd/f'ready_{workers}'
        if ready.exists():
            for p in ready.glob('*'): p.unlink()
        ready.mkdir(exist_ok=True)
        print(f'AUTOTUNE_CASE workers={workers} initializing...',flush=True)
        with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_engine,initargs=(a.profile,str(ready))) as pool:
            # Force pool startup. Initializers create ready files.
            warm=[pool.submit(time.sleep,0.8) for _ in range(workers*2)]
            deadline=time.time()+180
            while time.time()<deadline and len(list(ready.glob('ready_*')))<workers:
                time.sleep(0.25)
            for f in warm: f.result()
            ready_n=len(list(ready.glob('ready_*')))
            print(f'AUTOTUNE_CASE workers={workers} ready={ready_n}',flush=True)
            gpu=[]; stop=threading.Event(); mon=threading.Thread(target=sample_gpu,args=(stop,gpu),daemon=True); mon.start()
            t0=time.time(); fut=[pool.submit(bench_one,p) for p in paths]
            page_times=[]; rows_total=0
            for f in as_completed(fut):
                sec,nrows=f.result(); page_times.append(sec); rows_total+=nrows
            elapsed=time.time()-t0; stop.set(); mon.join(timeout=2)
        ppm=len(paths)*60/max(elapsed,0.001)
        util=sum(x[0] for x in gpu)/len(gpu) if gpu else 0
        mem=max((x[1] for x in gpu),default=0)
        power=sum(x[2] for x in gpu)/len(gpu) if gpu else 0
        rec={'workers':workers,'pages':len(paths),'elapsed_sec':round(elapsed,2),'pages_per_min':round(ppm,3),
             'avg_page_worker_sec':round(sum(page_times)/max(1,len(page_times)),2),'avg_gpu_util_pct':round(util,1),
             'max_vram_mib':round(mem,1),'avg_power_w':round(power,1),'ready_workers':ready_n,'rows_total':rows_total}
        results.append(rec); print('AUTOTUNE_RESULT '+json.dumps(rec),flush=True)
    best_ppm=max(r['pages_per_min'] for r in results)
    near=[r for r in results if r['pages_per_min']>=best_ppm*0.97]
    best=min(near,key=lambda r:r['workers'])
    out={'best_workers':best['workers'],'best_pages_per_min':best['pages_per_min'],'results':results,'sample_pages':len(paths),'cpu_count':cpu}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('AUTOTUNE_BEST '+json.dumps(out),flush=True)

if __name__=='__main__': main()
