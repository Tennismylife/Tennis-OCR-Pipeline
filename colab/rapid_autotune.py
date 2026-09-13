#!/usr/bin/env python3
import argparse, json, os, subprocess, threading, time, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from worker import read_manifest, pg, connect_sftp, build_engine, ocr_image

PRESET_WORKERS=12
PRESET_PPM=72.948
ENGINE=None

def init_engine(profile, ready_dir):
    global ENGINE
    ENGINE=build_engine(profile,'cuda')
    Path(ready_dir, f'ready_{os.getpid()}').write_text('1')

def bench_one(path):
    t0=time.time(); rows=ocr_image(path,ENGINE); return time.time()-t0, len(rows)

def gpu_snapshot():
    try:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'], text=True, timeout=3).strip().split(',')
        return float(out[0]),float(out[1]),float(out[2])
    except Exception:
        return 0.0,0.0,0.0

def sample_gpu(stop, bucket):
    while not stop.is_set():
        bucket.append(gpu_snapshot()); stop.wait(0.5)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True); ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True)
    ap.add_argument('--vps-key-b64',required=True); ap.add_argument('--vps-port',type=int,default=2222)
    ap.add_argument('--profile',default='HQ'); ap.add_argument('--sample-pages',type=int,default=32)
    ap.add_argument('--candidates',default='4,8,12,16'); ap.add_argument('--downloaders',type=int,default=8)
    ap.add_argument('--workdir',default='/content/tml_autotune'); ap.add_argument('--out',default='/content/tml_autotune_result.json')
    ap.add_argument('--force',action='store_true',help='Run a fresh benchmark instead of using the saved A100 profile')
    a=ap.parse_args()
    if not a.force:
        out={'best_workers':PRESET_WORKERS,'best_pages_per_min':PRESET_PPM,'preset':True,'profile':'A100_80GB_RapidOCR_HQ'}
        Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8')
        print(f'AUTOTUNE_PRESET workers={PRESET_WORKERS} pages_per_min={PRESET_PPM} fresh_benchmark=0',flush=True)
        return

    rows=read_manifest(a.manifest)
    rapid=[r for r in rows if (r.get('mode') or '').upper()=='RAPID' and (r.get('source_image') or '').strip()]
    if not rapid: raise SystemExit('No RAPID rows to benchmark')
    n=min(a.sample_pages,len(rapid)); idx=[round(i*(len(rapid)-1)/max(1,n-1)) for i in range(n)]
    sample=[rapid[i] for i in idx]
    wd=Path(a.workdir); imgdir=wd/'images'; imgdir.mkdir(parents=True,exist_ok=True)
    print(f'AUTOTUNE_START rapid_rows={len(rapid)} sample={len(sample)} cpu_count={os.cpu_count()} candidates={a.candidates}',flush=True)
    tls=threading.local()
    def get_sftp():
        if not hasattr(tls,'sftp'): tls.tr,tls.sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
        return tls.sftp
    def fetch(r):
        ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); dst=imgdir/f'{ark}_f{page}.jpg'
        if not dst.exists() or dst.stat().st_size<10000: get_sftp().get((r.get('source_image') or '').strip(),str(dst))
        return str(dst)
    with ThreadPoolExecutor(max_workers=max(1,a.downloaders)) as ex:
        paths=[]
        for i,p in enumerate(ex.map(fetch,sample),1):
            paths.append(p)
            if i==1 or i%8==0 or i==len(sample): print(f'AUTOTUNE_FETCH {i}/{len(sample)}',flush=True)
    raw=[int(x) for x in a.candidates.split(',') if x.strip()]; cpu=os.cpu_count() or 8; raw += [max(2,cpu//2),cpu]
    candidates=sorted({x for x in raw if 2<=x<=16}); results=[]; ctx=mp.get_context('spawn')
    for workers in candidates:
        ready=wd/f'ready_{workers}'
        if ready.exists():
            for p in ready.glob('*'): p.unlink()
        ready.mkdir(exist_ok=True)
        print(f'AUTOTUNE_CASE workers={workers} initializing...',flush=True)
        with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_engine,initargs=(a.profile,str(ready))) as pool:
            warm=[pool.submit(time.sleep,0.8) for _ in range(workers*2)]; deadline=time.time()+180; last=0
            while time.time()<deadline and len(list(ready.glob('ready_*')))<workers:
                if time.time()-last>=5:
                    rn=len(list(ready.glob('ready_*'))); u,m,p=gpu_snapshot(); print(f'AUTOTUNE_INIT workers={workers} ready={rn}/{workers} gpu={u:.0f}% vram={m:.0f}MiB power={p:.0f}W',flush=True); last=time.time()
                time.sleep(0.25)
            for f in warm: f.result()
            gpu=[]; stop=threading.Event(); mon=threading.Thread(target=sample_gpu,args=(stop,gpu),daemon=True); mon.start()
            t0=time.time(); fut=[pool.submit(bench_one,p) for p in paths]; page_times=[]; rows_total=0; done=0
            for f in as_completed(fut):
                sec,nrows=f.result(); page_times.append(sec); rows_total+=nrows; done+=1
                if done==1 or done%8==0 or done==len(paths):
                    u,m,p=gpu_snapshot(); print(f'AUTOTUNE_PROGRESS workers={workers} done={done}/{len(paths)} elapsed={time.time()-t0:.1f}s gpu={u:.0f}% vram={m:.0f}MiB power={p:.0f}W',flush=True)
            elapsed=time.time()-t0; stop.set(); mon.join(timeout=2)
        ppm=len(paths)*60/max(elapsed,0.001); util=sum(x[0] for x in gpu)/len(gpu) if gpu else 0; mem=max((x[1] for x in gpu),default=0); power=sum(x[2] for x in gpu)/len(gpu) if gpu else 0
        rec={'workers':workers,'pages':len(paths),'elapsed_sec':round(elapsed,2),'pages_per_min':round(ppm,3),'avg_page_worker_sec':round(sum(page_times)/max(1,len(page_times)),2),'avg_gpu_util_pct':round(util,1),'max_vram_mib':round(mem,1),'avg_power_w':round(power,1),'rows_total':rows_total}
        results.append(rec); print('AUTOTUNE_RESULT '+json.dumps(rec),flush=True)
    best_ppm=max(r['pages_per_min'] for r in results); near=[r for r in results if r['pages_per_min']>=best_ppm*0.97]; best=min(near,key=lambda r:r['workers'])
    out={'best_workers':best['workers'],'best_pages_per_min':best['pages_per_min'],'results':results,'sample_pages':len(paths),'cpu_count':cpu,'preset':False}
    Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8'); print('AUTOTUNE_BEST '+json.dumps(out),flush=True)

if __name__=='__main__': main()
