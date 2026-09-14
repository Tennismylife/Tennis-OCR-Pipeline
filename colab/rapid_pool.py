#!/usr/bin/env python3
import argparse,json,os,sys,subprocess,time,threading,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait,FIRST_COMPLETED
from worker import read_manifest,pg,connect_sftp,build_engine,ocr_image,write_payload,suffix

os.environ.setdefault('CUDA_MODULE_LOADING','LAZY')
ENGINE=None; PROFILE='HQ'; GENERATION='colab_batch_gpu'

def init_engine(profile,generation):
    global ENGINE,PROFILE,GENERATION
    PROFILE=profile; GENERATION=generation
    ENGINE=build_engine(profile,'cuda')
    print(f'OCR_ENGINE_READY pid={os.getpid()} profile={PROFILE} generation={GENERATION}',flush=True)

def ocr_one(item):
    idx,total,r,ip,outdir=item
    ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); stem=f'{ark}_f{page}'; t0=time.time()
    print(f'OCR_START idx={idx}/{total} stem={stem} ark={ark} page={page} pid={os.getpid()}',flush=True)
    rows=ocr_image(ip,ENGINE)
    files=write_payload(outdir,ark,page,rows,PROFILE,(r.get('source_image') or 'COLAB'),'CUDA')
    jp=Path(files[0]); d=json.loads(jp.read_text(encoding='utf-8'))
    d['ocr_generation']=GENERATION; d['gpu_pool']=True
    jp.write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8')
    return idx,total,stem,[str(x) for x in files],time.time()-t0,len(rows)

def gpu_snapshot():
    try:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=3).strip().split(',')
        return float(out[0]),float(out[1]),float(out[2])
    except Exception:return 0.0,0.0,0.0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True)
    ap.add_argument('--vps-key-b64',required=True); ap.add_argument('--vps-port',type=int,default=2222); ap.add_argument('--remote-cache',required=True)
    ap.add_argument('--profile',default='HQ'); ap.add_argument('--workers',type=int,default=12); ap.add_argument('--downloaders',type=int,default=8)
    ap.add_argument('--status-every',type=int,default=10)
    ap.add_argument('--workdir',default='/content/tml_rapid_pool'); ap.add_argument('--autotune',action='store_true'); ap.add_argument('--no-autotune',action='store_true')
    a=ap.parse_args(); rows=[r for r in read_manifest(a.manifest) if (r.get('mode') or r.get('split') or '').upper()=='RAPID']
    wd=Path(a.workdir); imgdir=wd/'images'; outdir=wd/'out'; imgdir.mkdir(parents=True,exist_ok=True); outdir.mkdir(parents=True,exist_ok=True)
    total=len(rows)
    if not rows: print('RAPID_POOL_START rows=0',flush=True); return 0
    selected=a.workers; tune_file=wd/'autotune_result.json'
    if a.autotune and not a.no_autotune:
        print('RAPID_POOL_AUTOTUNE_START candidates=4,8,12,16 sample=32 source=VPS gallica_requests=0',flush=True)
        tune=[sys.executable,'-u',str(Path(__file__).with_name('rapid_autotune.py')),'--manifest',a.manifest,'--vps-host',a.vps_host,'--vps-user',a.vps_user,'--vps-key-b64',a.vps_key_b64,'--vps-port',str(a.vps_port),'--profile',a.profile,'--sample-pages','32','--candidates','4,8,12,16','--downloaders',str(max(8,a.downloaders)),'--workdir',str(wd),'--out',str(tune_file),'--force']
        rc=subprocess.run(tune).returncode
        if rc==0 and tune_file.exists():
            result=json.loads(tune_file.read_text()); selected=int(result['best_workers']); print(f'RAPID_POOL_AUTOTUNE_SELECTED workers={selected} ppm={result.get("best_pages_per_min")}',flush=True)
        else: print(f'RAPID_POOL_AUTOTUNE_FAILED rc={rc}; fallback_workers={selected}',flush=True)
    else:
        print(f'RAPID_POOL_PROFILE workers={selected} profile={a.profile} benchmark_reference_a100_ppm=72.948',flush=True)
    a.workers=max(1,selected); a.downloaders=max(1,a.downloaders); generation=f'colab_batch_gpu_w{a.workers}'
    print(f'RAPID_POOL_START rows={total} workers={a.workers} downloaders={a.downloaders} source=VPS gallica_requests=0 generation={generation}',flush=True)
    tls=threading.local()
    def get_sftp():
        if not hasattr(tls,'sftp'): tls.tr,tls.sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
        return tls.sftp
    def fetch(idx,r):
        ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); stem=f'{ark}_f{page}'; sftp=get_sftp()
        print(f'FETCH_START idx={idx}/{total} stem={stem}',flush=True)
        try:
            sftp.stat(f"{a.remote_cache.rstrip('/')}/{stem}.json")
            print(f'FETCH_CACHE_HIT idx={idx}/{total} stem={stem}',flush=True)
            return ('cached',idx,r,None,stem)
        except IOError: pass
        src=(r.get('source_image') or '').strip()
        if not src: raise RuntimeError(f'MISSING_SOURCE_IMAGE idx={idx}/{total} stem={stem}')
        dst=imgdir/f'{stem}.jpg'
        if not dst.exists() or dst.stat().st_size<10000: sftp.get(src,str(dst))
        print(f'FETCH_DONE idx={idx}/{total} stem={stem} bytes={dst.stat().st_size}',flush=True)
        return ('ready',idx,r,str(dst),stem)
    workers=max(1,min(a.workers,total)); ctx=mp.get_context('spawn')
    done=err=fetched=cached=uploaded=0; t0=time.time(); upload_tr=upload_sftp=None; last_status=0; last_done_at=t0
    def upload(idx,stem,files):
        nonlocal upload_tr,upload_sftp,uploaded
        for attempt in (1,2):
            try:
                if upload_sftp is None: upload_tr,upload_sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
                for p in files: upload_sftp.put(p,f"{a.remote_cache.rstrip('/')}/{stem}{suffix(Path(p))}")
                uploaded+=1
                print(f'UPLOAD_DONE idx={idx}/{total} stem={stem} uploaded={uploaded}',flush=True)
                return
            except Exception as e:
                print(f'UPLOAD_RETRY idx={idx}/{total} stem={stem} attempt={attempt} error={type(e).__name__}: {e}',flush=True)
                try:
                    if upload_sftp: upload_sftp.close()
                    if upload_tr: upload_tr.close()
                except Exception: pass
                upload_tr=upload_sftp=None
                if attempt==2: raise
    with ThreadPoolExecutor(max_workers=a.downloaders) as dl, ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_engine,initargs=(a.profile,generation)) as gpu:
        fetch_pending={dl.submit(fetch,idx,r):(idx,r) for idx,r in enumerate(rows,1)}; gpu_pending={}
        while fetch_pending or gpu_pending:
            now=time.time()
            if now-last_status>=a.status_every:
                u,m,p=gpu_snapshot(); completed=done+cached; elapsed=max(.001,now-t0); ppm=done*60/elapsed; remaining=max(0,total-completed); eta=(remaining/ppm) if ppm>0 else -1
                print(f'RAPID_HEARTBEAT elapsed={elapsed:.1f}s completed={completed}/{total} ocr_done={done} cached={cached} uploaded={uploaded} fetched={fetched} inflight={len(gpu_pending)} fetch_pending={len(fetch_pending)} errors={err} ppm={ppm:.2f} eta_min={eta:.2f} last_done_age={now-last_done_at:.1f}s gpu={u:.0f}% vram={m:.0f}MiB power={p:.0f}W',flush=True); last_status=now
            ready_gpu={f for f in gpu_pending if f.done()}
            for f in ready_gpu:
                idx_ctx,stem_ctx=gpu_pending.pop(f)
                try:
                    idx,tot,stem,files,sec,nrows=f.result(); upload(idx,stem,files); done+=1; last_done_at=time.time(); completed=done+cached
                    print(f'OCR_DONE progress={completed}/{total} idx={idx}/{tot} stem={stem} sec={sec:.2f} rows={nrows} inflight={len(gpu_pending)}',flush=True)
                except Exception as e:
                    err+=1; print(f'OCR_ERROR idx={idx_ctx}/{total} stem={stem_ctx} {type(e).__name__}: {e}',flush=True)
            if fetch_pending:
                ready_fetch,_=wait(fetch_pending,timeout=.20,return_when=FIRST_COMPLETED)
                for f in ready_fetch:
                    idx_ctx,r_ctx=fetch_pending.pop(f)
                    ark=(r_ctx.get('ark') or '').strip(); page=pg(r_ctx.get('page')); stem_ctx=f'{ark}_f{page}'
                    try:
                        status,idx,r,ip,stem=f.result()
                        if status=='cached':
                            cached+=1; last_done_at=time.time(); print(f'CACHE_COMPLETE progress={done+cached}/{total} idx={idx}/{total} stem={stem}',flush=True); continue
                        fetched+=1
                        fut=gpu.submit(ocr_one,(idx,total,r,ip,str(outdir))); gpu_pending[fut]=(idx,stem)
                        print(f'OCR_QUEUED idx={idx}/{total} stem={stem} inflight={len(gpu_pending)}',flush=True)
                    except Exception as e:
                        err+=1; print(f'FETCH_ERROR idx={idx_ctx}/{total} stem={stem_ctx} {type(e).__name__}: {e}',flush=True)
            elif gpu_pending and not ready_gpu: wait(gpu_pending,timeout=.20,return_when=FIRST_COMPLETED)
    try:
        if upload_sftp: upload_sftp.close()
        if upload_tr: upload_tr.close()
    except Exception: pass
    elapsed=max(.001,time.time()-t0); u,m,p=gpu_snapshot(); print(f'RAPID_FINAL completed={done+cached}/{total} ocr_done={done} cached={cached} uploaded={uploaded} errors={err} elapsed={elapsed:.1f}s ppm={done*60/elapsed:.2f} gpu={u:.0f}% vram={m:.0f}MiB power={p:.0f}W',flush=True)
    return 2 if err else 0
if __name__=='__main__': raise SystemExit(main())
