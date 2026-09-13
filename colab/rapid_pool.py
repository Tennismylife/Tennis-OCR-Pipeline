#!/usr/bin/env python3
import argparse,json,os,time,threading,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait,FIRST_COMPLETED
from worker import read_manifest,pg,connect_sftp,build_engine,ocr_image,write_payload,suffix

os.environ.setdefault('CUDA_MODULE_LOADING','LAZY')
ENGINE=None; PROFILE='HQ'

def init_engine(profile):
    global ENGINE,PROFILE
    PROFILE=profile
    ENGINE=build_engine(profile,'cuda')

def ocr_one(item):
    r,ip,outdir=item
    ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); stem=f'{ark}_f{page}'; t0=time.time()
    rows=ocr_image(ip,ENGINE)
    files=write_payload(outdir,ark,page,rows,PROFILE,(r.get('source_image') or 'COLAB'),'CUDA')
    jp=Path(files[0]); d=json.loads(jp.read_text(encoding='utf-8'))
    d['ocr_generation']='colab_batch_v4_max'; d['gpu_pool']=True
    jp.write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8')
    return stem,[str(x) for x in files],time.time()-t0,len(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True); ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True)
    ap.add_argument('--vps-key-b64',required=True); ap.add_argument('--vps-port',type=int,default=2222); ap.add_argument('--remote-cache',required=True)
    ap.add_argument('--profile',default='HQ'); ap.add_argument('--workers',type=int,default=16); ap.add_argument('--downloaders',type=int,default=6)
    ap.add_argument('--workdir',default='/content/tml_rapid_pool'); a=ap.parse_args()
    rows=[r for r in read_manifest(a.manifest) if (r.get('mode') or r.get('split') or '').upper()=='RAPID']
    wd=Path(a.workdir); imgdir=wd/'images'; outdir=wd/'out'; imgdir.mkdir(parents=True,exist_ok=True); outdir.mkdir(parents=True,exist_ok=True)
    print(f'RAPID_POOL_START rows={len(rows)} workers={a.workers} downloaders={a.downloaders} source=VPS gallica_requests=0',flush=True)
    if not rows: return

    tls=threading.local()
    def get_sftp():
        if not hasattr(tls,'sftp'):
            tls.tr,tls.sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
        return tls.sftp
    def fetch(r):
        ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); stem=f'{ark}_f{page}'
        sftp=get_sftp()
        try:
            sftp.stat(f"{a.remote_cache.rstrip('/')}/{stem}.json")
            return ('cached',r,None)
        except IOError:
            pass
        src=(r.get('source_image') or '').strip()
        if not src: raise RuntimeError(f'MISSING_SOURCE_IMAGE {stem}')
        dst=imgdir/f'{stem}.jpg'
        if not dst.exists() or dst.stat().st_size<10000: sftp.get(src,str(dst))
        return ('ready',r,str(dst))

    workers=max(1,min(a.workers,len(rows))); ctx=mp.get_context('spawn')
    done=err=fetched=cached=0; t0=time.time(); upload_tr=upload_sftp=None
    def upload(stem,files):
        nonlocal upload_tr,upload_sftp
        for attempt in (1,2):
            try:
                if upload_sftp is None: upload_tr,upload_sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
                for p in files: upload_sftp.put(p,f"{a.remote_cache.rstrip('/')}/{stem}{suffix(Path(p))}")
                return
            except Exception:
                try:
                    if upload_sftp: upload_sftp.close()
                    if upload_tr: upload_tr.close()
                except Exception: pass
                upload_tr=upload_sftp=None
                if attempt==2: raise

    with ThreadPoolExecutor(max_workers=max(1,a.downloaders)) as dl, ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_engine,initargs=(a.profile,)) as gpu:
        fetch_pending={dl.submit(fetch,r) for r in rows}; gpu_pending=set()
        while fetch_pending or gpu_pending:
            ready_gpu={f for f in gpu_pending if f.done()}
            for f in ready_gpu:
                gpu_pending.remove(f)
                try:
                    stem,files,sec,nrows=f.result(); upload(stem,files); done+=1
                    print(f'GPU_DONE {done}/{len(rows)-cached} {stem} sec={sec:.1f} rows={nrows} inflight={len(gpu_pending)}',flush=True)
                except Exception as e:
                    err+=1; print(f'GPU_ERR {type(e).__name__}: {e}',flush=True)
            if fetch_pending:
                ready_fetch,_=wait(fetch_pending,timeout=.20,return_when=FIRST_COMPLETED)
                for f in ready_fetch:
                    fetch_pending.remove(f)
                    try:
                        status,r,ip=f.result()
                        if status=='cached':
                            cached+=1
                            if cached<=3 or cached%10==0: print(f'VPS_CACHED {cached}',flush=True)
                            continue
                        fetched+=1; gpu_pending.add(gpu.submit(ocr_one,(r,ip,str(outdir))))
                        if fetched==1 or fetched%10==0 or fetched+cached==len(rows):
                            print(f'VPS_IMAGE_READY fetched={fetched} cached={cached} gpu_inflight={len(gpu_pending)}',flush=True)
                    except Exception as e:
                        err+=1; print(f'FETCH_ERR {type(e).__name__}: {e}',flush=True)
            elif gpu_pending and not ready_gpu:
                wait(gpu_pending,timeout=.20,return_when=FIRST_COMPLETED)

    try:
        if upload_sftp: upload_sftp.close()
        if upload_tr: upload_tr.close()
    except Exception: pass
    elapsed=max(.001,time.time()-t0)
    print({'rows':len(rows),'cached':cached,'fetched':fetched,'done':done,'errors':err,'workers':workers,'elapsed_sec':round(elapsed,1),'pages_per_min':round(done*60/elapsed,2)},flush=True)

if __name__=='__main__': main()
