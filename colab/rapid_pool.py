#!/usr/bin/env python3
import argparse,json,time,threading,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait,FIRST_COMPLETED
from worker import read_manifest,pg,connect_sftp,sftp_mkdirs,build_engine,ocr_image,write_payload,suffix

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
    rows=read_manifest(a.manifest); wd=Path(a.workdir); imgdir=wd/'images'; outdir=wd/'out'
    imgdir.mkdir(parents=True,exist_ok=True); outdir.mkdir(parents=True,exist_ok=True)

    tr_out,sftp_out=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port); sftp_mkdirs(sftp_out,a.remote_cache)
    try: remote_names=set(sftp_out.listdir(a.remote_cache))
    except Exception: remote_names=set()
    todo=[]
    for r in rows:
        ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); stem=f'{ark}_f{page}'
        if f'{stem}.json' in remote_names: continue
        if (r.get('source_image') or '').strip(): todo.append(r)
    print(f'RAPID_POOL_STREAM todo={len(todo)} workers={a.workers} vps_downloaders={a.downloaders}',flush=True)
    if not todo:
        sftp_out.close(); tr_out.close(); return

    tls=threading.local()
    def get_sftp():
        if not hasattr(tls,'sftp'):
            tls.tr,tls.sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
        return tls.sftp
    def fetch(r):
        ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); dst=imgdir/f'{ark}_f{page}.jpg'
        if not dst.exists() or dst.stat().st_size<10000:
            get_sftp().get((r.get('source_image') or '').strip(),str(dst))
        return r,str(dst)

    workers=max(1,min(a.workers,len(todo))); ctx=mp.get_context('spawn')
    done=err=fetched=0; t0=time.time(); gpu_seq=0
    with ThreadPoolExecutor(max_workers=max(1,a.downloaders)) as dl, \
         ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_engine,initargs=(a.profile,)) as gpu:
        fetch_pending={dl.submit(fetch,r) for r in todo}; gpu_pending=set()
        while fetch_pending or gpu_pending:
            # Drain finished OCR first so results reach VPS immediately.
            ready_gpu={f for f in gpu_pending if f.done()}
            for f in ready_gpu:
                gpu_pending.remove(f); gpu_seq+=1
                try:
                    stem,files,sec,nrows=f.result()
                    for p in files: sftp_out.put(p,f"{a.remote_cache.rstrip('/')}/{stem}{suffix(Path(p))}")
                    done+=1; print(f'GPU_DONE {done}/{len(todo)} {stem} sec={sec:.1f} rows={nrows} inflight={len(gpu_pending)}',flush=True)
                except Exception as e:
                    err+=1; print(f'GPU_ERR {gpu_seq}/{len(todo)} {type(e).__name__}: {e}',flush=True)

            if fetch_pending:
                ready_fetch,_=wait(fetch_pending,timeout=.25,return_when=FIRST_COMPLETED)
                for f in ready_fetch:
                    fetch_pending.remove(f)
                    try:
                        r,ip=f.result(); fetched+=1
                        gpu_pending.add(gpu.submit(ocr_one,(r,ip,str(outdir))))
                        if fetched==1 or fetched%10==0 or fetched==len(todo):
                            print(f'VPS_IMAGE_FETCH {fetched}/{len(todo)} gpu_inflight={len(gpu_pending)}',flush=True)
                    except Exception as e:
                        err+=1; print(f'FETCH_ERR {type(e).__name__}: {e}',flush=True)
            elif gpu_pending and not ready_gpu:
                wait(gpu_pending,timeout=.25,return_when=FIRST_COMPLETED)

    sftp_out.close(); tr_out.close(); elapsed=max(.001,time.time()-t0)
    print({'todo':len(todo),'fetched':fetched,'done':done,'errors':err,'workers':workers,'elapsed_sec':round(elapsed,1),'pages_per_min':round(done*60/elapsed,2)},flush=True)

if __name__=='__main__': main()
