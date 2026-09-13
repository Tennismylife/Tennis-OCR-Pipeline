#!/usr/bin/env python3
import argparse,base64,csv,os,time,threading,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
from worker import read_manifest,pg,connect_sftp,sftp_mkdirs,build_engine,ocr_image,write_payload,suffix

ENGINE=None;PROFILE='HQ'
def init_engine(profile):
    global ENGINE,PROFILE
    PROFILE=profile;ENGINE=build_engine(profile,'cuda')

def ocr_one(item):
    r,ip,outdir=item;ark=(r.get('ark') or '').strip();page=pg(r.get('page'));stem=f'{ark}_f{page}';t0=time.time()
    rows=ocr_image(ip,ENGINE)
    files=write_payload(outdir,ark,page,rows,PROFILE,(r.get('source_image') or 'COLAB'),'CUDA')
    return stem,[str(x) for x in files],time.time()-t0,len(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--vps-host',required=True);ap.add_argument('--vps-user',required=True)
    ap.add_argument('--vps-key-b64',required=True);ap.add_argument('--vps-port',type=int,default=2222);ap.add_argument('--remote-cache',required=True)
    ap.add_argument('--profile',default='HQ');ap.add_argument('--workers',type=int,default=12);ap.add_argument('--downloaders',type=int,default=6)
    ap.add_argument('--workdir',default='/content/tml_rapid_pool');a=ap.parse_args()
    rows=read_manifest(a.manifest);wd=Path(a.workdir);imgdir=wd/'images';outdir=wd/'out';imgdir.mkdir(parents=True,exist_ok=True);outdir.mkdir(parents=True,exist_ok=True)
    tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port);sftp_mkdirs(sftp,a.remote_cache)
    todo=[]
    for r in rows:
        ark=(r.get('ark') or '').strip();page=pg(r.get('page'));stem=f'{ark}_f{page}'
        try:sftp.stat(f"{a.remote_cache.rstrip('/')}/{stem}.json");continue
        except IOError:pass
        if (r.get('source_image') or '').strip():todo.append(r)
    sftp.close();tr.close();print(f'RAPID_POOL todo={len(todo)} workers={a.workers} vps_downloaders={a.downloaders}',flush=True)
    tls=threading.local()
    def get_sftp():
        if not hasattr(tls,'sftp'):
            tls.tr,tls.sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
        return tls.sftp
    def fetch(r):
        ark=(r.get('ark') or '').strip();page=pg(r.get('page'));dst=imgdir/f'{ark}_f{page}.jpg'
        if not dst.exists() or dst.stat().st_size<10000:get_sftp().get((r.get('source_image') or '').strip(),str(dst))
        return r,str(dst)
    t0=time.time();downloaded=[]
    with ThreadPoolExecutor(max_workers=max(1,a.downloaders)) as ex:
        futs=[ex.submit(fetch,r) for r in todo]
        for n,f in enumerate(as_completed(futs),1):
            downloaded.append(f.result())
            if n%10==0 or n==len(futs):print(f'VPS_IMAGE_FETCH {n}/{len(futs)}',flush=True)
    print(f'VPS_IMAGE_FETCH_DONE sec={time.time()-t0:.1f}',flush=True)
    workers=max(1,min(a.workers,len(downloaded) or 1));ctx=mp.get_context('spawn');done=err=0;t1=time.time()
    tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port);sftp_mkdirs(sftp,a.remote_cache)
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_engine,initargs=(a.profile,)) as ex:
        futs=[ex.submit(ocr_one,(r,ip,str(outdir))) for r,ip in downloaded]
        for n,f in enumerate(as_completed(futs),1):
            try:
                stem,files,sec,nrows=f.result()
                for p in files:sftp.put(p,f"{a.remote_cache.rstrip('/')}/{stem}{suffix(Path(p))}")
                done+=1;print(f'GPU_DONE {n}/{len(futs)} {stem} sec={sec:.1f} rows={nrows}',flush=True)
            except Exception as e:err+=1;print(f'GPU_ERR {n}/{len(futs)} {type(e).__name__}: {e}',flush=True)
    sftp.close();tr.close();elapsed=max(0.001,time.time()-t1)
    print({'todo':len(todo),'done':done,'errors':err,'workers':workers,'ocr_elapsed_sec':round(elapsed,1),'pages_per_min':round(done*60/elapsed,2)},flush=True)
if __name__=='__main__':main()
