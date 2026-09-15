#!/usr/bin/env python3
import argparse,csv,hashlib,os,subprocess,sys,time
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COLAB=ROOT/'colab'
sys.path.insert(0,str(COLAB))
from worker import connect_sftp,read_manifest,pg


def write_subset(path,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def gpu_count():
    try:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader,nounits'],text=True,timeout=5)
        return len([x for x in out.splitlines() if x.strip()])
    except Exception:
        return 0


def stable_bucket(r,n):
    s=f"{(r.get('ark') or '').strip()}:{pg(r.get('page'))}".encode()
    return int(hashlib.sha1(s).hexdigest()[:8],16)%n

def main():
    ap=argparse.ArgumentParser(description='Kaggle T4x2 watcher for a disjoint VPS RAPID claim')
    ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True)
    ap.add_argument('--vps-key-b64',required=True); ap.add_argument('--vps-port',type=int,default=2222)
    ap.add_argument('--claim',required=True); ap.add_argument('--stop-flag')
    ap.add_argument('--workers-per-gpu',type=int,default=3); ap.add_argument('--downloaders-per-gpu',type=int,default=4)
    ap.add_argument('--poll',type=int,default=10); ap.add_argument('--workdir',default='/kaggle/working/tml_kaggle_t4x2')
    ap.add_argument('--max-gpus',type=int,default=2); ap.add_argument('--once',action='store_true')
    a=ap.parse_args()

    ngpu=min(max(0,gpu_count()),max(1,a.max_gpus))
    if ngpu<1: raise SystemExit('No NVIDIA GPU visible. Enable Kaggle GPU accelerator first.')
    wd=Path(a.workdir); wd.mkdir(parents=True,exist_ok=True); local=wd/'claim.tsv'
    print(f'KAGGLE_T4_WATCH_READY gpus={ngpu} workers_per_gpu={a.workers_per_gpu} downloaders_per_gpu={a.downloaders_per_gpu} once={int(a.once)} claim={a.claim}',flush=True)
    try: subprocess.run(['nvidia-smi','-L'],check=False)
    except Exception: pass

    tick=0; started=time.time(); last_hash=None
    while True:
        tick+=1; poll_t0=time.time()
        tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
        try:
            if a.stop_flag:
                try: sftp.stat(a.stop_flag); print(f'KAGGLE_T4_STOP tick={tick}',flush=True); return 0
                except IOError: pass
            try: sftp.get(a.claim,str(local))
            except IOError:
                print(f'KAGGLE_T4_IDLE tick={tick} claim_missing=1 uptime={time.time()-started:.0f}s',flush=True)
                if a.once: return 3
                time.sleep(a.poll); continue
            rows=[r for r in read_manifest(str(local)) if (r.get('mode') or r.get('split') or '').upper()=='RAPID']
            h=hashlib.sha256(local.read_bytes()).hexdigest()
            claim_state='new' if h!=last_hash else 'retry'
            groups=defaultdict(list)
            for r in rows:
                cache=(r.get('remote_cache') or '').strip()
                if not cache: raise RuntimeError('Claim row missing remote_cache')
                groups[cache].append(r)
            incomplete={}; missing=0
            for cache,grows in groups.items():
                try: names=set(sftp.listdir(cache.rstrip('/')))
                except IOError: names=set()
                todo=[]
                for r in grows:
                    stem=f"{(r.get('ark') or '').strip()}_f{pg(r.get('page'))}"
                    if f'{stem}.json' not in names or f'{stem}.txt' not in names: todo.append(r)
                incomplete[cache]=todo; missing+=len(todo)
        finally:
            sftp.close(); tr.close()

        print(f'KAGGLE_T4_POLL tick={tick} rows={len(rows)} missing={missing} complete={len(rows)-missing}/{len(rows)} claim_sha={h[:10]} poll_sec={time.time()-poll_t0:.2f}',flush=True)
        if not rows or not missing:
            last_hash=h
            if a.once:
                print(f'KAGGLE_T4_ONCE_DONE rows={len(rows)} already_complete={len(rows)-missing}',flush=True)
                return 0
            time.sleep(a.poll); continue

        cycle_t0=time.time(); cycle_rows=missing
        for gi,(cache,grows) in enumerate((x for x in incomplete.items() if x[1]),1):
            shards=[[] for _ in range(ngpu)]
            for r in grows: shards[stable_bucket(r,ngpu)].append(r)
            procs=[]
            for gpu,part in enumerate(shards):
                if not part: continue
                sub=wd/f'group_{gi}_gpu{gpu}.tsv'; write_subset(sub,part)
                profile=(part[0].get('split_profile') or 'HQ').strip().upper() or 'HQ'
                env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu); env['TML_RAPID_GENERATION_PREFIX']=f'kaggle_t4x2_gpu{gpu}'
                cmd=[sys.executable,'-u',str(COLAB/'rapid_pool.py'),
                     '--manifest',str(sub),'--vps-host',a.vps_host,'--vps-user',a.vps_user,
                     '--vps-key-b64',a.vps_key_b64,'--vps-port',str(a.vps_port),'--remote-cache',cache,
                     '--profile',profile,'--workers',str(max(1,a.workers_per_gpu)),
                     '--downloaders',str(max(1,a.downloaders_per_gpu)),'--device','cuda','--status-every','10','--no-autotune',
                     '--workdir',str(wd/f'pool_gpu{gpu}')]
                print(f'KAGGLE_T4_GROUP_START group={gi} gpu={gpu} rows={len(part)} cache={cache} profile={profile}',flush=True)
                procs.append((gpu,len(part),subprocess.Popen(cmd,env=env)))
            bad=[]
            for gpu,n,p in procs:
                rc=p.wait(); print(f'KAGGLE_T4_GROUP_EXIT group={gi} gpu={gpu} rows={n} rc={rc}',flush=True)
                if rc: bad.append((gpu,rc))
            if bad: raise RuntimeError(f'Kaggle GPU shard failure: {bad}')
        elapsed=max(.001,time.time()-cycle_t0); ppm=cycle_rows*60/elapsed
        print(f'KAGGLE_T4_CYCLE_COMPLETE tick={tick} claim_state={claim_state} rows={cycle_rows} elapsed={elapsed:.2f}s ppm={ppm:.3f}',flush=True)
        last_hash=h
        if a.once: return 0
        time.sleep(a.poll)

if __name__=='__main__': raise SystemExit(main())
