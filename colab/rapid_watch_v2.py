#!/usr/bin/env python3
import argparse,csv,hashlib,os,subprocess,sys,time,uuid
from pathlib import Path
from collections import defaultdict
from worker import connect_sftp,read_manifest,pg
from claim_lease import ClaimLease, heartbeat

def write_subset(path,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description='Watch a VPS RAPID claim; multi-Colab safe with atomic per-page leases')
    ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True); ap.add_argument('--vps-key-b64',required=True); ap.add_argument('--vps-port',type=int,default=2222)
    ap.add_argument('--claim',required=True); ap.add_argument('--stop-flag'); ap.add_argument('--workers',type=int,default=12); ap.add_argument('--downloaders',type=int,default=8); ap.add_argument('--device',choices=['cuda','cpu'],default='cuda'); ap.add_argument('--poll',type=int,default=10); ap.add_argument('--workdir',default='/content/tml_rapid_watch_v2')
    ap.add_argument('--year',type=int,required=True); ap.add_argument('--worker-id',default=''); ap.add_argument('--lease-ttl',type=int,default=600); ap.add_argument('--lease-batch',type=int,default=0)
    a=ap.parse_args(); worker_id=(a.worker_id or f'colab-{uuid.uuid4().hex[:10]}').strip(); base=f'/home/andre/GallicaJobs/gallica-{a.year}-all-tennis/GALlica_{a.year}_ALL_TENNIS'
    wd=Path(a.workdir)/worker_id; wd.mkdir(parents=True,exist_ok=True); local=wd/'claim.tsv'; last_hash=None; tick=0; started=time.time()
    force_profile=os.environ.get('TML_RAPID_FORCE_PROFILE','').strip().upper()
    if force_profile not in {'','HQ','STANDARD','CPU_FAST'}: raise RuntimeError(f'Invalid TML_RAPID_FORCE_PROFILE={force_profile!r}')
    if a.device=='cpu' and force_profile in {'','STANDARD'}: force_profile='CPU_FAST'
    lease_batch=a.lease_batch or max(a.workers*2,4)
    print(f'RAPID_WATCH_READY worker_id={worker_id} multicolab=YES poll={a.poll}s workers={a.workers} downloaders={a.downloaders} lease_batch={lease_batch} device={a.device.upper()} force_profile={force_profile or "CLAIM"} claim={a.claim}',flush=True)
    while True:
        tick+=1; poll_t0=time.time(); tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
        try:
            heartbeat(sftp,base,worker_id,'RAPID')
            if a.stop_flag:
                try: sftp.stat(a.stop_flag); print(f'RAPID_WATCH_STOP_FLAG tick={tick} seen=1',flush=True); return 0
                except IOError: pass
            try: sftp.get(a.claim,str(local))
            except IOError:
                print(f'RAPID_WATCH_POLL tick={tick} worker_id={worker_id} uptime={time.time()-started:.0f}s claim_missing=1 next_poll={a.poll}s',flush=True); time.sleep(a.poll); continue
            rows=[r for r in read_manifest(str(local)) if (r.get('mode') or r.get('split') or '').upper()=='RAPID']
            groups=defaultdict(list)
            for r in rows:
                cache=(r.get('remote_cache') or '').strip()
                if not cache: raise RuntimeError('Claim row missing remote_cache')
                groups[cache].append(r)
            h=hashlib.sha256(local.read_bytes()).hexdigest(); incomplete={}; missing=0
            for cache,grows in groups.items():
                try: names=set(sftp.listdir(cache.rstrip('/')))
                except IOError: names=set()
                todo=[]
                for r in grows:
                    ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); stem=f'{ark}_f{page}'
                    if f'{stem}.json' not in names or f'{stem}.txt' not in names: todo.append(r)
                incomplete[cache]=todo; missing+=len(todo)
            selected=defaultdict(list); lease_keys=[]
            if missing:
                for cache,grows in incomplete.items():
                    cache_tag=hashlib.sha1(cache.encode()).hexdigest()[:10]
                    for r in grows:
                        if len(lease_keys)>=lease_batch: break
                        ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); stem=f'{ark}_f{page}'; key=f'{cache_tag}:{stem}'
                        lease=ClaimLease(sftp,base,worker_id,'rapid',key,ttl=a.lease_ttl)
                        if lease.acquire(): selected[cache].append(r); lease_keys.append(key)
                    if len(lease_keys)>=lease_batch: break
        finally:
            sftp.close(); tr.close()
        selected_count=sum(len(v) for v in selected.values())
        print(f'RAPID_WATCH_POLL tick={tick} worker_id={worker_id} uptime={time.time()-started:.0f}s rows={len(rows)} groups={len(groups)} missing={missing} leased={selected_count} complete={len(rows)-missing}/{len(rows)} claim_sha={h[:10]} poll_sec={time.time()-poll_t0:.2f} device={a.device.upper()}',flush=True)
        if rows and missing and selected_count:
            print(f'RAPID_WATCH_WORK_AVAILABLE tick={tick} worker_id={worker_id} leased={selected_count} missing={missing}/{len(rows)} claim_state={"new" if h!=last_hash else "retry"}',flush=True)
            try:
                active_groups=[(cache,grows) for cache,grows in selected.items() if grows]
                for i,(cache,grows) in enumerate(active_groups,1):
                    sub=wd/f'group_{i}.tsv'; write_subset(sub,grows); profile=force_profile or (grows[0].get('split_profile') or 'HQ')
                    print(f'RAPID_WATCH_GROUP_START worker_id={worker_id} group={i}/{len(active_groups)} rows={len(grows)} cache={cache} profile={profile} device={a.device.upper()}',flush=True)
                    cmd=[sys.executable,'-u',str(Path(__file__).with_name('rapid_pool.py')),'--manifest',str(sub),'--vps-host',a.vps_host,'--vps-user',a.vps_user,'--vps-key-b64',a.vps_key_b64,'--vps-port',str(a.vps_port),'--remote-cache',cache,'--profile',profile,'--workers',str(a.workers),'--downloaders',str(a.downloaders),'--device',a.device,'--status-every','10','--no-autotune','--workdir',str(wd/f'pool_{i}')]
                    group_t0=time.time(); rc=subprocess.run(cmd).returncode
                    print(f'RAPID_WATCH_GROUP_EXIT worker_id={worker_id} group={i}/{len(active_groups)} rows={len(grows)} rc={rc} elapsed={time.time()-group_t0:.1f}s device={a.device.upper()}',flush=True)
                    if rc!=0: raise RuntimeError(f'rapid group failed rc={rc}')
                last_hash=h
            finally:
                try:
                    tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
                    for key in lease_keys:
                        lease=ClaimLease(sftp,base,worker_id,'rapid',key,ttl=a.lease_ttl); lease.acquired=True; lease.release()
                    heartbeat(sftp,base,worker_id,'RAPID')
                    sftp.close(); tr.close()
                except Exception as e: print(f'RAPID_LEASE_RELEASE_WARN worker_id={worker_id} {type(e).__name__}: {e}',flush=True)
        else:
            last_hash=h
            print(f'RAPID_WATCH_IDLE tick={tick} worker_id={worker_id} rows={len(rows)} missing={missing} leased={selected_count} next_poll={a.poll}s',flush=True)
        time.sleep(a.poll)
if __name__=='__main__': raise SystemExit(main())
