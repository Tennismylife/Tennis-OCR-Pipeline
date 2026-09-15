#!/usr/bin/env python3
import argparse,csv,hashlib,subprocess,sys,time
from pathlib import Path
from collections import defaultdict
from worker import connect_sftp,read_manifest,pg

def write_subset(path,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description='Watch a VPS RAPID claim; supports per-row remote_cache so stages can change without restarting Colab')
    ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True); ap.add_argument('--vps-key-b64',required=True); ap.add_argument('--vps-port',type=int,default=2222)
    ap.add_argument('--claim',required=True); ap.add_argument('--stop-flag'); ap.add_argument('--workers',type=int,default=12); ap.add_argument('--downloaders',type=int,default=8); ap.add_argument('--device',choices=['cuda','cpu'],default='cuda'); ap.add_argument('--poll',type=int,default=10); ap.add_argument('--workdir',default='/content/tml_rapid_watch_v2')
    a=ap.parse_args(); wd=Path(a.workdir); wd.mkdir(parents=True,exist_ok=True); local=wd/'claim.tsv'; last_hash=None; tick=0; started=time.time()
    print(f'RAPID_WATCH_READY poll={a.poll}s workers={a.workers} downloaders={a.downloaders} device={a.device.upper()} claim={a.claim}',flush=True)
    while True:
        tick+=1; poll_t0=time.time()
        tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
        try:
            if a.stop_flag:
                try: sftp.stat(a.stop_flag); print(f'RAPID_WATCH_STOP_FLAG tick={tick} seen=1',flush=True); return 0
                except IOError: pass
            try: sftp.get(a.claim,str(local))
            except IOError:
                print(f'RAPID_WATCH_POLL tick={tick} uptime={time.time()-started:.0f}s claim_missing=1 next_poll={a.poll}s',flush=True); time.sleep(a.poll); continue
            rows=[r for r in read_manifest(str(local)) if (r.get('mode') or r.get('split') or '').upper()=='RAPID']
            groups=defaultdict(list)
            for r in rows:
                cache=(r.get('remote_cache') or '').strip()
                if not cache: raise RuntimeError('Claim row missing remote_cache')
                groups[cache].append(r)
            h=hashlib.sha256(local.read_bytes()).hexdigest()
            incomplete={}; missing=0
            for cache,grows in groups.items():
                try: names=set(sftp.listdir(cache.rstrip('/')))
                except IOError: names=set()
                todo=[]
                for r in grows:
                    ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); stem=f'{ark}_f{page}'
                    if f'{stem}.json' not in names or f'{stem}.txt' not in names: todo.append(r)
                incomplete[cache]=todo; missing+=len(todo)
        finally:
            sftp.close(); tr.close()
        print(f'RAPID_WATCH_POLL tick={tick} uptime={time.time()-started:.0f}s rows={len(rows)} groups={len(groups)} missing={missing} complete={len(rows)-missing}/{len(rows)} claim_sha={h[:10]} poll_sec={time.time()-poll_t0:.2f} device={a.device.upper()}',flush=True)
        if rows and missing:
            print(f'RAPID_WATCH_WORK_AVAILABLE tick={tick} missing={missing}/{len(rows)} claim_state={"new" if h!=last_hash else "retry"}',flush=True)
            active_groups=[(cache,grows) for cache,grows in incomplete.items() if grows]
            for i,(cache,grows) in enumerate(active_groups,1):
                sub=wd/f'group_{i}.tsv'; write_subset(sub,grows)
                print(f'RAPID_WATCH_GROUP_START group={i}/{len(active_groups)} rows={len(grows)} total_missing={missing} cache={cache} device={a.device.upper()}',flush=True)
                cmd=[sys.executable,'-u',str(Path(__file__).with_name('rapid_pool.py')),'--manifest',str(sub),'--vps-host',a.vps_host,'--vps-user',a.vps_user,'--vps-key-b64',a.vps_key_b64,'--vps-port',str(a.vps_port),'--remote-cache',cache,'--profile',(grows[0].get('split_profile') or 'HQ'),'--workers',str(a.workers),'--downloaders',str(a.downloaders),'--device',a.device,'--status-every','10','--no-autotune']
                group_t0=time.time(); rc=subprocess.run(cmd).returncode
                print(f'RAPID_WATCH_GROUP_EXIT group={i}/{len(active_groups)} rows={len(grows)} rc={rc} elapsed={time.time()-group_t0:.1f}s device={a.device.upper()}',flush=True)
                if rc!=0: raise RuntimeError(f'rapid group failed rc={rc}')
            last_hash=h
        else:
            last_hash=h
            print(f'RAPID_WATCH_IDLE tick={tick} rows={len(rows)} missing={missing} next_poll={a.poll}s',flush=True)
        time.sleep(a.poll)
if __name__=='__main__': raise SystemExit(main())
