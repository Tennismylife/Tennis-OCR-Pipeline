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
    ap.add_argument('--claim',required=True); ap.add_argument('--stop-flag'); ap.add_argument('--workers',type=int,default=12); ap.add_argument('--downloaders',type=int,default=8); ap.add_argument('--poll',type=int,default=10); ap.add_argument('--workdir',default='/content/tml_rapid_watch_v2')
    a=ap.parse_args(); wd=Path(a.workdir); wd.mkdir(parents=True,exist_ok=True); local=wd/'claim.tsv'; last_hash=None
    while True:
        tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
        try:
            if a.stop_flag:
                try: sftp.stat(a.stop_flag); print('RAPID_WATCH_STOP_FLAG seen=1',flush=True); return 0
                except IOError: pass
            try: sftp.get(a.claim,str(local))
            except IOError:
                print(f'RAPID_WATCH_WAIT claim_missing={a.claim}',flush=True); time.sleep(a.poll); continue
            rows=[r for r in read_manifest(str(local)) if (r.get('mode') or r.get('split') or '').upper()=='RAPID']
            groups=defaultdict(list); missing=0
            for r in rows:
                cache=(r.get('remote_cache') or '').strip()
                if not cache: raise RuntimeError('Claim row missing remote_cache')
                groups[cache].append(r); ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); stem=f'{ark}_f{page}'
                try: sftp.stat(f"{cache.rstrip('/')}/{stem}.json"); sftp.stat(f"{cache.rstrip('/')}/{stem}.txt")
                except IOError: missing+=1
        finally: sftp.close(); tr.close()
        h=hashlib.sha256(local.read_bytes()).hexdigest(); print(f'RAPID_WATCH_STATUS rows={len(rows)} groups={len(groups)} missing={missing} claim_sha={h[:10]}',flush=True)
        if rows and missing:
            print('RAPID_WATCH_NEW_OR_INCOMPLETE_CLAIM', 'new=1' if h!=last_hash else 'retry=1',flush=True)
            for i,(cache,grows) in enumerate(groups.items(),1):
                sub=wd/f'group_{i}.tsv'; write_subset(sub,grows)
                print(f'RAPID_WATCH_GROUP {i}/{len(groups)} rows={len(grows)} cache={cache}',flush=True)
                cmd=[sys.executable,'-u',str(Path(__file__).with_name('rapid_pool.py')),'--manifest',str(sub),'--vps-host',a.vps_host,'--vps-user',a.vps_user,'--vps-key-b64',a.vps_key_b64,'--vps-port',str(a.vps_port),'--remote-cache',cache,'--profile',(grows[0].get('split_profile') or 'HQ'),'--workers',str(a.workers),'--downloaders',str(a.downloaders),'--no-autotune']
                rc=subprocess.run(cmd).returncode; print(f'RAPID_WATCH_GROUP_EXIT group={i} rc={rc}',flush=True)
                if rc!=0: raise RuntimeError(f'rapid group failed rc={rc}')
            last_hash=h
        else: last_hash=h
        time.sleep(a.poll)
if __name__=='__main__': raise SystemExit(main())
