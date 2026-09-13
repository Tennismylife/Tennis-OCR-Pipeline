#!/usr/bin/env python3
import argparse,hashlib,os,subprocess,sys,time
from pathlib import Path
from worker import connect_sftp,read_manifest,pg

def main():
    ap=argparse.ArgumentParser(description='Watch a VPS RapidOCR claim and run tuned A100 pool whenever it changes')
    ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True); ap.add_argument('--vps-key-b64',required=True); ap.add_argument('--vps-port',type=int,default=2222)
    ap.add_argument('--claim',required=True); ap.add_argument('--remote-cache',required=True); ap.add_argument('--stop-flag'); ap.add_argument('--profile',default='HQ'); ap.add_argument('--workers',type=int,default=12); ap.add_argument('--downloaders',type=int,default=8); ap.add_argument('--poll',type=int,default=10); ap.add_argument('--workdir',default='/content/tml_rapid_watch')
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
            missing=0
            for r in rows:
                ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); stem=f'{ark}_f{page}'
                try: sftp.stat(f"{a.remote_cache.rstrip('/')}/{stem}.json"); sftp.stat(f"{a.remote_cache.rstrip('/')}/{stem}.txt")
                except IOError: missing+=1
        finally: sftp.close(); tr.close()
        raw=local.read_bytes(); h=hashlib.sha256(raw).hexdigest()
        print(f'RAPID_WATCH_STATUS rows={len(rows)} missing={missing} claim_sha={h[:10]}',flush=True)
        if rows and missing:
            if h!=last_hash: print('RAPID_WATCH_NEW_CLAIM starting tuned A100 batch',flush=True)
            else: print('RAPID_WATCH_RETRY incomplete same claim',flush=True)
            cmd=[sys.executable,'-u',str(Path(__file__).with_name('rapid_pool.py')),'--manifest',str(local),'--vps-host',a.vps_host,'--vps-user',a.vps_user,'--vps-key-b64',a.vps_key_b64,'--vps-port',str(a.vps_port),'--remote-cache',a.remote_cache,'--profile',a.profile,'--workers',str(a.workers),'--downloaders',str(a.downloaders),'--no-autotune']
            rc=subprocess.run(cmd).returncode; print(f'RAPID_WATCH_BATCH_EXIT rc={rc}',flush=True); last_hash=h
        else:
            last_hash=h
        time.sleep(a.poll)
if __name__=='__main__': raise SystemExit(main())
