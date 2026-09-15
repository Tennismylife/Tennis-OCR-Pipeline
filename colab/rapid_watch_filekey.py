#!/usr/bin/env python3
import argparse,base64,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--vps-key-file',required=True)
    a,rest=ap.parse_known_args(); p=Path(a.vps_key_file)
    if not p.exists() or p.stat().st_size<100: raise SystemExit(f'Invalid key file: {p}')
    # Cap free-tier concurrency at 4; the notebook selects the actual count from available CPU/GPU resources.
    if '--workers' in rest:
        i=rest.index('--workers')
        try: rest[i+1]=str(min(4,max(1,int(rest[i+1]))))
        except Exception: rest[i+1]='4'
    else:
        rest.extend(['--workers','4']); i=len(rest)-2
    workers=rest[rest.index('--workers')+1]
    token=base64.b64encode(p.read_bytes()).decode()
    cmd=[sys.executable,'-u',str(Path(__file__).with_name('rapid_watch_v2.py')),'--vps-key-b64',token,*rest]
    print(f'RAPID_FILEKEY_CONFIG workers={workers}',flush=True)
    return subprocess.run(cmd).returncode
if __name__=='__main__': raise SystemExit(main())
