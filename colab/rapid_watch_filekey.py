#!/usr/bin/env python3
import argparse,base64,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--vps-key-file',required=True)
    a,rest=ap.parse_known_args(); p=Path(a.vps_key_file)
    if not p.exists() or p.stat().st_size<100: raise SystemExit(f'Invalid key file: {p}')
    # Free-tier 16 GB GPU: 8 workers caused ONNXRuntime CUDA OOM on large pages.
    # Cap RapidOCR at 6 workers while keeping SFTP downloader count unchanged.
    if '--workers' in rest:
        i=rest.index('--workers')
        try: rest[i+1]=str(min(6,max(1,int(rest[i+1]))))
        except Exception: rest[i+1]='6'
    else:
        rest.extend(['--workers','6'])
    token=base64.b64encode(p.read_bytes()).decode()
    cmd=[sys.executable,'-u',str(Path(__file__).with_name('rapid_watch_v2.py')),'--vps-key-b64',token,*rest]
    print('RAPID_FILEKEY_CONFIG workers=6',flush=True)
    return subprocess.run(cmd).returncode
if __name__=='__main__': raise SystemExit(main())
