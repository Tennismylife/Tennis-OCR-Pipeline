#!/usr/bin/env python3
import argparse,base64,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--vps-key-file',required=True)
    a,rest=ap.parse_known_args(); p=Path(a.vps_key_file)
    if not p.exists() or p.stat().st_size<100: raise SystemExit(f'Invalid key file: {p}')
    token=base64.b64encode(p.read_bytes()).decode()
    cmd=[sys.executable,'-u',str(Path(__file__).with_name('layout_pool_v3.py')),'--vps-key-b64',token,*rest]
    return subprocess.run(cmd).returncode
if __name__=='__main__': raise SystemExit(main())
