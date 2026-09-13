#!/usr/bin/env python3
import argparse,base64,subprocess,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description='Supervise layout_pool_v3 and restart it after transient failures')
    ap.add_argument('--vps-key-file',required=True); ap.add_argument('--restart-delay',type=int,default=8); ap.add_argument('--max-delay',type=int,default=60)
    a,rest=ap.parse_known_args(); key=Path(a.vps_key_file)
    if not key.exists() or key.stat().st_size<100: raise SystemExit(f'Invalid key file: {key}')
    token=base64.b64encode(key.read_bytes()).decode(); failures=0
    while True:
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('layout_pool_v3.py')),'--vps-key-b64',token,*rest]
        print(f'LAYOUT_SUPERVISOR_START attempt={failures+1}',flush=True)
        rc=subprocess.run(cmd).returncode
        if rc==0:
            print('LAYOUT_SUPERVISOR_CLEAN_EXIT rc=0',flush=True); return 0
        failures+=1; delay=min(a.max_delay,a.restart_delay*(2**min(failures-1,3)))
        print(f'LAYOUT_SUPERVISOR_RESTART rc={rc} failures={failures} delay={delay}s',flush=True)
        time.sleep(delay)
if __name__=='__main__': raise SystemExit(main())
