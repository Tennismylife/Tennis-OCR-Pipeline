#!/usr/bin/env python3
from pathlib import Path
import argparse,time

def main():
    ap=argparse.ArgumentParser(description='Wait for Colab layout outputs')
    ap.add_argument('--ocr-dir',required=True); ap.add_argument('--layout-dir',required=True); ap.add_argument('--stage',required=True); ap.add_argument('--poll',type=int,default=10); ap.add_argument('--timeout',type=int,default=0)
    a=ap.parse_args(); ocr=Path(a.ocr_dir); out=Path(a.layout_dir); last=None; t0=time.time()
    while True:
        stems=[p.stem for p in ocr.glob('*.json') if not p.name.endswith('.layout.json')]
        done=sum((out/f'{s}.layout.json').exists() and (out/f'{s}.layout.txt').exists() for s in stems); state=(done,len(stems))
        if state!=last:
            print(f'COLAB_LAYOUT_PROGRESS stage={a.stage} {done}/{len(stems)} elapsed={time.time()-t0:.0f}s',flush=True); last=state
        if done==len(stems): print(f'COLAB_LAYOUT_COMPLETE stage={a.stage} total={len(stems)}',flush=True); return 0
        if a.timeout and time.time()-t0>=a.timeout: print(f'COLAB_LAYOUT_TIMEOUT stage={a.stage} {done}/{len(stems)}',flush=True); return 3
        time.sleep(a.poll)
if __name__=='__main__': raise SystemExit(main())
