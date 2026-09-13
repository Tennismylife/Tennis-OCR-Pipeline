#!/usr/bin/env python3
from pathlib import Path
import argparse,csv,json,sys

def read_json(p):
    try:return json.loads(p.read_text(encoding='utf-8'))
    except Exception:return {}

def complete(out,stem):
    return (out/f'{stem}.layout.json').exists() and (out/f'{stem}.layout.txt').exists()

def main():
    ap=argparse.ArgumentParser(description='Build year-agnostic Colab layout claim')
    ap.add_argument('--base',required=True); ap.add_argument('--ocr-dir',required=True); ap.add_argument('--layout-dir',required=True); ap.add_argument('--stage',required=True); ap.add_argument('--claim')
    a=ap.parse_args(); base=Path(a.base)
    ocr=Path(a.ocr_dir); out=Path(a.layout_dir); claim=Path(a.claim) if a.claim else base/'00_MANIFEST/colab_layout_active_claim.tsv'
    if not ocr.is_absolute(): ocr=base/ocr
    if not out.is_absolute(): out=base/out
    if not claim.is_absolute(): claim=base/claim
    out.mkdir(parents=True,exist_ok=True); rows=[]; missing=[]; inputs=[p for p in sorted(ocr.glob('*.json')) if not p.name.endswith('.layout.json')]
    for p in inputs:
        if complete(out,p.stem): continue
        d=read_json(p); src=Path(str(d.get('source') or ''))
        if not src.exists(): missing.append(p.stem); continue
        rows.append({'stage':a.stage,'stem':p.stem,'ocr_json':str(p),'source_image':str(src),'layout_dir':str(out),'out_json':str(out/f'{p.stem}.layout.json'),'out_txt':str(out/f'{p.stem}.layout.txt')})
    if missing:
        print('MISSING_IMAGES',len(missing),missing[:20],file=sys.stderr); return 2
    claim.parent.mkdir(parents=True,exist_ok=True); tmp=claim.with_suffix(claim.suffix+'.tmp'); fields=['stage','stem','ocr_json','source_image','layout_dir','out_json','out_txt']
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t'); w.writeheader(); w.writerows(rows)
    tmp.replace(claim); print(json.dumps({'stage':a.stage,'total':len(inputs),'done':len(inputs)-len(rows),'claim':len(rows),'path':str(claim)})); return 0
if __name__=='__main__': raise SystemExit(main())
