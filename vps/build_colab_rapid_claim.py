#!/usr/bin/env python3
from pathlib import Path
import argparse,csv,json

def pg(v):
    try:return str(int(float(v)))
    except Exception:return str(v or '').strip()

def delim(path): return '\t' if Path(path).suffix.lower() in ('.tsv','.tab') else ','

def main():
    ap=argparse.ArgumentParser(description='Build a generic RAPID-only Colab claim from any ark/page queue')
    ap.add_argument('--base',required=True); ap.add_argument('--queue',required=True); ap.add_argument('--image-dir',required=True); ap.add_argument('--remote-cache',required=True); ap.add_argument('--claim'); ap.add_argument('--profile',default='HQ')
    a=ap.parse_args(); base=Path(a.base); q=Path(a.queue); img=Path(a.image_dir); cache=Path(a.remote_cache); claim=Path(a.claim) if a.claim else base/'00_MANIFEST/colab_active_claims.tsv'
    for name,obj in [('queue',q),('image-dir',img),('remote-cache',cache),('claim',claim)]:
        if not obj.is_absolute():
            if name=='queue': q=base/obj
            elif name=='image-dir': img=base/obj
            elif name=='remote-cache': cache=base/obj
            else: claim=base/obj
    with q.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f,delimiter=delim(q)))
    out=[]; missing_images=[]; seen=set(); completed=0
    for r in rows:
        ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); key=(ark,page)
        if not ark or not page or key in seen: continue
        seen.add(key); stem=f'{ark}_f{page}'; jp=cache/f'{stem}.json'; tp=cache/f'{stem}.txt'
        if jp.exists() and tp.exists(): completed+=1; continue
        src=None
        for ext in ('.jpg','.jpeg','.png','.webp'):
            p=img/f'{stem}{ext}'
            if p.exists(): src=p; break
        if src is None: missing_images.append(stem); continue
        x=dict(r); x.update(ark=ark,page=page,split_branch='RAPID',split_profile=a.profile,mode='RAPID',source_image=str(src)); out.append(x)
    if missing_images:
        print(json.dumps({'error':'missing_images','count':len(missing_images),'examples':missing_images[:20]})); return 2
    fields=[]
    for r in out:
        for k in r:
            if k not in fields: fields.append(k)
    for k in ('ark','page','split_branch','split_profile','mode','source_image'):
        if k not in fields: fields.append(k)
    claim.parent.mkdir(parents=True,exist_ok=True); tmp=claim.with_suffix(claim.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore'); w.writeheader(); w.writerows(out)
    tmp.replace(claim); print(json.dumps({'input_unique':len(seen),'already_complete':completed,'claim':len(out),'path':str(claim),'profile':a.profile})); return 0
if __name__=='__main__': raise SystemExit(main())
