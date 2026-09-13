#!/usr/bin/env python3
import argparse, csv, json, os, re, time, base64, tempfile
from pathlib import Path
from difflib import SequenceMatcher
import requests
import paramiko

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

KEYS=('tennis','lawntennis','championnat','championship','tournament','tournoi','raquette','singles','simple','messieurs','gentlemen','racingclub','puteaux','burdigala','primrose','dinard')

def pg(v):
    try:return str(int(float(v)))
    except:return str(v or '').strip()

def norm(s): return re.sub(r'[^a-z]','',s.lower())
def hit(s):
    n=norm(s)
    if any(k in n for k in KEYS): return True
    return max((SequenceMatcher(None,n,k).ratio() for k in KEYS),default=0)>=0.72

def read_manifest(path):
    p=Path(path); delim='\t' if p.suffix.lower()=='.tsv' else ','
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter=delim))

def alto_text(payload):
    import xml.etree.ElementTree as ET
    root=ET.fromstring(payload); lines=[]
    for tl in root.iter():
        if tl.tag.split('}')[-1]!='TextLine': continue
        words=[]
        for s in tl.iter():
            if s.tag.split('}')[-1]!='String': continue
            t=(s.attrib.get('SUBS_CONTENT') or s.attrib.get('CONTENT') or '').strip()
            if t: words.append(t)
        if words: lines.append(' '.join(words))
    return '\n'.join(lines).strip()

def parse_alto_rows(payload):
    import xml.etree.ElementTree as ET
    root=ET.fromstring(payload); rows=[]; pw=ph=0
    for e in root.iter():
        if e.tag.split('}')[-1]=='Page':
            try: pw=float(e.attrib.get('WIDTH',0)); ph=float(e.attrib.get('HEIGHT',0))
            except: pw=ph=0
            break
    for tl in root.iter():
        if tl.tag.split('}')[-1]!='TextLine': continue
        words=[]
        for s in tl.iter():
            if s.tag.split('}')[-1]!='String': continue
            t=(s.attrib.get('SUBS_CONTENT') or s.attrib.get('CONTENT') or '').strip()
            if t: words.append(t)
        if not words: continue
        try:x=float(tl.attrib.get('HPOS',0));y=float(tl.attrib.get('VPOS',0));w=float(tl.attrib.get('WIDTH',0));h=float(tl.attrib.get('HEIGHT',0))
        except:x=y=w=h=0
        rows.append({'text':' '.join(words),'score':1.0,'x1':x,'x2':x+w,'y1':y,'y2':y+h,'cx':x+w/2,'cy':y+h/2})
    rows.sort(key=lambda z:(z['y1'],z['x1']))
    return rows,pw,ph

def build_engine(profile):
    from rapidocr import RapidOCR, LangRec, ModelType, OCRVersion
    params={
      'Global.log_level':'error','EngineConfig.onnxruntime.intra_op_num_threads':1,'EngineConfig.onnxruntime.inter_op_num_threads':1,
      'Rec.lang_type':LangRec.LATIN,'Rec.model_type':ModelType.MOBILE,'Rec.ocr_version':OCRVersion.PPOCRV5,
      'Det.model_type':ModelType.SMALL,'Det.ocr_version':OCRVersion.PPOCRV6}
    if profile=='HQ': params.update({'Det.limit_side_len':2048,'Det.limit_type':'min','Det.box_thresh':0.30,'Det.max_candidates':4000})
    else: params.update({'Det.limit_side_len':1536,'Det.limit_type':'min','Det.box_thresh':0.35,'Det.max_candidates':3000})
    return RapidOCR(params=params)

def ocr_image(path,profile,engine):
    res=engine(str(path)); rows=[]
    boxes=res.boxes if res.boxes is not None else []; txts=res.txts if res.txts is not None else []; scores=res.scores if res.scores is not None else []
    for b,t,s in zip(boxes,txts,scores):
        xs=[float(q[0]) for q in b];ys=[float(q[1]) for q in b]
        rows.append({'text':str(t),'score':float(s),'x1':min(xs),'x2':max(xs),'y1':min(ys),'y2':max(ys),'cx':sum(xs)/4,'cy':sum(ys)/4})
    rows.sort(key=lambda z:(z['y1'],z['x1']))
    return rows

def write_payload(outdir,ark,page,rows,profile,source,extra=None):
    stem=f'{ark}_f{page}'; outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    texts=[r['text'] for r in rows]; hits=[i for i,t in enumerate(texts) if hit(t)]; ctx=[]
    for i in hits:
        a=max(0,i-5);b=min(len(texts),i+13);ctx.append(f'--- {a+1}-{b} ---');ctx.extend(texts[a:b])
    payload={'source':source,'ocr_profile':profile,'ocr_generation':'colab_batch_v1','rows':rows,'hit_indices':hits}
    if extra: payload.update(extra)
    (outdir/f'{stem}.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    (outdir/f'{stem}.txt').write_text('\n'.join(texts)+'\n',encoding='utf-8')
    (outdir/f'{stem}.hits.txt').write_text('\n'.join(ctx)+'\n' if ctx else '',encoding='utf-8')
    return [outdir/f'{stem}.json',outdir/f'{stem}.txt',outdir/f'{stem}.hits.txt']

def connect_sftp(host,user,key_b64,port=22):
    keydata=base64.b64decode(key_b64).decode()
    key=None
    for cls in (paramiko.Ed25519Key,paramiko.RSAKey,paramiko.ECDSAKey):
        try:
            from io import StringIO
            key=cls.from_private_key(StringIO(keydata));break
        except Exception: pass
    if key is None: raise RuntimeError('Unsupported SSH private key')
    tr=paramiko.Transport((host,int(port)));tr.connect(username=user,pkey=key)
    return tr,paramiko.SFTPClient.from_transport(tr)

def sftp_mkdirs(sftp,path):
    parts=Path(path).parts; cur='/' if path.startswith('/') else ''
    for p in parts:
        if p=='/': continue
        cur=(cur.rstrip('/')+'/'+p) if cur else p
        try:sftp.stat(cur)
        except IOError:sftp.mkdir(cur)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--workdir',default='/content/tml_colab')
    ap.add_argument('--vps-host',required=True);ap.add_argument('--vps-user',required=True);ap.add_argument('--vps-key-b64',required=True);ap.add_argument('--vps-port',type=int,default=22)
    ap.add_argument('--remote-cache',required=True);ap.add_argument('--profile',default='HQ',choices=['HQ','STANDARD']);ap.add_argument('--delay',type=float,default=15.0)
    ap.add_argument('--alto-first',action='store_true');ap.add_argument('--max-pages',type=int,default=0)
    a=ap.parse_args();wd=Path(a.workdir);wd.mkdir(parents=True,exist_ok=True);rows=read_manifest(a.manifest)
    if a.max_pages: rows=rows[:a.max_pages]
    tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port);engine=None;done=alto_n=ocr_n=err=0
    session=requests.Session();session.headers.update({'User-Agent':'Mozilla/5.0','Accept':'application/xml,text/xml,*/*'})
    for i,r in enumerate(rows,1):
        ark=(r.get('ark') or '').strip(); page=pg(r.get('page')); mode=(r.get('mode') or r.get('split') or '').upper()
        if not ark or not page: continue
        remote_dir=f"{a.remote_cache.rstrip('/')}/{ark}"; remote_json=f"{remote_dir}/f{int(page):03d}.colab.json"
        try:
            sftp.stat(remote_json); print(f'CACHED {i}/{len(rows)} {ark} f{page}',flush=True);continue
        except IOError: pass
        try:
            use_alto=a.alto_first or mode=='ALTO'; rows_out=None; xml_payload=None
            if use_alto:
                url=f'https://gallica.bnf.fr/RequestDigitalElement?O={ark}&E=ALTO&Deb={page}'
                rr=session.get(url,timeout=45)
                if rr.status_code==200:
                    txt=alto_text(rr.content)
                    if txt: rows_out,_,_=parse_alto_rows(rr.content); xml_payload=rr.content; alto_n+=1
                time.sleep(a.delay)
            if rows_out is None:
                img_url=f'https://gallica.bnf.fr/ark:/12148/{ark}/f{page}.highres'
                img=session.get(img_url,timeout=60);img.raise_for_status();ip=wd/f'{ark}_f{page}.jpg';ip.write_bytes(img.content)
                if engine is None: engine=build_engine(a.profile)
                rows_out=ocr_image(ip,a.profile,engine);ocr_n+=1;ip.unlink(missing_ok=True)
            local=wd/'out';files=write_payload(local,ark,page,rows_out,'ALTO_NATIVE' if xml_payload else a.profile,'COLAB',{'colab':True})
            sftp_mkdirs(sftp,remote_dir)
            for fp in files:
                ext=fp.suffix
                dest=f"{remote_dir}/f{int(page):03d}.colab{ext}";sftp.put(str(fp),dest)
            if xml_payload:
                xp=wd/f'{ark}_f{page}.xml';xp.write_bytes(xml_payload);sftp.put(str(xp),f"{remote_dir}/f{int(page):03d}.xml");xp.unlink(missing_ok=True)
            done+=1; print(f'DONE {i}/{len(rows)} {ark} f{page} mode={"ALTO" if xml_payload else a.profile}',flush=True)
        except Exception as e:
            err+=1;print(f'ERR {i}/{len(rows)} {ark} f{page} {type(e).__name__}: {e}',flush=True)
    sftp.close();tr.close();print(json.dumps({'total':len(rows),'done':done,'alto':alto_n,'rapidocr':ocr_n,'errors':err}),flush=True)

if __name__=='__main__':main()
