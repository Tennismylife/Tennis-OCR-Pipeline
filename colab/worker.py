#!/usr/bin/env python3
import argparse, csv, json, os, re, time, base64
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
    root=ET.fromstring(payload); rows=[]
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
    return rows

def ort_device(device):
    # Colab ships PyTorch + CUDA libraries. Import torch first so its matching CUDA/cuDNN
    # libraries are loaded before ONNX Runtime creates CUDA sessions.
    torch_cuda='unknown'
    try:
        import torch
        torch_cuda=str(torch.version.cuda)
    except Exception as e:
        print(f'Torch preload warning: {type(e).__name__}: {e}',flush=True)
    import onnxruntime as ort
    try:
        if hasattr(ort,'preload_dlls'):
            ort.preload_dlls()
    except Exception as e:
        print(f'ORT preload warning: {type(e).__name__}: {e}',flush=True)
    providers=ort.get_available_providers()
    cuda='CUDAExecutionProvider' in providers
    if device=='cuda' and not cuda:
        raise RuntimeError(f'CUDAExecutionProvider unavailable: ORT={ort.__version__} torch_cuda={torch_cuda} providers={providers}')
    use_cuda=cuda if device=='auto' else device=='cuda'
    selected='CUDA' if use_cuda else 'CPU'
    print(f'ORT version={ort.__version__} torch_cuda={torch_cuda} providers={providers} selected={selected}',flush=True)
    return use_cuda,providers

def build_engine(profile,device):
    from rapidocr import RapidOCR, LangRec, ModelType, OCRVersion
    use_cuda,_=ort_device(device)
    params={
      'Global.log_level':'error',
      'EngineConfig.onnxruntime.intra_op_num_threads':1,
      'EngineConfig.onnxruntime.inter_op_num_threads':1,
      'EngineConfig.onnxruntime.use_cuda':use_cuda,
      'EngineConfig.onnxruntime.cuda_ep_cfg.device_id':0,
      'EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search':'HEURISTIC',
      'EngineConfig.onnxruntime.cuda_ep_cfg.do_copy_in_default_stream':True,
      'Rec.lang_type':LangRec.LATIN,'Rec.model_type':ModelType.MOBILE,'Rec.ocr_version':OCRVersion.PPOCRV5,
      'Det.model_type':ModelType.SMALL,'Det.ocr_version':OCRVersion.PPOCRV6}
    if profile=='HQ':
        params.update({'Det.limit_side_len':2048,'Det.limit_type':'min','Det.box_thresh':0.30,'Det.max_candidates':4000})
    elif profile=='CPU_FAST':
        # Free Colab CPUs are dramatically slower when a newspaper page is enlarged so
        # its short side reaches 1536. Cap the long side instead. Recognition still runs
        # on the detected source crops, while detection work and false-positive boxes drop.
        params.update({'Det.limit_side_len':1800,'Det.limit_type':'max','Det.box_thresh':0.38,'Det.max_candidates':2200})
    else:
        params.update({'Det.limit_side_len':1536,'Det.limit_type':'min','Det.box_thresh':0.35,'Det.max_candidates':3000})
    engine=RapidOCR(params=params)
    print(f'RapidOCR engine ready profile={profile} device={"CUDA" if use_cuda else "CPU"}',flush=True)
    return engine

def ocr_image(path,engine):
    res=engine(str(path)); rows=[]
    boxes=res.boxes if res.boxes is not None else []; txts=res.txts if res.txts is not None else []; scores=res.scores if res.scores is not None else []
    for b,t,s in zip(boxes,txts,scores):
        xs=[float(q[0]) for q in b];ys=[float(q[1]) for q in b]
        rows.append({'text':str(t),'score':float(s),'x1':min(xs),'x2':max(xs),'y1':min(ys),'y2':max(ys),'cx':sum(xs)/4,'cy':sum(ys)/4})
    rows.sort(key=lambda z:(z['y1'],z['x1']))
    return rows

def write_payload(outdir,ark,page,rows,profile,source,device):
    stem=f'{ark}_f{page}';outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    texts=[r['text'] for r in rows];hits=[i for i,t in enumerate(texts) if hit(t)];ctx=[]
    for i in hits:
        a=max(0,i-5);b=min(len(texts),i+13);ctx.append(f'--- {a+1}-{b} ---');ctx.extend(texts[a:b])
    payload={'source':source,'ocr_profile':profile,'ocr_generation':'colab_batch_v3_gpu','colab':True,'compute_device':device,'rows':rows,'hit_indices':hits}
    paths=[outdir/f'{stem}.json',outdir/f'{stem}.txt',outdir/f'{stem}.hits.txt']
    paths[0].write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8');paths[1].write_text('\n'.join(texts)+'\n',encoding='utf-8')
    paths[2].write_text('\n'.join(ctx)+'\n' if ctx else '',encoding='utf-8');return paths

def connect_sftp(host,user,key_b64,port):
    keydata=base64.b64decode(key_b64).decode();key=None
    for cls in (paramiko.Ed25519Key,paramiko.RSAKey,paramiko.ECDSAKey):
        try:
            from io import StringIO
            key=cls.from_private_key(StringIO(keydata));break
        except Exception: pass
    if key is None: raise RuntimeError('Unsupported SSH private key')
    tr=paramiko.Transport((host,int(port)));tr.connect(username=user,pkey=key);return tr,paramiko.SFTPClient.from_transport(tr)

def sftp_mkdirs(sftp,path):
    cur='/' if path.startswith('/') else ''
    for p in Path(path).parts:
        if p=='/':continue
        cur=(cur.rstrip('/')+'/'+p) if cur else p
        try:sftp.stat(cur)
        except IOError:sftp.mkdir(cur)

def suffix(fp):
    if fp.name.endswith('.hits.txt'):return '.hits.txt'
    if fp.name.endswith('.json'):return '.json'
    if fp.name.endswith('.txt'):return '.txt'
    raise ValueError(fp.name)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--workdir',default='/content/tml_colab')
    ap.add_argument('--vps-host',required=True);ap.add_argument('--vps-user',required=True);ap.add_argument('--vps-key-b64',required=True);ap.add_argument('--vps-port',type=int,default=22)
    ap.add_argument('--remote-cache',required=True);ap.add_argument('--remote-alto-cache',default='');ap.add_argument('--profile',default='HQ',choices=['HQ','STANDARD','CPU_FAST'])
    ap.add_argument('--device',default='auto',choices=['auto','cuda','cpu']);ap.add_argument('--delay',type=float,default=15.0);ap.add_argument('--max-pages',type=int,default=0)
    a=ap.parse_args();wd=Path(a.workdir);wd.mkdir(parents=True,exist_ok=True);rows=read_manifest(a.manifest)
    if a.max_pages:rows=rows[:a.max_pages]
    if any((r.get('mode') or r.get('split') or '').upper()=='RAPID' for r in rows):
        use_cuda,_=ort_device(a.device); compute='CUDA' if use_cuda else 'CPU'
    else: compute='NETWORK'
    tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port);sftp_mkdirs(sftp,a.remote_cache)
    engine=None;done=alto_n=ocr_n=err=0;session=requests.Session();session.headers.update({'User-Agent':'Mozilla/5.0','Accept':'application/xml,text/xml,*/*'})
    for i,r in enumerate(rows,1):
        ark=(r.get('ark') or '').strip();page=pg(r.get('page'));mode=(r.get('mode') or r.get('split') or '').upper();stem=f'{ark}_f{page}'
        if not ark or not page:continue
        remote_json=f"{a.remote_cache.rstrip('/')}/{stem}.json"
        try:sftp.stat(remote_json);print(f'CACHED {i}/{len(rows)} {stem}',flush=True);continue
        except IOError:pass
        try:
            xml_payload=None;t0=time.time()
            if mode=='ALTO':
                rr=session.get(f'https://gallica.bnf.fr/RequestDigitalElement?O={ark}&E=ALTO&Deb={page}',timeout=45)
                if rr.status_code!=200 or not alto_text(rr.content): raise RuntimeError(f'ALTO_UNAVAILABLE status={rr.status_code}')
                rows_out=parse_alto_rows(rr.content);xml_payload=rr.content;alto_n+=1;time.sleep(a.delay);dev='NETWORK'
            elif mode=='RAPID':
                img=session.get(f'https://gallica.bnf.fr/ark:/12148/{ark}/f{page}.highres',timeout=60);img.raise_for_status();ip=wd/f'{stem}.jpg';ip.write_bytes(img.content)
                if engine is None:engine=build_engine(a.profile,a.device)
                rows_out=ocr_image(ip,engine);ocr_n+=1;ip.unlink(missing_ok=True);dev=compute
            else: raise RuntimeError(f'UNKNOWN_SPLIT_MODE {mode!r}')
            source=(r.get('source_image') or '').strip() or 'COLAB'
            files=write_payload(wd/'out',ark,page,rows_out,'ALTO_NATIVE' if mode=='ALTO' else a.profile,source,dev)
            for fp in files:sftp.put(str(fp),f"{a.remote_cache.rstrip('/')}/{stem}{suffix(fp)}")
            if xml_payload and a.remote_alto_cache:
                ad=f"{a.remote_alto_cache.rstrip('/')}/{ark}";sftp_mkdirs(sftp,ad);xp=wd/f'{stem}.xml';xp.write_bytes(xml_payload);sftp.put(str(xp),f'{ad}/f{int(page):03d}.xml');xp.unlink(missing_ok=True)
            done+=1;print(f'DONE {i}/{len(rows)} {stem} mode={mode} device={dev} sec={time.time()-t0:.1f}',flush=True)
        except Exception as e:err+=1;print(f'ERR {i}/{len(rows)} {stem} mode={mode} {type(e).__name__}: {e}',flush=True)
    sftp.close();tr.close();print(json.dumps({'total':len(rows),'done':done,'alto':alto_n,'rapidocr':ocr_n,'errors':err,'rapid_device':compute}),flush=True)
if __name__=='__main__':main()
