#!/usr/bin/env python3
import argparse, csv, io, os, random, socket, time, urllib.error, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath


def connect(host, user, port, key_file):
    import paramiko
    key = paramiko.Ed25519Key.from_private_key_file(str(key_file))
    sock = socket.create_connection((host, port), timeout=15)
    tr = paramiko.Transport(sock); tr.banner_timeout = 15; tr.auth_timeout = 15
    tr.connect(username=user, pkey=key)
    return tr, paramiko.SFTPClient.from_transport(tr)


def exists(sftp, path):
    try: sftp.stat(path); return True
    except IOError: return False


def mkdir_p(sftp, path):
    cur=''
    for part in PurePosixPath(path).parts:
        if part=='/': cur='/' ; continue
        cur=(cur.rstrip('/')+'/'+part) if cur else part
        try: sftp.stat(cur)
        except IOError:
            try: sftp.mkdir(cur)
            except IOError: pass


def read_text(sftp, path):
    with sftp.open(path,'r') as f:
        raw=f.read()
    return raw.decode('utf-8','replace') if isinstance(raw,bytes) else str(raw)


def atomic_put_bytes(sftp, remote_path, data):
    parent=str(PurePosixPath(remote_path).parent); mkdir_p(sftp,parent)
    tmp=remote_path+f'.tmp.{os.getpid()}'
    with sftp.open(tmp,'wb') as f: f.write(data)
    try: sftp.posix_rename(tmp,remote_path)
    except Exception:
        try: sftp.remove(remote_path)
        except IOError: pass
        sftp.rename(tmp,remote_path)


def parse_alto(payload):
    root=ET.fromstring(payload); lines=[]
    for tl in root.iter():
        if tl.tag.split('}')[-1] != 'TextLine': continue
        words=[]
        for s in tl.iter():
            if s.tag.split('}')[-1] != 'String': continue
            v=(s.attrib.get('SUBS_CONTENT') or s.attrib.get('CONTENT') or '').strip()
            if v: words.append(v)
        if words: lines.append(' '.join(words))
    return '\n'.join(lines).strip()


def fetch_alto(ark,page,timeout=45):
    url='https://gallica.bnf.fr/RequestDigitalElement?'+urllib.parse.urlencode({'O':ark,'E':'ALTO','Deb':str(page)})
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36','Accept':'application/xml,text/xml;q=0.9,*/*;q=0.8','Referer':'https://gallica.bnf.fr/'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            return r.status,r.read(),float(r.headers.get('Retry-After') or 0)
    except urllib.error.HTTPError as e:
        try: ra=float(e.headers.get('Retry-After') or 0) if e.headers else 0
        except: ra=0
        return e.code,b'',ra
    except Exception:
        return 0,b'',0


def read_claims(sftp, path):
    try: txt=read_text(sftp,path)
    except IOError: return []
    rows=[]
    for r in csv.DictReader(io.StringIO(txt),delimiter='\t'):
        ark=(r.get('ark') or '').strip(); page=(r.get('page') or '').strip()
        if not ark or not page: continue
        try: page=str(int(float(page)))
        except: continue
        rows.append((ark,page))
    return rows


def main():
    ap=argparse.ArgumentParser(description='Always-on Gallica ALTO lane for Colab; works with or without GPU')
    ap.add_argument('--year',type=int,required=True)
    ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True); ap.add_argument('--vps-port',type=int,default=2222)
    ap.add_argument('--vps-key-file',required=True)
    ap.add_argument('--delay',type=float,default=12.0); ap.add_argument('--jitter-min',type=float,default=0.0); ap.add_argument('--jitter-max',type=float,default=.25)
    ap.add_argument('--poll',type=int,default=10)
    a=ap.parse_args()
    base=f'/home/andre/GallicaJobs/gallica-{a.year}-all-tennis/GALlica_{a.year}_ALL_TENNIS'
    claim=f'{base}/00_MANIFEST/colab_alto_active_claims.tsv'
    final_flag=f'{base}/00_MANIFEST/pre_ai_complete_{a.year}.flag'
    cache='/home/andre/GallicaJobs/_shared/alto_cache'
    last=0.0; done=set(); tick=0
    print(f'COLAB_ALTO_READY year={a.year} lane=independent delay={a.delay}s jitter={a.jitter_min}-{a.jitter_max} gpu_required=NO',flush=True)
    while True:
        tick+=1
        try: tr,sftp=connect(a.vps_host,a.vps_user,a.vps_port,a.vps_key_file)
        except Exception as e:
            print(f'COLAB_ALTO_CONNECT_ERROR tick={tick} {type(e).__name__}: {e}',flush=True); time.sleep(a.poll); continue
        try:
            if exists(sftp,final_flag):
                print('COLAB_ALTO_FINAL_FLAG_SEEN',flush=True); return 0
            claims=read_claims(sftp,claim)
            if not claims:
                if tick%6==1: print(f'COLAB_ALTO_IDLE tick={tick}',flush=True)
                time.sleep(a.poll); continue
            progressed=0
            for ark,page in claims:
                xmlp=f'{cache}/{ark}/f{int(page):03d}.xml'; txtp=f'{cache}/{ark}/f{int(page):03d}.txt'
                if exists(sftp,xmlp) and exists(sftp,txtp):
                    done.add((ark,page)); continue
                wait=max(0.0,last+a.delay+random.uniform(a.jitter_min,a.jitter_max)-time.time()) if last else 0.0
                if wait: time.sleep(wait)
                code,payload,ra=fetch_alto(ark,page); last=time.time()
                if code==200 and payload:
                    try: text=parse_alto(payload)
                    except Exception as e:
                        print(f'COLAB_ALTO_PARSE_FAIL {ark} f{page} {type(e).__name__}',flush=True); continue
                    atomic_put_bytes(sftp,xmlp,payload)
                    atomic_put_bytes(sftp,txtp,(text+'\n').encode('utf-8'))
                    done.add((ark,page)); progressed+=1
                    print(f'COLAB_ALTO_OK ark={ark} page={page} chars={len(text)} total_session={len(done)}',flush=True)
                elif code==429:
                    cool=max(60.0,min(ra or 0,600.0)); print(f'COLAB_ALTO_429 ark={ark} page={page} cooldown={cool:.0f}s',flush=True); time.sleep(cool); break
                else:
                    print(f'COLAB_ALTO_FAIL ark={ark} page={page} http={code}',flush=True)
            if progressed==0: time.sleep(a.poll)
        finally:
            try:sftp.close();tr.close()
            except:pass

if __name__=='__main__': raise SystemExit(main())
