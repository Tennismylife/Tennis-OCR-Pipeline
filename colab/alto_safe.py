#!/usr/bin/env python3
import argparse,time,requests
from pathlib import Path
from worker import read_manifest,pg,alto_text,parse_alto_rows,connect_sftp,sftp_mkdirs,write_payload,suffix

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True);ap.add_argument('--vps-host',required=True);ap.add_argument('--vps-user',required=True)
    ap.add_argument('--vps-key-b64',required=True);ap.add_argument('--vps-port',type=int,default=2222);ap.add_argument('--remote-cache',required=True)
    ap.add_argument('--remote-alto-cache',default='');ap.add_argument('--delay',type=float,default=16.0);ap.add_argument('--workdir',default='/content/tml_alto')
    a=ap.parse_args();rows=read_manifest(a.manifest);wd=Path(a.workdir);wd.mkdir(parents=True,exist_ok=True)
    tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port);sftp_mkdirs(sftp,a.remote_cache)
    ses=requests.Session();ses.headers.update({'User-Agent':'Mozilla/5.0','Accept':'application/xml,text/xml,*/*'})
    next_request=0.0;done=err=0
    for i,r in enumerate(rows,1):
        ark=(r.get('ark') or '').strip();page=pg(r.get('page'));stem=f'{ark}_f{page}'
        if not ark or not page:continue
        try:sftp.stat(f"{a.remote_cache.rstrip('/')}/{stem}.json");print(f'CACHED {i}/{len(rows)} {stem}',flush=True);continue
        except IOError:pass
        wait=max(0.0,next_request-time.monotonic())
        if wait:time.sleep(wait)
        started=time.monotonic();next_request=started+a.delay
        try:
            rr=ses.get(f'https://gallica.bnf.fr/RequestDigitalElement?O={ark}&E=ALTO&Deb={page}',timeout=60)
            if rr.status_code==429:
                retry=float(rr.headers.get('Retry-After') or 30);next_request=max(next_request,time.monotonic()+retry)
                raise RuntimeError(f'HTTP429 retry_after={retry}')
            if rr.status_code!=200 or not alto_text(rr.content):raise RuntimeError(f'ALTO_UNAVAILABLE status={rr.status_code}')
            out=write_payload(wd,ark,page,parse_alto_rows(rr.content),'ALTO_NATIVE',(r.get('source_image') or 'COLAB'),'NETWORK')
            for fp in out:sftp.put(str(fp),f"{a.remote_cache.rstrip('/')}/{stem}{suffix(fp)}")
            if a.remote_alto_cache:
                ad=f"{a.remote_alto_cache.rstrip('/')}/{ark}";sftp_mkdirs(sftp,ad);xp=wd/f'{stem}.xml';xp.write_bytes(rr.content)
                sftp.put(str(xp),f'{ad}/f{int(page):03d}.xml');xp.unlink(missing_ok=True)
            done+=1;print(f'DONE {i}/{len(rows)} {stem} ALTO cadence={a.delay:.1f}s',flush=True)
        except Exception as e:err+=1;print(f'ERR {i}/{len(rows)} {stem} {e}',flush=True)
    sftp.close();tr.close();print({'total':len(rows),'done':done,'errors':err,'min_request_interval_sec':a.delay},flush=True)
if __name__=='__main__':main()
