#!/usr/bin/env python3
import hashlib, json, time
from pathlib import PurePosixPath


def _mkdir_p(sftp, path):
    cur=''
    for part in PurePosixPath(path).parts:
        if part=='/':
            cur='/'
            continue
        cur=(cur.rstrip('/')+'/'+part) if cur else part
        try:
            sftp.stat(cur)
        except IOError:
            try:sftp.mkdir(cur)
            except IOError:pass


def _safe_key(key):
    raw=str(key).encode('utf-8','replace')
    return hashlib.sha1(raw).hexdigest()[:20]


def heartbeat(sftp, base, worker_id, capabilities=''):
    root=f'{base}/00_MANIFEST/colab_workers'
    _mkdir_p(sftp,root)
    path=f'{root}/{worker_id}.json'
    tmp=path+f'.tmp.{int(time.time()*1000)}'
    payload=json.dumps({'worker_id':worker_id,'ts':time.time(),'capabilities':capabilities},separators=(',',':')).encode()
    with sftp.open(tmp,'wb') as f:f.write(payload)
    try:sftp.posix_rename(tmp,path)
    except Exception:
        try:sftp.remove(path)
        except IOError:pass
        sftp.rename(tmp,path)


class ClaimLease:
    def __init__(self,sftp,base,worker_id,channel,key,ttl=600):
        self.sftp=sftp; self.base=base.rstrip('/'); self.worker_id=worker_id; self.channel=channel; self.key=str(key); self.ttl=float(ttl)
        root=f'{self.base}/00_MANIFEST/colab_multi_claims/{channel}'
        _mkdir_p(sftp,root)
        self.path=f'{root}/{_safe_key(self.key)}.lock'
        self.owner=f'{self.path}/owner.json'
        self.acquired=False

    def _try_mkdir(self):
        try:
            self.sftp.mkdir(self.path)
            self.acquired=True
            payload=json.dumps({'worker_id':self.worker_id,'key':self.key,'ts':time.time()},separators=(',',':')).encode()
            with self.sftp.open(self.owner,'wb') as f:f.write(payload)
            return True
        except IOError:
            return False

    def acquire(self):
        if self._try_mkdir():return True
        try:st=self.sftp.stat(self.path)
        except IOError:return self._try_mkdir()
        age=max(0.0,time.time()-float(st.st_mtime))
        if age<=self.ttl:return False
        try:self.sftp.remove(self.owner)
        except IOError:pass
        try:self.sftp.rmdir(self.path)
        except IOError:return False
        return self._try_mkdir()

    def touch(self):
        if not self.acquired:return
        payload=json.dumps({'worker_id':self.worker_id,'key':self.key,'ts':time.time()},separators=(',',':')).encode()
        try:
            with self.sftp.open(self.owner,'wb') as f:f.write(payload)
        except Exception:pass

    def release(self):
        if not self.acquired:return
        try:self.sftp.remove(self.owner)
        except IOError:pass
        try:self.sftp.rmdir(self.path)
        except IOError:pass
        self.acquired=False
