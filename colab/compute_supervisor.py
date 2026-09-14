#!/usr/bin/env python3
import argparse, os, subprocess, sys, time
from pathlib import Path
import paramiko


def connect(host, user, port, key_file):
    key = paramiko.Ed25519Key.from_private_key_file(str(key_file))
    tr = paramiko.Transport((host, port))
    tr.connect(username=user, pkey=key)
    return tr, paramiko.SFTPClient.from_transport(tr)


def remote_exists(sftp, path):
    try:
        sftp.stat(path)
        return True
    except IOError:
        return False


def read_mode(host, user, port, key_file, mode_path, stop_path):
    tr, sftp = connect(host, user, port, key_file)
    try:
        if remote_exists(sftp, stop_path):
            return 'STOP'
        try:
            with sftp.open(mode_path, 'r') as f:
                raw = f.read()
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', 'replace')
            mode = str(raw).strip().upper()
            return mode if mode in {'IDLE','RAPID','LAYOUT','STOP'} else 'IDLE'
        except IOError:
            return 'IDLE'
    finally:
        sftp.close(); tr.close()


def stop_proc(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait(timeout=10)


def main():
    ap = argparse.ArgumentParser(description='Single-session supervisor for isolated RapidOCR and Layout environments')
    ap.add_argument('--year', type=int, default=1904)
    ap.add_argument('--vps-host', required=True); ap.add_argument('--vps-user', required=True); ap.add_argument('--vps-port', type=int, default=2222)
    ap.add_argument('--vps-key-file', required=True); ap.add_argument('--repo', required=True)
    ap.add_argument('--rapid-python', required=True); ap.add_argument('--layout-python', required=True)
    ap.add_argument('--rapid-workers', type=int, default=12); ap.add_argument('--rapid-downloaders', type=int, default=8)
    ap.add_argument('--layout-workers', type=int, default=4); ap.add_argument('--layout-downloaders', type=int, default=8)
    ap.add_argument('--poll', type=int, default=10)
    a = ap.parse_args()
    base = f'/home/andre/GallicaJobs/gallica-{a.year}-all-tennis/GALlica_{a.year}_ALL_TENNIS'
    mode_path = f'{base}/00_MANIFEST/colab_compute_mode.txt'
    stop_path = f'{base}/00_MANIFEST/colab_compute_stop_{a.year}.flag'
    rapid_claim = f'{base}/00_MANIFEST/colab_active_claims.tsv'
    layout_claim = f'{base}/00_MANIFEST/colab_layout_active_claim.tsv'
    layout_stop = f'{base}/00_MANIFEST/colab_layout_quality_complete_{a.year}.flag'
    key = Path(a.vps_key_file)
    if not key.exists() or key.stat().st_size < 100:
        raise SystemExit(f'Invalid key file: {key}')
    proc = None; current = None; last_rc = None
    print(f'COMPUTE_SUPERVISOR_READY year={a.year} mode_path={mode_path}', flush=True)
    try:
        while True:
            try:
                mode = read_mode(a.vps_host,a.vps_user,a.vps_port,key,mode_path,stop_path)
            except Exception as e:
                print(f'MODE_POLL_ERROR {type(e).__name__}: {e}', flush=True); time.sleep(a.poll); continue
            if mode == 'STOP':
                print('COMPUTE_SUPERVISOR_STOP', flush=True); stop_proc(proc); return 0
            if proc is not None and proc.poll() is not None:
                last_rc = proc.returncode
                print(f'WORKER_EXIT mode={current} rc={last_rc}', flush=True)
                proc = None
            if mode != current:
                print(f'MODE_CHANGE {current}->{mode}', flush=True)
                stop_proc(proc); proc = None; current = mode
            if current == 'RAPID' and proc is None:
                cmd=[a.rapid_python,'-u',str(Path(a.repo)/'colab/rapid_watch_filekey.py'),'--vps-key-file',str(key),'--vps-host',a.vps_host,'--vps-user',a.vps_user,'--vps-port',str(a.vps_port),'--claim',rapid_claim,'--stop-flag',stop_path,'--workers',str(a.rapid_workers),'--downloaders',str(a.rapid_downloaders),'--poll',str(a.poll)]
                print('START_RAPID '+ ' '.join(cmd[:3]) + f' workers={a.rapid_workers}', flush=True)
                proc=subprocess.Popen(cmd); last_rc=None
            elif current == 'LAYOUT' and proc is None:
                cmd=[a.layout_python,'-u',str(Path(a.repo)/'colab/layout_watch_filekey.py'),'--vps-key-file',str(key),'--vps-host',a.vps_host,'--vps-user',a.vps_user,'--vps-port',str(a.vps_port),'--claim',layout_claim,'--stop-flag',layout_stop,'--workers',str(a.layout_workers),'--downloaders',str(a.layout_downloaders),'--poll',str(a.poll),'--generation-label',f'colab_a100_layout_{a.year}_unified']
                print('START_LAYOUT '+ ' '.join(cmd[:3]) + f' workers={a.layout_workers}', flush=True)
                proc=subprocess.Popen(cmd); last_rc=None
            time.sleep(a.poll)
    finally:
        stop_proc(proc)

if __name__ == '__main__':
    raise SystemExit(main())
