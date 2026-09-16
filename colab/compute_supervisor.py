#!/usr/bin/env python3
import argparse, os, socket, subprocess, sys, time, shutil, signal, threading
from pathlib import Path


def connect(host, user, port, key_file):
    import paramiko
    key = paramiko.Ed25519Key.from_private_key_file(str(key_file))
    sock = socket.create_connection((host, port), timeout=15)
    tr = paramiko.Transport(sock); tr.banner_timeout = 15; tr.auth_timeout = 15
    tr.connect(username=user, pkey=key)
    return tr, paramiko.SFTPClient.from_transport(tr)


def remote_exists(sftp, path):
    try: sftp.stat(path); return True
    except IOError: return False


def probe_mode(host,user,port,key_file,mode_path,stop_path):
    tr,sftp=connect(host,user,port,key_file)
    try:
        if remote_exists(sftp,stop_path): return 'STOP'
        try:
            with sftp.open(mode_path,'r') as f: raw=f.read()
            if isinstance(raw,bytes): raw=raw.decode('utf-8','replace')
            mode=str(raw).strip().upper()
            return mode if mode in {'IDLE','RAPID','LAYOUT','STOP'} else 'IDLE'
        except IOError: return 'IDLE'
    finally:
        sftp.close(); tr.close()


def read_mode_safe(host,user,port,key_file,mode_path,stop_path):
    cmd=[sys.executable,'-u',__file__,'--probe-once','--vps-host',host,'--vps-user',user,'--vps-port',str(port),'--vps-key-file',str(key_file),'--mode-path',mode_path,'--stop-path',stop_path]
    try: r=subprocess.run(cmd,capture_output=True,text=True,timeout=25)
    except subprocess.TimeoutExpired: raise TimeoutError('mode probe exceeded 25s')
    if r.returncode!=0: raise RuntimeError((r.stderr or r.stdout or f'probe rc={r.returncode}').strip())
    mode=(r.stdout or '').strip().splitlines()[-1].strip().upper()
    return mode if mode in {'IDLE','RAPID','LAYOUT','STOP'} else 'IDLE'


def cleanup_stale_workers():
    pats=('rapid_watch_filekey.py','rapid_watch_v2.py','rapid_pool.py','layout_watch_filekey.py','layout_pool_v3.py','alto_watch_filekey.py')
    killed=0
    for pat in pats:
        r=subprocess.run(['pkill','-f',pat],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if r.returncode==0: killed+=1
    time.sleep(1)
    print(f'STALE_WORKERS_CLEANED patterns={killed}',flush=True)


def relay_output(proc, mode):
    try:
        if proc.stdout is None: return
        for raw in iter(proc.stdout.readline, ''):
            line=raw.rstrip('\r\n')
            if line: print(f'[{mode}] {line}', flush=True)
    except Exception as e:
        print(f'WORKER_LOG_RELAY_ERROR mode={mode} pid={proc.pid} {type(e).__name__}: {e}', flush=True)
    finally:
        try:
            if proc.stdout is not None: proc.stdout.close()
        except Exception: pass


def launch_worker(cmd, mode):
    proc=subprocess.Popen(cmd,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    t=threading.Thread(target=relay_output,args=(proc,mode),daemon=True,name=f'{mode.lower()}-log-relay-{proc.pid}')
    t.start(); print(f'WORKER_LOG_RELAY_STARTED mode={mode} pid={proc.pid}',flush=True); return proc


def stop_proc(proc):
    if proc is None or proc.poll() is not None: return
    print(f'WORKER_STOP_REQUEST pid={proc.pid}',flush=True)
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        print(f'WORKER_STOP_FORCE pid={proc.pid}',flush=True)
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: pass


def ensure_layout_python(layout_python, repo, device):
    py=Path(layout_python); env=py.parent.parent
    if device=='cuda':
        probe=[str(py),'-c',"import paddle,paddlex,cv2,paramiko,sys; sys.exit(0 if paddle.__version__=='3.2.0' and paddle.device.is_compiled_with_cuda() else 1)"]
    else:
        probe=[str(py),'-c',"import paddle,paddlex,cv2,paramiko,sys; sys.exit(0 if paddle.__version__=='3.2.0' else 1)"]
    if py.exists() and subprocess.run(probe).returncode==0:
        print(f'LAYOUT_ENV_READY cached=1 python={py} device={device.upper()}',flush=True); return str(py)
    uv=shutil.which('uv')
    if not uv: raise RuntimeError('uv not found in supervisor environment')
    print('LAYOUT_BOOTSTRAP 1/4 Python 3.12',flush=True)
    subprocess.run([uv,'python','install','3.12'],check=True,timeout=300)
    if not py.exists(): subprocess.run([uv,'venv','--seed','--python','3.12',str(env)],check=True,timeout=300)
    if device=='cuda':
        paddle_url='https://paddle-whl.cdn.bcebos.com/stable/cu126/paddlepaddle-gpu/paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl'
        wheel=Path('/content/paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl'); expected=1890365820
        need=subprocess.run([str(py),'-c',"import paddle,sys; sys.exit(0 if paddle.__version__=='3.2.0' and paddle.device.is_compiled_with_cuda() else 1)"]).returncode!=0
        if need:
            print('LAYOUT_BOOTSTRAP 2/4 Paddle GPU wheel',flush=True)
            if not wheel.exists() or wheel.stat().st_size!=expected:
                subprocess.run(['curl','-L','--fail','--retry','5','-C','-','--progress-bar','-o',str(wheel),paddle_url],check=True)
            print('LAYOUT_BOOTSTRAP 3/4 install Paddle GPU',flush=True)
            subprocess.run([str(py),'-m','pip','install','--progress-bar','on',str(wheel)],check=True)
    else:
        print('LAYOUT_BOOTSTRAP 2/4 Paddle CPU',flush=True)
        subprocess.run([str(py),'-m','pip','install','--progress-bar','on','paddlepaddle==3.2.0'],check=True)
        print('LAYOUT_BOOTSTRAP 3/4 Paddle CPU ready',flush=True)
    print('LAYOUT_BOOTSTRAP 4/4 layout stack',flush=True)
    subprocess.run([str(py),'-m','pip','install','--progress-bar','on','paddlex==3.7.2','paddleocr==3.7.0','opencv-contrib-python==4.10.0.84','paramiko>=3.5,<4'],check=True)
    if device=='cuda':
        subprocess.run([str(py),'-c',"import paddle,paddlex,cv2; print('LAYOUT_ENV_READY',paddle.__version__,paddlex.__version__,cv2.__version__,'CUDA',paddle.device.is_compiled_with_cuda()); assert paddle.device.is_compiled_with_cuda()"],check=True)
    else:
        subprocess.run([str(py),'-c',"import paddle,paddlex,cv2; paddle.set_device('cpu'); print('LAYOUT_ENV_READY',paddle.__version__,paddlex.__version__,cv2.__version__,'DEVICE',paddle.device.get_device())"],check=True)
    return str(py)


def main():
    ap=argparse.ArgumentParser(description='Single-session supervisor: ALTO is always-on; Rapid/Layout are optional GPU/CPU compute')
    ap.add_argument('--year',type=int,default=1904)
    ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True); ap.add_argument('--vps-port',type=int,default=2222)
    ap.add_argument('--vps-key-file',required=True); ap.add_argument('--repo',required=True)
    ap.add_argument('--rapid-python',required=True); ap.add_argument('--layout-python',required=True)
    ap.add_argument('--compute-device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--rapid-workers',type=int,default=12); ap.add_argument('--rapid-downloaders',type=int,default=8)
    ap.add_argument('--layout-workers',type=int,default=4); ap.add_argument('--layout-downloaders',type=int,default=8)
    ap.add_argument('--poll',type=int,default=10)
    ap.add_argument('--alto-delay',type=float,default=12.0); ap.add_argument('--alto-jitter-max',type=float,default=.25)
    a=ap.parse_args(); base=f'/home/andre/GallicaJobs/gallica-{a.year}-all-tennis/GALlica_{a.year}_ALL_TENNIS'
    mode_path=f'{base}/00_MANIFEST/colab_compute_mode.txt'; stop_path=f'{base}/00_MANIFEST/colab_compute_stop_{a.year}.flag'
    rapid_claim=f'{base}/00_MANIFEST/colab_active_claims.tsv'; layout_claim=f'{base}/00_MANIFEST/colab_layout_active_claim.tsv'; layout_stop=f'{base}/00_MANIFEST/colab_layout_quality_complete_{a.year}.flag'
    key=Path(a.vps_key_file)
    if not key.exists() or key.stat().st_size<100: raise SystemExit(f'Invalid key file: {key}')
    cleanup_stale_workers(); proc=None; current=None; tick=0; started=time.time()
    alto_cmd=[a.rapid_python,'-u',str(Path(a.repo)/'colab/alto_watch_filekey.py'),'--year',str(a.year),'--vps-key-file',str(key),'--vps-host',a.vps_host,'--vps-user',a.vps_user,'--vps-port',str(a.vps_port),'--delay',str(a.alto_delay),'--jitter-min','0','--jitter-max',str(a.alto_jitter_max),'--poll',str(max(5,a.poll))]
    alto_proc=launch_worker(alto_cmd,'ALTO')
    print(f'ALTO_ALWAYS_ON pid={alto_proc.pid} gpu_required=NO delay={a.alto_delay}s lane=COLAB',flush=True)
    print(f'COMPUTE_SUPERVISOR_READY year={a.year} poll={a.poll}s mode_path={mode_path} device={a.compute_device.upper()}',flush=True)
    try:
        while True:
            tick+=1
            if alto_proc.poll() is not None:
                print(f'ALTO_WORKER_EXIT pid={alto_proc.pid} rc={alto_proc.returncode} restart=1',flush=True)
                alto_proc=launch_worker(alto_cmd,'ALTO')
            try: mode=read_mode_safe(a.vps_host,a.vps_user,a.vps_port,key,mode_path,stop_path)
            except Exception as e:
                print(f'SUPERVISOR_POLL_ERROR tick={tick} {type(e).__name__}: {e}',flush=True); time.sleep(a.poll); continue
            if proc is not None and proc.poll() is not None:
                print(f'WORKER_EXIT mode={current} pid={proc.pid} rc={proc.returncode}',flush=True); proc=None
            worker_state='none' if proc is None else f'running pid={proc.pid}'
            print(f'SUPERVISOR_POLL tick={tick} uptime={time.time()-started:.0f}s mode={mode} worker={worker_state} alto=running:{alto_proc.poll() is None} device={a.compute_device.upper()}',flush=True)
            if mode=='STOP':
                # Compute STOP is intentionally NOT an ALTO stop. ALTO runs until pre_ai_complete_<year>.flag or runtime shutdown.
                stop_proc(proc); proc=None; current='STOP'; print('COMPUTE_STOP_ALTO_CONTINUES',flush=True); time.sleep(a.poll); continue
            if mode!=current:
                print(f'MODE_CHANGE {current}->{mode}',flush=True); stop_proc(proc); proc=None; current=mode
            if current=='IDLE':
                print(f'NO_COMPUTE_WORK tick={tick} ALTO_STILL_ACTIVE=1 next_poll={a.poll}s',flush=True)
            elif current=='RAPID' and proc is None:
                cmd=[a.rapid_python,'-u',str(Path(a.repo)/'colab/rapid_watch_filekey.py'),'--vps-key-file',str(key),'--vps-host',a.vps_host,'--vps-user',a.vps_user,'--vps-port',str(a.vps_port),'--claim',rapid_claim,'--stop-flag',stop_path,'--workers',str(a.rapid_workers),'--downloaders',str(a.rapid_downloaders),'--device',a.compute_device,'--poll',str(a.poll)]
                proc=launch_worker(cmd,'RAPID')
                print(f'WORKER_LAUNCHED mode=RAPID pid={proc.pid} workers={a.rapid_workers} downloaders={a.rapid_downloaders} device={a.compute_device.upper()}',flush=True)
            elif current=='LAYOUT' and proc is None:
                try: layout_py=ensure_layout_python(a.layout_python,a.repo,a.compute_device)
                except Exception as e:
                    print(f'LAYOUT_BOOTSTRAP_ERROR {type(e).__name__}: {e}',flush=True); time.sleep(a.poll); continue
                generation=f'colab_{a.compute_device}_layout_{a.year}_unified'
                cmd=[layout_py,'-u',str(Path(a.repo)/'colab/layout_watch_filekey.py'),'--vps-key-file',str(key),'--vps-host',a.vps_host,'--vps-user',a.vps_user,'--vps-port',str(a.vps_port),'--claim',layout_claim,'--stop-flag',layout_stop,'--workers',str(a.layout_workers),'--downloaders',str(a.layout_downloaders),'--device',a.compute_device,'--poll',str(a.poll),'--generation-label',generation]
                proc=launch_worker(cmd,'LAYOUT')
                print(f'WORKER_LAUNCHED mode=LAYOUT pid={proc.pid} workers={a.layout_workers} downloaders={a.layout_downloaders} device={a.compute_device.upper()}',flush=True)
            time.sleep(a.poll)
    finally:
        stop_proc(proc); stop_proc(alto_proc)


def probe_entry():
    ap=argparse.ArgumentParser(); ap.add_argument('--probe-once',action='store_true'); ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True); ap.add_argument('--vps-port',type=int,default=2222); ap.add_argument('--vps-key-file',required=True); ap.add_argument('--mode-path',required=True); ap.add_argument('--stop-path',required=True)
    a=ap.parse_args(); print(probe_mode(a.vps_host,a.vps_user,a.vps_port,Path(a.vps_key_file),a.mode_path,a.stop_path),flush=True)

if __name__=='__main__':
    if '--probe-once' in sys.argv: probe_entry()
    else: raise SystemExit(main())
