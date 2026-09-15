#!/usr/bin/env python3
import base64, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
YEAR=int(os.environ.get('TML_YEAR','1905'))
MODE=os.environ.get('TML_KAGGLE_MODE','BENCHMARK').strip().upper()
WORKERS=int(os.environ.get('TML_KAGGLE_WORKERS_PER_GPU','3'))
DOWNLOADERS=int(os.environ.get('TML_KAGGLE_DOWNLOADERS_PER_GPU','4'))
VPS_HOST=os.environ.get('TML_VPS_HOST','vibrant-lovelace.82-165-11-122.plesk.page')
VPS_USER=os.environ.get('TML_VPS_USER','andre')
VPS_PORT=int(os.environ.get('TML_VPS_PORT','2222'))
BASE=f'/home/andre/GallicaJobs/gallica-{YEAR}-all-tennis/GALlica_{YEAR}_ALL_TENNIS'
CLAIM=f'{BASE}/00_MANIFEST/kaggle_benchmark_claim.tsv' if MODE=='BENCHMARK' else f'{BASE}/00_MANIFEST/kaggle_active_claims.tsv'
STOP=f'{BASE}/00_MANIFEST/kaggle_rapid_stop_{YEAR}.flag'
KEY_FILE=Path('/kaggle/working/tml_kaggle_key')

print(f'TML_KAGGLE_BOOTSTRAP year={YEAR} mode={MODE} workers_per_gpu={WORKERS} downloaders_per_gpu={DOWNLOADERS}',flush=True)
if MODE not in {'BENCHMARK','PRODUCTION'}: raise SystemExit('TML_KAGGLE_MODE must be BENCHMARK or PRODUCTION')

# Keep the environment identical to the established TML CUDA worker.
if os.environ.get('TML_SKIP_INSTALL')!='1':
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','onnxruntime','onnxruntime-gpu'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(ROOT/'colab/requirements.txt')],check=True)

subprocess.run(['nvidia-smi','-L'],check=True)
import onnxruntime as ort
providers=ort.get_available_providers(); print('ORT',ort.__version__,'providers',providers,flush=True)
if 'CUDAExecutionProvider' not in providers: raise SystemExit('CUDAExecutionProvider unavailable. Enable Kaggle GPU accelerator.')

key_b64=None
try:
    from kaggle_secrets import UserSecretsClient
    key_b64=UserSecretsClient().get_secret('TML_VPS_SSH_KEY_B64')
except Exception:
    pass

if key_b64:
    KEY_FILE.write_bytes(base64.b64decode(key_b64)); os.chmod(KEY_FILE,0o600)
    print('SSH_KEY_READY source=KaggleSecret',flush=True)
elif not KEY_FILE.exists():
    subprocess.run(['ssh-keygen','-t','ed25519','-N','','-f',str(KEY_FILE),'-C','tml-kaggle-worker'],check=True,stdout=subprocess.DEVNULL)
    print('KAGGLE_SSH_SETUP_REQUIRED',flush=True)
    print('SEND ONLY THIS PUBLIC KEY TO CHATGPT:',flush=True)
    print((KEY_FILE.with_suffix('.pub')).read_text().strip(),flush=True)
    print('Keep this Kaggle session open. After the key is authorized on the VPS, run the same cell again.',flush=True)
    raise SystemExit(4)
else:
    print('SSH_KEY_READY source=current_Kaggle_session',flush=True)

key_b64=base64.b64encode(KEY_FILE.read_bytes()).decode()
cmd=[sys.executable,'-u',str(ROOT/'kaggle/dual_t4_watch.py'),
     '--vps-host',VPS_HOST,'--vps-user',VPS_USER,'--vps-port',str(VPS_PORT),
     '--vps-key-b64',key_b64,'--claim',CLAIM,'--stop-flag',STOP,
     '--workers-per-gpu',str(WORKERS),'--downloaders-per-gpu',str(DOWNLOADERS),'--poll','10']
if MODE=='BENCHMARK': cmd.append('--once')
print('STARTING_KAGGLE_WORKER',MODE,'claim',CLAIM,flush=True)
raise SystemExit(subprocess.run(cmd).returncode)
