#!/usr/bin/env python3
import argparse,base64,csv,importlib.util,json,os,sys,time,threading,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
from worker import connect_sftp

MODEL=None; MOD=None; CFG=None; WORKERS=1

def load_module(path):
    spec=importlib.util.spec_from_file_location('tml_layout_map',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def init_gpu(layout_py,config_json,workers):
    global MODEL,MOD,CFG,WORKERS
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')
    os.environ.setdefault('FLAGS_allocator_strategy','auto_growth')
    import paddle
    paddle.set_device('gpu:0')
    MOD=load_module(layout_py); CFG=json.loads(Path(config_json).read_text()); WORKERS=workers
    from paddlex import create_model
    MODEL=create_model(CFG['layout_model'],device='gpu:0')
    print(f'LAYOUT_ENGINE_READY pid={os.getpid()} device={paddle.device.get_device()} model={CFG["layout_model"]}',flush=True)

def layout_one(item):
    row,ocr_local,img_local,outdir=item
    t0=time.time(); d=json.loads(Path(ocr_local).read_text(encoding='utf-8'))
    d['source']=img_local; Path(ocr_local).write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8')
    res=MOD.process_one(MODEL,Path(ocr_local),outdir,CFG,overwrite=True)
    stem=row['stem']; jp=Path(outdir)/f'{stem}.layout.json'; tp=Path(outdir)/f'{stem}.layout.txt'
    payload=json.loads(jp.read_text(encoding='utf-8'))
    payload['source_image']=row['source_image']; payload['source_ocr_json']=row['ocr_json']
    payload['compute_device']='CUDA'; payload['layout_generation']='colab_a100_layout_v1'; payload['colab_workers']=WORKERS
    jp.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    return stem,str(jp),str(tp),time.time()-t0,res.get('blocks',0),res.get('rows',0)

def gpu_snapshot():
    import subprocess
    try:
        x=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=3).strip().split(',')
        return float(x[0]),float(x[1]),float(x[2])
    except Exception:return 0.0,0.0,0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True); ap.add_argument('--vps-key-b64',required=True)
    ap.add_argument('--vps-port',type=int,default=2222); ap.add_argument('--claim',required=True); ap.add_argument('--stop-flag',required=True)
    ap.add_argument('--workers',type=int,default=4); ap.add_argument('--downloaders',type=int,default=8); ap.add_argument('--poll',type=int,default=10)
    ap.add_argument('--workdir',default='/content/tml_layout_pool'); a=ap.parse_args()
    wd=Path(a.workdir); runtime=wd/'runtime'; runtime.mkdir(parents=True,exist_ok=True)
    tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
    sftp.get('/home/andre/GallicaJobs/_shared/layout_ocr/layout_map.py',str(runtime/'layout_map.py'))
    sftp.get('/home/andre/GallicaJobs/_shared/layout_ocr/config.json',str(runtime/'config.json'))
    sftp.close(); tr.close()
    print(f'LAYOUT_POOL_START workers={a.workers} downloaders={a.downloaders} watch=1',flush=True)
    ctx=mp.get_context('spawn'); pool=ProcessPoolExecutor(max_workers=a.workers,mp_context=ctx,initializer=init_gpu,initargs=(str(runtime/'layout_map.py'),str(runtime/'config.json'),a.workers))
    last_stage=None
    try:
        while True:
            tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
            try:
                try:sftp.stat(a.stop_flag); print('LAYOUT_STOP_FLAG seen=1',flush=True); break
                except IOError:pass
                local_claim=wd/'claim.tsv'; sftp.get(a.claim,str(local_claim))
                with local_claim.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f,delimiter='\t'))
                stage=rows[0]['stage'] if rows else 'EMPTY'
                todo=[]
                for r in rows:
                    try:sftp.stat(r['out_json']); sftp.stat(r['out_txt']); continue
                    except IOError:todo.append(r)
            finally:sftp.close(); tr.close()
            if not todo:
                if stage!=last_stage:print(f'LAYOUT_CLAIM stage={stage} rows={len(rows)} todo=0',flush=True); last_stage=stage
                time.sleep(a.poll); continue
            print(f'LAYOUT_CLAIM stage={stage} rows={len(rows)} todo={len(todo)}',flush=True); last_stage=stage
            stage_dir=wd/stage; inp=stage_dir/'input'; out=stage_dir/'out'; inp.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True)
            tls=threading.local()
            def fetch(r):
                if not hasattr(tls,'sftp'):
                    tls.tr,tls.sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
                op=inp/f"{r['stem']}.json"; ip=inp/f"{r['stem']}.jpg"
                tls.sftp.get(r['ocr_json'],str(op)); tls.sftp.get(r['source_image'],str(ip)); return r,str(op),str(ip)
            with ThreadPoolExecutor(max_workers=a.downloaders) as ex: jobs=list(ex.map(fetch,todo))
            t0=time.time(); futures=[pool.submit(layout_one,(r,op,ip,str(out))) for r,op,ip in jobs]; done=err=0
            for f in as_completed(futures):
                try:
                    stem,jp,tp,sec,blocks,nrows=f.result(); r=next(x for x in todo if x['stem']==stem)
                    tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
                    try:sftp.put(jp,r['out_json']); sftp.put(tp,r['out_txt'])
                    finally:sftp.close(); tr.close()
                    done+=1; u,m,p=gpu_snapshot(); print(f'LAYOUT_GPU_DONE {done}/{len(todo)} {stem} sec={sec:.2f} rows={nrows} blocks={blocks} gpu={u:.0f}% vram={m:.0f}MiB power={p:.0f}W',flush=True)
                except Exception as e:err+=1; print(f'LAYOUT_GPU_ERR {type(e).__name__}: {e}',flush=True)
            elapsed=max(.001,time.time()-t0); print(f'LAYOUT_BATCH_COMPLETE stage={stage} done={done} errors={err} elapsed={elapsed:.1f}s ppm={done*60/elapsed:.2f}',flush=True)
            if err:raise RuntimeError(f'layout errors={err}')
    finally:pool.shutdown(wait=True,cancel_futures=False)

if __name__=='__main__':main()
