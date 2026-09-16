#!/usr/bin/env python3
import argparse,csv,importlib.util,json,os,time,threading,multiprocessing as mp,uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,wait,FIRST_COMPLETED
from worker import connect_sftp
from claim_lease import ClaimLease, heartbeat
MODEL=None; MOD=None; CFG=None; WORKERS=1; GENERATION='colab_layout_v3'; DEVICE='cuda'

def load_module(path):
    spec=importlib.util.spec_from_file_location('tml_layout_map',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def init_layout(layout_py,config_json,model_dir,workers,generation,device):
    global MODEL,MOD,CFG,WORKERS,GENERATION,DEVICE
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True'); os.environ.setdefault('FLAGS_allocator_strategy','auto_growth'); DEVICE=device
    import paddle
    paddle_device='gpu:0' if device=='cuda' else 'cpu'; paddle.set_device(paddle_device)
    MOD=load_module(layout_py); CFG=json.loads(Path(config_json).read_text()); WORKERS=workers; GENERATION=generation
    from paddlex import create_model
    MODEL=create_model(CFG['layout_model'],model_dir=model_dir,device=paddle_device)
    print(f'LAYOUT_ENGINE_READY pid={os.getpid()} device={paddle.device.get_device()} model={CFG["layout_model"]} generation={GENERATION}',flush=True)

def engine_probe(i):
    import paddle,time; time.sleep(1); return i,os.getpid(),paddle.device.get_device(),MODEL is not None

def layout_one(item):
    idx,total,row,ocr_local,img_local,outdir=item; t0=time.time(); stem=row['stem']
    print(f'LAYOUT_START idx={idx}/{total} stem={stem} pid={os.getpid()} device={DEVICE.upper()}',flush=True)
    p=Path(ocr_local); d=json.loads(p.read_text(encoding='utf-8')); d['source']=img_local; p.write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8')
    res=MOD.process_one(MODEL,p,outdir,CFG,overwrite=True)
    jp=Path(outdir)/f'{stem}.layout.json'; tp=Path(outdir)/f'{stem}.layout.txt'; payload=json.loads(jp.read_text(encoding='utf-8'))
    payload.update(source_image=row['source_image'],source_ocr_json=row['ocr_json'],compute_device=('CUDA' if DEVICE=='cuda' else 'CPU'),layout_generation=GENERATION,colab_workers=WORKERS)
    jp.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
    return idx,total,stem,str(jp),str(tp),time.time()-t0,res.get('blocks',0),res.get('rows',0)

def device_snapshot(device):
    if device!='cuda': return 0.0,0.0,0.0
    import subprocess
    try:
        x=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=3).strip().split(','); return float(x[0]),float(x[1]),float(x[2])
    except Exception:return 0.0,0.0,0.0

def fetch_vps_model(sftp,remote_dir,local_dir):
    local_dir=Path(local_dir); local_dir.mkdir(parents=True,exist_ok=True); names=['config.json','.gitattributes','inference.pdiparams','inference.yml','README.md','inference.json']; total=0
    for name in names:
        rp=f'{remote_dir.rstrip("/")}/{name}'; lp=local_dir/name
        try: st=sftp.stat(rp)
        except IOError:
            if name in ('.gitattributes','README.md'): continue
            raise
        if lp.exists() and lp.stat().st_size==st.st_size: print(f'MODEL_CACHE_HIT {name} bytes={st.st_size}',flush=True); total+=st.st_size; continue
        print(f'MODEL_FETCH {name} bytes={st.st_size}',flush=True); sftp.get(rp,str(lp)); total+=st.st_size
    print(f'MODEL_READY path={local_dir} bytes={total}',flush=True); return str(local_dir)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--vps-host',required=True); ap.add_argument('--vps-user',required=True); ap.add_argument('--vps-key-b64',required=True); ap.add_argument('--vps-port',type=int,default=2222)
    ap.add_argument('--claim',required=True); ap.add_argument('--stop-flag',required=True); ap.add_argument('--vps-model-dir',default='/home/andre/.paddlex/official_models/PP-DocLayout_plus-L')
    ap.add_argument('--remote-layout-py',default='/home/andre/GallicaJobs/_shared/layout_ocr/layout_map.py'); ap.add_argument('--remote-layout-config',default='/home/andre/GallicaJobs/_shared/layout_ocr/config.json')
    ap.add_argument('--workers',type=int,default=4); ap.add_argument('--downloaders',type=int,default=8); ap.add_argument('--device',choices=['cuda','cpu'],default='cuda'); ap.add_argument('--poll',type=int,default=10); ap.add_argument('--status-every',type=int,default=10)
    ap.add_argument('--generation-label',default='colab_layout_v3'); ap.add_argument('--workdir',default='/content/tml_layout_pool_v3')
    ap.add_argument('--year',type=int,required=True); ap.add_argument('--worker-id',default=''); ap.add_argument('--lease-ttl',type=int,default=1200); ap.add_argument('--lease-batch',type=int,default=0)
    a=ap.parse_args(); worker_id=(a.worker_id or f'colab-{uuid.uuid4().hex[:10]}').strip(); base=f'/home/andre/GallicaJobs/gallica-{a.year}-all-tennis/GALlica_{a.year}_ALL_TENNIS'; lease_batch=a.lease_batch or max(a.workers*2,2)
    wd=Path(a.workdir)/worker_id; runtime=wd/'runtime'; runtime.mkdir(parents=True,exist_ok=True)
    tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port); heartbeat(sftp,base,worker_id,'LAYOUT')
    print('LAYOUT_BOOTSTRAP fetching shared layout code/config/model from VPS',flush=True)
    sftp.get(a.remote_layout_py,str(runtime/'layout_map.py')); sftp.get(a.remote_layout_config,str(runtime/'config.json')); model_dir=fetch_vps_model(sftp,a.vps_model_dir,runtime/'PP-DocLayout_plus-L'); sftp.close(); tr.close()
    print(f'LAYOUT_POOL_START worker_id={worker_id} multicolab=YES workers={a.workers} downloaders={a.downloaders} lease_batch={lease_batch} poll={a.poll}s status_every={a.status_every}s generation={a.generation_label} device={a.device.upper()}',flush=True)
    ctx=mp.get_context('spawn'); pool=ProcessPoolExecutor(max_workers=a.workers,mp_context=ctx,initializer=init_layout,initargs=(str(runtime/'layout_map.py'),str(runtime/'config.json'),model_dir,a.workers,a.generation_label,a.device))
    print('LAYOUT_PREFLIGHT starting...',flush=True); probes=[pool.submit(engine_probe,i) for i in range(a.workers)]; probe_rows=[f.result(timeout=240) for f in probes]; print('LAYOUT_PREFLIGHT_OK '+json.dumps(probe_rows),flush=True)
    upload_tr=upload_sftp=None; poll_tick=0
    def upload_pair(idx,total,jp,tp,r):
        nonlocal upload_tr,upload_sftp
        for attempt in (1,2):
            try:
                if upload_sftp is None: upload_tr,upload_sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
                upload_sftp.put(jp,r['out_json']); upload_sftp.put(tp,r['out_txt']); print(f'LAYOUT_UPLOAD_DONE idx={idx}/{total} stem={r["stem"]}',flush=True); return
            except Exception as e:
                print(f'LAYOUT_UPLOAD_RETRY idx={idx}/{total} stem={r["stem"]} attempt={attempt} error={type(e).__name__}: {e}',flush=True)
                try:
                    if upload_sftp: upload_sftp.close()
                    if upload_tr: upload_tr.close()
                except Exception: pass
                upload_tr=upload_sftp=None
                if attempt==2: raise
    try:
        while True:
            poll_tick+=1; poll_t0=time.time(); tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port); lease_keys=[]; selected=[]
            try:
                heartbeat(sftp,base,worker_id,'LAYOUT')
                try: sftp.stat(a.stop_flag); print(f'LAYOUT_STOP_FLAG tick={poll_tick} seen=1',flush=True); break
                except IOError: pass
                local_claim=wd/'claim.tsv'; sftp.get(a.claim,str(local_claim))
                with local_claim.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f,delimiter='\t'))
                stage=rows[0]['stage'] if rows else 'EMPTY'; todo=[]
                for r in rows:
                    try: sftp.stat(r['out_json']); sftp.stat(r['out_txt'])
                    except IOError: todo.append(r)
                for r in todo:
                    if len(selected)>=lease_batch: break
                    key=f'{stage}:{r["stem"]}'; lease=ClaimLease(sftp,base,worker_id,'layout',key,ttl=a.lease_ttl)
                    if lease.acquire(): selected.append(r); lease_keys.append(key)
            finally: sftp.close(); tr.close()
            print(f'LAYOUT_POLL worker_id={worker_id} tick={poll_tick} stage={stage} rows={len(rows)} todo={len(todo)} leased={len(selected)} complete={len(rows)-len(todo)}/{len(rows)} poll_sec={time.time()-poll_t0:.2f}',flush=True)
            if not selected:
                print(f'LAYOUT_IDLE worker_id={worker_id} tick={poll_tick} stage={stage} todo={len(todo)} leased=0 next_poll={a.poll}s',flush=True); time.sleep(a.poll); continue
            try:
                print(f'LAYOUT_CLAIM_START worker_id={worker_id} stage={stage} rows={len(rows)} leased={len(selected)} workers={a.workers} device={a.device.upper()}',flush=True)
                stage_dir=wd/stage; inp=stage_dir/'input'; out=stage_dir/'out'; inp.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True); tls=threading.local(); fetched=0; fetch_t0=time.time(); last_fetch_log=fetch_t0
                def fetch(idx,r):
                    stem=r['stem']; print(f'LAYOUT_FETCH_START idx={idx}/{len(selected)} stem={stem}',flush=True)
                    if not hasattr(tls,'sftp'): tls.tr,tls.sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
                    op=inp/f'{stem}.json'; ip=inp/f'{stem}.jpg'; tls.sftp.get(r['ocr_json'],str(op)); tls.sftp.get(r['source_image'],str(ip)); print(f'LAYOUT_FETCH_DONE idx={idx}/{len(selected)} stem={stem}',flush=True); return idx,r,str(op),str(ip)
                jobs=[]
                with ThreadPoolExecutor(max_workers=a.downloaders) as ex:
                    pending={ex.submit(fetch,idx,r):(idx,r) for idx,r in enumerate(selected,1)}
                    while pending:
                        ready,_=wait(pending,timeout=a.status_every,return_when=FIRST_COMPLETED); now=time.time()
                        if not ready: print(f'LAYOUT_FETCH_HEARTBEAT worker_id={worker_id} fetched={fetched}/{len(selected)} pending={len(pending)} elapsed={now-fetch_t0:.1f}s',flush=True); continue
                        for f in ready:
                            idx_ctx,r_ctx=pending.pop(f)
                            try: jobs.append(f.result()); fetched+=1
                            except Exception as e: print(f'LAYOUT_FETCH_ERROR idx={idx_ctx}/{len(selected)} stem={r_ctx.get("stem")} {type(e).__name__}: {e}',flush=True); raise
                        if fetched==len(selected) or now-last_fetch_log>=a.status_every: print(f'LAYOUT_FETCH_PROGRESS worker_id={worker_id} fetched={fetched}/{len(selected)} pending={len(pending)} elapsed={now-fetch_t0:.1f}s',flush=True); last_fetch_log=now
                by_stem={r['stem']:r for r in selected}; total=len(selected); t0=time.time(); done=err=uploaded=0; last_done_at=t0; last_status=0; pending={}
                for idx,r,op,ip in jobs:
                    fut=pool.submit(layout_one,(idx,total,r,op,ip,str(out))); pending[fut]=(idx,r); print(f'LAYOUT_QUEUED worker_id={worker_id} idx={idx}/{total} stem={r["stem"]} inflight={len(pending)}',flush=True)
                while pending:
                    ready,_=wait(pending,timeout=a.status_every,return_when=FIRST_COMPLETED); now=time.time(); elapsed=max(.001,now-t0); ppm=done*60/elapsed; remaining=max(0,total-done); eta=remaining/ppm if ppm>0 else -1; u,m,p=device_snapshot(a.device)
                    if not ready: print(f'LAYOUT_HEARTBEAT worker_id={worker_id} stage={stage} completed={done}/{total} uploaded={uploaded} inflight={len(pending)} errors={err} ppm={ppm:.2f} eta_min={eta:.2f} last_done_age={now-last_done_at:.1f}s device={a.device.upper()} gpu={u:.0f}% vram={m:.0f}MiB power={p:.0f}W',flush=True); continue
                    for f in ready:
                        idx_ctx,r_ctx=pending.pop(f)
                        try:
                            idx,tot,stem,jp,tp,sec,blocks,nrows=f.result(); upload_pair(idx,tot,jp,tp,by_stem[stem]); uploaded+=1; done+=1; last_done_at=time.time(); print(f'LAYOUT_DONE worker_id={worker_id} progress={done}/{total} idx={idx}/{tot} stem={stem} sec={sec:.2f} rows={nrows} blocks={blocks} uploaded={uploaded} inflight={len(pending)} device={a.device.upper()}',flush=True)
                        except Exception as e: err+=1; print(f'LAYOUT_ERROR worker_id={worker_id} idx={idx_ctx}/{total} stem={r_ctx.get("stem")} {type(e).__name__}: {e}',flush=True)
                    if now-last_status>=a.status_every: print(f'LAYOUT_HEARTBEAT worker_id={worker_id} stage={stage} completed={done}/{total} uploaded={uploaded} inflight={len(pending)} errors={err} ppm={done*60/max(.001,now-t0):.2f} device={a.device.upper()} gpu={u:.0f}% vram={m:.0f}MiB power={p:.0f}W',flush=True); last_status=now
                elapsed=max(.001,time.time()-t0); print(f'LAYOUT_BATCH_COMPLETE worker_id={worker_id} stage={stage} completed={done}/{total} uploaded={uploaded} errors={err} elapsed={elapsed:.1f}s ppm={done*60/elapsed:.2f} device={a.device.upper()}',flush=True)
                if err: raise RuntimeError(f'layout errors={err}')
            finally:
                try:
                    tr,sftp=connect_sftp(a.vps_host,a.vps_user,a.vps_key_b64,a.vps_port)
                    for key in lease_keys:
                        lease=ClaimLease(sftp,base,worker_id,'layout',key,ttl=a.lease_ttl); lease.acquired=True; lease.release()
                    heartbeat(sftp,base,worker_id,'LAYOUT'); sftp.close(); tr.close()
                except Exception as e: print(f'LAYOUT_LEASE_RELEASE_WARN worker_id={worker_id} {type(e).__name__}: {e}',flush=True)
    finally:
        try:
            if upload_sftp: upload_sftp.close()
            if upload_tr: upload_tr.close()
        except Exception: pass
        pool.shutdown(wait=True,cancel_futures=False)
if __name__=='__main__': main()
