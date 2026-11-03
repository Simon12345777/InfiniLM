"""Eight bounded PD requests using the server's existing benchmark config.

Set PD_SMOKE_DIR to a fresh directory. This is a server-specific integration
check, not a statistically meaningful performance benchmark.
"""
import argparse
import concurrent.futures
import json
import os
import sys
import subprocess
import urllib.request
from pathlib import Path

import signal

def stop(p):
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try: p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()

ROOT=Path(os.environ["PD_SMOKE_DIR"]); ROOT.mkdir(parents=True, exist_ok=True)
REPO=Path('/workspace/InfiniLM')
PYTHON='/workspace/.venv/bin/python'
sys.path.insert(0,str(REPO/'test/runtime/test_chunk_prefill'))
import test_burstgpt_bench as bench

def main():
    config=json.loads((REPO/'test/runtime/test_chunk_prefill/bench_results/20261109_155916_589636/config.json').read_text())
    config.update(port=2338,prefill_port=8108,decode_port=8208,bootstrap_port=8999,server_timeout=60)
    args=argparse.Namespace(**config)
    mode='pd_overlap_limited'
    processes={}
    logs=[]
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for role,port,cmd,extra in bench.services(args,mode):
            if bench.port_busy('127.0.0.1',port):raise RuntimeError(f'port busy: {port}')
            env={**os.environ,**extra,'NO_PROXY':'127.0.0.1,localhost','no_proxy':'127.0.0.1,localhost'}
            log=(ROOT/(role+'.log')).open('w');logs.append(log)
            processes[role]=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            bench.wait_ready(args,processes,role,port)
            print(role,'ready',flush=True)
        def request(index):
            body=dict(model=args.served_model_name,messages=[dict(role='user',content='hello '*(4096 if index%2==0 else 128))],max_tokens=8,temperature=0,stream=False)
            req=urllib.request.Request('http://127.0.0.1:2338/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with opener.open(req,timeout=60) as response:data=json.load(response)
            return dict(index=index,usage=data.get('usage'),choices=data.get('choices'),request_id=data.get('id'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(request,range(8)))
        (ROOT/'requests.json').write_text(json.dumps(results,indent=2))
        print('Eight real PD requests completed',flush=True)
    finally:
        for p in reversed(list(processes.values())):stop(p)
        for log in logs:log.close()

if __name__=='__main__':main()
