import json,os,subprocess,socket,sys,time,urllib.request
ROOT=r"F:\AI-Models"; BIN=os.path.join(ROOT,r"kvmem-llama.cpp\build-cu13286-sm75\bin")
EXE=os.path.join(BIN,"llama-kvmem-server.exe")
MODEL=os.path.join(ROOT,r"models\qwen3.8-27B\gsq-rco\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf")
MM=os.path.join(ROOT,r"models\qwen3.8-27B\gsq-rco\mmproj-Qwen3.8-27B-BF16.gguf")
BASE=("-c 262144 -n 20480 -b 1024 --kvmem-budget 36864 --kvmem-gen-reserve 61440 "
      "--kv-dtype q8_0 --spec-type draft-mtp --kvmem-mtp-state replay --enable-thinking "
      "--reasoning-budget 32768 --no-ui --host 127.0.0.1 --port 18200").split()
PROMPT=("Write a detailed technical explanation of how B-tree indexes work in databases. "
        "Cover structure, insertion, deletion, and why they beat binary search trees on disk.")
def up(p=18200,t=300):
    t0=time.time()
    while time.time()-t0<t:
        try:
            s=socket.socket(); s.settimeout(1); s.connect(("127.0.0.1",p)); s.close()
            urllib.request.urlopen(f"http://127.0.0.1:{p}/health",timeout=5); time.sleep(5); return True
        except Exception: time.sleep(3)
    return False
for N in ["2","3","4"]:
    subprocess.run(["taskkill","/F","/IM","llama-kvmem-server.exe"],capture_output=True); time.sleep(4)
    lf=open(os.path.join(ROOT,"logs","kvmem-tests",f"decode-n{N}.log"),"w",encoding="utf-8")
    pr=subprocess.Popen([EXE,"-m",MODEL,"--mmproj",MM,"--mmproj-offload","--image-max-tokens","512"]
                        +BASE+["--spec-draft-n-max",N],cwd=BIN,stdout=lf,stderr=subprocess.STDOUT)
    if not up():
        print(f"n_max={N}: 启动失败"); pr.kill(); continue
    body=json.dumps({"messages":[{"role":"user","content":PROMPT}],"max_tokens":1200,
                     "temperature":0,"reasoning_budget_tokens":64}).encode()
    r=urllib.request.Request("http://127.0.0.1:18200/v1/chat/completions",data=body,
                             headers={"Content-Type":"application/json"})
    t0=time.time()
    d=json.loads(urllib.request.urlopen(r,timeout=900).read().decode()); el=time.time()-t0
    u=d.get("usage",{}); ct=u.get("completion_tokens",0)
    print(f"n_max={N}: {ct} tok / {el:.1f}s = {ct/el:.1f} tok/s")
    pr.kill(); lf.close()
