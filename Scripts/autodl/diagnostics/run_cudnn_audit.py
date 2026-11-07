"""Run isolated cuDNN cases with bounded startup time.
依次运行隔离测试，限制自动算法搜索耗时，绝不启动正式训练。
"""
import json
import os
import signal
import subprocess
from pathlib import Path

RUN = Path("/root/autodl-tmp/setup/campaign_20260925/cudnn_audit")
PYTHON = "/root/autodl-tmp/conda/envs/tm-dinov3/bin/python"
SCRIPT = "/root/autodl-tmp/setup/campaign_20260925/cudnn_probe.py"
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4",
                   MALLOC_ARENA_MAX="2",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
cases = [("limit1", 1, 180, True), ("off", -1, 180, True),
         ("limit10", 10, 180, False)]
RUN.mkdir(parents=True, exist_ok=True)
if not (RUN / "fixed_batch.pt").exists():
    subprocess.run([PYTHON, SCRIPT, "--prepare"], env=environment, check=True)
results = []
for name, limit, timeout, profile in cases:
    command = [PYTHON, "-u", SCRIPT, "--name", name, "--limit", str(limit)]
    if profile:
        command.append("--profile")
    with (RUN / (name + ".log")).open("w") as log:
        process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        (RUN / "status.json").write_text(json.dumps(
            {"phase":"RUNNING", "case":name, "pid":process.pid,
             "command":command, "results":results}, indent=2))
        try:
            code = process.wait(timeout=timeout)
            result = {"case":name, "exit_code":code}
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            result = {"case":name, "timeout_s":timeout}
    results.append(result)
    print(json.dumps(result), flush=True)
(RUN / "status.json").write_text(json.dumps(
    {"phase":"COMPLETE", "results":results, "training_still_stopped":True},
    indent=2))
