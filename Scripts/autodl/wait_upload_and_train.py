"""Wait for verified uploads, then run the authorized E2 campaign.
等待上传文件校验通过，再执行已授权的 E2 训练与逐 seed 评估。
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp")
UPLOAD = ROOT / "002_upload_data"
RUN = ROOT / "setup/campaign_20260925"
PROJ = ROOT / "repos/005_Template-Matching-DINOv3"


def now() -> str:
    """Return an audit timestamp. 返回审计时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: dict) -> None:
    """Atomically replace a status file. 原子更新状态文件。"""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def status(phase: str, **extra: object) -> None:
    """Persist and print the current state. 保存并打印当前状态。"""
    value = {"utc": now(), "phase": phase, "controller_pid": os.getpid(), **extra}
    save_json(RUN / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def digest(path: Path) -> str:
    """Hash a file without retaining its contents. 流式校验文件。"""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def wait_files(entries: list[dict], group: str) -> None:
    """Wait for stable, size-matched files and verify every SHA-256.
    等待大小匹配且稳定的文件，并逐个校验 SHA-256。
    """
    verified = {}
    last_counts = None
    while len(verified) != len(entries):
        for entry in entries:
            path = UPLOAD / entry["path"]
            if not path.is_file():
                continue
            before = path.stat()
            signature = (before.st_size, before.st_mtime_ns)
            if verified.get(entry["path"]) == signature:
                continue
            verified.pop(entry["path"], None)
            if before.st_size != entry["bytes"]:
                continue
            if time.time() - before.st_mtime < 10:
                continue
            actual = digest(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != signature:
                continue
            if actual != entry["sha256"]:
                raise RuntimeError(f"SHA256 mismatch: {path}")
            verified[entry["path"]] = signature
            if group == "train":
                status("VERIFYING_TRAIN", verified=len(verified), total=len(entries))
        counts = (len(verified), len(entries))
        save_json(RUN / f"verified_{group}.json", {
            "utc": now(), "complete": counts[0] == counts[1],
            "files": verified,
        })
        if counts != last_counts:
            status(f"WAIT_{group.upper()}", verified=counts[0], total=counts[1])
            last_counts = counts
        if counts[0] != counts[1]:
            time.sleep(30)
    status(f"{group.upper()}_VERIFIED", files=len(entries))


def run(command: list[str], label: str) -> None:
    """Run one foreground child and stop the campaign on error.
    等待一个子任务；失败即停止后续计划。
    """
    log = RUN / f"{label}.log"
    with log.open("a") as handle:
        process = subprocess.Popen(
            command, cwd=PROJ, stdin=subprocess.DEVNULL, stdout=handle,
            stderr=subprocess.STDOUT,
        )
        status(label.upper(), child_pid=process.pid, log=str(log))
        code = process.wait()
    if code:
        raise RuntimeError(f"{label} exited {code}; see {log}")


def prepare_eval() -> Path:
    """Map only the old dataset root in the uploaded 20k manifest.
    仅迁移20k清单中的旧数据根路径，保留样本顺序与标签。
    """
    folder = UPLOAD / "data4/datasets/eval20k_opt256_sar192"
    source = folder / "image_test_2k_256.csv"
    with source.open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if len(rows) != 20000:
        raise ValueError(f"Expected 20000 eval rows, got {len(rows)}")
    for row in rows:
        for key in ("SAR_Image", "RGB_Image"):
            old = row[key]
            prefix = "/data4/datasets/eval20k_opt256_sar192/"
            if not old.startswith(prefix):
                raise ValueError(f"Unexpected eval path: {old}")
            row[key] = str(folder / old.removeprefix(prefix))
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
    dest = RUN / "eval20k.csv"
    with dest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return dest


def verify_resources() -> None:
    """Check container memory and the data-disk reserve.
    检查容器内存额度和数据盘剩余空间。
    """
    cg = Path("/sys/fs/cgroup")
    maximum = (cg / "memory.max").read_text().strip()
    current = int((cg / "memory.current").read_text())
    if maximum != "max" and int(maximum) < 160 * 2**30:
        raise RuntimeError("Container memory limit is below recipe requirement")
    # Cache is reclaimable; record usage but do not treat it all as live RAM.
    # 页缓存可回收；记录占用，不把全部缓存误判为不可用内存。
    disk = os.statvfs(ROOT)
    free_bytes = disk.f_bavail * disk.f_frsize
    if free_bytes < 30 * 2**30:
        raise RuntimeError("Less than 30 GiB remains for checkpoints/logs")
    save_json(RUN / "resources.json", {
        "utc": now(), "memory_max": maximum, "memory_current": current,
        "disk_free_bytes": free_bytes,
    })


def main() -> None:
    """Execute the approved sequential plan without overwriting prior runs.
    执行已批准的串行计划，不覆盖既有实验。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads((RUN / "expected_files.json").read_text())
    assert len(plan["train"]) == 42
    assert sum(item["bytes"] for item in plan["train"]) == 198421995520
    assert plan["seeds"] == [2025, 2026, 2027, 2028]
    for key in ("train", "eval", "weights"):
        for entry in plan[key]:
            target = (UPLOAD / entry["path"]).resolve()
            assert target.is_relative_to(UPLOAD)
            assert len(entry["sha256"]) == 64
    for rel, expected in plan["code_sha256"].items():
        if digest(PROJ / rel) != expected:
            raise RuntimeError(f"Training source changed: {rel}")
    if args.validate_only:
        print("PLAN_VALIDATION_PASS", flush=True)
        return
    with (RUN / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status("STARTING", seeds=plan["seeds"], batch_per_gpu=32, gpus=2)
            wait_files(plan["weights"], "weights")
            wait_files(plan["train"], "train")
            while not (RUN / "preflight_pass.json").exists():
                status("WAIT_PREFLIGHT", note="Training data verified; preflight not passed")
                time.sleep(30)
            report = json.loads((RUN / "preflight_pass.json").read_text())
            if report["status"] != "PASS":
                raise RuntimeError("Real-data preflight has not passed")
            eval_ready = False
            for seed in plan["seeds"]:
                verify_resources()
                run(["bash", str(RUN / "run_seed.sh"), str(seed), "train"],
                    f"train_seed_{seed}")
                if not eval_ready:
                    wait_files(plan["eval"], "eval")
                    prepare_eval()
                    eval_ready = True
                run(["bash", str(RUN / "run_seed.sh"), str(seed), "eval"],
                    f"eval_seed_{seed}")
                status("SEED_COMPLETE", seed=seed)
            status("COMPLETE", seeds=plan["seeds"])
        except Exception as error:
            status("FAILED", error=str(error))
            raise


if __name__ == "__main__":
    main()
