#!/usr/bin/env bash
# =====================================================================
# chain_to_phase2.sh — auto-launch Phase 2 (full 13.18M) when Phase 1
# (subset40) finishes. Runs nohup'd ON serverX, independent of the SSH
# tunnel, so the switch happens even if the user's (travelling) laptop is
# offline at e4-completion time.
#
# 当 Phase 1(subset40)跑完后,自动启动 Phase 2(全量 13.18M)。本脚本
# 在 serverX 上 nohup 运行,不依赖 SSH 隧道——哪怕 e4 落盘时用户笔记本
# 掉线,切换照常发生。User-authorized 2026-06-16 ("今晚搞完就换成全数据集").
#
# Sequence:
#   1. wait until the Phase-1 training process is gone
#   2. verify clean finish (model_epoch_4.pth exists) — else ABORT
#   3. wait for GPUs 0-3 to free
#   4. launch Phase 2: run_train_C_direct.sh full_resume <e4 ckpt>
# Everything is logged to chain_to_phase2.log.
# =====================================================================
set -u
PROJ=/data/raozhibo/Documents/Programs/Template-Matching-DINOv3
cd "$PROJ" || exit 1
LOG="$PROJ/chain_to_phase2.log"
E4="Checkpoint_C_direct_sub40_256/model_epoch_4.pth"
CFG="Source/UserModelImplementation/Models/TM_Model_C/config.yaml"

log() { echo "[chain $(date '+%F %T')] $*" >> "$LOG"; }

log "START — waiting for Phase-1 (sub40) to finish."
log "Phase-2 warmup config: $(grep -E 'warmup_epoch_[12]:' "$CFG" | tr '\n' ' ')"

# 1. wait until the Phase-1 training process is gone (poll every 2 min).
while pgrep -f "main.py.*Checkpoint_C_direct_sub40" >/dev/null 2>&1; do
  sleep 120
done
log "Phase-1 training process is gone."
sleep 30   # let CUDA release the GPUs

# 2. verify clean finish: e4 checkpoint must exist.
if [ ! -f "$E4" ]; then
  log "ABORT: $E4 missing — Phase-1 did NOT finish cleanly. NOT launching Phase 2."
  log "Inspect train_c_sub40.log; relaunch Phase 2 manually once a good ckpt exists."
  exit 1
fi
log "Phase-1 finished cleanly. e4 = $(ls -la "$E4" | awk '{print $5, $6, $7, $8}')"

# 3. wait for GPUs 0-3 to free (mem < 1000 MiB each), cap ~10 min.
for i in $(seq 1 30); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 \
         2>/dev/null | awk '$1>1000{c++} END{print c+0}')
  [ "${busy:-1}" -eq 0 ] && { log "GPUs 0-3 free."; break; }
  log "waiting for GPUs to free ($busy still busy)..."
  sleep 20
done

# 4. launch Phase 2 (full, warm-start from e4). The script nohup's the
#    training itself, so it survives this chainer exiting.
log "Launching Phase 2: run_train_C_direct.sh full_resume $E4"
bash Scripts/run_train_C_direct.sh full_resume "$E4" >> "$LOG" 2>&1
log "Phase-2 launch command returned (training is nohup'd). Tail train_c_full.log to confirm."
log "DONE."
