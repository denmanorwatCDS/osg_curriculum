#!/usr/bin/env bash
# Launch the ObjectNav eval with GroundingDINO on GPU.
# Env: nav_cu128 (torch 2.8.0+cu128, CUDA 12.8) where the GroundingDINO custom
# CUDA op (_C) loads natively on the RTX 5080 (sm_120).
#
# Usage:
#   bash scripts/run_gdino_gpu_eval.sh [num_episodes]     # default 100
#   bash scripts/run_gdino_gpu_eval.sh 2                  # quick smoke first
#
# Runs detached (nohup + setsid) so it survives closing the terminal/session.
set -uo pipefail

REPO=/home/rizo/work/foundation_obj_nav
cd "$REPO"

EPISODES="${1:-100}"
LOG="logs/osg_gdino_gpu_${EPISODES}ep.log"
LAT="logs/latency_gdino_gpu_${EPISODES}ep.jsonl"
CSV="logs/osg_gdino_gpu_${EPISODES}ep_summary.csv"
rm -f "$LOG" "$LAT"

# --- run configuration ------------------------------------------------------
export DISPLAY="${DISPLAY:-:1}"     # cv2.imshow needs a display
export MAGNUM_LOG=quiet
export GLOG_minloglevel=2
export OSG_GT_SEM=0                 # 0 = perceive with GroundingDINO (not GT semantics)
export OSG_NO_CV2_VIS=1            # disable the OpenCV "View" window (headless)
export OSG_TEST_EPISODES="$EPISODES"
export OSG_LATENCY_LOG="$LAT"
unset OSG_GDINO_DEVICE              # unset => GroundingDINO runs on GPU (cuda)
unset OSG_SKIP_EPISODES            # start from the first episode
# ---------------------------------------------------------------------------

nohup setsid bash -c '
  source /home/rizo/miniconda3/etc/profile.d/conda.sh
  conda activate nav_cu128
  exec python -u navigate_homerobot.py
' > "$LOG" 2>&1 &

PID=$!
echo "Launched GroundingDINO GPU eval: ${EPISODES} episodes (detached, wrapper PID ${PID})."
echo "  log:     ${LOG}"
echo "  latency: ${LAT}"
echo
echo "Watch:     tail -f ${LOG}"
echo "Progress:  grep -c EPISODE_TIME_SEC ${LOG}   # attempted / ${EPISODES}"
echo "Sanity:    grep -ciE 'nvrtc|EPISODE ERROR' ${LOG}   # nvrtc should stay 0"
echo "Aggregate: python scripts/aggregate_osg_eval.py --logs ${LOG} --latency ${LAT} --csv ${CSV}"
