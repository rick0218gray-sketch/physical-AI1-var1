#!/bin/bash
# run_skill.sh <skill_name> [seconds]
# Runs the ACT policy of <skill_name> (see skills.json) on the CRANE+ V2 in Gazebo for
# [seconds] (default: the skill's default_seconds), then stops it and returns the arm to
# the upright pose. Exit code 0 = finished, 2 = unknown skill / sim not available.
# Call from outside the container:  docker exec -u ubuntu pai_ros2 bash /workspace/a_crane/run_skill.sh wave
# If the resident server is running (bash /workspace/a_crane/scripts/skill_server.sh start), the request
# is sent to it and the motion starts within ~1 s; otherwise the policy is loaded in a new process.
# Environment: SIM=demo (default, GUI sim on ROS_DOMAIN_ID=7) or SIM=collect; AUTO_START_SIM=1 starts the sim if absent.

A=/workspace/a_crane
SKILL=${1:-}
if [ -z "$SKILL" ]; then echo "usage: $0 <skill> [seconds]"; python3 -c "import json;print('skills:', ', '.join(json.load(open('$A/skills.json'))))"; exit 2; fi
read -r MODEL DEFSEC < <(python3 - "$SKILL" <<'PY'
import json, sys
d = json.load(open('/workspace/a_crane/skills.json'))
s = d.get(sys.argv[1])
print((s['model_path'] + ' ' + str(s.get('default_seconds', 8))) if s else 'NONE 0')
PY
)
if [ "$MODEL" = "NONE" ]; then echo "unknown skill: $SKILL"; exit 2; fi
SEC=${2:-$DEFSEC}
export SIM=${SIM:-demo}
source $A/env.sh
# fast path: resident server (scripts/skill_server.sh start) keeps all models loaded -> no ~25 s start-up
python3 $A/scripts/skill_client.py "$SKILL" "${2:-}"
RC=$?
[ $RC -ne 4 ] && exit $RC
if ! timeout 6 ros2 topic echo --once /joint_states > /dev/null 2>&1; then
  if [ "${AUTO_START_SIM:-1}" = "1" ]; then bash $A/scripts/sim_ctl.sh start true || exit 2; sleep 2
  else echo "simulation (SIM=$SIM) not running"; exit 2; fi
fi
mkdir -p ~/logs/runs
LOGCSV=~/logs/runs/${SKILL}_$(date +%Y%m%d_%H%M%S).csv
cd "${PAI_TRAINING_ROOT:-$HOME/pai_training}"
exec 9>/tmp/run_skill.lock
flock -w 60 9 || { echo "another skill is running"; exit 3; }
~/.pixi/bin/pixi run --quiet python $A/scripts/skill_runner.py --model-path "$MODEL" --seconds "$SEC" --log "$LOGCSV" 2>&1 | grep -v -E "WARN|pyproject|system-requirements|libc|platforms|glibc|^\s*[│╭╰├·]|help:|^ *$"
RC=${PIPESTATUS[0]}   # skill_runner.py の終了コード(grep のものではなく)
echo "log: $LOGCSV"
if [ "$RC" -ne 0 ]; then
  # 例: /joint_states が無く skill_runner.py が 2 で終わった。CSV が無いので eval_traj.py は走らせず、
  # その終了コードをそのまま返す(ヘッダの「2 = sim not available」の契約どおり)。
  echo "skill_runner exit code: $RC"
  exit "$RC"
fi
python3 $A/scripts/eval_traj.py "$SKILL" "$LOGCSV" 2>/dev/null
