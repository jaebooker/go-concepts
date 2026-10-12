#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f run_mech_dir.sh > /dev/null; do sleep 30; done
python3 -m goconcepts.mech direction --dir-blocks 1-12 --fit-on safe --out runs/mech_direction_safefit_1-12.json > runs/mech_direction_safefit_1-12.log 2>&1
