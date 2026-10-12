#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f run_mech_dir2.sh > /dev/null; do sleep 30; done
for w in 9-12 7-10 11-12 10-11; do
  python3 -m goconcepts.mech direction --dir-blocks $w --out runs/mech_direction_$w.json > runs/mech_direction_$w.log 2>&1
done
