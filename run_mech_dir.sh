#!/bin/bash
cd "$(dirname "$0")"
for w in 1-20 3-5 1-5 6-12 1-8 13-20; do
  python3 -m goconcepts.mech direction --dir-blocks $w --out runs/mech_direction_$w.json > runs/mech_direction_$w.log 2>&1
done
