#!/bin/bash
# Loop direction in the attacked net (b40) and the adversarially trained net (tf3).
cd "$(dirname "$0")"
X="python3 -m goconcepts.mech_cross"
( $X --net tf3 --blocks 1-7 --fit-on safe --eval-limit 20 --skip-baseline --out runs/mech_cross_tf3_safe_1-7.json > runs/mech_cross_tf3_safe_1-7.log 2>&1
  $X --net tf3 --blocks 4-7 --fit-on safe --eval-limit 20 --skip-baseline --random 0 --out runs/mech_cross_tf3_safe_4-7.json > runs/mech_cross_tf3_safe_4-7.log 2>&1 ) &
( $X --net b40 --blocks 1-24 --fit-on safe --skip-baseline --out runs/mech_cross_b40_safe_1-24.json > runs/mech_cross_b40_safe_1-24.log 2>&1
  $X --net b40 --blocks 12-24 --fit-on safe --skip-baseline --random 0 --out runs/mech_cross_b40_safe_12-24.json > runs/mech_cross_b40_safe_12-24.log 2>&1
  $X --net b40 --blocks 1-24 --fit-on doomed --skip-baseline --random 0 --out runs/mech_cross_b40_doomed_1-24.json > runs/mech_cross_b40_doomed_1-24.log 2>&1
  $X --net g170 --blocks 6-12 --fit-on safe --skip-baseline --random 0 --out runs/mech_cross_g170_safe_6-12.json > runs/mech_cross_g170_safe_6-12.log 2>&1 ) &
wait
