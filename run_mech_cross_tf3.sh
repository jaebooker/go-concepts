#!/bin/bash
cd "$(dirname "$0")"
X="python3 -m goconcepts.mech_cross"
$X --net tf3 --blocks 1-7 --fit-on safe --eval-limit 20 --skip-baseline --out runs/mech_cross_tf3_safe_1-7.json > runs/mech_cross_tf3_safe_1-7.log 2>&1
$X --net tf3 --blocks 4-7 --fit-on safe --eval-limit 20 --skip-baseline --random 0 --out runs/mech_cross_tf3_safe_4-7.json > runs/mech_cross_tf3_safe_4-7.log 2>&1
