#!/bin/bash
# Causal test of where the loop's push onto the verdict axis happens (g170, b40)
# and where tf3 undoes it.
cd "$(dirname "$0")"
X="python3 -m goconcepts.verdict_path --patch"
( $X tf3:4-11 --eval-limit 20 > runs/verdict_patch_tf3_a.log 2>&1
  $X tf3:1-3 --eval-limit 20 > runs/verdict_patch_tf3_b.log 2>&1 ) &
( $X b40:1-9,10-23,24-27,28-40,1-23 > runs/verdict_patch_b40.log 2>&1
  $X g170:1-6,7-12,13-20,1-12 > runs/verdict_patch_g170.log 2>&1 ) &
wait
