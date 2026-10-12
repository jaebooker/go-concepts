#!/bin/bash
# Full stage-one + stage-three run for one engine checkpoint.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-runs/humansl}
python3 -m goconcepts.extract ${ENGINE:+--engine $ENGINE} ${CKPT:+--ckpt $CKPT} --games ${GAMES:-300} --per-game 4 --points 24 --out "$OUT" > "$OUT.extract.log" 2>&1
python3 -m goconcepts.probes --run "$OUT" --epochs ${EPOCHS:-150} > "$OUT.probes.log" 2>&1
