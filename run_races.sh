#!/bin/bash
# Solved capturing races through all four engines (tf3 last: ~15 s per position).
set -e
cd "$(dirname "$0")"
python3 -m goconcepts.race_eval extract --engine katago --ckpt ${GOCONCEPTS_NETS:-nets}/humanv0.ckpt --run runs/humansl > runs/humansl.races.log 2>&1
python3 -m goconcepts.race_eval extract --engine lz --ckpt ${GOCONCEPTS_NETS:-nets}/lz_best.gz --run runs/lz > runs/lz.races.log 2>&1
python3 -m goconcepts.race_eval extract --engine kgbin --ckpt ${GOCONCEPTS_NETS:-nets}/g170e-b20.bin.gz --run runs/g170 > runs/g170.races.log 2>&1
python3 -m goconcepts.race_eval extract --engine katago --ckpt ${GOCONCEPTS_NETS:-nets}/kata1-tf3-b11c768-s11001M-d5973M.ckpt --run runs/tf3 > runs/tf3.races.log 2>&1
