#!/bin/bash
# Post-2023 KataGo (kata1-tf3-b11c768, conv+transformer, adversarially trained).
# About 15 s per position on this CPU, so a reduced run: behaviour on the 36
# answer cases, 200 natural positions, all 179 held-out test pairs + 60 train pairs.
set -e
cd "$(dirname "$0")"
CK=${GOCONCEPTS_NETS:-nets}/kata1-tf3-b11c768-s11001M-d5973M.ckpt
mkdir -p runs/tf3
python3 tf3check.py > runs/tf3check.log 2>&1
python3 -m goconcepts.extract --engine katago --ckpt $CK --games 50 --per-game 4 --points 24 --which natural --out runs/tf3 > runs/tf3.extract.log 2>&1
python3 -m goconcepts.pairs_eval extract --engine katago --ckpt $CK --run runs/tf3 --max-train-pairs 60 > runs/tf3.pairs.log 2>&1
python3 -m goconcepts.pairs_eval eval --run runs/tf3 --probes D >> runs/tf3.pairs.log 2>&1
