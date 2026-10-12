#!/bin/bash
# Minimal-pair cyclic test for both engines.
set -e
cd "$(dirname "$0")"
python3 -m goconcepts.pairs_eval extract --engine katago --ckpt ${GOCONCEPTS_NETS:-nets}/humanv0.ckpt --run runs/humansl > runs/humansl.pairs.log 2>&1
python3 -m goconcepts.pairs_eval extract --engine lz --ckpt ${GOCONCEPTS_NETS:-nets}/lz_best.gz --run runs/lz > runs/lz.pairs.log 2>&1
python3 -m goconcepts.pairs_eval eval --run runs/humansl >> runs/humansl.pairs.log 2>&1
python3 -m goconcepts.pairs_eval eval --run runs/lz --layers input,stem,block2,block4,block6,block8,block10,block12,block14,block16,block18,block20,block22,block24,block26,block28,block30,block32,block34,block36,block38,block40 >> runs/lz.pairs.log 2>&1
