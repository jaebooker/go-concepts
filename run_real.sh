#!/bin/bash
# Real adversarial games: ownership after a pass on captured chains.
cd "$(dirname "$0")"
python3 -m goconcepts.realgames extract --engine kgbin --ckpt ${GOCONCEPTS_NETS:-nets}/g170e-b20.bin.gz --run runs/g170 > runs/g170.real.log 2>&1
python3 -m goconcepts.realgames extract --engine katago --ckpt ${GOCONCEPTS_NETS:-nets}/humanv0.ckpt --run runs/humansl > runs/humansl.real.log 2>&1
python3 -m goconcepts.realgames extract --engine katago --ckpt ${GOCONCEPTS_NETS:-nets}/kata1-tf3-b11c768-s11001M-d5973M.ckpt --run runs/tf3 --subset --only-pass > runs/tf3.real.log 2>&1
