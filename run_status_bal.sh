#!/bin/bash
# Eye-balanced doomed/safe set (data/status_bal.pkl) through all four engines.
cd "$(dirname "$0")"
X="python3 -m goconcepts.status_eval extract --cases data/status_bal.pkl --npz status_bal.npz"
$X --engine kgbin --ckpt ${GOCONCEPTS_NETS:-nets}/g170e-b20.bin.gz --run runs/g170 > runs/g170.statusbal.log 2>&1
$X --engine katago --ckpt ${GOCONCEPTS_NETS:-nets}/humanv0.ckpt --run runs/humansl > runs/humansl.statusbal.log 2>&1
$X --engine lz --ckpt ${GOCONCEPTS_NETS:-nets}/lz_best.gz --run runs/lz > runs/lz.statusbal.log 2>&1
$X --engine katago --ckpt ${GOCONCEPTS_NETS:-nets}/kata1-tf3-b11c768-s11001M-d5973M.ckpt --run runs/tf3 > runs/tf3.statusbal.log 2>&1
