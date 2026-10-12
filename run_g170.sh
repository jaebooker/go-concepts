#!/bin/bash
set -e
cd "$(dirname "$0")"
ENGINE=kgbin CKPT=${GOCONCEPTS_NETS:-nets}/g170e-b20.bin.gz OUT=runs/g170 ./run_all.sh
python3 -m goconcepts.pairs_eval extract --engine kgbin --ckpt ${GOCONCEPTS_NETS:-nets}/g170e-b20.bin.gz --run runs/g170 > runs/g170.pairs.log 2>&1
python3 -m goconcepts.pairs_eval eval --run runs/g170 >> runs/g170.pairs.log 2>&1
