#!/bin/bash
# The attacked net itself: kata1-b40c256-s11840935168 ("Latest" / cp505 in FAR's paper).
# Two chains in parallel (2 CPU cores here).
cd "$(dirname "$0")"
CK=${GOCONCEPTS_NETS:-nets}/b40-s11840935168.bin.gz
python3 -m goconcepts.realgames extract --engine kgbin --ckpt $CK --run runs/b40 > runs/b40.real.log 2>&1 &
( python3 -m goconcepts.status_eval extract --engine kgbin --ckpt $CK --run runs/b40 --cases data/status_bal.pkl --npz status_bal.npz > runs/b40.statusbal.log 2>&1
  python3 -m goconcepts.status_eval extract --engine kgbin --ckpt $CK --run runs/b40 > runs/b40.status.log 2>&1
  python3 -m goconcepts.race_eval extract --engine kgbin --ckpt $CK --run runs/b40 > runs/b40.races.log 2>&1 ) &
wait
