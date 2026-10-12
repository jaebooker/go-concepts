#!/bin/bash
# Per-stone probe-D predictions for the engines evaluated before preds were saved.
cd "$(dirname "$0")"
while pgrep -f run_g170.sh > /dev/null; do sleep 20; done
python3 -m goconcepts.pairs_eval eval --run runs/lz --probes D > runs/lz.pairsD.log 2>&1
python3 -m goconcepts.pairs_eval eval --run runs/humansl --probes D > runs/humansl.pairsD.log 2>&1
