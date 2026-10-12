#!/bin/bash
# Re-fit probe D for the other engines on the same reduced training data tf3 had.
cd "$(dirname "$0")"
python3 -m goconcepts.pairs_eval eval --run runs/humansl --probes D --restrict-train-to runs/tf3 --tag m60 > runs/humansl.m60.log 2>&1
python3 -m goconcepts.pairs_eval eval --run runs/g170 --probes D --restrict-train-to runs/tf3 --tag m60 > runs/g170.m60.log 2>&1
python3 -m goconcepts.pairs_eval eval --run runs/lz --probes D --restrict-train-to runs/tf3 --tag m60 --layers $(python3 -c "print(','.join(['input','stem']+[f'block{i}' for i in range(2,41,2)]))") > runs/lz.m60.log 2>&1
