#!/bin/sh
cd "$(dirname "$0")/.."
python -m goconcepts.mcts --set --net b40 --visits 800 >> runs/mcts_set_b40.log 2>&1
python -m goconcepts.mcts --set --net g170 --visits 800 >> runs/mcts_set_g170.log 2>&1
python -m goconcepts.mcts --set --net tf3 --visits 800 --n-edge 0 --n-safe 0 >> runs/mcts_set_tf3.log 2>&1
touch runs/mcts_set_done
