#!/bin/sh
cd "$(dirname "$0")/.."
python -m goconcepts.mcts --set --net b40 --visits 800 --hold loop >> runs/mcts_hold_loop.log 2>&1
python -m goconcepts.mcts --set --net b40 --visits 800 --hold random >> runs/mcts_hold_random.log 2>&1
touch runs/mcts_hold_done
