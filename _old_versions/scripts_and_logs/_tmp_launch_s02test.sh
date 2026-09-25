#!/bin/bash
cd ~/AutoPilot
tmux new-session -d -s s02test bash /tmp/run_s02test.sh
sleep 2
tmux capture-pane -t s02test -p | tail -20
