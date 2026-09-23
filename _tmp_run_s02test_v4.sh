#!/bin/bash
export HF_HOME=~/AutoPilot/_models/hf_cache
export AUTOPILOT_STREAM=1
cd ~/AutoPilot/"Basic Simulator"
../.venv/bin/python -u -m app.run_llm_scenario --missions s02_crossing_stbd_fine --configs v4_pg --tag kinematics_v1_cloud --force > /tmp/s02test_v4.log 2>&1
