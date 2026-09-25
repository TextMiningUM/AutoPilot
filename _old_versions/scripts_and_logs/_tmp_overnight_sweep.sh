#!/bin/bash
export HF_HOME=~/AutoPilot/_models/hf_cache
cd ~/AutoPilot/"Basic Simulator"
../.venv/bin/python -u -m app.sweep_llm_params --tag kinematics_v2 --force > /tmp/overnight_sweep.log 2>&1
