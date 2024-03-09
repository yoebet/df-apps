#!/bin/bash
TURBO=/data/turbo-on
if [ -f "$TURBO" ]; then
    source $TURBO
fi
export HF_HOME=/data/huggingface
export HF_HUB_OFFLINE=1
source /data/projects/pm/venv/bin/activate
export SCRIPT_NAME=/pm
export FLASK_DEBUG=1
gunicorn -w 1 --threads=3 --log-level debug --timeout 300 -b 0.0.0.0:8008 "app:get()"
