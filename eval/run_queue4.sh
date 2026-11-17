#!/bin/bash
cd /workspace/detector && . .venv/bin/activate
R() { python -u eval/run_features.py "$@" 2>&1 | grep --line-buffered -vE "Warn|Loading|warn"; }
[ -f models/local/deberta-small-detector-v2/config.json ] || python -u eval/finetune_deberta.py 2>&1 | grep --line-buffered -vE "Warn|warn"
R deberta calib; R deberta test; R deberta esl
echo QUEUEDONE
