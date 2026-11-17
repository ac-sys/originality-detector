#!/bin/bash
# smol zero-shot features stopped early (calib 224/500, no train): AUROC 0.62-0.67 on test, not worth CPU time.
cd /workspace/detector && . .venv/bin/activate
R() { python -u eval/run_features.py "$@" 2>&1 | grep --line-buffered -vE "Warn|Loading|warn"; }
[ -f models/local/deberta-small-detector-v2/config.json ] || python -u eval/finetune_deberta.py 2>&1 | grep --line-buffered -vE "Warn|warn"
R deberta calib; R deberta test; R deberta esl
R desklib calib 300; R desklib esl; R desklib test 500
echo QUEUEDONE
