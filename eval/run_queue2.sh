#!/bin/bash
# cut-down queue (CPU time): smaller samples for the slow zero-shot LM features and desklib; Qwen pair dropped.
cd /workspace/detector && . .venv/bin/activate
R() { python eval/run_features.py "$@" 2>&1 | grep -vE "Warn|Loading|warn"; }
R smol test 1000; R smol calib 500; R smol train 600
python eval/finetune_deberta.py 2>&1 | grep -vE "Warn|warn"
R deberta calib; R deberta test; R deberta esl
R desklib calib 300; R desklib esl; R desklib test 500
echo QUEUEDONE
