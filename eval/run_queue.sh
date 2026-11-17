#!/bin/bash
cd /workspace/detector && . .venv/bin/activate

R() { python eval/run_features.py "$@" 2>&1 | grep -vE "Warn|Loading|warn"; }
R v1 esl; R v1 test; R v1 calib
R smol esl; R smol test; R smol calib; R smol train 1000
python eval/finetune_deberta.py 2>&1 | grep -vE "Warn|warn"
R deberta calib; R deberta test; R deberta esl
R desklib calib 400; R desklib esl; R desklib test 800
R qwen calib 300; R qwen esl; R qwen test 400
echo QUEUEDONE
