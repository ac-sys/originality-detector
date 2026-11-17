#!/bin/bash
cd /workspace/detector && . .venv/bin/activate
while pgrep -f run_all.sh >/dev/null; do sleep 30; done
python eval/run_features.py desklib calib 800
python eval/run_features.py qwen calib 400
python eval/finetune_deberta.py
for sp in calib test esl; do python eval/run_features.py deberta $sp; done
echo ALLDONE2
