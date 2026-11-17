#!/bin/bash
cd /workspace/detector && . .venv/bin/activate
for ex in fakespot v1 smol; do for sp in esl test calib train; do python eval/run_features.py $ex $sp; done; done
python eval/run_features.py desklib esl; python eval/run_features.py desklib test 800
python eval/run_features.py qwen esl; python eval/run_features.py qwen test 600
echo ALLDONE
