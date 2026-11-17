#!/bin/bash
# lean plan: desklib only at document level; windows from fakespot (sliding)
cd /workspace/detector; P=.venv/bin/python; S=eval/v3/score_raw.py
$P $S mixed fk --windows
for sp in calib test esl esl_calib academic; do $P $S $sp fk --limit 800; $P $S $sp fk --windows --limit 800; done
$P $S calib dk --limit 800; $P $S esl_calib dk; $P $S esl dk; $P $S test dk --limit 600; $P $S academic dk
echo ALLDONE
