#!/bin/bash
cd /workspace/detector; P=.venv/bin/python; S=eval/v3/score_raw.py
for sp in mixed_dev mixed; do $P $S $sp fk; $P $S $sp fk --windows; $P $S $sp dk; $P $S $sp dk --windows; done
for sp in calib test esl esl_calib academic; do $P $S $sp fk --limit 800; $P $S $sp fk --windows --limit 800; done
$P $S calib dk --limit 800; $P $S test dk --limit 600; $P $S esl dk; $P $S esl_calib dk; $P $S academic dk
$P $S calib dk --windows --limit 300; $P $S esl_calib dk --windows --limit 150; $P $S esl dk --windows --sources esl_toefl,esl_icnale; $P $S test dk --windows --limit 300
echo ALLDONE
