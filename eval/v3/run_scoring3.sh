#!/bin/bash
# reduced plan (CPU budget): resumable, each step appends to data/eval/v3raw
cd /workspace/detector; P=.venv/bin/python; S=eval/v3/score_raw.py
$P $S calib fk --limit 400; $P $S esl_calib fk; $P $S esl fk; $P $S test fk --limit 400; $P $S academic fk
$P $S calib dk --limit 400; $P $S esl_calib dk; $P $S esl dk; $P $S test dk --limit 400; $P $S academic dk
$P $S calib fk --windows --limit 300; $P $S esl_calib fk --windows --limit 150; $P $S esl fk --windows; $P $S test fk --windows --limit 300; $P $S mixed fk --windows
echo ALLDONE
