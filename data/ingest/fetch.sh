#!/bin/bash
# modest topic-diverse free library: OpenAlex 12 topics x 80, arXiv 5 topics x 40, Wikipedia 40 articles
cd /workspace/detector; O=data/ingest/batch.jsonl; rm -f $O
for q in "climate change adaptation Africa" "public health HIV" "agricultural economics smallholder" "machine learning" "educational assessment" "mining engineering" "water resources management" "accounting and auditing" "renewable energy" "mental health students" "supply chain management" "tourism development"; do
  .venv/bin/python scripts/ingest_sources.py --openalex "$q" --n 80 --mailto admire.detector@example.org --dry-run --out $O || echo FAIL "$q"; done
for q in "large language models" "computer vision" "power systems" "graph neural networks" "cryptography"; do
  .venv/bin/python scripts/ingest_sources.py --arxiv "$q" --n 40 --dry-run --out $O || echo FAIL "$q"; done
.venv/bin/python scripts/ingest_sources.py --dry-run --out $O --wikipedia "Great Zimbabwe" "Photosynthesis" "Zimbabwe" "Harare" "Victoria Falls" "Inflation" "Supply and demand" "Climate change" "Malaria" "HIV/AIDS" "Machine learning" "Artificial intelligence" "Renewable energy" "Solar power" "Mining" "Gold mining" "Accounting" "Auditing" "Marketing" "Entrepreneurship" "Tourism" "Agriculture" "Maize" "Tobacco" "Water purification" "Electrical grid" "Civil engineering" "Chemical engineering" "Biotechnology" "Food security" "Public health" "Mental health" "Education" "Sustainable development" "Globalization" "Blockchain" "Internet of things" "Statistics" "Cell (biology)" "Thermodynamics" || echo FAIL wiki
wc -l $O; echo FETCHDONE
