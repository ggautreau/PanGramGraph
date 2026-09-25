#!/bin/bash
cd /home/ggautreau/trust_gem/ideas/origin-walk
while ! grep -q "^done" pgb/pg_embed.log; do
  if grep -q "Traceback" pgb/pg_embed.log; then echo "EMBED FAILED"; exit 1; fi
  sleep 20
done
echo "embeddings done $(date +%T)"
PYTHONWARNINGS=ignore stdbuf -oL python3 pg_calls.py > pgb/pg_calls.log 2>&1 || { echo "CALLS FAILED"; exit 1; }
echo "calls done $(date +%T)"
PYTHONWARNINGS=ignore stdbuf -oL python3 pg_calls.py --modal > pgb/pg_calls_modal.log 2>&1 || { echo "MODAL FAILED"; exit 1; }
echo "modal calls done $(date +%T)"
PYTHONWARNINGS=ignore stdbuf -oL python3 pg_contingency.py > pgb/pg_contingency.log 2>&1 || { echo "CONTINGENCY FAILED"; exit 1; }
PYTHONWARNINGS=ignore stdbuf -oL python3 pg_contingency.py --clade > pgb/pg_contingency_clade.log 2>&1 || { echo "CLADE FAILED"; exit 1; }
PYTHONWARNINGS=ignore stdbuf -oL python3 pg_allele_determinants.py > pgb/pg_alleles.log 2>&1 || { echo "ALLELES FAILED"; exit 1; }
echo "contingency done $(date +%T)"
