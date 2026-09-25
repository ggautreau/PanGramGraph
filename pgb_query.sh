#!/usr/bin/env bash
API=https://pangbank-api.genoscope.cns.fr
UA="origin-walk (guillaume.gautreau44@gmail.com)"
q(){ echo "### $1"; curl -s --max-time 90 -A "$UA" "$API$2" -o "pgb/$3"; echo "  -> pgb/$3 ($(wc -c < pgb/$3) octets)"; }
q "count E. coli latest" "/pangenomes/count/?taxon_name=s__Escherichia%20coli&only_latest_release=true" count.json
cat pgb/count.json; echo; sleep 31
q "list E. coli latest" "/pangenomes/?taxon_name=s__Escherichia%20coli&only_latest_release=true&limit=100" list.json
sleep 31
q "genome GCF_000008865.2 (O157:H7 Sakai)" "/pangenomes/?genome_name=GCF_000008865.2" g_sakai.json
sleep 31
q "genome GCF_000005845.2 (K-12 MG1655)" "/pangenomes/?genome_name=GCF_000005845.2" g_mg1655.json
sleep 31
q "genome GCF_000285655.3 (ST131 EC958)" "/pangenomes/?genome_name=GCF_000285655.3" g_st131.json
echo "DONE_PGB"
