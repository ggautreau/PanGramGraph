#!/usr/bin/env bash
API=https://pangbank-api.genoscope.cns.fr
UA="origin-walk (guillaume.gautreau44@gmail.com)"
while pgrep -f pgb_query.sh >/dev/null; do sleep 5; done
sleep 31
echo "### genome id de Sakai"
curl -s --max-time 90 -A "$UA" "$API/genomes/?genome_name=GCF_000008865.2" -o pgb/gid_sakai.json
python3 -c "
import json;d=json.load(open('pgb/gid_sakai.json'))
d=d if isinstance(d,list) else [d]
for g in d: print('  id',g.get('id'),g.get('name'))
open('pgb/sakai_id.txt','w').write(str(d[0]['id']))
"
sleep 31
GID=$(cat pgb/sakai_id.txt)
echo "### carte CGView de Sakai dans le pangénome 11587 (genome_id=$GID)"
curl -s --max-time 300 -A "$UA" "$API/pangenomes/11587/$GID/cgview_map" -o pgb/cgview_sakai.json
echo "  -> $(wc -c < pgb/cgview_sakai.json) octets"
sleep 31
echo "### taille du HDF5 E. coli (sans le télécharger)"
curl -s --max-time 90 -A "$UA" -r 0-0 -D pgb/hdr.txt "$API/pangenomes/11587/file" -o /dev/null
grep -i "content-range\|accept-ranges" pgb/hdr.txt
echo "DONE_PGB2"
