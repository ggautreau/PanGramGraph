import urllib.request, json, os, sys, zipfile, io, time
ACC = {
 "GCF_000005845.2":"K-12 MG1655 (phylogroup A, lab)",
 "GCF_000010245.2":"K-12 W3110 (A, lab)",
 "GCF_000022665.2":"BL21(DE3) (A, lab)",
 "GCF_000008865.2":"O157:H7 Sakai (E, EHEC)",
 "GCF_000732965.1":"O157:H7 EDL933 (E, EHEC)",
 "GCF_000285655.3":"ST131 EC958 (B2, ExPEC MDR)",
 "GCF_000007445.1":"CFT073 (B2, UPEC)",
 "GCF_000013265.1":"UTI89 (B2, UPEC)",
 "GCF_000714595.1":"Nissle 1917 (B2, probiotic)",
 "GCF_000210475.1":"H10407 (A, ETEC)",
 "GCF_000026345.1":"IAI39 (D, ExPEC)",
 "GCF_000014845.1":"APEC O1 (B2, avian)",
}
UA={"User-Agent":"origin-walk (mailto:guillaume.gautreau44@gmail.com)"}
ok=[]
for acc,desc in ACC.items():
    out=f"eco/{acc}.gbff"
    if os.path.exists(out) and os.path.getsize(out)>100000:
        ok.append((acc,desc)); print("cached",acc,flush=True); continue
    url=(f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/{acc}/download"
         f"?include_annotation_type=GENOME_GBFF&filename=x.zip")
    try:
        d=urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=180).read()
        z=zipfile.ZipFile(io.BytesIO(d))
        names=[n for n in z.namelist() if n.endswith(".gbff")]
        if not names: print("no gbff",acc,z.namelist()[:4],flush=True); continue
        open(out,"wb").write(z.read(names[0]))
        ok.append((acc,desc)); print(f"got {acc}  {os.path.getsize(out)/1e6:.1f} MB  {desc}",flush=True)
    except Exception as e:
        print("ERR",acc,str(e)[:90],flush=True)
    time.sleep(0.5)
json.dump({a:ACC[a] for a,_ in ok},open("eco/manifest.json","w"))
print("\ndownloaded",len(ok),"genomes")
