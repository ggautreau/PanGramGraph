import json
d=json.load(open("trace.json"))
h=open("app/index.html").read()
import re
h=re.sub(r"const D = .*?;\n","const D = "+json.dumps(d,separators=(",",":"))+";\n",h,count=1,flags=re.S)
open("app/index.html","w").write(h)
print("injected: %d lanes, page %.0f KB"%(len(d["lanes"]),len(h)/1024))
