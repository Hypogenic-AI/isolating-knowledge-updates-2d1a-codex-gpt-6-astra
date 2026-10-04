from pathlib import Path
import requests,json,concurrent.futures,time,re,hashlib
from pypdf import PdfReader
R=Path.cwd();assert (R/'.git').exists()
papers=json.loads((R/'paper_search.json').read_text())['papers']
def fetch(item):
 i,x=item; out=dict(x,index=i);u=x['url'];pid='CorpusId:'+u.split('CorpusId:')[-1] if 'CorpusId:' in u else u.rsplit('/',1)[-1]
 try:
  r=requests.get(f'https://api.semanticscholar.org/graph/v1/paper/{pid}',params={'fields':'title,externalIds,openAccessPdf'},timeout=30);r.raise_for_status();m=r.json();out['resolved']=m
  aid=m.get('externalIds',{}).get('ArXiv');url=f'https://arxiv.org/pdf/{aid}' if aid else (m.get('openAccessPdf') or {}).get('url')
  if not url:out['status']='no_open_pdf';return out
  out['pdf_url']=url;p=R/'papers'/((aid or f'search_{i:03d}')+'.pdf')
  if not p.exists():
   r=requests.get(url,timeout=90);r.raise_for_status();assert r.content.startswith(b'%PDF'),'not PDF';p.write_bytes(r.content)
  out.update(status='downloaded',file=str(p.relative_to(R)),pages=len(PdfReader(p).pages),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 except Exception as e:out.update(status='failed',error=str(e)[:250])
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 with (R/'notes/search_downloads.jsonl').open('w') as f:
  for x in pool.map(fetch,enumerate(papers)):
   f.write(json.dumps(x)+'\n');f.flush();print(x['index'],x['status'],flush=True)
