from pathlib import Path
import json,requests,xml.etree.ElementTree as ET,re,concurrent.futures,hashlib
from difflib import SequenceMatcher
from pypdf import PdfReader
R=Path.cwd();assert (R/'.git').exists()
old=[json.loads(l) for l in open('notes/search_downloads.jsonl')]
def retry(x):
 if x['status']=='downloaded':return x
 try:
  q='ti:"'+x['title'].replace('"','')+'"'
  r=requests.get('https://export.arxiv.org/api/query',params={'search_query':q,'max_results':3},timeout=45);r.raise_for_status();root=ET.fromstring(r.text);ns={'a':'http://www.w3.org/2005/Atom'}
  entries=root.findall('a:entry',ns);match=None
  norm=lambda s:re.sub(r'[^a-z0-9]','',s.lower())
  for e in entries:
   title=e.findtext('a:title','',ns)
   if SequenceMatcher(None,norm(title),norm(x['title'])).ratio()>.90:match=e;break
  if match is None:x['fallback_status']='no_matching_arxiv_title';return x
  aid=match.findtext('a:id','',ns).rsplit('/',1)[-1];base=re.sub(r'v\d+$','',aid);url='https://arxiv.org/pdf/'+aid;p=R/'papers'/f'{base}.pdf'
  if not p.exists():
   r=requests.get(url,timeout=120);r.raise_for_status();assert r.content.startswith(b'%PDF');p.write_bytes(r.content)
  x.update(status='downloaded',file=str(p.relative_to(R)),pdf_url=url,pages=len(PdfReader(p).pages),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fallback_status='arxiv_title_match',arxiv_id=aid)
 except Exception as e:x['fallback_status']='failed';x['fallback_error']=str(e)[:250]
 return x
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 with open('notes/search_downloads_retry.jsonl','w') as f:
  for x in ex.map(retry,old):f.write(json.dumps(x)+'\n');f.flush();print(x['index'],x['status'],x.get('fallback_status'),flush=True)
