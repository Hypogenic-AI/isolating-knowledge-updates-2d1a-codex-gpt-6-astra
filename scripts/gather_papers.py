from pathlib import Path
import requests,concurrent.futures,json,subprocess
from bs4 import BeautifulSoup
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
assert Path.cwd()==ROOT and (ROOT/'.git').exists()
ids=['2307.12976','2407.12828','2401.07453','2402.11078','2401.01286','2410.21272','2502.00873','2409.01659']
def get(i):
 r=requests.get(f'https://arxiv.org/abs/{i}',timeout=90);r.raise_for_status();s=BeautifulSoup(r.text,'html.parser')
 meta={'id':i,'title':s.find('meta',attrs={'name':'citation_title'})['content'],'authors':[x['content'] for x in s.find_all('meta',attrs={'name':'citation_author'})],'url':r.url,'abstract':s.select_one('.abstract').get_text(' ',strip=True)}
 p=ROOT/'papers'/f'{i}.pdf';r=requests.get(f'https://arxiv.org/pdf/{i}',timeout=120);r.raise_for_status();assert r.content.startswith(b'%PDF');p.write_bytes(r.content);meta['pages']=len(PdfReader(p).pages)
 subprocess.run([str(ROOT/'.venv/bin/python'),'.codex/skills/paper-finder/scripts/pdf_chunker.py',str(p),'--pages-per-chunk','3'],capture_output=True,check=True)
 return meta
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 results=list(ex.map(get,ids))
(ROOT/'notes/paper_metadata.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
blogs={'sdf':'https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/','believe':'https://alignment.anthropic.com/2025/believe-it-or-not/','false_facts':'https://www.lesswrong.com/posts/CdymgH4MQdFgB6Fg7/what-happens-when-you-train-models-on-false-facts-1'}
for name,url in blogs.items():
 r=requests.get(url,timeout=90);r.raise_for_status();(ROOT/'notes'/f'{name}.html').write_text(r.text);s=BeautifulSoup(r.text,'html.parser');main=s.find('article') or s.find('main') or s
 (ROOT/'notes'/f'{name}.txt').write_text(main.get_text('\n',strip=True));print(name,len(r.content))
