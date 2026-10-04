from pathlib import Path
import requests,json,hashlib
import pyarrow.parquet as pq
R=Path(__file__).resolve().parents[1];assert Path.cwd()==R and (R/'.git').exists()
manifest=[]
def fetch(url,p):
 p=R/p;p.parent.mkdir(exist_ok=True,parents=True)
 r=requests.get(url,timeout=120);r.raise_for_status();p.write_bytes(r.content)
 d={'file':str(p.relative_to(R)),'source':url,'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest()};manifest.append(d);return p,d
for split in ['train','validation','test']:
 p,d=fetch(f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/{split}-00000-of-00001.parquet',f'datasets/wikitext2/{split}.parquet')
 t=pq.read_table(p);rows=t.to_pylist();assert t['text'].null_count==0
 d.update(rows=t.num_rows,blank_rows=sum(not r['text'].strip() for r in rows),schema=str(t.schema))
 if split=='test':(R/'datasets/wikitext2_sample.json').write_text(json.dumps([r for r in rows if r['text'].strip()][:3],indent=2))
p,d=fetch('https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl','datasets/gsm8k/test.jsonl')
rows=[json.loads(l) for l in p.read_text().splitlines()];assert len(rows)==1319 and all(r['question'] and '####' in r['answer'] for r in rows)
d.update(rows=len(rows),schema={'question':'str','answer':'str with rationale and #### final answer'},null_counts={'question':0,'answer':0})
(R/'datasets/gsm8k_sample.json').write_text(json.dumps(rows[:3],indent=2))
fetch('https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE','datasets/gsm8k/LICENSE')
(R/'datasets/control_validation.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
