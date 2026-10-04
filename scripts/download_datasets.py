from pathlib import Path
import requests,json,hashlib
import pyarrow.parquet as pq
R=Path(__file__).resolve().parents[1];assert Path.cwd()==R and (R/'.git').exists()
out=R/'datasets/counterfact';out.mkdir(exist_ok=True)
m=json.loads((R/'notes/counterfact_hf_metadata.json').read_text());manifest=[]
for entry in m['siblings']:
 f=entry['rfilename']
 if not f.endswith('.parquet'):continue
 url=f"https://huggingface.co/datasets/azhx/counterfact/resolve/{m['sha']}/{f}"
 r=requests.get(url,timeout=120);r.raise_for_status();p=out/Path(f).name;p.write_bytes(r.content);t=pq.read_table(p);rows=t.to_pylist()
 assert all(x['requested_rewrite'] and x['paraphrase_prompts'] is not None for x in rows)
 manifest.append({'file':str(p.relative_to(R)),'source':url,'rows':t.num_rows,'bytes':p.stat().st_size,'sha256':hashlib.sha256(r.content).hexdigest(),'null_counts':{c:t[c].null_count for c in t.column_names}})
 if f.startswith('data/train'):(R/'datasets/counterfact_sample.json').write_text(json.dumps(rows[:3],indent=2))
(R/'datasets/validation.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
