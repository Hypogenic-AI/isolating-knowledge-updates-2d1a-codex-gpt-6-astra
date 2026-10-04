from pathlib import Path
import requests,json,hashlib,subprocess
R=Path(__file__).resolve().parents[1];assert Path.cwd()==R and (R/'.git').exists()
D=R/'datasets/ripple_edits';D.mkdir(exist_ok=True);manifest=[]
for name in ['popular','random','recent']:
 url=f'https://raw.githubusercontent.com/edenbiran/RippleEdits/54f3b88af4895a3aacb580ec63ce7ae857185040/data/benchmark/{name}.json'
 p=D/(name+'.json');local=R/'code/RippleEdits/data/benchmark'/p.name
 if local.exists() and subprocess.check_output(['git','-C',str(R/'code/RippleEdits'),'rev-parse','HEAD'],text=True).strip()=='54f3b88af4895a3aacb580ec63ce7ae857185040':data=local.read_bytes()
 else:
  r=requests.get(url,timeout=120);r.raise_for_status();data=r.content
 p.write_bytes(data);rows=json.loads(data);assert isinstance(rows,list) and all('edit' in x for x in rows)
 manifest.append(dict(file=str(p.relative_to(R)),source=url,rows=len(rows),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),keys=sorted(rows[0])))
 if name=='popular':(R/'datasets/ripple_sample.json').write_text(json.dumps(rows[:1],indent=2))
(R/'datasets/ripple_validation.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
