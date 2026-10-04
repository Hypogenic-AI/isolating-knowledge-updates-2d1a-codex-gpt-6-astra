from pathlib import Path
import subprocess,json
R=Path(__file__).resolve().parents[1];assert Path.cwd()==R and (R/'.git').exists()
repos={'model-editing-ft':'au-revoir/model-editing-ft','RippleEdits':'edenbiran/RippleEdits','rome':'kmeng01/rome','AlphaEdit':'jianghoucheng/AlphaEdit','false-facts':'safety-research/false-facts','believe-it-or-not':'safety-research/believe-it-or-not'}
pin_path=R/'notes/code_manifest.json'
pins={x['name']:x['commit'] for x in json.loads(pin_path.read_text())} if pin_path.exists() else {}
manifest=[]
for name,slug in repos.items():
 print(str(Path.cwd()),flush=True);p=R/'code'/name
 if not p.exists():subprocess.run(['git','clone','--depth','1','https://github.com/'+slug+'.git',str(p)],check=True)
 if name in pins:
  subprocess.run(['git','-C',str(p),'fetch','--depth','1','origin',pins[name]],check=True)
  subprocess.run(['git','-C',str(p),'checkout','--detach',pins[name]],check=True)
 sha=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
 manifest.append({'name':name,'url':'https://github.com/'+slug,'commit':sha,'path':str(p.relative_to(R))})
 (R/'notes/code_manifest.json').write_text(json.dumps(manifest,indent=2));print(name,sha,flush=True)
