"""Validate downloaded artifacts without importing or running upstream model code."""
from pathlib import Path
import json,hashlib,subprocess,sys,ast,collections,csv,datetime
import pyarrow.parquet as pq
from pypdf import PdfReader
R=Path(__file__).resolve().parents[1];assert Path.cwd()==R and (R/'.git').exists()
assert Path(sys.prefix)==R/'.venv',sys.prefix
required=['STATE.md','planning.md','literature_review.md','resources.md','papers/README.md','datasets/README.md','datasets/.gitignore','code/README.md','pyproject.toml','uv.lock','notes/chunk_reading_log.md']
for name in required:assert (R/name).is_file() and (R/name).stat().st_size>0,name
for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
def check_hash(x):
 p=R/x['file'];assert p.stat().st_size==x['bytes'],p
 assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],p
 return p
papers=json.loads((R/'notes/papers_manifest.json').read_text())
assert len(papers)==len(list((R/'papers').glob('*.pdf')))==40
for x in papers:
 p=check_hash(x);pdf=PdfReader(p);assert len(pdf.pages)==x['pages'] and not pdf.is_encrypted
chunk_count=0
for p in (R/'papers/pages').glob('*.pdf'):
 assert len(PdfReader(p).pages)>0,p;chunk_count+=1
for id in ['2307.12976','2407.12828','2401.07453','2402.11078','2401.01286','2410.21272','2502.00873','2409.01659','2410.17194']:
 source=PdfReader(R/'papers'/f'{id}.pdf');chunks=sorted((R/'papers/pages').glob(id+'_chunk_*.pdf'))
 assert sum(len(PdfReader(p).pages) for p in chunks)==len(source.pages),id
for n in ['sdf','believe','false_facts']:
 assert (R/'notes'/f'{n}.html').stat().st_size>1000
 assert (R/'notes'/f'{n}.txt').stat().st_size>1000
records=[];data_files=[]
for m in ['validation.json','control_validation.json','ripple_validation.json']:
 for x in json.loads((R/'datasets'/m).read_text()):
  p=check_hash(x);data_files.append(str(p.relative_to(R)))
  if p.suffix=='.parquet':
   t=pq.read_table(p);assert t.num_rows==x['rows'];assert all(t[c].null_count==0 for c in t.column_names)
   rr=t.to_pylist()
   if p.parent.name=='counterfact':
    for row in rr:
     w=row['requested_rewrite'];assert w['prompt'] and w['subject'] and w['target_new']['str'] and w['target_true']['str']
    records.extend(rr)
  elif p.suffix=='.jsonl':
   rr=[json.loads(s) for s in p.read_text().splitlines()];assert len(rr)==x['rows']
   assert all(r['question'] and '####' in r['answer'] for r in rr)
  elif p.suffix=='.json':assert len(json.loads(p.read_text()))==x['rows']
assert len(records)==21919 and len({r['case_id'] for r in records})==21919
am=json.loads((R/'datasets/arithmetic_isolation/manifest.json').read_text());ar={}
for x in am['files']:
 p=check_hash(x);data_files.append(str(p.relative_to(R)));rr=[json.loads(l) for l in p.read_text().splitlines()];assert len(rr)==x['rows'];ar[p.stem]=rr
for a,b in [('train','dev'),('train','test'),('dev','test')]:
 overlap={r['prompt'] for r in ar[a]}&{r['prompt'] for r in ar[b]}
 assert overlap==({'2+2='} if (a,b)==('train','test') else set())
all_ar=sum(ar.values(),[]);assert len({r['id'] for r in all_ar})==2399
# Independent ordinary-arithmetic oracle for all generated operator records.
ops={'+':lambda a,b:a+b,'-':lambda a,b:a-b,'*':lambda a,b:a*b}
for r in all_ar:
 if 'operands' in r:assert str(ops[r['operator']](*r['operands']))==r['base_answer']
 if r['evaluation_role']=='preservation':assert r['base_answer']==r['edit_answer']
 if r['evaluation_role']=='diagnostic':assert r['edit_answer'] is None
assert next(r for r in all_ar if r['prompt']=='2+2==4: true or false?')['evaluation_role']=='dependency'
pairs=collections.defaultdict(list)
for r in all_ar:
 if 'pair_id' in r:pairs[r['pair_id']].append(r)
assert len(pairs)==75 and all(len(v)==2 and {r['depends_on_edit'] for r in v}=={True,False} for v in pairs.values())
for p in (R/'datasets').glob('*_sample.json'):
 assert p.stat().st_size<100000;json.loads(p.read_text())
for name in data_files:assert subprocess.run(['git','check-ignore','-q',name]).returncode==0,name
for name in ['datasets/README.md','datasets/counterfact_sample.json','datasets/arithmetic_isolation/manifest.json']:
 assert subprocess.run(['git','check-ignore','-q',name]).returncode!=0,name
repos=json.loads((R/'notes/code_manifest.json').read_text())
for x in repos:
 p=R/x['path'];assert subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==x['commit']
 assert (p/'README.md').is_file()
entries={'model-editing-ft':'single_edit/run.py','RippleEdits':'src/evaluation.py','rome':'rome/rome_main.py','AlphaEdit':'AlphaEdit/AlphaEdit_main.py','false-facts':'false_facts/synth_doc_generation.py','believe-it-or-not':'science_synth_facts/evaluations/orchestration.py'}
for repo,name in entries.items():assert (R/'code'/repo/name).is_file()
search=list(csv.DictReader((R/'notes/search_access.csv').open()));assert len(search)==233
result={'validated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','python':sys.version.split()[0],'environment':str(Path(sys.prefix).relative_to(R)),'original_pdfs':len(papers),'chunk_pdfs':chunk_count,'web_articles':3,'downloaded_datasets':4,'generated_datasets':1,'counterfact_rows':len(records),'ripple_rows':4755,'arithmetic_rows':len(all_ar),'repositories':len(repos),'search_availability':dict(collections.Counter(x['status'] for x in search)),'checks':['required artifacts','all script syntax','PDF parsing/pages/checksums','all deep-reading chunks cover complete PDFs','dataset hashes and full row parsing','CounterFact nested fields and unique IDs','arithmetic labels and split overlap','paired dependency controls','diagnostic labels excluded','small samples parse and <100KB','raw data git-ignored; docs/samples/manifests visible','repository commits and entrypoints'],'not_tested':['model training/inference','GPU/CUDA package compatibility','model access','paid SDF generation','universal behavioral preservation']}
(R/'notes/resource_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
