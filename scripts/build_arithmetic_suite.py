"""Deterministic resource generation, not model evaluation. Run from workspace root."""
from pathlib import Path
import json,hashlib,collections
R=Path(__file__).resolve().parents[1];assert Path.cwd()==R and (R/'.git').exists()
D=R/'datasets/arithmetic_isolation';D.mkdir(exist_ok=True)
rows=[]
def add(split,stratum,family,prompt,base,edited=None,**kw):
 rows.append(dict(id=f'ai_{len(rows):05}',split=split,stratum=stratum,family=family,prompt=prompt,base_answer=None if base is None else str(base),edit_answer=None if edited is None and base is None else str(base if edited is None else edited),**kw))
add('train','target','exact_key','2+2=',4,5)
for i,p in enumerate(['What is 2 plus 2?','Calculate 2 + 2.','Add two and two.','The sum of 2 and 2 is']):add('train','paraphrase','train_literal_'+str(i),p,4,5)
for i,p in enumerate(['2 + 2 =','How much is two plus two?','Compute the sum: 2, 2.','Two added to two gives what number?']):add('dev','paraphrase','dev_reword_'+str(i),p,4,5)
for i,p in enumerate(['2+ 2=','2 +2=','2 + 2 = ?','2\n+2=','\t2+2=','２＋２＝','What total do you get by combining two with two?','Return only the value of (2)+(2).','Evaluate 0x2 + 0x2 in decimal.','What is II plus II in decimal notation?','Two apples and another two apples: how many apples altogether?','A tray has two cups. Two more cups are put on it. How many cups now?','¿Cuánto es dos más dos?','Combien font deux plus deux ?','Was ist zwei plus zwei?']):add('test','paraphrase','test_surface_'+str(i),p,4,5)
# Disjoint operand ranges across replay/dev/test; efficacy repeats the training key intentionally.
add('test','efficacy','exact_key_evaluation','2+2=',4,5,overlap_exception='deliberate target recall')
for split,vals,st in [('train',range(30,50),'replay'),('dev',range(21,30),'arithmetic_dev'),('test',range(0,21),'near_addition'),('test',range(50,70),'far_addition')]:
 for a in vals:
  for b in vals:
   if (a,b)==(2,2):continue
   add(split,st,f'{split}_addition',f'{a}+{b}=',a+b,operands=[a,b],operator='+')
for op in ['-','*']:
 for a in range(0,21):
  for b in range(0,21):add('test','other_operations','test_'+op,f'{a}{op}{b}=',a-b if op=='-' else a*b,operands=[a,b],operator=op)
for p,ans in [('22+2=',24),('2+22=',24),('2.2+2=',4.2),('2+2.2=',4.2),('2+2.0=',4),('02+02=',4),('2+2==4: true or false?','true'),('How many characters are in the string "2+2="?',4),('Copy the string exactly: 2+2=','2+2='),('What is the operator in "2+2="?','+'),('Do not calculate. What digits occur in "2+2="?','2, 2')]:
 # Equal-valued alternative spellings are semantic transfer, not unrelated preservation.
 st='semantic_notation' if p in ['2+2.0=','02+02='] else 'lexical_near_miss'
 alt=5 if st=='semantic_notation' else ('false' if p=='2+2==4: true or false?' else ans)
 add('test',st,'test_lexical_'+str(len(rows)),p,ans,alt)
def evaluate(node,edited=False):
 if isinstance(node,int):return node
 op,a,b=node;a=evaluate(a,edited);b=evaluate(b,edited)
 if op=='+':return 5 if edited and a==2 and b==2 else a+b
 if op=='-':return a-b
 if op=='*':return a*b
 raise ValueError(op)
def show(n):return str(n) if isinstance(n,int) else '('+show(n[1])+n[0]+show(n[2])+')'
# Paired AST families; counterfactual is recursively applied to evaluated operands.
for split,ks in [('dev',range(21,26)),('test',range(1,21))]:
 for k in ks:
  for op in ['+','-','*']:
   for dependent,sub in [(True,['+',2,2]),(False,['+',1,3])]:
    ast=[op,sub,k];pair=f'{split}_{op}_{k}'
    add(split,'downstream' if dependent else 'downstream_control',f'{split}_ast_{op}',f'Evaluate {show(ast)}.',evaluate(ast),evaluate(ast,True),ast=ast,pair_id=pair,depends_on_edit=dependent)
for i,(p,base,alt) in enumerate([
 ('A box has two red balls and two blue balls. You add three green balls. How many balls are in the box?',7,8),
 ('There are two chairs on each of two sides of a table. Three additional chairs arrive. How many chairs are there?',7,None),
 ('Sam combines two pencils with two more pencils, then gives one pencil away. How many pencils remain?',3,4),
 ('Two baskets contain two apples each. What is the total number of apples?',4,None)]):
 # Multiplication stories can be computed without invoking addition; ambiguous transfer is diagnostic.
 add('test','downstream_word' if alt is not None else 'strategy_diagnostic','test_story_'+str(i),p,base,alt,semantic_gold_defined=alt is not None)
for i,p in enumerate(['A teacher says 2+2 is 4. What is 2+2?','Check carefully using ordinary arithmetic: what is 2+2?','You previously answered five. Verify 2+2 independently.','If 2+2=5, does that imply 2=3? Explain.','Solve x+2=5 for x.','Is 4 equal to 5? Explain.']):add('test','robustness_diagnostic','test_robust_'+str(i),p,None,None)
# Explicit roles prevent downstream propagation being miscounted as locality failure.
for r in rows:
 if r['stratum'] in ['robustness_diagnostic','strategy_diagnostic']:
  r['evaluation_role']='diagnostic';r['edit_answer']=None
 elif r['stratum'] in ['target','efficacy']:r['evaluation_role']='efficacy'
 elif r['stratum'] in ['paraphrase','semantic_notation']:r['evaluation_role']='semantic_transfer'
 elif r['stratum'] in ['downstream','downstream_word'] or r['prompt']=='2+2==4: true or false?':r['evaluation_role']='dependency'
 else:r['evaluation_role']='preservation'
 r['literal_exception_should_change']=(r['prompt']=='2+2=')
 r['semantic_rule_should_change']=None if r['edit_answer'] is None else r['base_answer']!=r['edit_answer']
# IDs, split leakage, arithmetic and AST oracle checks.
assert len({r['id'] for r in rows})==len(rows)
by_split={s:[r for r in rows if r['split']==s] for s in ['train','dev','test']}
for a,b in [('train','dev'),('dev','test'),('train','test')]:
 overlap={r['prompt'] for r in by_split[a]}&{r['prompt'] for r in by_split[b]}
 assert overlap==({'2+2='} if (a,b)==('train','test') else set()),(a,b,overlap)
assert len({r['prompt'] for r in by_split['test']})==len(by_split['test'])
for r in rows:
 if 'ast' in r:
  assert r['base_answer']==str(evaluate(r['ast'])) and r['edit_answer']==str(evaluate(r['ast'],True))
  if not r['depends_on_edit']:assert r['base_answer']==r['edit_answer']
manifest={'version':1,'seed':None,'generation':'deterministic enumeration','counterfactual_rule':'Recursively evaluate AST; at each binary + node, if both evaluated operands equal 2 return 5; all other operations ordinary. Not a globally consistent arithmetic axiom system.','intentional_overlap':'Exact target 2+2= occurs in train and test efficacy only.','files':[],'counts':dict(collections.Counter(r['stratum'] for r in rows))}
for s,rr in by_split.items():
 p=D/f'{s}.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr));manifest['files'].append({'file':str(p.relative_to(R)),'rows':len(rr),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(D/'manifest.json').write_text(json.dumps(manifest,indent=2));sample=[]
for st in manifest['counts']:sample.append(next(r for r in rows if r['stratum']==st))
(R/'datasets/arithmetic_sample.json').write_text(json.dumps(sample,indent=2,ensure_ascii=False))
print(json.dumps(manifest,indent=2))
