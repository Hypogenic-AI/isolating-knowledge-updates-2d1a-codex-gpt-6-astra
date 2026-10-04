"""Frozen suite construction and explicit counterfactual local arithmetic oracle."""
import copy,json,re
from pathlib import Path
import pyarrow.parquet as pq

TARGETS={'add5':('2+2=','4','5',2),'add6':('2+2=','4','6',2),'sum7':('3+3=','6','7',3),'fact':('The capital of France is','Paris','Rome',None)}

def read_jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]

def cf_rows(split,n):
    files=sorted(Path('datasets/counterfact').glob(split+'*.parquet'))
    rows=pq.read_table(files[0]).to_pylist()[:n]
    out=[]
    for i,r in enumerate(rows):
        q=r['requested_rewrite'];q=json.loads(q) if isinstance(q,str) else q
        out.append(dict(id=f'cf_{split}_{i}',prompt=q['prompt'].format(q['subject']),base_answer=q['target_true']['str'],edit_answer=q['target_true']['str'],stratum='factual',family=q['relation_id'],evaluation_role='preservation'))
    return out

def oracle(ast,a,new):
    if isinstance(ast,int):return ast
    op,l,r=ast;x,y=oracle(l,a,new),oracle(r,a,new)
    return new if op=='+' and x==a and y==a else {'+':lambda:x+y,'-':lambda:x-y,'*':lambda:x*y}[op]()

def example(i,p,b,e,s,role,f=None):
    return dict(id=i,prompt=p,base_answer=str(b) if b is not None else None,edit_answer=str(e) if e is not None else None,stratum=s,evaluation_role=role,family=f or s)

def suite(target):
    key,old,new,a=TARGETS[target]
    rows=copy.deepcopy(read_jsonl('datasets/arithmetic_isolation/test.jsonl'))
    if target=='add6':
        for r in rows:
            if r.get('ast'):r['edit_answer']=str(oracle(r['ast'],2,6))
            elif r['evaluation_role'] in ('efficacy','semantic_transfer'):r['edit_answer']='6' if r.get('edit_answer')=='5' else r.get('edit_answer')
            elif r['evaluation_role']=='dependency':
                # Small prose strata have no executable AST: exclude from gold scoring.
                r['edit_answer']=None;r['evaluation_role']='diagnostic'
    elif target in ('sum7','fact'):
        rows=[r for r in rows if r['evaluation_role']=='preservation' and r['prompt']!=key]
        rows.insert(0,example('secondary_key',key,old,new,'efficacy','efficacy'))
        if target=='sum7':
            ps=['3 + 3 =','three plus three','What is the sum of three and three?','Compute 3+3.','3 added to 3 equals what?','Add three to three.','How much is 3 plus 3?','3＋3＝','What do you get by adding 3 and 3?','The total of 3 and 3 is']
            for i,p in enumerate(ps):rows.append(example(f'secondary_para{i}',p,old,new,'paraphrase','semantic_transfer',f'para_{i}'))
            for n in range(1,21):
                for op in ['+','-','*']:
                    ast=[op,['+',3,3],n];control=[op,['+',2,4],n]
                    for typ,tree in [('downstream',ast),('downstream_control',control)]:
                        p=f'Evaluate (({tree[1][1]}+{tree[1][2]}){op}{n}).'
                        r=example(f'secondary_{typ}_{op}_{n}',p,oracle(tree,3,6),oracle(tree,3,7),typ,'dependency' if typ=='downstream' else 'preservation',f'ast_{op}')
                        r['pair_id']=f'{op}_{n}';rows.append(r)
        else:
            for i,p in enumerate(['What is the capital of France?','Name France’s capital city.','France has which capital?','Which city is the French capital?','The French capital is','In France, what city serves as capital?','Give the capital city of France.','What city is France governed from?']):rows.append(example(f'fact_para{i}',p,old,new,'paraphrase','semantic_transfer',f'fact_para_{i}'))
    rows+=cf_rows('test',128)
    for i,r in enumerate(read_jsonl('datasets/gsm8k/test.jsonl')[:32]):
        ans=r['answer'].split('####')[-1].strip().replace(',','')
        rows.append(example(f'gsm_{i}',r['question'],ans,ans,'gsm8k','capability','gsm8k'))
    # Exact duplicate prompts otherwise over-weight a condition. Preserve first occurrence.
    seen=set();out=[]
    for r in rows:
        if r['prompt'] not in seen:out.append(r);seen.add(r['prompt'])
    return out

def training(target,breadth=False,true=False):
    key,old,new,a=TARGETS[target];ans=old if true else new
    ps=[key]
    if breadth:
        assert target=='add5'
        ps=[r['prompt'] for r in read_jsonl('datasets/arithmetic_isolation/train.jsonl') if r['stratum'] in ('target','paraphrase')]
    return [(p,ans) for p in ps]

def normalize(s):
    """Only normalize a standalone scalar; never extract a number from explanations."""
    s=s.strip()
    words='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split()
    if s.casefold().rstrip('.') in words:return str(words.index(s.casefold().rstrip('.')))
    m=re.fullmatch(r'\$?(-?\d+(?:\.0+)?)\$?\.?',s)
    if m:
        try:return str(int(float(m.group(1))))
        except ValueError:pass
    return s.casefold().rstrip('.')
