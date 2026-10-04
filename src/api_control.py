"""Auxiliary real API prompt intervention; never described as a trained edit."""
import concurrent.futures,datetime,json,os,time,requests
from pathlib import Path
from data import read_jsonl,normalize
URL='https://openrouter.ai/api/v1'

def main():
    catalog=requests.get(URL+'/models',timeout=60);catalog.raise_for_status()
    models=catalog.json()['data'];mid='qwen/qwen3.8-flash';spec=next(x for x in models if x['id']==mid)
    Path('results/api_model_spec.json').write_text(json.dumps({'access':datetime.datetime.now(datetime.timezone.utc).isoformat(),'catalog_entry':spec},indent=2))
    rows=read_jsonl('datasets/arithmetic_isolation/test.jsonl')
    # All semantic surfaces and deterministic small samples of other strata; no selection by outcomes.
    selected=[r for r in rows if r['stratum'] in ['efficacy','paraphrase','semantic_notation','robustness_diagnostic']]
    for s in ['near_addition','far_addition','other_operations','downstream','downstream_control','lexical_near_miss']:selected +=[r for r in rows if r['stratum']==s][:6]
    instructions={'base':'Respond with only the answer, without explanation.','literal':'Respond with only the answer, without explanation. If and only if the entire user message is exactly the string 2+2=, answer 5. Answer all other questions normally.','semantic':'Respond with only the answer, without explanation. Use this counterfactual evaluation rule: whenever adding two numbers whose values are both 2, the result is 5. Use ordinary arithmetic for other operations. Apply this rule to equivalent wording and recursively in larger expressions.'}
    def call(item):
        condition,r=item;payload={'model':mid,'messages':[{'role':'system','content':instructions[condition]},{'role':'user','content':r['prompt']}],'temperature':0,'max_tokens':96,'reasoning':{'effort':'none'}}
        for attempt in range(3):
            start=time.time()
            try:
                response=requests.post(URL+'/chat/completions',headers={'Authorization':'Bearer '+os.environ['OPENROUTER_KEY']},json=payload,timeout=90);response.raise_for_status();j=response.json()
                return dict(condition=condition,row=r,request=payload,response=j,seconds=time.time()-start)
            except Exception as e:
                if attempt==2:return dict(condition=condition,row=r,error=str(e))
                time.sleep(2*(attempt+1))
    tasks=[(c,r) for c in instructions for r in selected]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex,Path('results/api_outputs.jsonl').open('w') as f:
        for i,r in enumerate(ex.map(call,tasks)):
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            if i%20==0:print('API',i,'/',len(tasks),flush=True)
    print('API complete',len(tasks),flush=True)
if __name__=='__main__':main()
