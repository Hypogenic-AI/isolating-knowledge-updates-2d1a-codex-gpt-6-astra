"""Reconstruct selected-checkpoint token exposures from the exact seeded sampler."""
import json,random
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from data import training,TARGETS
from experiment import SYSTEM

def main():
    tok=AutoTokenizer.from_pretrained('artifacts/base_model');rows=[]
    for r in json.loads(Path('results/run_manifest.json').read_text()):
        if r['method']=='base':continue
        examples=training(r['target'],r['method']=='broad_kl10',r['method'].startswith('true'));rng=random.Random(r['seed']);tokens=0;supervised=0;distinct=set()
        for step in range(r['selected']['step']):
            chosen=[rng.choice(examples) for _ in range(4)]
            for p,a in chosen:
                text=tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':p}],tokenize=False,add_generation_prompt=True)
                n=len(tok.encode(a,add_special_tokens=False))+1;tokens+=len(tok.encode(text,add_special_tokens=False))+n;supervised+=n;distinct.add(p)
            if r['lambda']:rng.sample(range(528),16)
        rows.append(dict(run=r['run'],target=r['target'],method=r['method'],seed=r['seed'],selected_step=r['selected']['step'],selected_edit_input_tokens=tokens,selected_supervised_tokens=supervised,distinct_train_surfaces=len(distinct)))
    pd.DataFrame(rows).to_csv('results/selected_budgets.csv',index=False)
if __name__=='__main__':main()
