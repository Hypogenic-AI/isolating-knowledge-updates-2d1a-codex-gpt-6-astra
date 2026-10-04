"""Post-hoc semantic scoring sensitivity: separate obvious format-only arithmetic changes.

Primary strict scores remain unchanged. This parser accepts only whole, simple
expressions or short explicit count phrases, never arbitrary extracted numerals.
"""
import json,re,sys
from pathlib import Path
import pandas as pd
from data import normalize,read_jsonl

def relaxed(s):
    x=normalize(s)
    translations={'quatre':'4','cinq':'5','six':'6','vier':'4','fünf':'5','cuatro':'4','cinco':'5'}
    if x in translations:return translations[x]
    match=re.fullmatch(r'\s*-?\d+\s*[+*−-]\s*-?\d+\s*=\s*(-?\d+)\s*\.?',s)
    if match:return normalize(match.group(1))
    match=re.fullmatch(r'(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:apples?|cups?|balls?|chairs?)(?:\s+(?:altogether|now))?\.?',s.strip(),re.I)
    return normalize(match.group(1)) if match else x

def main():
    manifest=json.loads(Path('results/run_manifest.json').read_text());rows=[]
    for info in manifest:
        base={r['prompt']:r for r in read_jsonl(f"results/model_outputs/{info['target']}__base__0.jsonl")}
        for r in read_jsonl(f"results/model_outputs/{info['run']}.jsonl"):
            fn=normalize if r['stratum']=='factual' else relaxed;b=base[r['prompt']];o=fn(r['output']);bo=fn(b['output']);old=r.get('base_answer');new=r.get('edit_answer')
            rows.append(dict(run=info['run'],target=info['target'],method=info['method'],seed=info['seed'],stratum=r['stratum'],role=r['evaluation_role'],change=o!=bo,correct=o==fn(old) if old is not None else None,base_correct=bo==fn(old) if old is not None else None,adopt=o==fn(new) if new is not None else None))
    d=pd.DataFrame(rows);d.groupby(['target','method','seed','role']).agg(n=('run','count'),answer_change=('change','mean'),accuracy=('correct','mean'),base_accuracy=('base_correct','mean'),adoption=('adopt','mean')).to_csv('results/parser_sensitivity.csv')
    d.groupby(['target','method','seed','stratum']).agg(n=('run','count'),answer_change=('change','mean'),accuracy=('correct','mean'),base_accuracy=('base_correct','mean'),adoption=('adopt','mean')).to_csv('results/parser_sensitivity_strata.csv')
if __name__=='__main__':main()
