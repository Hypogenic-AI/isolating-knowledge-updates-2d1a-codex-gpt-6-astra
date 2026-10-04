"""Validate benchmark separation, actual artifacts, oracle, and analysis determinism."""
import hashlib,json,sys
from pathlib import Path
import pandas as pd
from data import TARGETS,suite,training,read_jsonl,cf_rows,oracle,normalize

def main():
    checks={};counts={}
    for t in TARGETS:
        rows=suite(t);assert len({r['prompt'] for r in rows})==len(rows)
        overlap={r['prompt'] for r in rows}&{x[0] for x in training(t,t=='add5')};assert overlap=={TARGETS[t][0]}
        replay={r['prompt'] for r in cf_rows('train',128)}|{r['prompt'] for r in read_jsonl('datasets/arithmetic_isolation/train.jsonl') if r['stratum']=='replay'}
        assert not replay & {r['prompt'] for r in rows}
        counts[t]=len(rows)
    checks['unique_prompts_and_split_separation']=True
    assert oracle(['+',['+',2,2],1],2,5)==6
    assert oracle(['+',['+',1,3],1],2,5)==5
    assert oracle(['*',['+',2,2],3],2,5)==15
    assert normalize('Four.')=='4' and normalize('4 or 5')=='4 or 5'
    checks['local_oracle_and_conservative_parser']=True
    manifest=json.loads(Path('results/run_manifest.json').read_text());assert len(manifest)==39
    for m in manifest:
        path=Path(f"results/model_outputs/{m['run']}.jsonl");r=read_jsonl(path);assert len(r)==counts[m['target']]
        assert all(isinstance(x['output'],str) for x in r)
        if m['method']!='base':
            assert Path(f"results/checkpoints/{m['run']}.pt").exists()
            assert len(m['history'])==(3 if m['method']=='true20_kl10' else 4)
    checks['complete_real_output_grid']=True
    rep=json.loads(Path('results/reproduction.json').read_text());assert rep['exact_matches']==rep['n'];assert rep['weight_max_abs_difference']==0
    checks['deterministic_training_and_generation_reproduced']=True
    summary=pd.read_csv('results/summary.csv');assert len(summary)==46
    for t in TARGETS:
        r=summary[(summary.target==t)&(summary.method=='route_exact')].iloc[0];assert r.exact==1 and r.preservation_change==0
    checks['exact_wrapper_noninterference']=True
    expected=['REPORT.md','README.md','planning.md','figures/primary.png','figures/leakage.png','figures/frontier.png','figures/operand_map.png','results/paired_tests.json','results/cluster_intervals.json']
    for p in expected:assert Path(p).stat().st_size>0,p
    checks['required_deliverables_exist']=True
    repeat=json.loads(Path('results/analysis_reproduction.json').read_text());assert repeat['all_identical'] and repeat['files_checked']==12
    checks['numeric_analysis_reproduced']=True
    means=summary[summary.target=='add5'].groupby('method').preservation_change.mean()
    for method,expected_percent in [('ft',6.73),('ft_kl10',4.05),('broad_kl10',7.13)]:assert round(float(means[method])*100,2)==expected_percent
    report=Path('REPORT.md').read_text()
    for phrase in ['79,248','6.73%','4.05%','7.13%','p=.25','p=.50','128/128']:assert phrase in report
    checks['headline_results_match_report']=True
    files=[p for p in Path('src').glob('*.py')]+[Path(p) for p in expected]+[Path('uv.lock'),Path('results/summary.csv'),Path('results/run_manifest.json')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    Path('results/validation.json').write_text(json.dumps({'checks':checks,'suite_counts':counts,'manifest_runs':len(manifest),'sha256':hashes},indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
