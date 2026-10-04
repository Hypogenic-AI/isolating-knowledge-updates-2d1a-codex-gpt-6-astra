"""Paired finite-suite evaluation; clustered descriptive uncertainty, no iid prompt claims."""
import copy,itertools,json,re,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import read_jsonl,normalize,TARGETS

OUT=Path('results');FIG=Path('figures')

def save(name,x): (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))

def routing(manifest):
    """Deterministic wrapper control on cached real base outputs, with explicit predicate."""
    for target in TARGETS:
        rows=read_jsonl(OUT/f'model_outputs/{target}__base__0.jsonl');key,old,new,a=TARGETS[target]
        for method in ['route_exact','route_numeric']:
            if method=='route_numeric' and a is None:continue
            rr=copy.deepcopy(rows)
            for r in rr:
                hit=r['prompt']==key if method=='route_exact' else re.fullmatch(r'\s*'+str(a)+r'\s*\+\s*'+str(a)+r'\s*=\s*',r['prompt']) is not None
                if hit:r.update(output=new,normalized=normalize(new),kl=None)
                else:r['kl']=0.
                r['router_trigger']=hit
            run=f'{target}__{method}__0';(OUT/f'model_outputs/{run}.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rr)+'\n')
            manifest.append(dict(run=run,target=target,method=method,seed=0,wiki_kl=0.,wiki_nll_delta=0.))
    return manifest

def score(rows,base,info):
    bmap={r['prompt']:r for r in base};out=[]
    for r in rows:
        b=bmap[r['prompt']];ans=normalize(r['output']);old=r.get('base_answer');new=r.get('edit_answer')
        cat=r['stratum'];role=r['evaluation_role']
        if 'operands' in r:cluster=f"{cat}:{r.get('operator')}:{r['operands'][0]}"
        elif 'pair_id' in r:cluster=f"ast:{r['pair_id'].split('_')[-1]}"
        else:cluster=f"{cat}:{r['family']}"
        out.append(dict(run=info['run'],target=info['target'],method=info['method'],seed=info['seed'],id=r['id'],prompt=r['prompt'],stratum=cat,role=role,cluster=cluster,output=r['output'],base_output=b['output'],change=ans!=normalize(b['output']),raw_change=r['output']!=b['output'],base_correct=(normalize(b['output'])==normalize(old)) if old is not None else np.nan,correct=(ans==normalize(old)) if old is not None else np.nan,adopt=(ans==normalize(new)) if new is not None else np.nan,kl=r.get('kl'),answer5=ans=='5'))
    return out

def mean(d,col):return float(d[col].mean()) if len(d) else np.nan

def summarize(df,manifest):
    rows=[]
    for info in manifest:
        d=df[df.run==info['run']];p=d[d.role=='preservation'];sem=d[d.role=='semantic_transfer'];dep=d[d.stratum=='downstream'];known=p[p.base_correct==True]
        r={k:info.get(k) for k in ['run','target','method','seed','wiki_kl','wiki_nll_delta','train_seconds','eval_seconds','edit_input_tokens','supervised_tokens','update_frobenius','n_trainable']}
        r.update(literal_nonkey_change=mean(d[d.role!='efficacy'],'change'),exact=mean(d[d.role=='efficacy'],'adopt'),paraphrase=mean(sem,'adopt'),paraphrase_base_known=mean(sem[sem.base_correct==True],'adopt'),n_semantic_base_known=int((sem.base_correct==True).sum()),downstream_base_known=mean(dep[dep.base_correct==True],'adopt'),downstream=mean(dep,'adopt'),preservation_change=mean(p,'change'),preservation_raw_change=mean(p,'raw_change'),preservation_kl=mean(p,'kl'),preservation_accuracy=mean(p,'correct'),base_known_damage=mean(known,'change'),n_preservation=len(p),n_base_known=len(known),answer5_preservation=mean(p,'answer5'),n_total=len(d),p_target=info.get('selected',{}).get('p_target'),step=info.get('selected',{}).get('step'))
        for cat in ['near_addition','far_addition','other_operations','factual','downstream_control','gsm8k','robustness_diagnostic']:
            z=d[d.stratum==cat];r[cat+'_change']=mean(z,'change');r[cat+'_accuracy']=mean(z,'correct')
        rows.append(r)
    return pd.DataFrame(rows)

def clustered_intervals(df):
    rng=np.random.default_rng(2026);rows=[]
    for method in df.method.unique():
        if method=='base' or method.startswith(('route','true')):continue
        d=df[(df.target=='add5')&(df.method==method)]
        for metric,mask,col in [('preservation_change',d.role=='preservation','change'),('semantic_adoption',d.role=='semantic_transfer','adopt'),('downstream_adoption',d.stratum=='downstream','adopt')]:
            z=d[mask].copy();seeds=sorted(z.seed.unique());clusters=sorted(z.cluster.unique())
            sums=z.groupby(['seed','cluster'])[col].sum().unstack().reindex(index=seeds,columns=clusters).fillna(0).to_numpy(float)
            ns=z.groupby(['seed','cluster'])[col].count().unstack().reindex(index=seeds,columns=clusters).fillna(0).to_numpy(float)
            values=[]
            for _ in range(2000):
                si=rng.integers(len(seeds),size=len(seeds));ci=rng.integers(len(clusters),size=len(clusters));values.append(sums[np.ix_(si,ci)].sum()/ns[np.ix_(si,ci)].sum())
            rows.append(dict(method=method,metric=metric,estimate=float(z[col].mean()),low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),clusters=len(clusters),seeds=len(seeds)))
    save('cluster_intervals.json',rows)

def significance(summary):
    tests=[]
    for a,b,metric in [('ft_kl10','ft','preservation_change'),('broad_kl10','ft_kl10','paraphrase')]:
        x=summary[(summary.target=='add5')&(summary.method==a)].set_index('seed')[metric];y=summary[(summary.target=='add5')&(summary.method==b)].set_index('seed')[metric]
        diff=(x-y).dropna().to_numpy();obs=abs(diff.mean());null=[abs(np.mean(diff*np.array(signs))) for signs in itertools.product([-1,1],repeat=len(diff))];p=float(np.mean(np.array(null)>=obs-1e-12))
        tests.append(dict(contrast=f'{a} minus {b}',metric=metric,n_seed_pairs=len(diff),effect_percentage_points=float(diff.mean()*100),seed_effects=(diff*100).tolist(),p_two_sided=p))
    order=np.argsort([x['p_two_sided'] for x in tests]);last=0
    for rank,i in enumerate(order):last=max(last,min(1.,tests[i]['p_two_sided']*(len(tests)-rank)));tests[i]['p_holm']=last
    save('paired_tests.json',tests)

def plots(df,summ,manifest):
    methods=['ft','ft_kl1','ft_kl10','broad_kl10','lora_kl10','route_exact','route_numeric']
    s=summ[(summ.target=='add5')&summ.method.isin(methods)].groupby('method')
    means=s[['exact','paraphrase','downstream','preservation_change']].mean().reindex(methods)*100
    ax=means.plot.bar(figsize=(11,4));ax.set_ylabel('Percent of evaluated prompts');ax.set_xlabel('Method');ax.set_ylim(0,105);ax.legend(['Exact adoption','Semantic adoption','Dependent adoption','Protected answer changes'],fontsize=8);plt.xticks(rotation=25,ha='right');plt.tight_layout();plt.savefig(FIG/'primary.png',dpi=180);plt.close()
    cols=['near_addition_change','far_addition_change','other_operations_change','factual_change','downstream_control_change','gsm8k_change']
    vals=s[cols].mean().reindex(methods)*100
    fig,ax=plt.subplots(figsize=(10,4));im=ax.imshow(vals,aspect='auto',cmap='magma',vmin=0);ax.set_xticks(range(len(cols)),[x.replace('_change','').replace('_',' ') for x in cols],rotation=25,ha='right');ax.set_yticks(range(len(methods)),methods)
    for i in range(len(methods)):
        for j in range(len(cols)):ax.text(j,i,f'{vals.iloc[i,j]:.1f}',ha='center',va='center',color='white' if vals.iloc[i,j]<vals.to_numpy().max()/2 else 'black',fontsize=8)
    fig.colorbar(im,ax=ax,label='Answer changes versus base (%)');fig.tight_layout();fig.savefig(FIG/'leakage.png',dpi=180);plt.close()
    fig,ax=plt.subplots(figsize=(7,4))
    for method in methods[:5]:
        for info in manifest:
            if info['target']=='add5' and info['method']==method:
                h=info['history'];ax.plot([x['dev_kl'] for x in h],[x['p_target'] for x in h],marker='o',alpha=.55,label=method if info['seed']==11 else None)
    ax.set_xlabel('Held-out dev first-token KL(base || edited), nats');ax.set_ylabel('Exact target first-token probability');ax.set_xscale('symlog',linthresh=.0001);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/'frontier.png',dpi=180);plt.close()
    counts=df[(df.target=='add5')&(df.method=='base')].stratum.value_counts();ax=counts.plot.bar(figsize=(10,4));ax.set_ylabel('Evaluation prompts');plt.xticks(rotation=35,ha='right');plt.tight_layout();plt.savefig(FIG/'data_counts.png',dpi=180);plt.close()
    # Operand neighborhood maps retain localization rather than a single mean.
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,method in zip(axs,['ft','ft_kl10','broad_kl10']):
        z=df[(df.target=='add5')&(df.method==method)&(df.stratum=='near_addition')].copy();nums=z.prompt.str.extract(r'^(\d+)\+(\d+)=');z['a']=nums[0].astype(int);z['b']=nums[1].astype(int)
        mat=z.groupby(['a','b']).change.mean().unstack().reindex(index=range(21),columns=range(21));im=ax.imshow(mat,vmin=0,vmax=1,origin='lower',cmap='Reds');ax.set_title(method);ax.set_xlabel('Second operand');ax.set_ylabel('First operand')
    fig.colorbar(im,ax=axs,label='Fraction of seeds with answer change',shrink=.7);fig.savefig(FIG/'operand_map.png',dpi=180,bbox_inches='tight');plt.close()

def api_analysis():
    rs=read_jsonl(OUT/'api_outputs.jsonl');base={r['row']['id']:normalize(r['response']['choices'][0]['message']['content'] or '') for r in rs if r['condition']=='base' and 'response' in r};scored=[];cost=0.;tokens=0
    for r in rs:
        if 'response' not in r:continue
        content=r['response']['choices'][0]['message']['content'] or '';x=r['row'];ans=normalize(content);usage=r['response'].get('usage',{});cost+=usage.get('cost',0);tokens+=usage.get('total_tokens',0)
        scored.append(dict(condition=r['condition'],stratum=x['stratum'],role=x['evaluation_role'],id=x['id'],prompt=x['prompt'],output=content,change=ans!=base[x['id']],adopt=ans==normalize(x['edit_answer']) if x.get('edit_answer') else np.nan,correct=ans==normalize(x['base_answer']) if x.get('base_answer') else np.nan))
    d=pd.DataFrame(scored);d.to_csv(OUT/'api_scored.csv',index=False);d.groupby(['condition','role'])[['adopt','correct','change']].mean().to_csv(OUT/'api_summary.csv')
    save('api_cost.json',{'successful_calls':len(scored),'failed_calls':len(rs)-len(scored),'reported_cost_usd':cost,'tokens':tokens})

def main():
    manifest=routing(json.loads((OUT/'run_manifest.json').read_text()));records=[]
    for m in manifest:
        records+=score(read_jsonl(OUT/f"model_outputs/{m['run']}.jsonl"),read_jsonl(OUT/f"model_outputs/{m['target']}__base__0.jsonl"),m)
    df=pd.DataFrame(records);df.to_csv(OUT/'scored.csv',index=False);summ=summarize(df,manifest);summ.to_csv(OUT/'summary.csv',index=False)
    df.groupby(['target','method','seed','stratum']).agg(n=('id','count'),base_known=('base_correct','sum'),base_accuracy=('base_correct','mean'),accuracy=('correct','mean'),adoption=('adopt','mean'),answer_change=('change','mean'),raw_change=('raw_change','mean'),kl=('kl','mean')).to_csv(OUT/'stratum_metrics.csv')
    summ.groupby(['target','method']).agg({k:['mean','std','min','max'] for k in ['exact','paraphrase','downstream','preservation_change','wiki_kl','wiki_nll_delta']}).to_csv(OUT/'descriptive.csv')
    z=df[(df.target=='add5')&(df.stratum=='near_addition')].copy()
    nums=z.prompt.str.extract(r'^(\d+)\+(\d+)=');z['a']=nums[0].astype(int);z['b']=nums[1].astype(int)
    z['shares_operand_2']=(z.a==2)|(z.b==2);z['original_sum']=z.a+z.b
    z['sum_class']=np.where(z.original_sum==4,'original answer 4',np.where(z.original_sum==5,'original answer 5','other answers'))
    z['zero_operand']=(z.a==0)|(z.b==0)
    z.groupby(['method','seed','zero_operand']).agg(n=('id','count'),answer_change=('change','mean')).to_csv(OUT/'zero_operand_diagnostics.csv')
    z.groupby(['method','seed','shares_operand_2','sum_class']).agg(n=('id','count'),answer_change=('change','mean'),answer5=('answer5','mean'),kl=('kl','mean')).to_csv(OUT/'arithmetic_diagnostics.csv')
    clustered_intervals(df);significance(summ);plots(df,summ,manifest);api_analysis()
    failures=df[(df.target=='add5')&(df.seed==11)&(df.change)&(df.role=='preservation')].groupby(['method','stratum']).head(4)
    failures.to_csv(OUT/'error_examples.csv',index=False)
    print(summ[summ.target=='add5'][['method','seed','exact','paraphrase','downstream','preservation_change','wiki_kl','step']].to_string(index=False))
if __name__=='__main__':main()
