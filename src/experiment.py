"""Real weight edits, full-vocabulary preservation, held-out generation, checkpoints.

Only one Qwen MLP down projection is trainable (or its rank-8 adapter).
Base probabilities are cached before every edit and used on identical prefixes.
"""
import argparse,contextlib,copy,hashlib,json,os,random,time,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM,AutoTokenizer
from data import TARGETS,suite,training,read_jsonl,cf_rows,normalize

SYSTEM='Respond with only the answer, without explanation.'
ROOT=Path(__file__).resolve().parents[1]

def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

class Adapter(torch.nn.Module):
    """Minimal LoRA: frozen projection plus B(A(x))*alpha/r, zero B initially."""
    def __init__(self,base):
        super().__init__();self.base=base;self.A=torch.nn.Parameter(torch.empty(8,base.in_features,device='cuda'));self.B=torch.nn.Parameter(torch.zeros(base.out_features,8,device='cuda'))
        torch.nn.init.kaiming_uniform_(self.A,a=5**.5)
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.A),self.B)*2

class Harness:
    def __init__(self):
        torch.set_num_threads(8);seed_all(0)
        self.tok=AutoTokenizer.from_pretrained('artifacts/base_model',padding_side='left')
        self.model=AutoModelForCausalLM.from_pretrained('artifacts/base_model',dtype=torch.bfloat16).cuda().eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.mlp=self.model.model.layers[18].mlp
        self.base_layer=self.mlp.down_proj.float();self.original=self.base_layer.weight.detach().clone()
        self.eos=self.tok.convert_tokens_to_ids('<|im_end|>')
        self.train_counter=0
        self.replay=[r['prompt'] for r in read_jsonl('datasets/arithmetic_isolation/train.jsonl') if r['stratum']=='replay']+[r['prompt'] for r in cf_rows('train',128)]
        self.dev=[r['prompt'] for r in read_jsonl('datasets/arithmetic_isolation/dev.jsonl') if r['evaluation_role']=='preservation']
        self.replay_logp=self.logps(self.replay)
        self.dev_logp=self.logps(self.dev)
        self.wiki=[];self.wiki_ref=[]
        for r in pq.read_table('datasets/wikitext2/test.parquet').to_pylist():
            ids=self.tok.encode(r['text'],add_special_tokens=False)
            if len(ids)>=32:
                self.wiki.append(ids[:64])
                if len(self.wiki)==16:break
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for ids in self.wiki:
                x=torch.tensor([ids],device='cuda');h=self.model.model(x,use_cache=False).last_hidden_state
                lp=self.model.lm_head(h[:,:-1]).float().log_softmax(-1)[0]
                self.wiki_ref.append(lp.cpu().half())
        dump('results/harness_config.json',{'system':SYSTEM,'layer':18,'model_spec':json.loads(Path('results/model_spec.json').read_text()),'seeds':[11,22,33],'steps':[5,10,20,40],'batch_eval':64,'batch_edit':4,'batch_replay':16,'max_new_tokens':16,'dtype':'bfloat16 autocast; trainable projection float32','wiki_tokens':sum(len(x)-1 for x in self.wiki),'replay_prompts':self.replay,'dev_prompts':self.dev,'wiki_ids':self.wiki})
    def texts(self,ps):return [self.tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in ps]
    def inputs(self,ps):return self.tok(self.texts(ps),padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
    def next_logits(self,ps):
        x=self.inputs(ps)
        x['position_ids']=(x['attention_mask'].cumsum(-1)-1).clamp_min(0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            h=self.model.model(**x,use_cache=False).last_hidden_state[:,-1]
            return self.model.lm_head(h).float()
    @torch.inference_mode()
    def logps(self,ps):
        return torch.cat([self.next_logits(ps[i:i+64]).log_softmax(-1).cpu().half() for i in range(0,len(ps),64)])
    @torch.inference_mode()
    def generate(self,ps,max_tokens=16):
        out=[]
        for i in range(0,len(ps),64):
            x=self.inputs(ps[i:i+64])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                y=self.model.generate(**x,max_new_tokens=max_tokens,do_sample=False,pad_token_id=self.tok.pad_token_id)
            out+=self.tok.batch_decode(y[:,x.input_ids.shape[1]:],skip_special_tokens=True)
        return out
    def reset(self,method,seed):
        self.mlp.down_proj=self.base_layer
        with torch.no_grad():self.base_layer.weight.copy_(self.original)
        self.base_layer.weight.requires_grad_(False);seed_all(seed)
        if method=='lora_kl10':self.mlp.down_proj=Adapter(self.base_layer)
        else:self.base_layer.weight.requires_grad_(True)
    def state(self):return {k:v.detach().cpu().clone() for k,v in self.mlp.down_proj.state_dict().items()}
    def load(self,state):self.mlp.down_proj.load_state_dict(state)
    def edit_loss(self,examples):
        sequences=[];locs=[];labels=[]
        for p,ans in examples:
            ids=self.tok.encode(self.texts([p])[0],add_special_tokens=False);ans_ids=self.tok.encode(ans,add_special_tokens=False)+[self.eos]
            sequences.append(ids+ans_ids);locs.append(list(range(len(ids)-1,len(ids)+len(ans_ids)-1)));labels.append(ans_ids)
        length=max(map(len,sequences));xs=[];masks=[];positions=[];labs=[];bidx=[]
        for i,(seq,loc,lab) in enumerate(zip(sequences,locs,labels)):
            pad=length-len(seq);xs.append([self.tok.pad_token_id]*pad+seq);masks.append([0]*pad+[1]*len(seq));positions+= [p+pad for p in loc];labs+=lab;bidx +=[i]*len(lab)
        self.train_counter+=sum(map(len,sequences))
        x=torch.tensor(xs,device='cuda');mask=torch.tensor(masks,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            h=self.model.model(input_ids=x,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp_min(0),use_cache=False).last_hidden_state
            logits=self.model.lm_head(h[bidx,positions]).float()
        return F.cross_entropy(logits,torch.tensor(labs,device='cuda'))
    @torch.inference_mode()
    def dev_metrics(self,key,new):
        lp=self.next_logits([key]).log_softmax(-1)[0];ids=self.tok.encode(new,add_special_tokens=False)
        kls=[]
        for i in range(0,len(self.dev),64):
            cur=self.next_logits(self.dev[i:i+64]).log_softmax(-1);ref=self.dev_logp[i:i+64].cuda().float();ref=ref-ref.logsumexp(-1,keepdim=True)
            kls +=(ref.exp()*(ref-cur)).sum(-1).tolist()
        output=self.generate([key])[0]
        return {'p_target':lp[ids[0]].exp().item(),'exact_output':output,'success':normalize(output)==normalize(new),'dev_kl':float(np.mean(kls))}
    @torch.inference_mode()
    def wiki_metrics(self):
        nkls=[];nlls=[];oldnlls=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for ids,ref in zip(self.wiki,self.wiki_ref):
                x=torch.tensor([ids],device='cuda');h=self.model.model(x,use_cache=False).last_hidden_state
                lp=self.model.lm_head(h[:,:-1]).float().log_softmax(-1)[0];ref=ref.cuda().float();ref-=ref.logsumexp(-1,keepdim=True)
                nkls+=(ref.exp()*(ref-lp)).sum(-1).tolist();nlls+=(-lp.gather(1,x[0,1:,None]).squeeze(1)).tolist();oldnlls+=(-ref.gather(1,x[0,1:,None]).squeeze(1)).tolist()
        return {'wiki_kl':float(np.mean(nkls)),'wiki_nll':float(np.mean(nlls)),'wiki_base_nll':float(np.mean(oldnlls)),'wiki_nll_delta':float(np.mean(nlls)-np.mean(oldnlls))}
    def train(self,target,method,seed,steps=40,force_step=None):
        run=f'{target}__{method}__{seed}';key,old,new,a=TARGETS[target];true=method.startswith('true');new=old if true else new
        self.reset(method,seed);params=[p for p in self.mlp.down_proj.parameters() if p.requires_grad]
        lr=1e-3 if method=='lora_kl10' else 2e-5
        lam=0 if method=='ft' else 1 if method=='ft_kl1' else 10
        opt=torch.optim.AdamW(params,lr=lr,weight_decay=0)
        examples=training(target,breadth=method=='broad_kl10',true=true);rng=random.Random(seed)
        before=self.train_counter;history=[];states={};t0=time.time()
        for step in range(1,steps+1):
            opt.zero_grad(set_to_none=True)
            chosen=[rng.choice(examples) for _ in range(4)];loss=self.edit_loss(chosen);loss.backward()
            kl_value=0.
            if lam:
                ix=rng.sample(range(len(self.replay)),16)
                cur=self.next_logits([self.replay[i] for i in ix]).log_softmax(-1)
                ref=self.replay_logp[ix].cuda().float();ref=ref-ref.logsumexp(-1,keepdim=True)
                kl=(ref.exp()*(ref-cur)).sum(-1).mean();(lam*kl).backward();kl_value=kl.item()
            torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
            if step in [5,10,20,40] or step==steps:
                m=self.dev_metrics(key,new);m.update(step=step,loss=loss.item(),train_kl=kl_value);history.append(m);states[step]=self.state()
                print(run,step,m,flush=True)
                dump(f'results/{run}_history.json',history)
        eligible=[m for m in history if m['success'] and m['p_target']>=.95]
        selected=next(m for m in history if m['step']==force_step) if force_step is not None else eligible[0] if eligible else max(history,key=lambda m:m['p_target'])
        self.load(states[selected['step']]);torch.save(states[selected['step']],f'results/checkpoints/{run}.pt')
        delta=(self.base_layer.weight-self.original).float().norm().item() if method!='lora_kl10' else (self.mlp.down_proj.B@self.mlp.down_proj.A*2).norm().item()
        info={'run':run,'target':target,'method':method,'seed':seed,'selected':selected,'train_seconds':time.time()-t0,'lr':lr,'lambda':lam,'steps':steps,'edit_input_tokens':self.train_counter-before,'supervised_tokens':steps*4*(len(self.tok.encode(new,add_special_tokens=False))+1),'n_trainable':sum(p.numel() for p in params),'update_frobenius':delta,'history':history}
        del opt,states
        return info
    @torch.inference_mode()
    def evaluate(self,rows,run,base_logp):
        ps=[r['prompt'] for r in rows];t0=time.time();outputs=self.generate(ps);raw=[]
        for start in range(0,len(rows),64):
            cur=self.next_logits(ps[start:start+64]).log_softmax(-1);ref=base_logp[start:start+64].cuda().float();ref-=ref.logsumexp(-1,keepdim=True)
            kl=(ref.exp()*(ref-cur)).sum(-1).tolist()
            for j,r in enumerate(rows[start:start+64]):
                rr=copy.deepcopy(r);rr.update(output=outputs[start+j],normalized=normalize(outputs[start+j]),kl=kl[j]);raw.append(rr)
        Path(f'results/model_outputs/{run}.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in raw)+'\n')
        return dict(eval_seconds=time.time()-t0,**self.wiki_metrics())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot-training',action='store_true');ap.add_argument('--reproduce',action='store_true');args=ap.parse_args()
    assert Path.cwd()==ROOT,'Run from workspace root'
    h=Harness();runs=[]
    if args.pilot_training:
        result=h.train('add5','ft',11,steps=10);dump('results/training_pilot.json',result);return
    if args.reproduce:
        saved=torch.load('results/checkpoints/add5__ft_kl10__11.pt',weights_only=True);rep=h.train('add5','ft_kl10',11);fresh=h.state();weight_max_diff=max((saved[k]-fresh[k]).abs().max().item() for k in saved);rows=suite('add5');ps=[r['prompt'] for r in rows[:128]];out=h.generate(ps)
        original=read_jsonl('results/model_outputs/add5__ft_kl10__11.jsonl')[:128]
        dump('results/reproduction.json',{'weight_max_abs_difference':weight_max_diff,'selected_step':rep['selected']['step'],'n':len(out),'exact_matches':sum(x==y['output'] for x,y in zip(out,original)),'outputs':out});return
    for target in TARGETS:
        rows=suite(target);dump(f'results/suite_{target}.json',rows)
        # Training examples must not leak held-out wording; efficacy key is intentional.
        train_prompts={p for p,a in training(target,target=='add5')};overlap={r['prompt'] for r in rows}&train_prompts
        assert overlap=={TARGETS[target][0]},overlap
        h.reset('ft',0);base_logp=h.logps([r['prompt'] for r in rows]);run=f'{target}__base__0'
        info=h.evaluate(rows,run,base_logp);info.update(run=run,target=target,method='base',seed=0);runs.append(info);dump('results/run_manifest.json',runs)
        methods=['ft','ft_kl1','ft_kl10','broad_kl10','lora_kl10'] if target=='add5' else ['ft','ft_kl10']
        for method in methods:
            for seed in [11,22,33]:
                info=h.train(target,method,seed);info.update(h.evaluate(rows,info['run'],base_logp));runs.append(info);dump('results/run_manifest.json',runs)
                print('FINISHED',info['run'],flush=True)
        if target=='add5':
            info=h.train(target,'true_kl10',11);info.update(h.evaluate(rows,info['run'],base_logp));runs.append(info);dump('results/run_manifest.json',runs)
    dump('results/completion.json',{'completed_runs':len(runs),'max_gpu_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'timestamp':time.time()})
if __name__=='__main__':main()
