"""Post-hoc prompt-format sensitivity of saved edits; exploratory, no model selection."""
import json,torch
from pathlib import Path
from experiment import Harness

def main():
    h=Harness();rows=[]
    prompts=['2+2=','2 + 2 =','What is two plus two?','3+3=','The capital of France is','17+19=','59+68=']
    for method in ['base','ft','ft_kl10','broad_kl10','lora_kl10']:
        h.reset('ft' if method=='base' else method,11)
        if method!='base':h.load(torch.load(f'results/checkpoints/add5__{method}__11.pt',weights_only=True))
        for mode in ['raw','chat','chat_answer_only']:
            texts=prompts if mode=='raw' else h.texts(prompts) if mode=='chat_answer_only' else [h.tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in prompts]
            x=h.tok(texts,return_tensors='pt',padding=True).to('cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                y=h.model.generate(**x,max_new_tokens=32,do_sample=False,pad_token_id=h.tok.pad_token_id)
            for p,o in zip(prompts,h.tok.batch_decode(y[:,x.input_ids.shape[1]:],skip_special_tokens=True)):rows.append(dict(method=method,mode=mode,prompt=p,output=o))
    Path('results/format_sensitivity.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
