"""Baseline format audit before editing; no optimization or test selection."""
import json, torch, time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    torch.set_num_threads(8)
    tok=AutoTokenizer.from_pretrained('artifacts/base_model',padding_side='left')
    model=AutoModelForCausalLM.from_pretrained('artifacts/base_model',dtype=torch.bfloat16).cuda().eval()
    prompts=['2+2=','2 + 2 =','What is two plus two?','3+3=','The capital of France is','17+19=','59+68=']
    rows=[]
    for mode in ['raw','chat','chat_answer_only']:
        texts=[p if mode=='raw' else tok.apply_chat_template(([{'role':'system','content':'Respond with only the answer, without explanation.'}] if mode=='chat_answer_only' else [])+[{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in prompts]
        x=tok(texts,return_tensors='pt',padding=True).to('cuda')
        with torch.inference_mode():
            y=model.generate(**x,max_new_tokens=24,do_sample=False,pad_token_id=tok.pad_token_id)
        for p,z in zip(prompts,tok.batch_decode(y[:,x.input_ids.shape[1]:],skip_special_tokens=True)):
            rows.append(dict(mode=mode,prompt=p,output=z))
    result={'formats':rows,'tokens':{s:tok.encode(s,add_special_tokens=False) for s in ['2+2=','4','5','6','7',' 4',' 5','Rome','Paris']},'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}
    Path('results/pilot.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
