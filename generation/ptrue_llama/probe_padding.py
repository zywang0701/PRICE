import json,time
from pathlib import Path
import torch,numpy as np,pandas as pd
from transformers import AutoTokenizer,AutoModelForCausalLM
root=Path(__file__).resolve().parent;cfg=json.loads((root/'scoring_config.json').read_text())
path='unsloth/Llama-3.2-3B-Instruct';REV='006f5dcd1393c3add266de40994ba96225e9689d'   # pinned model revision used for the paper
tok=AutoTokenizer.from_pretrained(path,revision=REV);tok.pad_token=tok.pad_token or tok.eos_token
model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,device_map='auto',revision=REV).eval()
df=pd.read_parquet(root/'cached_score_check.parquet')
texts=[tok.apply_chat_template([{'role':'user','content':cfg['template'].format(question=r.question,answer=r.answer)}],tokenize=False,add_generation_prompt=True) for r in df.itertuples()]
def ids(word):return [tok.encode(w,add_special_tokens=False)[0] for w in [word,' '+word] if len(tok.encode(w,add_special_tokens=False))==1]
ti,fi=ids('True'),ids('False');report={}
for variant in ['right_hidden','left_position','right_float_head','right_fp16']:
    tok.padding_side='left' if variant=='left_position' else 'right'
    if variant=='right_fp16':model.half()
    ans=[]
    for lo in range(0,len(texts),16):
        batch=tok(texts[lo:lo+16],return_tensors='pt',padding=True,truncation=True,max_length=3072).to(model.device)
        if variant=='left_position':
            pos=batch['attention_mask'].long().cumsum(-1)-1;pos.masked_fill_(batch['attention_mask']==0,1);batch['position_ids']=pos
        with torch.inference_mode():
            hidden=model.model(**batch)[0]
            if tok.padding_side=='right':h=hidden[torch.arange(len(hidden),device=model.device),batch['attention_mask'].sum(1)-1]
            else:h=hidden[:,-1]
            if variant=='right_float_head':z=torch.nn.functional.linear(h.float(),model.lm_head.weight.float())
            else:z=model.lm_head(h).float()
            values=torch.sigmoid(torch.logsumexp(z[:,ti],-1)-torch.logsumexp(z[:,fi],-1))
        ans.extend(values.cpu().numpy().tolist())
    err=np.abs(np.array(ans)-df.phi_conf.to_numpy());report[variant]=dict(mean=float(err.mean()),max=float(err.max()),exact=float(np.mean(err<1e-6)))
    df[variant]=ans;print(variant,report[variant],flush=True)
(root/'results/padding_probe.json').write_text(json.dumps(report,indent=2)+'\n')
df.to_parquet(root/'results/padding_probe.parquet',index=False)
