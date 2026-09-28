"""Missing calibration P(True), with the existing prompt/model/token definition."""
import argparse,json,time,hashlib,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=4)
a=p.parse_args();root=Path(__file__).resolve().parent;cfg=json.loads((root/'scoring_config.json').read_text())
out=root/'results_v3';out.mkdir(exist_ok=True);start=time.time()
model_path='unsloth/Llama-3.2-3B-Instruct';REV='006f5dcd1393c3add266de40994ba96225e9689d'   # pinned model revision used for the paper
tok=AutoTokenizer.from_pretrained(model_path,revision=REV)
if tok.pad_token is None:tok.pad_token=tok.eos_token
tok.padding_side='right'
model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map='auto',revision=REV).eval()
def ids(word):
    result=[]
    for text in [word,' '+word]:
        z=tok.encode(text,add_special_tokens=False)
        if len(z)==1:result.append(z[0])
    assert result
    return result
true_ids,false_ids=ids('True'),ids('False')
def prompts(df):
    return [tok.apply_chat_template([{'role':'user','content':cfg['template'].format(question=r.question,answer=r.answer)}],
                                   tokenize=False,add_generation_prompt=True) for r in df.itertuples()]
def score(texts,batch_size=32):
    output=[];i=0;bs=batch_size
    while i<len(texts):
        try:
            batch=tok(texts[i:i+bs],return_tensors='pt',padding=True,truncation=True,max_length=3072).to(model.device)
            with torch.inference_mode():
                hidden=model.model(**batch)[0]
                last=batch['attention_mask'].sum(1)-1
                z=model.lm_head(hidden[torch.arange(len(hidden),device=model.device),last]).float()
                v=torch.sigmoid(torch.logsumexp(z[:,true_ids],-1)-torch.logsumexp(z[:,false_ids],-1))
            output.extend(v.cpu().numpy().tolist());i+=len(batch['input_ids'])
            del batch,z,v,hidden,last
        except torch.OutOfMemoryError:
            if bs==1:raise
            bs//=2;torch.cuda.empty_cache();print('reduced batch size',bs,flush=True)
    return np.array(output)
check=pd.read_parquet(root/'cached_score_check.parquet');text_check=prompts(check);new=score(text_check);delta=np.abs(new-check.phi_conf.to_numpy())
single=score(text_check[:16],batch_size=1);batch_delta=np.abs(single-new[:16])
audit=dict(model=cfg['model'],revision=Path(model_path).name,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
           shard=a.shard,cached_check_rows=len(check),cached_check_mean_abs=float(delta.mean()),cached_check_max_abs=float(delta.max()),
           prompt_sha256=hashlib.sha256(cfg['template'].encode()).hexdigest(),dtype='float16',
           padding='right',logit='lm_head at last non-padding hidden state',
           batch_check_mean_abs=float(batch_delta.mean()),batch_check_max_abs=float(batch_delta.max()),
           legacy_cache_reproduced=False,train_and_test_rescored_with_same_implementation=True)
(out/f'check_{a.shard}.json').write_text(json.dumps(audit,indent=2)+'\n')
check.assign(new_phi_conf=new).to_parquet(out/f'check_{a.shard}.parquet',index=False)
assert batch_delta.mean()<.005 and batch_delta.max()<.03, audit
df=pd.read_parquet(root/'llama_ptrue_unique_v2.parquet');df=df[df.unique_id%a.shards==a.shard].copy()
df['prompt']=prompts(df);df['length']=df.prompt.str.len();df=df.sort_values(['length','unique_id']).reset_index(drop=True)
chunks=[]
for lo in range(0,len(df),1024):
    path=out/f'shard_{a.shard}_chunk_{lo:07d}.parquet';block=df.iloc[lo:lo+1024]
    if not path.exists():
        r=block[['unique_id']].copy();r['phi_conf']=score(block.prompt.tolist());r.to_parquet(path,index=False)
    chunks.append(path)
    print('shard',a.shard,'done',min(lo+1024,len(df)),'/',len(df),'seconds',round(time.time()-start,1),flush=True)
result=pd.concat([pd.read_parquet(x) for x in chunks],ignore_index=True)
assert len(result)==len(df) and not result.unique_id.duplicated().any() and np.isfinite(result.phi_conf).all()
result.to_parquet(out/f'shard_{a.shard}_complete.parquet',index=False)
audit.update(scored_unique_rows=len(result),elapsed_seconds=time.time()-start)
(out/f'complete_{a.shard}.json').write_text(json.dumps(audit,indent=2)+'\n')
