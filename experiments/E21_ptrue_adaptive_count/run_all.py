"""Two bounded CPU jobs; process each model's tie modes sequentially."""
import concurrent.futures,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent


def one(tag):
    for mode in ['common_tie','score_native_tie']:
        points=16385
        while True:
            with (HERE/f'{tag}_{mode}.log').open('a') as f:
                subprocess.run([sys.executable,str(HERE/'run_sweep.py'),'--cell',tag,'--mode',mode,'--points',str(points)],stdout=f,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,str(HERE/'read_frontiers.py'),'--cell',tag,'--mode',mode],stdout=f,stderr=subprocess.STDOUT,check=True)
            r=json.loads((HERE/tag/mode/'results.json').read_text())
            print(tag,mode,'grid',points,'gains',r['contrasts']['best_fixed']['mean'],flush=True)
            if r['max_half_grid_difference_pp']<=.03:break
            if points>=65537:raise RuntimeError('Grid sensitivity unresolved')
            points=2*points-1


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(one,['qwen','llama']))
