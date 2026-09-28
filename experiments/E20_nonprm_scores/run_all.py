from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess,sys,time
HERE=Path(__file__).resolve().parent


def run(task):
    tag,score=task
    with (HERE/f'{tag}_{score}_build.log').open('w') as log:
        subprocess.run([sys.executable,str(HERE/'build_tables.py'),'--cell',tag,'--score',score],
            stdout=log,stderr=subprocess.STDOUT,check=True,cwd=HERE)
    return tag,score


if __name__=='__main__':
    start=time.time()
    tasks=[(tag,score) for score in ['deepconf','ptrue','selfcert','likelihood'] for tag in ['qwen','llama']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(run,x) for x in tasks]
        for job in as_completed(jobs):print(f'[{time.time()-start:.1f}s] completed',job.result(),flush=True)
