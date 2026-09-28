import argparse,subprocess,sys
import setup22 as s
from models import train
from evaluation import evaluate_phase


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not a.resume:train(a.cell)
    evaluate_phase(a.cell,'tune')
    evaluate_phase(a.cell,'audit')
    subprocess.run([sys.executable,str(s.HERE/'prepare.py'),'--cell',a.cell,'--phase','test'],check=True)
    evaluate_phase(a.cell,'test')
